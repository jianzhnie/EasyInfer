"""Fix zero expert handling on Ascend NPU for LongCat-Flash with EP (vllm >= 0.23).

Problem 1: vllm-ascend's native zero-expert path never runs
------------------------------------------------------------
vllm 0.23 moved the zero-expert config onto ``ZeroExpertRouter``, so
``AscendUnquantizedFusedMoEMethod.apply``'s gate
``getattr(layer, "zero_expert_num", 0) > 0`` is always False.  Without it,
top-k ids in [N, N+Z) (zero experts) reach the dispatch kernel -> aicore crash.

Problem 2: MC2 MoE comm is incompatible with zero-expert weight zeroing
-----------------------------------------------------------------------
``npu_moe_distribute_dispatch_v2`` drops zero-weight slots, so
``MoeDistributeCombineV2``'s shape check (expandX.dim0 >= tokens*topk)
fails, the op never launches, and later collectives hang (AllGather
timeout).  The ALLGATHER comm method computes MoE locally after a gather,
where zero-weight slots are harmless (same semantics as the GPU path).

Fix
---
0b. Mirror ``ZeroExpertRouter`` config onto ``AscendFusedMoE`` so the native
    zero-expert path in ``apply`` runs (id sanitization + result add);
    teach ``FusedExpertsResult`` ``+=`` for the native add.
0c. Optionally force the ALLGATHER MoE comm method
    (``EASYINFER_MOE_COMM=allgather``) to bypass MC2.
3.  Safety net in ``_maybe_add_zero_expert_output`` (the runner asserts on
    ``router.zero_expert_output``; inject a scalar zero as a no-op add,
    which also avoids double-counting the native contribution).
"""

from __future__ import annotations

import os
import torch
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch

# ===========================================================================
# Patch 0b: enable vllm-ascend's NATIVE zero-expert handling (>= 0.23)
# ===========================================================================
# vllm 0.23 stores zero-expert config on the ZeroExpertRouter, so
# ``AscendUnquantizedFusedMoEMethod.apply``'s gate
# ``getattr(layer, "zero_expert_num", 0) > 0`` is always False and the
# native path (id sanitization + zero_expert_result add) never runs.
# Re-enable it by mirroring the router's config onto the layer.


@register_patch(target="vllm_ascend.ops.fused_moe.fused_moe_0_23_0")
def patch_enable_native_zero_expert(module: object) -> None:
    _orig_init = module.AscendFusedMoE.__init__

    # ``apply`` adds the zero-expert result onto the value returned by
    # ``fused_experts``; in this version that value is a
    # ``FusedExpertsResult`` dataclass, not a tensor.  Teach it ``+=``
    # so the native add path works.
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

    def _fused_experts_result_iadd(self, other):
        self.routed_out += other
        return self

    FusedExpertsResult.__iadd__ = _fused_experts_result_iadd

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        router = getattr(self, "router", None)
        if (
            isinstance(router, ZeroExpertRouter)
            and router.zero_expert_type is not None
        ):
            # Zero-expert count is not stored on the router; derive it
            # from the routing bias width (real + zero) minus logical.
            n_zero = 0
            bias = getattr(router, "e_score_correction_bias", None)
            if bias is not None:
                n_zero = bias.shape[-1] - router.num_logical_experts
            self.zero_expert_num = max(n_zero, 1)
            self.zero_expert_type = router.zero_expert_type
            patch_logger.info(
                f"[fix_ep_zero_expert] Enabled native zero-expert path: "
                f"num={self.zero_expert_num}, type={self.zero_expert_type}"
            )

    module.AscendFusedMoE.__init__ = _init


# ===========================================================================
# Patch 0c: force ALLGATHER MoE comm (EASYINFER_MOE_COMM=allgather)
# ===========================================================================
# MC2 dispatch drops zero-weight (clamped zero-expert) slots, so the
# combine kernel's shape check ``expandX.dim0 >= tokens*topk`` fails and
# the op never launches, corrupting the stream and hanging later
# collectives.  The ALLGATHER comm method computes MoE locally after a
# gather, where zero-weight slots are harmless (same semantics as GPU).


@register_patch(target="vllm_ascend.ascend_forward_context")
def patch_force_allgather_moe_comm(module: object) -> None:
    if os.environ.get("EASYINFER_MOE_COMM", "").lower() != "allgather":
        return
    _orig = module.select_moe_comm_method
    _logged = False

    def _select(num_tokens, vllm_config, is_draft_model=False):
        nonlocal _logged
        selected = _orig(num_tokens, vllm_config, is_draft_model)
        if selected is not None:
            if not _logged:
                _logged = True
                patch_logger.info(
                    f"[fix_ep_zero_expert] MoE comm method overridden: "
                    f"{selected} -> ALLGATHER"
                )
            return module.MoECommType.ALLGATHER
        return selected

    module.select_moe_comm_method = _select


# ===========================================================================
# Patch 3: MoERunner._maybe_add_zero_expert_output — safety net
# ===========================================================================


@register_patch(target="vllm.model_executor.layers.fused_moe.runner.moe_runner")
def patch_moe_runner_zero_expert(module: object) -> None:
    MoERunner = module.MoERunner
    _orig_maybe = MoERunner._maybe_add_zero_expert_output

    # NOTE: on vllm-ascend >= 0.23 the zero-expert contribution is added
    # natively inside ``apply`` (enabled by ``patch_enable_native_zero_expert``).
    # The runner still asserts on ``router.zero_expert_output``, so inject a
    # scalar zero (no-op add) when it was never computed.
    def _maybe(self, result):
        if (
            isinstance(self.router, ZeroExpertRouter)
            and self.router._zero_expert_output is None
            and self.router.zero_expert_type is not None
        ):
            self.router._zero_expert_output = torch.tensor(
                0.0, device=result.device, dtype=result.dtype
            )
        return _orig_maybe(self, result)

    MoERunner._maybe_add_zero_expert_output = _maybe
    patch_logger.info(
        "[fix_ep_zero_expert] Patched MoERunner._maybe_add_zero_expert_output"
    )
