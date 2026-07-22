"""Fix zero expert handling and MLP compute on Ascend NPU for LongCat-Flash with EP.

Problem 1: Missing zero-expert output (AssertionError)
-------------------------------------------------------
``AscendFusedMoE`` never calls ``ZeroExpertRouter._compute_routing``,
so ``_zero_expert_output`` stays ``None`` and the runner asserts.

Problem 2: Out-of-bounds expert indices in EP dispatch (aicore crash)
----------------------------------------------------------------------
``ascend_select_experts`` returns top-k ids in [0, N+Z) where N=512
and Z=256 (zero).  The EP dispatch kernel expects ids in [0, 255]
(local experts).  Zero-expert ids cause out-of-bounds NPU access.

Problem 3: Large expert group MLP kernel crash
-----------------------------------------------
With 256 local experts per EP rank, the ``npu_grouped_matmul`` kernel
in ``unquant_apply_mlp`` may trigger ``fftsplus aivector error``.
Splitting experts into smaller chunks (max 64 per call) avoids this.

Fix
---
0. Patch ``unquant_apply_mlp`` to chunk experts (max 64 per call).
1. Patch ``TokenDispatcherWithMC2.token_dispatch`` to sanitize ids.
2. Patch ``MoERunner.forward`` to pre-compute zero expert output.
3. Safety net in ``_maybe_add_zero_expert_output``.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.fused_moe.router.zero_expert_router import (
    ZeroExpertRouter,
)
from vllm_ascend.ops.fused_moe.experts_selector import (
    select_experts as ascend_select_experts,
)
from vllm_ascend.ops.fused_moe.experts_selector import (
    zero_experts_compute as ascend_zero_experts_compute,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch

# ===========================================================================
# Patch 0: unquant_apply_mlp — chunk experts to avoid kernel limit
# ===========================================================================

_MAX_EXPERTS_PER_CHUNK = 64


@register_patch(target="vllm_ascend.ops.fused_moe.moe_mlp")
def patch_chunked_mlp(module: object) -> None:
    """Chunk large expert groups to avoid NPU grouped-matmul kernel crash."""

    _original = module.unquant_apply_mlp

    def _chunked(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        group_list: torch.Tensor,
        w1_bias: torch.Tensor | None = None,
        w2_bias: torch.Tensor | None = None,
        activation: str | None = None,
        group_list_type: int = 1,
        topk_scales: torch.Tensor | None = None,
        need_trans: bool = True,
        swiglu_limit: float = 0.0,
        lora_context: Any = None,
        expanded_row_idx: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
    ):
        num_experts = w1.shape[0]
        if num_experts <= _MAX_EXPERTS_PER_CHUNK:
            return _original(
                hidden_states,
                w1,
                w2,
                group_list,
                w1_bias,
                w2_bias,
                activation,
                group_list_type,
                topk_scales,
                need_trans,
                swiglu_limit=swiglu_limit,
                lora_context=lora_context,
                expanded_row_idx=expanded_row_idx,
                topk_ids=topk_ids,
            )

        outputs: list[torch.Tensor] = []
        chunks = (num_experts + _MAX_EXPERTS_PER_CHUNK - 1) // _MAX_EXPERTS_PER_CHUNK

        for ci in range(chunks):
            e0 = ci * _MAX_EXPERTS_PER_CHUNK
            e1 = min(e0 + _MAX_EXPERTS_PER_CHUNK, num_experts)

            w1c = w1[e0:e1]
            w2c = w2[e0:e1]
            b1c = w1_bias[e0:e1] if w1_bias is not None else None
            b2c = w2_bias[e0:e1] if w2_bias is not None else None

            if group_list_type == 0:  # CUMSUM
                t0 = 0 if e0 == 0 else int(group_list[e0 - 1].item())
                t1 = int(group_list[e1 - 1].item())
                glc = group_list[e0:e1].clone() - t0
            else:  # COUNT
                t0 = int(group_list[:e0].sum().item())
                t1 = t0 + int(group_list[e0:e1].sum().item())
                glc = group_list[e0:e1].clone()

            if t1 <= t0:
                continue

            hsc = hidden_states[t0:t1]
            cout, _ = _original(
                hsc,
                w1c,
                w2c,
                glc,
                b1c,
                b2c,
                activation,
                group_list_type,
                None,
                need_trans,
                swiglu_limit=swiglu_limit,
                lora_context=lora_context,
                expanded_row_idx=expanded_row_idx,
                topk_ids=topk_ids,
            )
            outputs.append(cout)

        if not outputs:
            return torch.zeros(
                0, w2.shape[-1], device=hidden_states.device, dtype=hidden_states.dtype
            ), None
        return torch.cat(outputs, dim=0), None

    module.unquant_apply_mlp = _chunked
    patch_logger.info(
        "[fix_ep_zero_expert] Patched unquant_apply_mlp (chunked experts)"
    )


# ===========================================================================
# Patch 1: TokenDispatcherWithMC2 — sanitize expert ids
# ===========================================================================


@register_patch(target="vllm_ascend.ops.fused_moe.token_dispatcher")
def patch_mc2_token_dispatch(module: object) -> None:
    """Clamp zero-expert indices to valid range before MC2 dispatch."""

    _original = TokenDispatcherWithMC2.token_dispatch

    def _patched(self: TokenDispatcherWithMC2, token_dispatch_input):
        ids = token_dispatch_input.topk_ids
        wts = token_dispatch_input.topk_weights
        n_logical = getattr(self, "num_experts", 0)
        if n_logical > 0:
            zm = ids >= n_logical
            if zm.any():
                ids = torch.where(zm, torch.zeros_like(ids), ids)
                wts = torch.where(zm, torch.zeros_like(wts), wts)
                object.__setattr__(token_dispatch_input, "topk_ids", ids)
                object.__setattr__(token_dispatch_input, "topk_weights", wts)
        return _original(self, token_dispatch_input)

    TokenDispatcherWithMC2.token_dispatch = _patched
    patch_logger.info(
        "[fix_ep_zero_expert] Patched TokenDispatcherWithMC2.token_dispatch"
    )


# ===========================================================================
# Patch 2 & 3: MoERunner — zero expert output
# ===========================================================================


def _compute_zero_output(
    router: ZeroExpertRouter,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    n_exp = router_logits.shape[-1]
    tw, ti = ascend_select_experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
        top_k=top_k,
        use_grouped_topk=False,
        renormalize=router.renormalize,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func=router.scoring_func,
        routed_scaling_factor=router.routed_scaling_factor,
        e_score_correction_bias=router.e_score_correction_bias,
        num_experts=n_exp,
    )
    _, _, zo = ascend_zero_experts_compute(
        expert_indices=ti,
        expert_scales=tw,
        num_experts=router.num_logical_experts,
        zero_expert_type=router.zero_expert_type,
        hidden_states=hidden_states,
    )
    return zo.to(hidden_states.dtype)


@register_patch(target="vllm.model_executor.layers.fused_moe.runner.moe_runner")
def patch_moe_runner_zero_expert(module: object) -> None:
    MoERunner = module.MoERunner
    _orig_fwd = MoERunner.forward
    _orig_maybe = MoERunner._maybe_add_zero_expert_output

    def _fwd(self, hs, rl, input_ids=None):
        if (
            isinstance(self.router, ZeroExpertRouter)
            and self.router.zero_expert_type is not None
        ):
            self.router._zero_expert_output = _compute_zero_output(
                self.router, hs, rl, self.moe_config.experts_per_token
            )
        return _orig_fwd(self, hs, rl, input_ids)

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

    MoERunner.forward = _fwd
    MoERunner._maybe_add_zero_expert_output = _maybe
    patch_logger.info(
        "[fix_ep_zero_expert] Patched MoERunner.forward + _maybe_add_zero_expert_output"
    )
