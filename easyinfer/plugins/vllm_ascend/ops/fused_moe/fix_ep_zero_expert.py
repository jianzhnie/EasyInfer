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

Problem 3: the native path adds the zero-expert result at the WRONG point
-------------------------------------------------------------------------
``AscendUnquantizedFusedMoEMethod.apply`` adds ``zero_expert_result`` onto
the fused-experts output *before* ``finalize`` and before the runner's
final TP/EP all-reduce.  With EP the MoE input is replicated across all
EP ranks, so every rank computes the SAME full identity contribution —
the downstream all-reduce then sums it ``world_size`` times (×64 on a
TP=EP=64 deployment), drowning the real output → garbled text (乱码).
Upstream adds the zero-expert output at the very END of
``MoERunner.forward`` (``_maybe_add_zero_expert_output``), AFTER
``_maybe_reduce_final_output``'s all-reduce — exactly once.

Fix
---
0b. Mirror ``ZeroExpertRouter`` config onto ``AscendFusedMoE`` so the native
    zero-expert path in ``apply`` runs (id sanitization is still needed);
    teach ``FusedExpertsResult`` ``+=`` for the native add.
0b2. Wrap ``zero_experts_compute`` in ``fused_moe.py``: stash the real
    identity contribution and return zeros instead, so the premature
    ``final_hidden_states += zero_expert_result`` in ``apply`` becomes a
    no-op (Problem 3).
0c. Optionally force the ALLGATHER MoE comm method
    (``EASYINFER_MOE_COMM=allgather``) to bypass MC2.
3.  In ``_maybe_add_zero_expert_output``, feed the stashed identity
    contribution to the runner so it is added once, at the end, after the
    final all-reduce (same semantics as upstream GPU).  Falls back to a
    scalar zero no-op when no stashed value is available (also satisfies
    the runner's ``assert zero_expert_output is not None``).
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
# Re-enable it by mirroring the router's config onto the layer.  The id
# sanitization is required; the premature add is neutralized by Patch 0b2.


@register_patch(target="vllm_ascend.ops.fused_moe.fused_moe_0_23_0")
def patch_enable_native_zero_expert(module: object) -> None:
    # Guard against repeated patching (e.g. module reload or multiple imports).
    if getattr(module.AscendFusedMoE, "_ez_patched", False):
        return
    module.AscendFusedMoE._ez_patched = True  # type: ignore[attr-defined]

    _orig_init = module.AscendFusedMoE.__init__

    # ``apply`` adds the zero-expert result onto the value returned by
    # ``fused_experts``; in that version the return value is a
    # ``FusedExpertsResult`` dataclass, not a tensor.  Give it an
    # ``__iadd__`` so ``result += zero_expert_result`` works.
    #
    # IMPORTANT: only inject ``__iadd__`` if the class doesn't already
    # define one (including inherited).  ``hasattr`` catches both own and
    # inherited definitions; if a parent class provides ``__iadd__`` we
    # must not silently replace it with our tensor-only version.
    from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult

    if not hasattr(FusedExpertsResult, "__iadd__"):
        def _fused_experts_result_iadd(self, other):
            # ``other`` is always a plain tensor in the current ``apply``
            # code path (the return value of ``zero_experts_compute``).
            # Guard against an unexpected non-tensor value to fail loudly
            # rather than silently corrupting state.
            if not isinstance(other, torch.Tensor):
                raise TypeError(
                    f"FusedExpertsResult.__iadd__ expected a Tensor, "
                    f"got {type(other).__name__}"
                )
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
            # Derive zero-expert count from the routing bias width
            # (real + zero) minus the logical (real) expert count.
            #
            # NOTE: both the bias (768 = 512 real + 256 zero) and
            # router.num_logical_experts (512) are GLOBAL, unsharded
            # values on every EP rank — vllm 0.23 builds FusedMoE with
            # num_experts = real experts only and passes the full-width
            # bias to the router.  Do NOT use self.global_num_experts
            # here: it includes EPLB redundant experts
            # (global_redundant_expert_num), which would undercount
            # n_zero when EPLB is enabled.
            n_zero = 0
            bias = getattr(router, "e_score_correction_bias", None)
            if bias is not None:
                # Guard against unexpected bias layouts (scalar, empty).
                if bias.ndim < 1:
                    raise RuntimeError(
                        "[fix_ep_zero_expert] e_score_correction_bias has "
                        "unexpected shape %s (ndim=%d).  Cannot derive "
                        "zero-expert count — vllm-ascend version may be "
                        "incompatible.",
                        tuple(bias.shape),
                        bias.ndim,
                    )
                n_zero = bias.shape[-1] - router.num_logical_experts
            # max(...,0): when n_zero==0 the model truly has no zero
            # experts — do NOT enable the native path (Bug #3 in review).
            # Previously ``max(n_zero, 1)`` forced the path on for every
            # ZeroExpertRouter, which is a false positive when the model
            # has no zero experts.
            n_zero = max(n_zero, 0)
            if n_zero > 0:
                self.zero_expert_num = n_zero
                self.zero_expert_type = router.zero_expert_type
                # Flag consumed by Patch 3: the runner must re-add the
                # stashed contribution (Patch 0b2) instead of its own.
                router._ez_native_handled = True  # type: ignore[attr-defined]
                patch_logger.info(
                    "[fix_ep_zero_expert] Enabled native zero-expert path: "
                    "num={}, type={}",
                    self.zero_expert_num,
                    self.zero_expert_type,
                )
            else:
                # zero_expert_type is set but n_zero derived as 0 —
                # inconsistent config or bias layout change.  Fail fast
                # because zero-expert IDs will NOT be sanitized and will
                # reach the dispatch kernel → aicore crash.
                raise RuntimeError(
                    "[fix_ep_zero_expert] zero_expert_type=%s is set but "
                    "derived n_zero=0 "
                    "(bias.shape=%s, global_num_experts=%s).  "
                    "Cannot enable native zero-expert path — the model "
                    "will crash without ID sanitization.  "
                    "The vllm-ascend version may be incompatible.",
                    router.zero_expert_type,
                    tuple(bias.shape) if bias is not None else "N/A",
                    self.global_num_experts,
                )

    module.AscendFusedMoE.__init__ = _init


# ===========================================================================
# Patch 0b2: relocate the native zero-expert add (fix Problem 3)
# ===========================================================================
# ``AscendUnquantizedFusedMoEMethod.apply`` (vllm_ascend/ops/fused_moe/
# fused_moe.py) adds ``zero_expert_result`` onto the fused-experts output
# BEFORE ``finalize`` and BEFORE the runner's final TP/EP all-reduce.
# With EP the MoE input is replicated across EP ranks, so every rank
# contributes the SAME full identity value and the downstream all-reduce
# sums it world_size times (×64 at TP=EP=64) → garbled output.
#
# The wrapper below keeps the native call (its id/weight sanitization is
# required to keep zero-expert ids out of the dispatch kernel) but stashes
# the real identity contribution and returns zeros, making the premature
# ``final_hidden_states += zero_expert_result`` a no-op.  Patch 3 then
# hands the stashed value to the runner, which adds it once at the very
# end of ``MoERunner.forward`` — the same point upstream uses on GPU.

# Stash for the identity contribution computed inside ``apply``.  Written
# by the ``zero_experts_compute`` wrapper, consumed (and cleared) by the
# ``_maybe_add_zero_expert_output`` wrapper — a strict 1:1 sequence per
# MoE layer per forward pass, so a single slot is sufficient.
_pending_zero_expert_output: torch.Tensor | None = None


@register_patch(target="vllm_ascend.ops.fused_moe.fused_moe")
def patch_relocate_zero_expert_add(module: object) -> None:
    # Guard against repeated patching (consistent with the other patches).
    if getattr(module, "_ez_reloc_patched", False):
        return
    module._ez_reloc_patched = True  # type: ignore[attr-defined]

    _orig_zec = module.zero_experts_compute

    def _zero_experts_compute_stashing(*args, **kwargs):
        global _pending_zero_expert_output
        expert_indices, expert_scales, result = _orig_zec(*args, **kwargs)
        _pending_zero_expert_output = result
        # Zeros, not the real result: ``apply`` unconditionally adds this
        # to the fused-experts output pre-finalize, where it would be
        # all-reduced world_size times (Problem 3).
        return expert_indices, expert_scales, torch.zeros_like(result)

    module.zero_experts_compute = _zero_experts_compute_stashing
    patch_logger.info(
        "[fix_ep_zero_expert] Wrapped zero_experts_compute: identity "
        "contribution relocated to the runner (post all-reduce)"
    )


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

    # Guard against repeated patching (consistent with Patch 0b and Patch 3).
    if getattr(module, "_ez_ag_patched", False):
        return
    module._ez_ag_patched = True  # type: ignore[attr-defined]

    _orig = module.select_moe_comm_method
    _logged = False

    def _select(num_tokens, vllm_config, is_draft_model=False):
        nonlocal _logged
        selected = _orig(num_tokens, vllm_config, is_draft_model)
        if selected is not None:
            if not _logged:
                _logged = True
                patch_logger.info(
                    "[fix_ep_zero_expert] MoE comm method overridden: "
                    "%s -> ALLGATHER",
                    selected,
                )
            return module.MoECommType.ALLGATHER
        return selected

    module.select_moe_comm_method = _select


# ===========================================================================
# Patch 3: MoERunner._maybe_add_zero_expert_output — add once, at the end
# ===========================================================================
# Upstream calls this at the very END of ``MoERunner.forward``, AFTER
# ``_maybe_reduce_final_output``'s TP/EP all-reduce.  That is the only
# correct place to add the zero-expert identity contribution (Problem 3).
# Here we hand the runner the real contribution stashed by Patch 0b2.


@register_patch(target="vllm.model_executor.layers.fused_moe.runner.moe_runner")
def patch_moe_runner_zero_expert(module: object) -> None:
    MoERunner = module.MoERunner

    # Guard against repeated patching.
    if getattr(MoERunner, "_ez_maybe_patched", False):
        return
    MoERunner._ez_maybe_patched = True  # type: ignore[attr-defined]

    _orig_maybe = MoERunner._maybe_add_zero_expert_output

    def _maybe(self, result):
        global _pending_zero_expert_output
        if (
            isinstance(self.router, ZeroExpertRouter)
            and self.router.zero_expert_type is not None
        ):
            # Only redirect the runner addition when the native path in
            # ``apply`` actually handled zero experts (flag set by
            # ``patch_enable_native_zero_expert``).  Otherwise preserve
            # the original ``_zero_expert_output`` so the runner can add
            # it normally (future versions where the native path is off).
            if getattr(self.router, "_ez_native_handled", False):
                # ``result`` may be a plain Tensor (current vllm-ascend)
                # or a FusedExpertsResult (future versions).
                if isinstance(result, torch.Tensor):
                    ref = result
                elif hasattr(result, "routed_out"):
                    ref = result.routed_out
                else:
                    raise TypeError(
                        "[fix_ep_zero_expert] Unexpected result type: "
                        "%s.  Expected Tensor or object with 'routed_out'."
                        % type(result).__name__
                    )

                stashed = _pending_zero_expert_output
                _pending_zero_expert_output = None
                if stashed is not None:
                    # The stashed identity was computed from the prepared
                    # (pre-finalize) hidden states; the runner result is
                    # post-finalize, post-all-reduce.  Both must cover the
                    # same tokens — true for DP=1 (prepare is a no-op).
                    # DP>1 + ALLGATHER gathers tokens in prepare, so the
                    # layouts diverge; fail fast instead of silently
                    # producing wrong output.
                    if stashed.shape != ref.shape:
                        raise RuntimeError(
                            "[fix_ep_zero_expert] Stashed zero-expert "
                            "output shape %s does not match the runner "
                            "result shape %s.  This deployment config "
                            "(e.g. DP>1 with ALLGATHER MoE comm) is not "
                            "supported by the zero-expert relocation."
                            % (tuple(stashed.shape), tuple(ref.shape))
                        )
                    # Added once by ``_orig_maybe`` below, after the final
                    # all-reduce — exactly the upstream GPU semantics.
                    self.router._zero_expert_output = stashed
                else:
                    # No stashed value (native path did not run this
                    # forward, e.g. non-MoE call).  Inject a scalar zero
                    # as a no-op add so the runner's
                    # ``assert zero_expert_output is not None`` holds.
                    self.router._zero_expert_output = torch.tensor(
                        0.0, device=ref.device, dtype=ref.dtype
                    )
        return _orig_maybe(self, result)

    MoERunner._maybe_add_zero_expert_output = _maybe
    patch_logger.info(
        "[fix_ep_zero_expert] Patched MoERunner._maybe_add_zero_expert_output"
    )
