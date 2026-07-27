"""Patches for vllm/model_executor/models/longcat_flash.py

1. **Grouped Routing** — optional, only when config sets ``use_group_routing``
   AND ``expert_expansion_factor > 1``.
2. **MTP weight filtering** — skips Multi-Token Prediction keys during weight
   loading (the built-in ``".mtp." in name`` check misses keys that start with
   ``mtp.`` after vLLM strips the ``model.`` prefix).

Supports Grouped Routing (分组路由) from the custom HuggingFace model
``modeling_longcat_flash_group.py``.

Routing logic (matching HF ``LongcatFlashTopkRouter.get_topk_indices``):

    Router outputs (N+Z) logits → reshape into *F* groups of (N+Z)/F experts
    → pick best expert within each group → top-k from the group winners
    → map back to original expert indices in [0, N+Z).

The router classifier dimension stays at N+Z (no expansion).  Expert
computation uses N real + Z zero experts, unchanged.

Compatibility
-------------
- **Without zero experts** (``zero_expert_type=None``): sets
  ``custom_routing_function`` on the ``FusedMoE`` so the router factory
  creates a ``CustomRoutingRouter``.
- **With zero experts** (``zero_expert_type="identity"``): directly
  monkey-patches ``ZeroExpertRouter._compute_routing`` on the instance,
  because the router factory gives ``ZeroExpertRouter`` higher priority
  and it ignores ``custom_routing_function``.

Ascend NPU note
---------------
vLLM 0.23 removed ``ZeroExpertFusedMoE``; ``LongcatMoe`` now builds a plain
``FusedMoE`` and vllm_ascend's ``AscendFusedMoE`` computes the zero-expert
contribution natively, so the old ``AscendZeroExpertFusedMoE`` OOT
replacement (and its grouped-routing ``select_experts`` hook) is gone.
The duck-typing marker it relied on (``_temporarily_set_attrs``) no longer
exists in vllm_ascend >= 0.23, so the Ascend-specific grouped-routing path
below is currently inactive; grouped routing is only wired up on GPU.
Both LongCat-Flash checkpoints ship without ``use_group_routing``, so this
path is dormant in practice.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from vllm.logger import init_logger

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch

target_logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Grouped routing function
# ---------------------------------------------------------------------------


def _grouped_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    expansion_factor: int,
    n_routed_experts: int,
    e_score_correction_bias: torch.Tensor | None = None,
    **_ignored: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped top-k routing matching HF ``LongcatFlashTopkRouter``.

    Args:
        hidden_states: *(tokens, hidden_dim)* — unused, for API compatibility.
        gating_output: *(tokens, N+Z)* — raw router logits.
        topk: Number of experts to select.
        renormalize: Whether to normalize top-k weights to sum to 1.
        expansion_factor: *F*, number of experts per group.
        n_routed_experts: N+Z, total router output dimension.
        e_score_correction_bias: Optional bias added **after** softmax
            (matching HF behaviour).

    Returns:
        *topk_weights*: *(tokens, topk)*
        *topk_ids*: *(tokens, topk)* — expert indices in [0, N+Z).
    """
    del hidden_states  # unused — kept for API compatibility with FusedMoE
    total_groups = n_routed_experts // expansion_factor  # (N+Z) / F

    if total_groups < topk:
        raise ValueError(
            f"Grouped routing: expansion_factor={expansion_factor} yields "
            f"{total_groups} groups from {n_routed_experts} experts, "
            f"but topk={topk}. Reduce expansion_factor to at most "
            f"{n_routed_experts // topk}."
        )

    # 1. Softmax across all logits → scores
    scores = gating_output.softmax(dim=-1)

    # 2. Add bias after softmax (HF-style: scores + bias)
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores

    # 3. Reshape into F groups: (tokens, N+Z) → (tokens, F, total_groups)
    #    → transpose → (tokens, total_groups, F)
    grouped_scores = scores_for_choice.view(
        -1, expansion_factor, total_groups
    ).transpose(-1, -2)

    # 4. Best expert within each group
    #    group_score_best: (tokens, total_groups)
    #    group_best_idx:   (tokens, total_groups) — offset in [0, F)
    group_score_best, group_best_idx = grouped_scores.max(dim=-1)

    # 5. Top-k from the total_groups winners
    _, topk_group_ids = torch.topk(group_score_best, k=topk, dim=-1, sorted=False)

    # 6. Map back to original expert index:
    #    expert_idx = group_id + best_offset * total_groups
    best_offsets = group_best_idx.gather(1, topk_group_ids)
    topk_ids = topk_group_ids + best_offsets * total_groups

    # 7. Gather weights from the ORIGINAL softmax scores (before bias)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator

    return topk_weights, topk_ids


# ---------------------------------------------------------------------------
# ZeroExpertRouter patch helper
# ---------------------------------------------------------------------------


def _patch_zero_expert_router(
    experts: Any,
    expansion_factor: int,
    n_routed: int,
) -> None:
    """Replace ``ZeroExpertRouter._compute_routing`` with grouped routing.

    The vLLM router factory prioritises ``ZeroExpertRouter`` over
    ``CustomRoutingRouter``, so ``custom_routing_function`` is ignored when
    ``zero_expert_type`` is set.  This helper directly replaces the routing
    logic while keeping the zero-expert bookkeeping intact.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        zero_experts_compute_triton,
    )

    router = experts.router  # ZeroExpertRouter instance

    def grouped_compute_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Grouped top-k (replaces fused_topk_bias call)
        topk_weights, topk_ids = _grouped_routing(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=router.top_k,
            renormalize=router.renormalize,
            expansion_factor=expansion_factor,
            n_routed_experts=n_routed,
            e_score_correction_bias=router.e_score_correction_bias,
        )
        topk_weights = topk_weights.to(torch.float32)
        topk_ids = topk_ids.to(
            torch.int32 if indices_type is None else indices_type,
        )

        # 2. Scaling (same as original ZeroExpertRouter)
        if router.routed_scaling_factor != 1.0:
            topk_weights *= router.routed_scaling_factor

        # 3. Zero-expert contribution (same as original ZeroExpertRouter)
        router._zero_expert_output = zero_experts_compute_triton(
            expert_indices=topk_ids.clone(),
            expert_scales=topk_weights.clone(),
            num_experts=router.num_logical_experts,
            zero_expert_type=router.zero_expert_type,
            hidden_states=hidden_states,
        )

        # 4. Mask zero-expert IDs so downstream MoE ignores them
        zero_mask = topk_ids >= router.num_logical_experts
        topk_ids[zero_mask] = 0
        topk_weights[zero_mask] = 0.0

        return topk_weights, topk_ids

    router._compute_routing = grouped_compute_routing


# ---------------------------------------------------------------------------
# Ascend select_experts patch helper
# ---------------------------------------------------------------------------


def _patch_ascend_select_experts(
    experts: Any,
    expansion_factor: int,
    n_routed: int,
) -> None:
    """Replace ``select_experts`` on Ascend with grouped routing.

    Legacy hook kept for older vllm_ascend releases whose
    ``AscendZeroExpertFusedMoE.forward()`` temporarily disabled
    ``custom_routing_function`` and called ``select_experts`` directly,
    bypassing both the router patch and ``custom_routing_function``.
    On vllm_ascend >= 0.23 that OOT class no longer exists and this
    helper is never invoked (see the module docstring).
    """

    def grouped_select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = _grouped_routing(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=experts.top_k,
            renormalize=experts.renormalize,
            expansion_factor=expansion_factor,
            n_routed_experts=n_routed,
            e_score_correction_bias=experts.e_score_correction_bias,
        )
        if experts.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * experts.routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    experts.select_experts = grouped_select_experts


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------


# ===========================================================================
# Patch 1: Grouped Routing (optional, config-driven)
# ===========================================================================


@register_patch(target="vllm.model_executor.models.longcat_flash")
def patch_longcat_flash_grouped_routing(module: Any) -> None:
    """Inject grouped routing into ``LongcatMoe`` when config enables it.

    Only activates when **both** ``use_group_routing=True`` and
    ``expert_expansion_factor > 1`` are present in the config.
    Otherwise this is a no-op.

    Three code paths (device-dependent):
    - **GPU, zero_expert_type set**: patches ``ZeroExpertRouter`` directly.
    - **GPU, zero_expert_type=None**: sets ``custom_routing_function``.
    - **Ascend NPU (legacy, vllm_ascend < 0.23 only)**: replaces
      ``select_experts`` on the fused-MoE instance.
    """

    original_moe_init = module.LongcatMoe.__init__

    def patched_moe_init(
        self: Any,
        config: Any,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        quant_config: Any | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        original_moe_init(
            self,
            config=config,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            enable_eplb=enable_eplb,
        )

        # ---- Guard: only activate when explicitly configured ----
        use_group_routing = getattr(config, "use_group_routing", False)
        expansion_factor = getattr(config, "expert_expansion_factor", 1)
        if not (use_group_routing and expansion_factor > 1):
            return
        # -----------------------------------------------------------

        n_routed = self.router.n_routed_experts
        zero_expert_type = getattr(config, "zero_expert_type", None)
        # Duck-typing marker of the old AscendZeroExpertFusedMoE OOT class.
        # vllm_ascend >= 0.23 removed it, so this is False there and Path C
        # below stays inactive (upstream AscendFusedMoE handles zero experts
        # natively).
        is_ascend = hasattr(self.experts, "_temporarily_set_attrs")

        if zero_expert_type is not None and not is_ascend:
            # ---- Path A: GPU + zero experts ----
            # Router factory created a ZeroExpertRouter which ignores
            # custom_routing_function → patch _compute_routing directly.
            _patch_zero_expert_router(self.experts, expansion_factor, n_routed)
            patch_logger.info(
                "[longcat_flash] Grouped routing (ZeroExpertRouter): "
                "prefix=%s expansion=%d n_routed=%d top_k=%d zero=%s",
                prefix,
                expansion_factor,
                n_routed,
                top_k,
                zero_expert_type,
            )
        elif zero_expert_type is None:
            # ---- Path B: no zero expert (any device) ----
            # Router factory will create CustomRoutingRouter.
            self.experts.custom_routing_function = (
                lambda hidden_states,
                gating_output,
                topk,
                renormalize,
                **kw: _grouped_routing(
                    hidden_states,
                    gating_output,
                    topk,
                    renormalize,
                    expansion_factor,
                    n_routed,
                    e_score_correction_bias=self.router.e_score_correction_bias,
                )
            )
            patch_logger.info(
                "[longcat_flash] Grouped routing (CustomRoutingRouter): "
                "prefix=%s expansion=%d n_routed=%d top_k=%d",
                prefix,
                expansion_factor,
                n_routed,
                top_k,
            )

        # ---- Path C: Ascend NPU (legacy, inactive on vllm_ascend >= 0.23) ----
        # Only reachable on older vllm_ascend releases whose
        # AscendZeroExpertFusedMoE.forward() temporarily disabled
        # custom_routing_function and called select_experts directly,
        # bypassing both Path A's router patch and Path B's
        # custom_routing_function.
        if is_ascend:
            _patch_ascend_select_experts(self.experts, expansion_factor, n_routed)
            patch_logger.info(
                "[longcat_flash] Grouped routing (Ascend select_experts): "
                "prefix=%s expansion=%d n_routed=%d top_k=%d",
                prefix,
                expansion_factor,
                n_routed,
                top_k,
            )

    module.LongcatMoe.__init__ = patched_moe_init
    patch_logger.info("[longcat_flash] Grouped Routing monkey patch applied")


# ===========================================================================
# Patch 2: MTP weight filtering
# ===========================================================================


@register_patch(target="vllm.model_executor.models.longcat_flash")
def patch_longcat_flash_mtp_filter(module: Any) -> None:
    """Filter MTP (Multi-Token Prediction) keys during weight loading.

    The built-in check ``".mtp." in name`` misses keys that start with
    ``mtp.`` after vLLM strips the ``model.`` prefix from checkpoint keys.
    We pre-filter the weight iterator to skip any key containing ``mtp.``.
    """
    original_load_weights = module.LongcatFlashForCausalLM.load_weights

    def patched_load_weights(self: Any, weights: Any) -> Any:
        with contextlib.suppress(TypeError, ValueError):
            weights = [
                (n, t)
                for n, t in weights
                if ".mtp." not in n and not n.startswith("mtp.")
            ]
        return original_load_weights(self, weights)

    module.LongcatFlashForCausalLM.load_weights = patched_load_weights
    patch_logger.info("[longcat_flash] MTP weight filter applied")
