"""Correct LongCat MoE selector bias and routed scaling semantics."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


def _longcat_selector_enabled(_module: Any) -> tuple[bool, str]:
    """Avoid changing selector semantics for unrelated MoE models.

    ``experts_selector`` is shared by every Ascend MoE architecture in a
    worker process.  This patch is only validated for LongCat's native
    selector path, so require the same explicit opt-in used by the launcher.
    """
    enabled = os.environ.get("VLLM_LONGCAT_PATCH") == "1"
    return enabled, "VLLM_LONGCAT_PATCH=1 is required"


def _fixed_native_select_experts(
    module: Any,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """v0.23 native selector with bias used only for choosing expert IDs.

    ``select_experts`` applies ``routed_scaling_factor`` once after this
    function returns.  Keeping this function unscaled preserves that API and
    avoids the double scaling present in custom LongCat routing callbacks.
    """
    del kwargs
    if scoring_func == "softmax":
        scores = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = router_logits.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = F.softplus(router_logits).sqrt()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if use_grouped_topk:
        return module._select_expert_use_group_topk(
            topk_weights=scores,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
        )

    if custom_routing_function is not None:
        weights, ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
        return weights, ids.to(torch.int32)

    if e_score_correction_bias is not None:
        bias = e_score_correction_bias.to(dtype=scores.dtype)
        ids = (scores + bias.unsqueeze(0)).topk(
            top_k, dim=-1
        ).indices
        weights = scores.gather(1, ids)
    else:
        weights, ids = scores.topk(top_k, dim=-1)

    weights = weights.to(hidden_states.dtype)
    weights = module._renormalize_topk_weights(weights, renormalize)
    return weights, ids.to(torch.int32)


@register_patch(
    target="vllm_ascend.ops.fused_moe.experts_selector",
    condition=_longcat_selector_enabled,
)
def patch_moe_selector(module: Any) -> None:
    if getattr(module, "_ez_longcat_selector_patched", False):
        return
    original_native_select = getattr(module, "_native_select_experts", None)

    def _native_select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        use_hash: bool = False,
        tid2eid: dict[int, int] | None = None,
        input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve Ascend's hash/expert-remapping path.  The LongCat fix only
        # changes correction-bias and scaling semantics for the ordinary
        # native selector; silently dropping these arguments would break
        # models that rely on hash routing when they share a worker process.
        # ``input_ids`` is also passed for ordinary requests, so it must not
        # by itself trigger the fallback or LongCat would miss this fix.
        if (
            original_native_select is not None
            and (use_hash or tid2eid is not None)
        ):
            return original_native_select(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=top_k,
                use_grouped_topk=use_grouped_topk,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
                use_hash=use_hash,
                tid2eid=tid2eid,
                input_ids=input_ids,
                **kwargs,
            )
        return _fixed_native_select_experts(
            module,
            hidden_states,
            router_logits,
            top_k,
            use_grouped_topk,
            renormalize,
            topk_group,
            num_expert_group,
            custom_routing_function,
            scoring_func,
            routed_scaling_factor,
            e_score_correction_bias,
            **kwargs,
        )

    module._native_select_experts = _native_select_experts
    module._ez_longcat_selector_patched = True
    patch_logger.info(
        "[fix_moe_selector] MoE correction bias/scaling semantics applied"
    )


__all__ = ["patch_moe_selector"]
