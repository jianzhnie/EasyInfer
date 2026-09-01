"""Per-process LongCat graph settings loaded by vLLM general plugins."""

from __future__ import annotations

import os
from typing import Any

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


def _longcat_process_enabled(_module: Any) -> tuple[bool, str]:
    return (
        os.environ.get("VLLM_LONGCAT_PATCH") == "1",
        "VLLM_LONGCAT_PATCH=1 is required",
    )


@register_patch(
    target="vllm_ascend",
    condition=_longcat_process_enabled,
)
def configure_longcat_process(module: Any) -> None:
    del module

    try:
        import torch
        import torch_npu  # noqa: F401

        config = getattr(getattr(torch, "npu", None), "config", None)
        if config is not None:
            config.allow_internal_format = False
    except (ImportError, AttributeError, RuntimeError) as exc:
        patch_logger.warning("[longcat_process] NPU format setting skipped: {}", exc)

    if os.environ.get("VLLM_LONGCAT_DISABLE_FUSED_GATING") == "1":
        try:
            import vllm_ascend.ops.fused_moe.experts_selector as selector

            if not getattr(selector, "_ez_longcat_fused_gating_disabled", False):
                selector.check_npu_moe_gating_top_k = lambda *args, **kwargs: False
                selector._ez_longcat_fused_gating_disabled = True
        except (ImportError, AttributeError) as exc:
            patch_logger.warning("[longcat_process] fused gating setting skipped: {}", exc)

    patch_logger.info("[longcat_process] LongCat process graph settings applied")


__all__ = ["configure_longcat_process"]
