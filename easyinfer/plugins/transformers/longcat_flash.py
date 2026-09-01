"""Register LongCat-Flash with HuggingFace transformers auto classes.

Loads the Grouped Routing variant and registers it so that
``AutoModelForCausalLM.from_pretrained()`` works without
``trust_remote_code=True``.
"""

from __future__ import annotations

from typing import Any

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


@register_patch(target="transformers.models.auto.configuration_auto")
def patch_register_longcat_flash(_module: Any) -> None:
    """Register LongCat-Flash config + model with transformers."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        from .configuration_longcat_flash import LongcatFlashConfig
        from .modeling_longcat_flash_group import LongcatFlashGroupForCausalLM

        model_type = LongcatFlashConfig.model_type

        # Register the canonical model_type.
        AutoConfig.register(model_type, LongcatFlashConfig, exist_ok=True)

        # Architecture names live in config.architectures and are not valid
        # AutoConfig model_type keys. vLLM registers LongcatCausalLM aliases
        # separately in its ModelRegistry.
        AutoModelForCausalLM.register(
            LongcatFlashConfig,
            LongcatFlashGroupForCausalLM,
            exist_ok=True,
        )

        patch_logger.success(
            "[transformers] Registered LongcatFlashGroupForCausalLM "
            "(model_type={})",
            model_type,
        )
    except ImportError as e:
        patch_logger.warning("[transformers] Could not register LongCat-Flash: {}", e)

__all__ = ["patch_register_longcat_flash"]
