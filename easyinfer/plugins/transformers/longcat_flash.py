"""Register LongCat-Flash with HuggingFace transformers auto classes.

Loads the Grouped Routing variant and registers it so that
``AutoModelForCausalLM.from_pretrained()`` works without
``trust_remote_code=True``.
"""

from __future__ import annotations

from typing import Any

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch

# Some checkpoints use "LongcatCausalLM" in config.json instead of
# "LongcatFlashForCausalLM".  Registering each alias as a distinct
# model_type in AutoConfig lets transformers resolve the config class
# regardless of which architecture name appears in config.json.
_ARCH_ALIASES = ("LongcatCausalLM",)


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

        # Register each architecture alias as an additional model_type
        # pointing to the same config class, so that checkpoints using
        # e.g. "LongcatCausalLM" in config.json are recognized.
        for alias in _ARCH_ALIASES:
            AutoConfig.register(alias, LongcatFlashConfig, exist_ok=True)

        # Map the config class to the model class (applies to all aliases).
        AutoModelForCausalLM.register(
            LongcatFlashConfig,
            LongcatFlashGroupForCausalLM,
            exist_ok=True,
        )

        patch_logger.success(
            "[transformers] Registered LongcatFlashGroupForCausalLM "
            "(model_type=%s, aliases=%s)",
            model_type,
            list(_ARCH_ALIASES),
        )
    except ImportError as e:
        patch_logger.warning("[transformers] Could not register LongCat-Flash: %s", e)

__all__ = [
    "patch_register_longcat_flash",
]
