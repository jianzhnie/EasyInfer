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
# "LongcatFlashForCausalLM".  Both resolve to the same model_type
# registered below, so AutoConfig handles them automatically once
# the canonical class is registered.
_ARCH_ALIASES = ("LongcatCausalLM",)


@register_patch(target="transformers.models.auto.configuration_auto")
def patch_register_longcat_flash(_module: Any) -> None:
    """Register LongCat-Flash config + model with transformers."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        from .configuration_longcat_flash import LongcatFlashConfig
        from .modeling_longcat_flash_group import LongcatFlashGroupForCausalLM

        model_type = LongcatFlashConfig.model_type

        AutoConfig.register(model_type, LongcatFlashConfig, exist_ok=True)
        AutoModelForCausalLM.register(
            LongcatFlashConfig,
            LongcatFlashGroupForCausalLM,
            exist_ok=True,
        )

        for _alias in _ARCH_ALIASES:
            AutoConfig.register(model_type, LongcatFlashConfig, exist_ok=True)
            AutoModelForCausalLM.register(
                LongcatFlashConfig,
                LongcatFlashGroupForCausalLM,
                exist_ok=True,
            )

        patch_logger.success(
            f"[transformers] Registered LongcatFlashGroupForCausalLM "
            f"(model_type={model_type}, aliases={list(_ARCH_ALIASES)})"
        )
    except ImportError as e:
        patch_logger.warning(f"[transformers] Could not register LongCat-Flash: {e}")
