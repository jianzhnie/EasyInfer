"""Patch vllm.transformers_utils.config for EasyInfer custom model types."""

from typing import Any

from vllm.logger import init_logger

from easyinfer.plugins.registry import register_patch

logger = init_logger(__name__)

# Idempotency guard: only patch PretrainedConfig.__init__ once globally.
_PRETRAINED_CONFIG_PATCHED = False


@register_patch(target="vllm.transformers_utils.config")
def patch_vllm_config_registry(module: Any) -> None:
    """Register EasyInfer config aliases and compatibility fixes.

    - Registers ``pcl_model`` → ``DeepseekV3Config`` in vLLM's config registry.
    - Patches ``PretrainedConfig.__init__`` to auto-populate
      ``num_hidden_layers`` from ``num_layers`` when the former is missing.
      This is needed because the custom ``LongcatConfig`` (shipped with
      Meituan LongCat-Flash checkpoints) uses ``num_layers`` instead of the
      HuggingFace-standard ``num_hidden_layers``, and vllm_ascend's MLA ops
      read ``num_hidden_layers`` directly.

    Notes on the PretrainedConfig patch:
        The patch is applied **globally** (across all ``transformers`` config
        types) but is intentionally minimal: it only fires when
        ``num_hidden_layers`` is genuinely missing **and** ``num_layers`` is
        present — a combination that practically only occurs for
        LongCat-derived checkpoints.  The patch is also idempotent (applied
        at most once per process) to avoid conflicts with other libraries
        that may wrap ``__init__``.
    """
    global _PRETRAINED_CONFIG_PATCHED

    # Register the pcl_model config alias
    if module._CONFIG_REGISTRY.get("pcl_model") != "DeepseekV3Config":
        module._CONFIG_REGISTRY["pcl_model"] = "DeepseekV3Config"
        logger.info("Registered vLLM config alias: pcl_model -> DeepseekV3Config")

    # NOTE: no config alias is needed for LongCat checkpoints.  Their
    # config.json carries no ``model_type``, so vLLM never consults
    # ``_CONFIG_REGISTRY`` for them; with ``trust_remote_code`` the
    # checkpoint's own ``configuration_*.py`` is loaded instead.

    # ---- PretrainedConfig num_hidden_layers compatibility patch ----
    if _PRETRAINED_CONFIG_PATCHED:
        return
    _PRETRAINED_CONFIG_PATCHED = True

    try:
        from transformers import PretrainedConfig as _PretrainedConfig
    except ImportError:
        logger.warning(
            "transformers not installed; skipping PretrainedConfig compatibility patch"
        )
        return

    _original_pretrained_init = _PretrainedConfig.__init__

    def _patched_pretrained_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _original_pretrained_init(self, *args, **kwargs)
        # Only auto-populate num_hidden_layers when truly needed.
        # This guard ensures the patch is a no-op for all standard HF models
        # and only activates for checkpoints whose config uses the non-standard
        # ``num_layers`` field (e.g. LongCat-Flash).
        if not hasattr(self, "num_hidden_layers"):
            num_layers = getattr(self, "num_layers", None)
            if num_layers is not None:
                object.__setattr__(self, "num_hidden_layers", num_layers)

    _PretrainedConfig.__init__ = _patched_pretrained_init
    logger.info(
        "Patched PretrainedConfig.__init__ to auto-set num_hidden_layers "
        "from num_layers (LongCat compatibility)"
    )
