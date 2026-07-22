"""Patch vllm.transformers_utils.config for EasyInfer custom model types."""

from typing import Any

from vllm.logger import init_logger

from easyinfer.plugins.registry import register_patch

logger = init_logger(__name__)


@register_patch(target="vllm.transformers_utils.config")
def patch_vllm_config_registry(module: Any) -> None:
    """Register EasyInfer config aliases in vLLM config registry."""
    # Map the custom model_type to the same base config class used by kimi_k2.
    # This keeps compatibility with existing fields while giving us an isolated
    # runtime model_type for plugin routing.
    if module._CONFIG_REGISTRY.get("pcl_model") != "DeepseekV3Config":
        module._CONFIG_REGISTRY["pcl_model"] = "DeepseekV3Config"
        logger.info("Registered vLLM config alias: pcl_model -> DeepseekV3Config")

    # NOTE: no config alias is needed for LongCat checkpoints.  Their
    # config.json carries no ``model_type``, so vLLM never consults
    # ``_CONFIG_REGISTRY`` for them; with ``trust_remote_code`` the
    # checkpoint's own ``configuration_*.py`` is loaded instead.

    # The custom ``LongcatConfig`` uses ``num_layers`` instead of HF's
    # standard ``num_hidden_layers``.  vllm_ascend's MLA ops
    # (``MultiHeadLatentAttentionWrapper``) reads ``num_hidden_layers``
    # directly from the HF config.  We patch ``PretrainedConfig.__init__``
    # so that any config instance missing ``num_hidden_layers`` gets it
    # auto-populated from ``num_layers``.
    try:
        from transformers import PretrainedConfig as _PretrainedConfig
    except ImportError:
        _PretrainedConfig = None

    if _PretrainedConfig is not None:
        _original_pretrained_init = _PretrainedConfig.__init__

        def _patched_pretrained_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _original_pretrained_init(self, *args, **kwargs)
            if not hasattr(self, "num_hidden_layers"):
                num_layers = getattr(self, "num_layers", None)
                if num_layers is not None:
                    object.__setattr__(self, "num_hidden_layers", num_layers)

        _PretrainedConfig.__init__ = _patched_pretrained_init
        logger.info(
            "Patched PretrainedConfig.__init__ to auto-set num_hidden_layers from num_layers"
        )
