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
