"""Patch vllm.transformers_utils.model_arch_config_convertor for EasyInfer models."""

from typing import Any, cast

from vllm.logger import init_logger

from easyinfer.plugins.registry import register_patch

logger = init_logger(__name__)


@register_patch(target="vllm.transformers_utils.model_arch_config_convertor")
def patch_model_arch_convertors(module: Any) -> None:
    """Register ModelArchitectureConfig convertor for kimi_k2_mcore."""

    class PCLModelArchConfigConvertor(module.ModelArchConfigConvertorBase):
        """Convertor for EasyInfer Kimi-K2 MCore converted checkpoints."""

        def get_head_size(self) -> int:
            # EasyInfer MCore checkpoints use kv_channels as attention head dim.
            kv_channels = getattr(self.hf_text_config, "kv_channels", None)
            if kv_channels is not None:
                return cast(int, kv_channels)
            return cast(int, super().get_head_size())

        def get_total_num_kv_heads(self) -> int:
            # qkv projection is grouped-query style with `num_query_groups`.
            num_query_groups = getattr(self.hf_text_config, "num_query_groups", None)
            if num_query_groups is not None:
                return cast(int, num_query_groups)
            return cast(int, super().get_total_num_kv_heads())

        def is_deepseek_mla(self) -> bool:
            # EasyInfer MCore flavor is q/k/v projection based (non-MLA).
            return False

    module.MODEL_ARCH_CONFIG_CONVERTORS["pcl_model"] = PCLModelArchConfigConvertor
    logger.info("Registered ModelArchitectureConfig convertor for pcl_model")
