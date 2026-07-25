"""Fix ``extract_layer_index`` for dual-attention models in vllm_ascend.

LongCat-Flash uses dual self-attention (2 x :class:`DeepseekV2MLAAttention`)
per decoder layer, stored in a ``nn.ModuleList``.  The vLLM built-in
``FlashDecoderLayer`` passes ``prefix="...self_attn.{i}"`` (with the
ModuleList index) so weight loading can match the checkpoint keys
``model.layers.N.self_attn.0.*`` and ``model.layers.N.self_attn.1.*``.

vllm_ascend modules that call ``extract_layer_index(prefix)`` with
``num_attn_module=1`` fail because the prefix contains *two* integers
(the layer index *and* the attention index)::

    AssertionError: layer name ... should only contain one integer

Affected vllm_ascend modules:
- ``patch_deepseek_v2`` — attention __init__
- ``patch_qwen3_next_mtp`` — KV cache binding

This patch auto-detects the dual-attention pattern and returns only the
**layer-level** integer (first integer found).  Both attention sub-layers
get the same ``layer_id``.

Each target module gets its own independently named patch function so that
log messages clearly identify which module was patched.
"""

from __future__ import annotations

from typing import Any

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


@register_patch(target="vllm.model_executor.models.utils")
def fix_dual_attention_utils(module: Any) -> None:
    """Patch the source definition of ``extract_layer_index``."""
    _patch_extract_layer_index(module, "vllm.model_executor.models.utils")


@register_patch(target="vllm_ascend.patch.worker.patch_deepseek_v2")
def fix_dual_attention_deepseek_v2(module: Any) -> None:
    """Rebind ``extract_layer_index`` in the DeepSeek V2 patch worker."""
    _patch_extract_layer_index(module, "vllm_ascend.patch.worker.patch_deepseek_v2")


@register_patch(target="vllm_ascend.patch.worker.patch_qwen3_next_mtp")
def fix_dual_attention_qwen3_mtp(module: Any) -> None:
    """Rebind ``extract_layer_index`` in the Qwen3 Next MTP patch worker."""
    _patch_extract_layer_index(module, "vllm_ascend.patch.worker.patch_qwen3_next_mtp")


def _patch_extract_layer_index(module: Any, target_name: str) -> None:
    """Wrap module-level ``extract_layer_index`` for dual-attention support.

    When the layer name contains two or more integer components and
    ``num_attn_module=1``, only the first integer (the layer index) is
    returned.  Otherwise the original function is called unchanged.
    """
    original_extract = module.extract_layer_index

    def patched_extract(layer_name: str, num_attn_module: int = 1) -> int:
        if num_attn_module == 1 and "attn" in layer_name:
            int_count = sum(1 for p in layer_name.split(".") if p.isdigit())
            if int_count >= 2:
                for part in layer_name.split("."):
                    if part.isdigit():
                        return int(part)
        return original_extract(layer_name, num_attn_module)

    module.extract_layer_index = patched_extract
    patch_logger.info(
        "[fix_dual_attention] Patched extract_layer_index in %s", target_name
    )

__all__ = [
    "fix_dual_attention_utils",
    "fix_dual_attention_deepseek_v2",
    "fix_dual_attention_qwen3_mtp",
]
