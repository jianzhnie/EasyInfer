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
"""

from __future__ import annotations

from typing import Any

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch

_TARGETS = [
    # Patch the source function so any late importers get the fix.
    "vllm.model_executor.models.utils",
    # Also patch consumer modules to rebind their local "from X import Y" references.
    "vllm_ascend.patch.worker.patch_deepseek_v2",
    "vllm_ascend.patch.worker.patch_qwen3_next_mtp",
]


def _make_patch(target: str):
    @register_patch(target=target)
    def fix_dual_attention(module: Any) -> None:
        """Wrap module-level ``extract_layer_index`` for dual-attention."""
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
            "[fix_dual_attention] Patched extract_layer_index in %s", target
        )

    return fix_dual_attention


for _target in _TARGETS:
    _make_patch(_target)
