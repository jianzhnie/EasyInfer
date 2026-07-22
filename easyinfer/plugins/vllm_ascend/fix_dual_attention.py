"""Fix ``extract_layer_index`` for dual-attention models in vllm_ascend.

LongCat-Flash uses dual self-attention (2 x :class:`DeepseekV2MLAAttention`)
per decoder layer, stored in a ``nn.ModuleList``.  The vLLM built-in
``FlashDecoderLayer`` passes ``prefix="...self_attn.{i}"`` (with the
ModuleList index) so weight loading can match the checkpoint keys
``model.layers.N.self_attn.0.*`` and ``model.layers.N.self_attn.1.*``.

However, ``vllm_ascend.patch.worker.patch_deepseek_v2`` replaces
``DeepseekV2MLAAttention.__init__`` and calls::

    layer_id = extract_layer_index(prefix)

The prefix ``model.layers.0.self_attn.0`` contains *two* integers (the
layer index *and* the attention index), so the default ``num_attn_module=1``
path raises::

    AssertionError: layer name ... should only contain one integer

This patch auto-detects the dual-attention pattern and forces the
extraction to return only the **layer-level** integer (the one that
appears immediately before ``self_attn``).  Both attention sub-layers
get the same ``layer_id``, which is the correct behaviour for per-layer
decisions such as ``_skip_topk`` and ``indexer_types``.
"""

from __future__ import annotations

from typing import Any

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


@register_patch(target="vllm_ascend.patch.worker.patch_deepseek_v2")
def fix_dual_attention_extract_layer_index(module: Any) -> None:
    """Wrap the module-level ``extract_layer_index`` to handle dual-attention.

    ``module.extract_layer_index`` is a local binding (``from vllm...utils
    import extract_layer_index``).  Replacing it in the module's global
    namespace updates ``_deepseek_v2_mla_attention_init.__globals__`` so
    every call site inside the module picks up the patched version.
    """
    original_extract = module.extract_layer_index

    def patched_extract(layer_name: str, num_attn_module: int = 1) -> int:
        # If the caller already specified num_attn_module > 1, don't interfere
        if num_attn_module == 1 and "attn" in layer_name:
            # Count integer parts in the dotted name
            int_count = sum(1 for part in layer_name.split(".") if part.isdigit())
            if int_count >= 2:
                # Dual-attention prefix like "model.layers.0.self_attn.0":
                # return the layer-level integer (the first one).
                for part in layer_name.split("."):
                    if part.isdigit():
                        return int(part)

        return original_extract(layer_name, num_attn_module)

    module.extract_layer_index = patched_extract
    patch_logger.info(
        "[fix_dual_attention] Patched extract_layer_index in "
        "vllm_ascend.patch.worker.patch_deepseek_v2"
    )
