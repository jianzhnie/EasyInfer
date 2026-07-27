"""Fix ``extract_layer_index`` for dual-attention / multi-sub-layer models.

LongCat-Flash uses ``nn.ModuleList`` for dual-attention and multi-MLP,
producing layer names like ``model.layers.0.self_attn.0`` or
``model.layers.0.mlps.0.gate_up_proj`` that contain two integers.

Strategy: replace the entire ``_deepseek_v2_mla_attention_init`` function
on ``vllm_ascend.patch.worker.patch_deepseek_v2`` with a version that
extracts the layer index from multi-integer prefixes.  This bypasses the
fragile ``from X import Y`` binding issue entirely.
"""

from __future__ import annotations

from typing import Any

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


def _extract_layer_index_safe(prefix: str, num_attn_module: int = 1) -> int:
    """Like vllm's extract_layer_index but tolerates multi-integer prefixes."""
    int_vals = [int(p) for p in prefix.split(".") if p.lstrip("-").isdigit()]
    if num_attn_module == 1:
        if not int_vals:
            raise ValueError(f"No integer found in layer name: {prefix}")
        return int_vals[0]
    # Multi-attention: flatten.  Same logic as upstream vllm.
    if len(int_vals) <= 2:
        return (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
    raise ValueError(
        f"layer name {prefix} should contain at most two integers"
    )


# We store the original init function so we can call it from the wrapper.
_ORIG_INIT: Any = None


@register_patch(target="vllm_ascend.patch.worker.patch_deepseek_v2")
def fix_deepseek_v2_init(module: Any) -> None:
    """Replace ``_deepseek_v2_mla_attention_init`` with a prefix-tolerant version."""
    global _ORIG_INIT
    if _ORIG_INIT is not None:
        return

    _ORIG_INIT = module._deepseek_v2_mla_attention_init

    # Build wrapper source: we replace every
    #   layer_id = extract_layer_index(prefix)
    # with
    #   layer_id = _extract_layer_index_safe(prefix)
    # The wrapper simply delegates to _ORIG_INIT after toggling the import.
    #
    # Simpler: wrap _ORIG_INIT and intercept the call.

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Replace the module-level extract_layer_index *temporarily* inside
        # the called module so that _ORIG_INIT's globals resolve to our safe
        # version.  This avoids the `from X import Y` stale-reference problem.
        import vllm_ascend.patch.worker.patch_deepseek_v2 as _pdv2

        _save = _pdv2.extract_layer_index
        try:
            _pdv2.extract_layer_index = _extract_layer_index_safe
            return _ORIG_INIT(self, *args, **kwargs)
        finally:
            _pdv2.extract_layer_index = _save

    module._deepseek_v2_mla_attention_init = patched_init

    # vllm_ascend binds ``DeepseekV2MLAAttention.__init__`` to the original
    # function object at import time (see the bottom of patch_deepseek_v2.py).
    # Replacing the module attribute alone is therefore not enough: the class
    # keeps a direct reference to the original init.  Rebind the class too.
    try:
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

        DeepseekV2MLAAttention.__init__ = patched_init
    except ImportError:
        patch_logger.warning(
            "[fix_dual_attention] could not import DeepseekV2MLAAttention; "
            "only the module attribute was wrapped"
        )

    patch_logger.info(
        "[fix_dual_attention] _deepseek_v2_mla_attention_init wrapped"
    )
