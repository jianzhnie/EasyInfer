"""Fix ``get_cos_and_sin_mla`` for models where ``is_deepseek_mla`` is False.

LongCat-Flash uses DeepSeek MLA attention but its ``model_type`` is
``"longcat"``, which is not in ``is_deepseek_mla()``'s hardcoded list.
So ``use_mla=False``, ``set_cos_and_sin`` sets ``_cos``/``_sin`` (non-MLA
path), and ``_cos_mla`` stays ``None`` — crashing the MLA attention backend
when it calls ``get_cos_and_sin_mla(positions, use_cache=True)``.

The patch wraps ``get_cos_and_sin_mla`` so the ``_cos_mla`` / ``_sin_mla``
scratch buffers are allocated on demand and grown whenever a call needs
more tokens than the current buffer holds.  Buffer contents match the
ones/zeros initialisation upstream performs in ``set_cos_and_sin`` (the
original function overwrites the used slice on every call, so the buffers
are pure scratch space).

The single wrapper is installed on the defining module and re-exported
into the known from-import callers (``mla_v1`` / ``sfa_v1``), so every
call site shares one implementation instead of stacking nested wrappers.
"""

from __future__ import annotations

from typing import Any

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch

# The rotary_embedding module stores the global caches as module-level
# attributes.  We read/write them through the module object so we don't
# need ``global``.
_rotary_mod: Any = None

# Floor for the scratch buffer; matches the usual max_num_batched_tokens
# used by these deployments without having to reach for the scheduler
# config.  The buffer grows beyond this on demand.
_MIN_BUFFER_TOKENS = 8192


def _ensure_mla_caches(rotary_mod: Any, num_tokens: int) -> None:
    """Allocate ``_cos_mla`` / ``_sin_mla`` if missing, grow if too small.

    Upstream sizes the buffers once to ``max_num_batched_tokens`` in
    ``set_cos_and_sin``.  Here they are created lazily and re-allocated
    (doubled) whenever a call exceeds the current capacity, so a larger
    later batch can never overflow the slice assignment in the original
    ``get_cos_and_sin_mla``.
    """
    cos_buf = rotary_mod._cos_mla
    sin_buf = rotary_mod._sin_mla
    if cos_buf is not None and sin_buf is not None and cos_buf.size(0) >= num_tokens:
        return

    # ``_cos_cache`` is guaranteed non-None here: the original function
    # indexes it unconditionally, so it must exist for the call to work
    # at all.  Slice the first row only for shape/dtype/device.
    cos0 = rotary_mod._cos_cache[:1].unsqueeze(1).unsqueeze(2)
    rope_dim = cos0.shape[-1]
    new_size = max(num_tokens, _MIN_BUFFER_TOKENS)
    if cos_buf is not None:
        new_size = max(new_size, cos_buf.size(0) * 2)

    rotary_mod._cos_mla = cos0.new_ones(new_size, 1, 1, rope_dim)
    rotary_mod._sin_mla = cos0.new_zeros(new_size, 1, 1, rope_dim)
    patch_logger.info(
        "[fix_mla_rotary] (Re)allocated MLA cos/sin scratch (rope_dim=%d, tokens=%d)",
        rope_dim,
        new_size,
    )


# ---- rotary_embedding module (source definition) ----


@register_patch(target="vllm_ascend.ops.rotary_embedding")
def fix_rotary_embedding(module: Any) -> None:
    global _rotary_mod
    _rotary_mod = module

    original = module.get_cos_and_sin_mla

    def patched(positions, use_cache=False):
        if use_cache:
            _ensure_mla_caches(module, positions.size(0))
        return original(positions, use_cache)

    module.get_cos_and_sin_mla = patched
    patch_logger.info("[fix_mla_rotary] Patched rotary_embedding.get_cos_and_sin_mla")


# ---- mla_v1 / sfa_v1 (callers with local from-import bindings) ----


def _patch_caller(module: Any) -> None:
    """Rebind the caller's from-imported name to the patched function.

    ``mla_v1`` / ``sfa_v1`` hold their own reference from
    ``from vllm_ascend.ops.rotary_embedding import get_cos_and_sin_mla``,
    which the module-level patch cannot reach if the caller was imported
    earlier.  Rebinding (instead of wrapping again) keeps a single
    wrapper shared by all call sites.
    """
    if _rotary_mod is None:
        patch_logger.warning(
            "[fix_mla_rotary] rotary_embedding not patched; leaving %s untouched",
            module.__name__,
        )
        return
    module.get_cos_and_sin_mla = _rotary_mod.get_cos_and_sin_mla
    patch_logger.info(
        "[fix_mla_rotary] Rebound %s.get_cos_and_sin_mla to the patched function",
        module.__name__,
    )


@register_patch(target="vllm_ascend.attention.mla_v1")
def fix_mla_v1(module: Any) -> None:
    _patch_caller(module)


@register_patch(target="vllm_ascend.attention.sfa_v1")
def fix_sfa_v1(module: Any) -> None:
    _patch_caller(module)
