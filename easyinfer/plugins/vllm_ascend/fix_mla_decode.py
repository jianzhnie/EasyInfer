"""Graph-safe MLA decode and LongCat ND weight handling for Ascend.

vLLM-Ascend 0.23's generic MLA decode path uses ``torch.bmm`` and converts
``W_UK_T`` to FRACTAL_NZ.  LongCat's A3 graph path requires the fused NPU
batch-matmul with an ND weight instead.  The patch is model-gated and leaves
other MLA models unchanged.
"""

from __future__ import annotations

from typing import Any

import torch

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


def _is_longcat(self: Any) -> bool:
    config = getattr(getattr(self, "vllm_config", None), "model_config", None)
    text_config = getattr(config, "hf_text_config", None)
    if text_config is None:
        text_config = config
    # Checkpoints in the wild use both spellings.  The Meituan checkpoint
    # resolves to ``longcat_flash`` through its custom config, while older
    # exports used ``longcat`` directly.
    return getattr(text_config, "model_type", None) in {
        "longcat",
        "longcat_flash",
    }


@register_patch(target="vllm_ascend.attention.mla_v1")
def patch_longcat_mla_decode(module: Any) -> None:
    impl = getattr(module, "AscendMLAImpl", None)
    if impl is None or getattr(impl, "_ez_longcat_mla_patched", False):
        return

    original_qk = impl._q_proj_and_k_up_proj
    original_process = impl.process_weights_after_loading

    def _q_proj_and_k_up_proj(self: Any, x: torch.Tensor):
        if not _is_longcat(self):
            return original_qk(self, x)
        try:
            import torch_npu
        except ImportError:
            return original_qk(self, x)
        fused_bmm = getattr(torch_npu, "npu_transpose_batchmatmul", None)
        if fused_bmm is None:
            return original_qk(self, x)

        q_nope, q_pe = (
            self.q_proj(x)[0]
            .view(-1, self.num_heads, self.qk_head_dim)
            .split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        )
        ql_nope = fused_bmm(
            q_nope,
            self.W_UK_T,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
        return ql_nope, q_pe

    def _process_weights_after_loading(self: Any, act_dtype: torch.dtype):
        if not _is_longcat(self):
            return original_process(self, act_dtype)

        # The upstream method performs all required reshaping and then calls
        # its module-global ``maybe_trans_nz``.  Temporarily bypass only that
        # conversion so W_UK_T remains ND for the LongCat fused BMM.
        maybe_trans_nz = getattr(module, "maybe_trans_nz", None)
        if maybe_trans_nz is None:
            return original_process(self, act_dtype)
        module.maybe_trans_nz = lambda weight: weight
        try:
            return original_process(self, act_dtype)
        finally:
            module.maybe_trans_nz = maybe_trans_nz

    impl._q_proj_and_k_up_proj = _q_proj_and_k_up_proj
    impl.process_weights_after_loading = _process_weights_after_loading
    impl._ez_longcat_mla_patched = True
    patch_logger.info(
        "[fix_mla_decode] LongCat MLA decode BMM and ND weight handling applied"
    )


__all__ = ["patch_longcat_mla_decode"]
