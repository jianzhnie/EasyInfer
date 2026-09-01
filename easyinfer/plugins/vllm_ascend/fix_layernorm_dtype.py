"""Fix EZ1001 dtype mismatch in AscendRMSNorm.forward_oot on Ascend NPU.

On Ascend NPU, upstream kernels (MLA attention, MoE) may produce float32
tensors while the layer-norm weights stay bfloat16.  The
``torch.ops._C_ascend.npu_add_rms_norm_bias`` ACLNN operator requires all
inputs to share the same dtype (all bfloat16 / all float16 / all float32).

Rather than patching every individual model's forward method, we patch the
*operator* at the source — ``AscendRMSNorm.forward_oot`` — so that **all**
models benefit from the fix automatically.

What we do:

    Before the ACLNN call, cast ``x`` and ``residual`` (if any) to match
    ``self.weight.dtype``.  The ACLNN kernel then receives inputs that are
    guaranteed to be dtype-compatible with the weight tensor.

Graph mode (torch.compile) note
-------------------------------
``forward_oot`` wrapping alone does NOT cover graph mode out of the box:
vllm-ascend's fusion passes (``sequence_parallelism``,
``allreduce_rmsnorm_fusion_pass``, ``norm_quant_fusion_pass``) insert
*raw* ``npu_add_rms_norm_bias`` calls into the compiled FX graph,
bypassing the Python-level guard entirely (EZ1001 during graph capture,
2026-07-27).  Shimming the op on ``torch.ops._C_ascend`` was tried and
abandoned: any Python wrapper in the dynamo-traced path breaks
compilation (recursion / "Skip calling disabled function" / "function
marked as skipped" — three variants observed).

What works: disable the offending passes so the only op insertion site
left is the guarded ``forward_oot`` itself:

- ``VLLM_ASCEND_ENABLE_FLASHCOMM1=0`` → disables the SP passes
  (``pass_config.enable_sp`` follows FlashComm1);
- ``--additional-config '{"ascend_compilation_config":{"fuse_allreduce_rms":false}}'``
  → disables ``MatmulAllReduceAddRMSNormPass`` (on by default);
- ``norm_quant_fusion_pass`` is quant-only and irrelevant for BF16.

``run_vllm.sh`` applies both automatically when ``ENFORCE_EAGER=0``.
"""

from __future__ import annotations

from typing import Any

import torch

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch


@torch.library.custom_op("easyinfer::rms_norm_guard_x", mutates_args=())
def _rms_norm_guard_x(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Keep the dtype conversion opaque when vLLM captures a compiled graph."""
    if x.dtype != weight.dtype:
        return x.to(dtype=weight.dtype)
    return x.clone()


@_rms_norm_guard_x.register_fake
def _rms_norm_guard_x_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    # Fake implementations participate in torch.compile metadata
    # propagation.  The real custom op casts to ``weight.dtype``; returning
    # ``x`` unchanged here can make the compiled graph retain an invalid
    # FP32 -> BF16 edge and fail later at the ACLNN RMSNorm call.
    return x.to(dtype=weight.dtype)


@torch.library.custom_op("easyinfer::rms_norm_guard_residual", mutates_args=())
def _rms_norm_guard_residual(
    residual: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    if residual.dtype != weight.dtype:
        return residual.to(dtype=weight.dtype)
    return residual.clone()


@_rms_norm_guard_residual.register_fake
def _rms_norm_guard_residual_fake(
    residual: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return residual.to(dtype=weight.dtype)


@register_patch(target="vllm_ascend.ops.layernorm")
def fix_layernorm_forward_oot_dtype(module: Any) -> None:
    """Wrap AscendRMSNorm.forward_oot to cast inputs to the weight dtype."""
    _AscendRMSNorm = getattr(module, "AscendRMSNorm", None)
    if _AscendRMSNorm is None:
        return

    if getattr(_AscendRMSNorm, "_ez_lndtype_patched", False):
        return
    _AscendRMSNorm._ez_lndtype_patched = True  # type: ignore[attr-defined]

    _original_oot = _AscendRMSNorm.forward_oot

    def _dtype_safe_forward_oot(
        self: Any,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        target_dtype = self.weight.dtype
        is_compiling = getattr(torch, "compiler", None)
        is_compiling = bool(
            is_compiling is not None
            and getattr(is_compiling, "is_compiling", lambda: False)()
        )
        if is_compiling:
            x = torch.ops.easyinfer.rms_norm_guard_x(x, self.weight)
            if residual is not None:
                residual = torch.ops.easyinfer.rms_norm_guard_residual(
                    residual, self.weight
                )
        else:
            if x.dtype != target_dtype:
                x = x.to(dtype=target_dtype)
            if residual is not None and residual.dtype != target_dtype:
                residual = residual.to(dtype=target_dtype)
        return _original_oot(self, x, residual)

    _AscendRMSNorm.forward_oot = _dtype_safe_forward_oot
    patch_logger.info(
        "[fix_layernorm_dtype] AscendRMSNorm.forward_oot dtype guard applied"
    )
