"""Patch for vllm/model_executor/models/qwen3_moe.py

This module patches Qwen3MoeModel.load_weights to handle W4A16 quantization
where expert weights have _packed, _scale, _shape, _offset suffixes.

Root Cause:
- make_expert_params_mapping() generates param names like "experts.w2_weight"
- W4A16 quantization registers params as "experts.w2_weight_packed"
- load_weights fails because it looks for "w2_weight" but actual param is "w2_weight_packed"

Compatibility:
- vLLM < 0.20.1: W4A16 MoE loading is not natively supported → this patch is required.
- vLLM >= 0.20.1: upstream may have its own W4A16 MoE support, but the patched
  load_weights delegates to the original implementation for non-W4A16 models
  (via ``_is_w4a16_quantized``), so it is safe to apply on all versions.
"""

from collections.abc import Iterable
from typing import Any, cast

import torch
from vllm.logger import init_logger

from easyinfer.plugins.logging import patch_logger
from easyinfer.plugins.registry import register_patch

target_logger = init_logger(__name__)


def _is_w4a16_quantized(params_dict: dict) -> bool:
    """Check if the model uses W4A16 quantization for MoE experts.

    W4A16 registers params with _packed suffix (e.g., w2_weight_packed).
    """
    for name in params_dict:
        if "experts.w13_weight_packed" in name or "experts.w2_weight_packed" in name:
            return True
    return False


def _get_w4a16_weight_suffixes() -> list[str]:
    """Get the suffixes used for W4A16 quantized weight parameters."""
    return ["_packed", ""]  # Try _packed first, then no suffix


def _get_w4a16_aux_suffixes() -> list[str]:
    """Get the suffixes for auxiliary W4A16 parameters."""
    return ["_scale", "_shape", "_offset"]


@register_patch(target="vllm.model_executor.models.qwen3_moe")
def patch_qwen3_moe_load_weights(module: Any) -> None:
    """Patch Qwen3MoeModel.load_weights to handle W4A16 quantization.

    Applies to all vLLM versions.  The patched function internally checks
    whether the loaded model actually uses W4A16 quantization and falls
    back to the original load_weights implementation when it does not,
    making this patch safe across vLLM upgrades.
    """

    original_load_weights = module.Qwen3MoeModel.load_weights

    def patched_load_weights(
        self: Any, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        params_dict = dict(self.named_parameters())

        # Check if W4A16 quantization is used
        is_w4a16 = _is_w4a16_quantized(params_dict)

        if not is_w4a16:
            # Use original implementation for non-W4A16 models
            return cast(set[str], original_load_weights(self, weights))

        # W4A16-specific loading logic
        from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader,
            maybe_remap_kv_scale_name,
        )
        from vllm.model_executor.models.utils import is_pp_missing_parameter

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        w4a16_weight_suffixes = _get_w4a16_weight_suffixes()
        w4a16_aux_suffixes = _get_w4a16_aux_suffixes()

        for name, loaded_weight in weights:
            # Handle KV cache quantization scales
            if self.quant_config is not None:
                scale_name = self.quant_config.get_cache_scale(name)
                if scale_name is not None:
                    param = params_dict[scale_name]
                    kv_weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    assert loaded_weight.numel() == 1, (
                        f"KV scale numel {loaded_weight.numel()} != 1"
                    )
                    loaded_weight = loaded_weight.squeeze()
                    kv_weight_loader(param, loaded_weight)
                    loaded_params.add(scale_name)
                    continue

            # Handle stacked params (non-expert)
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue  # Experts handled below

                name_mapped = name.replace(weight_name, param_name)

                if (
                    name_mapped.endswith(ignore_suffixes)
                    and name_mapped not in params_dict
                ):
                    continue
                if is_pp_missing_parameter(name_mapped, self):
                    continue
                if name_mapped.endswith("scale"):
                    remapped = maybe_remap_kv_scale_name(name_mapped, params_dict)
                    if remapped is None:
                        continue
                    name_mapped = remapped

                if name_mapped in params_dict:
                    param = params_dict[name_mapped]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name_mapped)
                    handled = True
                    break

            if handled:
                continue

            # Handle expert parameters
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, shard_id, expert_id, expert_name = mapping
                if expert_name in name:
                    is_expert_weight = True
                    name_mapped = name.replace(expert_name, param_name)

                    # Try each W4A16 weight suffix
                    param_found = False
                    for weight_suffix in w4a16_weight_suffixes:
                        name_with_suffix = name_mapped + weight_suffix
                        if name_with_suffix in params_dict:
                            param = params_dict[name_with_suffix]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            try:
                                success = weight_loader(
                                    param,
                                    loaded_weight,
                                    name_with_suffix,
                                    shard_id=shard_id,
                                    expert_id=expert_id,
                                    return_success=True,
                                )
                                if success:
                                    loaded_params.add(name_with_suffix)
                                    param_found = True
                                    break
                            except TypeError:
                                # Fallback if weight_loader doesn't accept return_success
                                try:
                                    weight_loader(
                                        param,
                                        loaded_weight,
                                        name_with_suffix,
                                        shard_id=shard_id,
                                        expert_id=expert_id,
                                    )
                                    loaded_params.add(name_with_suffix)
                                    param_found = True
                                    break
                                except Exception as e:
                                    target_logger.debug(
                                        f"Failed to load {name_with_suffix}: {e}"
                                    )

                    # If not found as weight, try loading as auxiliary params
                    if not param_found:
                        for aux_suffix in w4a16_aux_suffixes:
                            name_with_aux = name_mapped + aux_suffix
                            if name_with_aux in params_dict:
                                param = params_dict[name_with_aux]
                                weight_loader = getattr(
                                    param, "weight_loader", default_weight_loader
                                )
                                try:
                                    success = weight_loader(
                                        param,
                                        loaded_weight,
                                        name_with_aux,
                                        shard_id=shard_id,
                                        expert_id=expert_id,
                                        return_success=True,
                                    )
                                    if success:
                                        loaded_params.add(name_with_aux)
                                        param_found = True
                                        break
                                except TypeError:
                                    try:
                                        weight_loader(
                                            param,
                                            loaded_weight,
                                            name_with_aux,
                                            shard_id=shard_id,
                                            expert_id=expert_id,
                                        )
                                        loaded_params.add(name_with_aux)
                                        param_found = True
                                        break
                                    except Exception as e:
                                        target_logger.debug(
                                            f"Failed to load {name_with_aux}: {e}"
                                        )

                    if param_found:
                        break

            else:
                if is_expert_weight:
                    # Expert weight not mapped locally, skip
                    continue

                # Handle non-expert weights
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("kv_scale"):
                    remapped_kv_scale_name = name.replace(".kv_scale", ".attn.kv_scale")
                    if remapped_kv_scale_name not in params_dict:
                        target_logger.warning_once(
                            "Found kv scale in checkpoint (e.g. %s), "
                            "but not found expected name in model (e.g. %s). "
                            "kv-scale is not loaded.",
                            name,
                            remapped_kv_scale_name,
                        )
                        continue
                    name = remapped_kv_scale_name

                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params

    module.Qwen3MoeModel.load_weights = patched_load_weights
    patch_logger.info("Patched Qwen3MoeModel.load_weights for W4A16 MoE support")

__all__ = [
    "patch_qwen3_moe_load_weights",
]
