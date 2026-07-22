"""Register custom architecture aliases in vLLM's model registry.

vLLM 0.18.0 resolves model architectures through ``_VLLM_MODELS`` in
``vllm.model_executor.models.registry``.  When an architecture is not found
there, vLLM falls back to ``trust_remote_code`` which imports the model
directory's ``modeling_*.py`` — a path that fails for LongCat checkpoints
because their ``modeling_longcat.py`` relies on ``transformers.utils.LossKwargs``
(added in transformers ≥ 4.52).

By registering the architecture here, vLLM uses its own built-in
``LongcatFlashForCausalLM`` implementation, avoiding the import altogether.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from easyinfer.plugins.registry import register_patch

logger = init_logger(__name__)

# Known architecture aliases: custom_name → (canonical_name)
_ARCH_ALIASES: dict[str, str] = {
    # LongCat-Flash checkpoints from Meituan use "LongcatCausalLM" in their
    # config.json but vLLM registers "LongcatFlashForCausalLM".  The two are
    # functionally identical — MLA + MoE + zero experts.
    "LongcatCausalLM": "LongcatFlashForCausalLM",
}


@register_patch(target="vllm.model_executor.models.registry")
def patch_vllm_model_registry(module: Any) -> None:
    """Register EasyInfer architecture aliases in vLLM's model registry.

    vLLM 0.23.0 introduced ``ModelRegistry``, a ``_ModelRegistry`` dataclass
    that snapshots ``_VLLM_MODELS`` into ``.models`` at import time.  The
    architecture validator (:meth:`_raise_for_unsupported`) checks
    ``ModelRegistry.models``, *not* ``_VLLM_MODELS`` directly.  We therefore
    register every alias in *both* places so the check passes.
    """

    for alias, canonical in _ARCH_ALIASES.items():
        # 1) Keep _VLLM_MODELS consistent (for older vLLM and for reference)
        if alias not in module._VLLM_MODELS:
            if canonical in module._VLLM_MODELS:
                module._VLLM_MODELS[alias] = module._VLLM_MODELS[canonical]
                logger.info(
                    "Registered arch alias in _VLLM_MODELS: %s -> %s (resolved to %s)",
                    alias,
                    canonical,
                    module._VLLM_MODELS[canonical],
                )
            else:
                module._VLLM_MODELS[alias] = ("longcat_flash", canonical)
                logger.info(
                    "Registered arch alias in _VLLM_MODELS: %s -> (longcat_flash, %s)",
                    alias,
                    canonical,
                )

        # 2) Register in ModelRegistry (required for vLLM >= 0.23.0).
        #    ModelRegistry.models is built once from _VLLM_MODELS at import
        #    time; later edits to _VLLM_MODELS are not visible to it.
        if alias not in module.ModelRegistry.models:
            # Resolve (mod_relname, cls_name) for the canonical architecture
            mod_relname, cls_name = module._VLLM_MODELS[alias]
            full_module_name = module._resolve_module_name(mod_relname)
            module.ModelRegistry.register_model(
                alias,
                f"{full_module_name}:{cls_name}",
            )
            logger.info(
                "Registered arch alias in ModelRegistry: %s -> %s:%s",
                alias,
                full_module_name,
                cls_name,
            )
