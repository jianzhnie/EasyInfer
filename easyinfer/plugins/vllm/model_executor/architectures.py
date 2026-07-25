"""Register custom architecture aliases in vLLM's model registry.

vLLM 0.18.0 resolves model architectures through ``_VLLM_MODELS`` in
``vllm.model_executor.models.registry``.  When an architecture is not found
there, vLLM falls back to ``trust_remote_code`` which imports the model
directory's ``modeling_*.py`` — a path that fails for LongCat checkpoints
because their ``modeling_longcat.py`` relies on ``transformers.utils.LossKwargs``
(added in transformers ≥ 4.52).

By registering the architecture here, vLLM uses its own built-in
``LongcatFlashForCausalLM`` implementation, avoiding the import altogether.

Custom architectures such as ``PCLForCausalLM`` (for Kimi-K2 MCore checkpoints)
are defined in EasyInfer's own plugin modules rather than inside
``vllm.model_executor.models``.  They are registered via the
``_CUSTOM_ARCHITECTURES`` table below, which bypasses vLLM's module-name
resolution so the full module path is preserved.
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

# Custom architectures whose implementation lives outside
# ``vllm.model_executor.models`` (i.e. inside EasyInfer's plugin tree).
# Each entry maps an architecture name to (full_module_path, class_name).
#
# Unlike aliases above, these do NOT go through ``_resolve_module_name``
# because that helper prepends ``vllm.model_executor.models.`` — which would
# produce a non-existent path for EasyInfer modules.
_CUSTOM_ARCHITECTURES: dict[str, tuple[str, str]] = {
    "PCLForCausalLM": (
        "easyinfer.plugins.vllm.model_executor.models.pcl_model",
        "PCLForCausalLM",
    ),
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

    # ---- Phase 1: architecture aliases (name → existing vLLM class) ----
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

    # ---- Phase 2: custom architectures (EasyInfer-owned implementations) ----
    for arch_name, (full_module, class_name) in _CUSTOM_ARCHITECTURES.items():
        if arch_name in module._VLLM_MODELS:
            continue  # already registered (idempotent)

        # Register in _VLLM_MODELS with the full module path (not a relative
        # name) so that older vLLM code paths can still find the module.
        module._VLLM_MODELS[arch_name] = (full_module, class_name)
        logger.info(
            "Registered custom arch in _VLLM_MODELS: %s -> (%s, %s)",
            arch_name,
            full_module,
            class_name,
        )

        # Register in ModelRegistry if available (vLLM >= 0.23.0).
        # We bypass _resolve_module_name because custom architectures use
        # absolute module paths.
        if hasattr(module, "ModelRegistry") and hasattr(
            module.ModelRegistry, "register_model"
        ):
            if arch_name not in module.ModelRegistry.models:
                module.ModelRegistry.register_model(
                    arch_name,
                    f"{full_module}:{class_name}",
                )
                logger.info(
                    "Registered custom arch in ModelRegistry: %s -> %s:%s",
                    arch_name,
                    full_module,
                    class_name,
                )
        else:
            logger.info(
                "ModelRegistry not available (vLLM < 0.23.0); "
                "custom arch %s registered in _VLLM_MODELS only",
                arch_name,
            )
