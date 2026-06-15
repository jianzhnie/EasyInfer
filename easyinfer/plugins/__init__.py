"""EasyInfer plugin system.

Provides automatic registration of EasyInfer plugins with deployment
backends such as vLLM and vLLM-Ascend.
"""

from __future__ import annotations

import importlib
import importlib.util

_REGISTERED = False


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _register_plugin(package_name: str) -> None:
    module = importlib.import_module(package_name)
    register_fn = getattr(module, "register", None)
    if callable(register_fn):
        register_fn()


def register() -> None:
    """Register all EasyInfer plugins with their respective frameworks.

    Call this once after installing easyinfer, or it happens automatically
    via entry points. This function is idempotent - multiple calls are safe.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    if _module_available("vllm"):
        _register_plugin("easyinfer.plugins.vllm")

    if _module_available("vllm_ascend"):
        _register_plugin("easyinfer.plugins.vllm_ascend")

    _REGISTERED = True


__all__ = ["register"]
