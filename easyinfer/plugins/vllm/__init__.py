"""EasyInfer vLLM Core Plugin.

Patches vLLM core modules (e.g., model_executor/models) for EasyInfer compatibility.
"""

from pathlib import Path

from easyinfer.plugins.logging import patch_logger


def register() -> None:
    """Register EasyInfer extensions with vLLM core.

    This function discovers and applies patches to vLLM core modules.
    """
    try:
        from easyinfer.plugins.registry import apply_all_patches, discover_modules

        # Discover vllm plugin modules (model_executor, etc.)
        plugin_dir = str(Path(__file__).parent)
        discover_modules("easyinfer.plugins.vllm", plugin_dir)
        # Apply all registered patches
        apply_all_patches()
        patch_logger.info("Registered EasyInfer with vLLM core")
    except ImportError as e:
        patch_logger.warning(f"Could not register EasyInfer with vLLM core: {e}")
