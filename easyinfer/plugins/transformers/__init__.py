"""EasyInfer Transformers Plugin.

Registers custom HuggingFace model architectures with transformers'
auto class system, enabling ``AutoModel.from_pretrained()`` and
``AutoConfig.from_pretrained()`` for EasyInfer custom models.
"""

from pathlib import Path

from easyinfer.plugins.logging import patch_logger


def register() -> None:
    """Register custom model architectures with HuggingFace transformers.

    Scans plugin modules for @register_auto_model decorators and
    applies them to the appropriate transformers auto classes.
    """
    try:
        from easyinfer.plugins.registry import apply_all_patches, discover_modules

        plugin_dir = str(Path(__file__).parent)
        discover_modules("easyinfer.plugins.transformers", plugin_dir)
        apply_all_patches()
        patch_logger.info("Registered EasyInfer custom models with transformers")
    except ImportError as e:
        patch_logger.warning(
            f"Could not register EasyInfer models with transformers: {e}"
        )
