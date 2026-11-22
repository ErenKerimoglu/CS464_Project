"""Model registry and architectures. Owners: Berkay (#6, #12), Eftelya (#8).

Every model exposes: ``forward(x) -> logits``, ``forward_features(x) -> embedding``,
``gradcam_target_layer`` and ``input_size`` (PROJECT_PLAN §4.2).

Importing this package registers every architecture whose dependencies are installed.
"""
from galaxy.models import small_cnn  # noqa: F401  (registers "small_cnn")
from galaxy.models.registry import build_model, list_models, register_model

# Optional architectures: registered only when their dependency (timm / escnn) is importable.
for _module in ("timm_models", "equivariant"):
    try:
        __import__(f"galaxy.models.{_module}")
    except ImportError:
        pass

__all__ = ["build_model", "list_models", "register_model"]
