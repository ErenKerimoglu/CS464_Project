"""Model registry. Issue #6 (owner: Berkay)."""
from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Decorator: ``@register_model("small_cnn")`` on a factory ``fn(num_classes, pretrained, **kw) -> nn.Module``."""
    raise NotImplementedError("Issue #6")


def build_model(name: str, num_classes: int = 10, pretrained: bool = True, **kwargs):
    """Names: small_cnn, resnet50, convnext_tiny, vit_small, escnn_d4, escnn_c8."""
    raise NotImplementedError("Issue #6")


def list_models() -> list[str]:
    raise NotImplementedError("Issue #6")
