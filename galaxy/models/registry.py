"""Model registry. Issue #6 (owner: Berkay).

Every architecture registers a factory ``fn(num_classes, pretrained, **kwargs) -> nn.Module`` under a name, and the
rest of the code only ever calls :func:`build_model`.
"""
from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


def register_model(name: str):
    """Decorator: ``@register_model("small_cnn")`` on a factory ``fn(num_classes, pretrained, **kw) -> nn.Module``."""

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY and _REGISTRY[name] is not fn:
            raise ValueError(f"Model name {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def build_model(name: str, num_classes: int = 10, pretrained: bool = True, **kwargs):
    """Names: small_cnn, resnet50, convnext_tiny, vit_small, escnn_d4, escnn_c8."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"Unknown model {name!r}. Available models: {available}. "
            "A missing name may mean its optional dependency (timm, escnn) is not installed."
        )
    return _REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kwargs)


def list_models() -> list[str]:
    """Sorted names of all registered models."""
    return sorted(_REGISTRY)
