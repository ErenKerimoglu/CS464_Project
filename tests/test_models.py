"""Generic model-contract tests (PROJECT_PLAN §4.2). Parametrized over every registered model, so models added
in #8 / #12 are covered automatically. CPU, tiny batch, no weight downloads (``pretrained=False``)."""
from __future__ import annotations

import pytest
import torch
from torch import nn

from galaxy.constants import NUM_CLASSES
from galaxy.models import build_model, list_models, register_model
from galaxy.models.utils import count_params, model_summary

ALL_MODELS = ["small_cnn", "resnet50", "convnext_tiny", "vit_small", "escnn_d4", "escnn_c8"]


def _build(name: str) -> nn.Module:
    """Build a model for testing. Skip when it is not registered (optional dependency missing)."""
    if name not in list_models():
        pytest.skip(f"{name} not registered (optional dependency missing or not implemented yet)")
    return build_model(name, num_classes=NUM_CLASSES, pretrained=False)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_model_contract(name):
    model = _build(name)
    size = model.input_size
    assert isinstance(size, int)
    assert isinstance(model.gradcam_target_layer, nn.Module)

    x = torch.randn(2, 3, size, size)
    model.train()
    logits = model(x)
    assert logits.shape == (2, NUM_CLASSES)
    feats = model.forward_features(x)
    assert feats.ndim == 2 and feats.shape[0] == 2

    logits.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("name", ALL_MODELS)
def test_eval_is_deterministic(name):
    model = _build(name).eval()
    x = torch.randn(2, 3, model.input_size, model.input_size)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


def test_small_cnn_size_and_input_size():
    model = build_model("small_cnn", pretrained=False)
    assert model.input_size == 128
    assert 1_000_000 <= count_params(model) <= 2_000_000
    assert build_model("small_cnn", pretrained=False, input_size=64).input_size == 64
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, NUM_CLASSES)


def test_small_cnn_accepts_config_kwargs():
    # keys that appear in the config's `model:` block must not break construction
    build_model("small_cnn", pretrained=False, drop_path=0.0, freeze_backbone=False)


def test_small_cnn_param_groups_partition():
    model = build_model("small_cnn", pretrained=False)
    ids = [id(p) for p in model.backbone_parameters() + model.head_parameters()]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))


def test_unknown_model_error_lists_available():
    with pytest.raises(KeyError, match="small_cnn"):
        build_model("does_not_exist")


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        register_model("small_cnn")(lambda **kw: None)


def test_list_models_contains_small_cnn_sorted():
    names = list_models()
    assert "small_cnn" in names and names == sorted(names)


def test_utils():
    model = build_model("small_cnn", pretrained=False)
    for p in model.backbone_parameters():
        p.requires_grad = False
    assert count_params(model, trainable_only=True) == count_params(model.fc)
    model.train()
    summary = model_summary(model)
    assert "SmallCNN" in summary and "(1, 10)" in summary
    assert model.training  # restored


# ---- timm backbones (#12) ----
TIMM_MODELS = ["resnet50", "convnext_tiny", "vit_small"]


@pytest.mark.parametrize("name", TIMM_MODELS)
def test_timm_embedding_and_param_groups(name):
    model = _build(name)
    dims = {"resnet50": 2048, "convnext_tiny": 768, "vit_small": 384}
    assert model.input_size == 224
    assert model.forward_features(torch.randn(2, 3, 224, 224)).shape == (2, dims[name])
    head, backbone = model.head_parameters(), model.backbone_parameters()
    assert len(head) + len(backbone) == len(list(model.parameters()))
    assert {id(p) for p in head}.isdisjoint({id(p) for p in backbone})
    assert head[0].shape[0] == NUM_CLASSES or head[0].shape[-1] == NUM_CLASSES


@pytest.mark.parametrize("name", TIMM_MODELS)
def test_timm_gradcam_layer_is_in_forward_path(name):
    model = _build(name).eval()
    seen = []
    handle = model.gradcam_target_layer.register_forward_hook(lambda m, i, o: seen.append(o))
    model(torch.randn(1, 3, 224, 224))
    handle.remove()
    assert len(seen) == 1
    if name == "vit_small":
        assert seen[0].shape == (1, 197, 384)
        assert model.reshape_transform(seen[0]).shape == (1, 384, 14, 14)
    else:
        assert seen[0].ndim == 4 and model.reshape_transform is None


def test_timm_drop_path_and_config_keys():
    model = build_model("convnext_tiny", pretrained=False, drop_path=0.1, freeze_backbone=False)
    assert model.net.stages[-1].blocks[-1].drop_path.drop_prob > 0
