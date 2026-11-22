"""Equivariance contract tests for the escnn architectures (#8). CPU/synthetic only."""
from __future__ import annotations

import pytest
import torch

from galaxy.models import build_model, list_models

pytest.importorskip("escnn")
pytestmark = pytest.mark.skipif(
    "escnn_d4" not in list_models() or "escnn_c8" not in list_models(),
    reason="escnn cannot be fully imported in this environment",
)


@pytest.fixture
def d4_model():
    model = build_model("escnn_d4", num_classes=10, pretrained=False)
    return model.eval()


@pytest.fixture
def batch():
    torch.manual_seed(464)
    return torch.randn(2, 3, 129, 129)


def test_d4_rot90_and_flip_invariance(d4_model, batch):
    with torch.no_grad():
        baseline = d4_model(batch)
        for transformed in (
            torch.rot90(batch, 1, dims=(2, 3)),
            torch.rot90(batch, 2, dims=(2, 3)),
            torch.rot90(batch, 3, dims=(2, 3)),
            torch.flip(batch, dims=(3,)),
        ):
            assert torch.allclose(d4_model(transformed), baseline, atol=1e-4, rtol=1e-4)


def test_c8_exact_rot90_invariance():
    model = build_model("escnn_c8", num_classes=10, pretrained=False).eval()
    torch.manual_seed(464)
    batch = torch.randn(2, 3, 129, 129)
    with torch.no_grad():
        baseline = model(batch)
        assert torch.allclose(model(torch.rot90(batch, 1, dims=(2, 3))), baseline, atol=1e-4, rtol=1e-4)


def test_report_45_degree_invariance_error(d4_model, batch, capsys):
    from torchvision import transforms

    with torch.no_grad():
        baseline = d4_model(batch)
        transformed = transforms.functional.rotate(batch, 45)
        error = (d4_model(transformed) - baseline).abs().max().item()
    print(f"D4 45-degree invariance error: {error:.6e}")
    assert "D4 45-degree invariance error:" in capsys.readouterr().out
