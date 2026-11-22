"""Small CNN trained from scratch. Issue #6 (owner: Berkay).

4 blocks of (conv3×3-BN-ReLU)×2 + maxpool, widths 32→64→128→256, GAP, dropout 0.3, linear head (~1–2 M params).
"""
from __future__ import annotations

import torch
from torch import nn

from galaxy.models.registry import register_model


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """From-scratch CNN baseline (~1.2 M parameters), following the model contract of PROJECT_PLAN §4.2."""

    def __init__(self, num_classes: int = 10, input_size: int = 128, in_channels: int = 3,
                 widths: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        chans = (in_channels, *widths)
        self.blocks = nn.ModuleList(_block(a, b) for a, b in zip(chans[:-1], chans[1:], strict=True))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(widths[-1], num_classes)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Last conv block (Grad-CAM target)."""
        return self.blocks[-1]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Global-average-pooled embedding, shape [B, widths[-1]]."""
        for block in self.blocks:
            x = block(x)
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(self.forward_features(x)))

    def backbone_parameters(self):
        """Everything except the classifier head (used for linear probing / per-group LR)."""
        return [p for n, p in self.named_parameters() if not n.startswith("fc.")]

    def head_parameters(self):
        return list(self.fc.parameters())


@register_model("small_cnn")
def small_cnn(num_classes: int = 10, pretrained: bool = False, input_size: int = 128, dropout: float = 0.3,
              drop_path: float = 0.0, freeze_backbone: bool = False, **kwargs) -> SmallCNN:
    """Factory. ``pretrained``, ``drop_path`` and ``freeze_backbone`` are the keys of the base config's ``model:``
    block; they are accepted so ``build_model(**cfg["model"])`` works, and ignored here (the trainer freezes)."""
    del pretrained, drop_path, freeze_backbone
    return SmallCNN(num_classes=num_classes, input_size=input_size, dropout=dropout, **kwargs)
