"""Rotation-equivariant CNN with escnn. Issue #8 (owner: Eftelya)."""
from __future__ import annotations

import torch
from torch import nn

try:
    from escnn import gspaces
    from escnn import nn as enn
except ImportError:  # Package is optional and registers no model when escnn is unavailable.
    gspaces = None
    enn = None

from galaxy.models.registry import register_model


class EquivariantCNN(nn.Module):
    """A regular-representation steerable CNN for D4 or C8 inputs."""

    def __init__(
        self,
        group: str = "d4",
        num_classes: int = 10,
        input_size: int = 129,
        width: int = 16,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        del pretrained, kwargs
        super().__init__()
        if enn is None or gspaces is None:
            raise ImportError("escnn is required to construct EquivariantCNN")
        if group not in {"d4", "c8"}:
            raise ValueError("group must be 'd4' or 'c8'")
        if input_size % 2 == 0:
            raise ValueError("input_size must be odd so 90-degree rotations stay exact")

        self.group_name = group
        self.input_size = input_size
        self.r2_act = gspaces.flipRot2dOnR2(N=4) if group == "d4" else gspaces.rot2dOnR2(N=8)

        trivial = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * 3)
        self.input_type = trivial
        self.mask = enn.MaskModule(trivial, S=input_size, margin=1)

        widths = [width, width, width * 2, width * 2, width * 4]
        in_type = trivial
        blocks = []
        for index, channels in enumerate(widths):
            out_type = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * channels)
            kernel_size = 7 if index == 0 else 5
            blocks.append(
                enn.SequentialModule(
                    enn.R2Conv(
                        in_type,
                        out_type,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    enn.InnerBatchNorm(out_type),
                    enn.ReLU(out_type, inplace=True),
                )
            )
            if index < len(widths) - 1:
                blocks[-1].append(enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.6, stride=2))
            in_type = out_type

        self.blocks = nn.ModuleList(blocks)
        self.group_type = in_type
        self.group_pool = enn.GroupPooling(in_type)
        pooled_width = widths[-1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_width, max(pooled_width, 128)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(max(pooled_width, 128), num_classes),
        )
        self.gradcam_target_layer = self.blocks[-1]

        if freeze_backbone:
            for parameter in self.backbone_parameters():
                parameter.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Invariant feature embedding before the classifier, shape [B, D]."""
        features = enn.GeometricTensor(x, self.input_type)
        features = self.mask(features)
        for block in self.blocks:
            features = block(features)
        features = self.group_pool(features)
        return self.gap(features.tensor).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def head_parameters(self):
        return list(self.head.parameters())

    def backbone_parameters(self):
        head_ids = {id(parameter) for parameter in self.head_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in head_ids]


def _factory(group: str):
    def build(
        num_classes: int = 10,
        pretrained: bool = False,
        input_size: int = 129,
        width: int = 16,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> EquivariantCNN:
        del kwargs
        return EquivariantCNN(
            group=group,
            num_classes=num_classes,
            pretrained=pretrained,
            input_size=input_size,
            width=width,
            freeze_backbone=freeze_backbone,
        )

    build.__name__ = f"escnn_{group}"
    return build


if enn is not None:
    register_model("escnn_d4")(_factory("d4"))
    register_model("escnn_c8")(_factory("c8"))
