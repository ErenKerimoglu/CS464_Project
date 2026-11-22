"""ImageNet-pretrained backbones via timm. Issue #12 (owner: Berkay).

| registry name | timm id                                       | Grad-CAM layer        |
|---------------|-----------------------------------------------|-----------------------|
| resnet50      | resnet50.tv_in1k                              | layer4[-1]            |
| convnext_tiny | convnext_tiny.fb_in1k                         | stages[-1]            |
| vit_small     | vit_small_patch16_224.augreg_in21k_ft_in1k    | blocks[-1].norm1 (+ reshape_transform) |

Inputs are standardized with our own per-band dataset statistics (``GalaxyPreprocess``), not the ImageNet mean/std:
fine-tuning adapts the first layer to the new input distribution, and the g/r/z bands of DECaLS are not RGB anyway.
"""
from __future__ import annotations

import timm
import torch
from torch import nn

from galaxy.models.registry import register_model

TIMM_IDS = {
    "resnet50": "resnet50.tv_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}
INPUT_SIZE = 224


def _gradcam_layer(name: str, net: nn.Module) -> nn.Module:
    if name == "resnet50":
        return net.layer4[-1]
    if name == "convnext_tiny":
        return net.stages[-1]
    return net.blocks[-1].norm1  # vit_small


def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """``reshape_transform`` for pytorch-grad-cam with ViT-S/16 at 224 px: [B, 1 + H*W, C] → [B, C, H, W].

    Drops the class token and puts the 14×14 patch grid back into a feature map.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


class TimmClassifier(nn.Module):
    """A timm backbone with a fresh 10-class head, following the model contract of PROJECT_PLAN §4.2."""

    def __init__(self, name: str, num_classes: int = 10, pretrained: bool = True, drop_path: float = 0.0,
                 drop_rate: float = 0.0, input_size: int = INPUT_SIZE):
        super().__init__()
        self.name = name
        self.input_size = input_size
        self.net = timm.create_model(TIMM_IDS[name], pretrained=pretrained, num_classes=num_classes,
                                     drop_path_rate=drop_path, drop_rate=drop_rate)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return _gradcam_layer(self.name, self.net)

    @property
    def reshape_transform(self):
        """Grad-CAM ``reshape_transform`` (ViT only; ``None`` for the CNNs)."""
        return vit_reshape_transform if self.name == "vit_small" else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled pre-logit embedding, shape [B, D] (2048 / 768 / 384)."""
        return self.net.forward_head(self.net.forward_features(x), pre_logits=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def head_parameters(self):
        """The new classifier layer (trained in linear probing; usually gets its own LR)."""
        return list(self.net.get_classifier().parameters())

    def backbone_parameters(self):
        """Everything except the classifier layer."""
        head = {id(p) for p in self.head_parameters()}
        return [p for p in self.parameters() if id(p) not in head]


def _factory(name: str):
    def build(num_classes: int = 10, pretrained: bool = True, drop_path: float = 0.0, drop_rate: float = 0.0,
              freeze_backbone: bool = False, **kwargs) -> TimmClassifier:
        # ``freeze_backbone`` is a key of the base config's model block; the trainer does the freezing.
        del freeze_backbone
        return TimmClassifier(name, num_classes=num_classes, pretrained=pretrained, drop_path=drop_path,
                              drop_rate=drop_rate, **kwargs)

    build.__name__ = name
    return build


for _name in TIMM_IDS:
    register_model(_name)(_factory(_name))
