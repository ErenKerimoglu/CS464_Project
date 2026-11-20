"""ImageNet-pretrained backbones via timm. Issue #12 (owner: Berkay).

| registry name | timm id                                       | Grad-CAM layer        |
|---------------|-----------------------------------------------|-----------------------|
| resnet50      | resnet50.tv_in1k                              | layer4[-1]            |
| convnext_tiny | convnext_tiny.fb_in1k                         | stages[-1]            |
| vit_small     | vit_small_patch16_224.augreg_in21k_ft_in1k    | blocks[-1].norm1 (+ reshape_transform) |
"""
TIMM_IDS = {
    "resnet50": "resnet50.tv_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}
