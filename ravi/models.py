"""Backbone loading. Everything goes through timm so one code path covers
ResNet, ConvNeXt, ViT/DeiT, Swin, CLIP, SigLIP, DINO, DINOv2/v3, MAE, BEiT, EVA.

Model spec (YAML):
  name:      timm model name with pretrained tag, e.g. "vit_base_patch14_dinov2.lvd142m"
  family:    grouping used for leave-one-family-out evaluation
  paradigm:  supervised | clip | ssl-distill | ssl-mim | ...
  keep_head: true -> keep the model's own output head (use for CLIP/SigLIP so the vector is
             the joint image-text embedding); default false -> pooled features (num_classes=0)
  img_size:  optional input size override (e.g. 224 for DINOv2, whose default is 518)
"""
from __future__ import annotations


def load_backbone(spec: dict, device: str, pretrained: bool = True):
    import timm
    import torch

    kwargs = {}
    if not spec.get("keep_head", False):
        kwargs["num_classes"] = 0
    if spec.get("img_size"):
        kwargs["img_size"] = int(spec["img_size"])
    model = timm.create_model(spec["name"], pretrained=pretrained, **kwargs)
    model.eval().to(device)
    if device.startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)
    data_cfg = timm.data.resolve_model_data_config(model)
    if spec.get("img_size"):
        s = int(spec["img_size"])
        data_cfg["input_size"] = (3, s, s)
    transform = timm.data.create_transform(**data_cfg, is_training=False)
    return model, transform, data_cfg


def forward_features(model, images, device: str, amp: bool):
    import torch

    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        if device.startswith("cuda"):
            images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(images)
        else:
            out = model(images)
        if out.ndim > 2:  # safety: pool anything that is not already a vector
            out = out.flatten(2).mean(-1) if out.ndim == 4 else out.mean(1)
        return out.float().cpu().numpy()
