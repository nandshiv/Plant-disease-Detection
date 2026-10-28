"""The two models we compare.

1. SimpleCNN      - a small CNN trained from scratch (baseline).
2. EfficientNet-B0 - pretrained on ImageNet, fine-tuned on our leaves
                     (transfer learning). Small (~4.1M params with our
                     38-class head) and fast on a T4.
"""

import torch
import torch.nn as nn
from torchvision import models


def conv_block(in_ch, out_ch):
    """Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool (halves H and W)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    """4 conv blocks: 224 -> 112 -> 56 -> 28 -> 14, then global average pooling.

    About 1.2M parameters. Global average pooling (instead of a big flatten +
    dense layer) keeps the parameter count low and reduces overfitting.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)          # (B, 256, 14, 14)
        x = self.pool(x).flatten(1)   # (B, 256)
        return self.classifier(x)     # (B, num_classes) raw scores (logits)


def build_efficientnet(num_classes, pretrained=True, freeze_backbone=False):
    """EfficientNet-B0 with its last layer replaced for our number of classes.

    pretrained=True     : start from ImageNet weights (this IS transfer learning).
    freeze_backbone=True: train only the new last layer ("feature extraction").
    freeze_backbone=False (default): update all layers ("fine-tuning").
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)  # new head, randomly initialised
    return model


def get_model(name, num_classes, pretrained=True):
    """name = "cnn" (from scratch) or "effnet" (transfer learning)."""
    if name == "cnn":
        return SimpleCNN(num_classes)
    if name == "effnet":
        return build_efficientnet(num_classes, pretrained=pretrained)
    raise ValueError(f"unknown model name: {name}")


def get_gradcam_layer(model, name):
    """The last convolutional layer: Grad-CAM uses its feature maps."""
    if name == "cnn":
        return model.features[-1]
    if name == "effnet":
        return model.features[-1]  # last Conv-BN-SiLU block, output (B, 1280, 7, 7)
    raise ValueError(f"unknown model name: {name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model, name, class_names, path, extra=None):
    """Save weights plus what the app needs to rebuild the model."""
    torch.save({"model_name": name, "class_names": class_names,
                "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild the model from a checkpoint. Returns (model, model_name, class_names)."""
    ckpt = torch.load(path, map_location=device)
    # pretrained=False: the weights come from the checkpoint, no need to download ImageNet weights.
    model = get_model(ckpt["model_name"], len(ckpt["class_names"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["model_name"], ckpt["class_names"]
