"""
src/model.py
ASL Sign Language Classifier — Model Definitions

Two options:
  - EfficientNetB3  (default, ~12M params, best accuracy/speed tradeoff)
  - ResNet50        (fallback, widely supported)

Usage:
    from model import build_model
    model = build_model("efficientnet_b3", num_classes=29)
"""

import torch
import torch.nn as nn
from torchvision import models


# ──────────────────────────────────────────────────────────────────
def build_model(arch: str = "efficientnet_b3",
                num_classes: int = 29,
                pretrained: bool = True,
                dropout: float = 0.4) -> nn.Module:
    """
    Build and return a model ready for fine-tuning.

    Args:
        arch:        'efficientnet_b3' | 'resnet50' | 'convnext_tiny'
        num_classes: number of output classes (29 for this dataset)
        pretrained:  use ImageNet weights
        dropout:     dropout rate on the classification head

    Returns:
        nn.Module with a custom classification head
    """
    arch = arch.lower()

    if arch == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model   = models.efficientnet_b3(weights=weights)
        in_feat = model.classifier[1].in_features          # 1536
        model.classifier = _make_head(in_feat, num_classes, dropout)

    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model   = models.resnet50(weights=weights)
        in_feat = model.fc.in_features                     # 2048
        model.fc = _make_head(in_feat, num_classes, dropout)

    elif arch == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model   = models.convnext_tiny(weights=weights)
        in_feat = model.classifier[2].in_features          # 768
        model.classifier[2] = _make_head(in_feat, num_classes, dropout)

    else:
        raise ValueError(f"Unknown architecture: {arch}. "
                         "Choose from efficientnet_b3 | resnet50 | convnext_tiny")

    return model


def _make_head(in_features: int, num_classes: int, dropout: float) -> nn.Module:
    """Two-layer MLP classification head."""
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=dropout / 2),
        nn.Linear(512, num_classes),
    )


# ──────────────────────────────────────────────────────────────────
class LabelSmoothingCE(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Smoothing prevents the model from becoming overconfident,
    which improves generalisation on unseen test data.
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing   = smoothing
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        # Build smooth target distribution
        smooth = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
        smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        return -(smooth * log_probs).sum(dim=-1).mean()


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity check
    for arch in ["efficientnet_b3", "resnet50", "convnext_tiny"]:
        m = build_model(arch, num_classes=29, pretrained=False)
        x = torch.randn(4, 3, 224, 224)
        y = m(x)
        print(f"{arch:20s}  output shape: {y.shape}   "
              f"params: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
