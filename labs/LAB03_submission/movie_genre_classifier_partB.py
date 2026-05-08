"""
Lab 3 — Part B: ResNet18 Drop-In Image Branch
==============================================
Copy your completed movie_genre_classifier.py into this file, then
replace the ImageBranch class with the ResNet18 version below.
Everything else (TabularBranch, FusionHead, Dataset, training) should
remain identical to Part A.
"""

# YOUR CODE HERE
# Paste your completed Part A code here, then replace ImageBranch
# with the class below. No other changes should be needed.


# =============================================================================
# PROVIDED: ResNet18 ImageBranch (replaces your Part A ImageBranch)
# =============================================================================

import torch
import torch.nn as nn
from torchvision import models


class ImageBranch(nn.Module):
    """
    Transfer learning image encoder: pretrained ResNet18 backbone
    with a small trainable projection head.

    The entire backbone is frozen by default (only the head trains).
    Optionally, the last residual block (layer4) can be unfrozen for
    fine-tuning once the head has converged.
    """

    BACKBONE_OUT_DIM = 512  # ResNet18 feature dimension after global average pool

    def __init__(self, out_dim=256, dropout=0.4, fine_tune=False):
        super().__init__()

        # Load ResNet18 with pretrained ImageNet weights
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Freeze ALL backbone parameters
        for param in backbone.parameters():
            param.requires_grad = False

        # (Optional) Unfreeze the last residual block for fine-tuning
        if fine_tune:
            for param in backbone.layer4.parameters():
                param.requires_grad = True

        # Remove the original FC classification head; keep up to avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output: (batch, 512, 1, 1)

        # Trainable projection head
        self.head = nn.Sequential(
            nn.Flatten(),                              # (batch, 512)
            nn.Dropout(dropout),
            nn.Linear(self.BACKBONE_OUT_DIM, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        features = self.backbone(x)   # (batch, 512, 1, 1) — frozen
        return self.head(features)    # (batch, out_dim)   — trained
