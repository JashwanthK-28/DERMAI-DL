"""
DermAI: Skin Disease Detector from Mobile Images Using Deep Learning
Model Architectures (ResNet50 & EfficientNet-B0) + Focal / Weighted Loss
"""

import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

DIAGNOSTIC_CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
NUM_CLASSES = len(DIAGNOSTIC_CLASSES)

CLASS_DESCRIPTIONS = {
    "ACK": "Actinic Keratosis (Pre-cancerous)",
    "BCC": "Basal Cell Carcinoma (Malignant)",
    "MEL": "Melanoma (Malignant - Critical)",
    "NEV": "Melanocytic Nevus (Benign)",
    "SCC": "Squamous Cell Carcinoma (Malignant)",
    "SEK": "Seborrheic Keratosis (Benign)",
}

RISK_MAPPING = {
    "MEL": "CRITICAL",
    "BCC": "HIGH",
    "SCC": "HIGH",
    "ACK": "MODERATE",
    "NEV": "LOW",
    "SEK": "LOW",
}


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Helps focus training on hard-to-classify samples (e.g. rare MEL and SCC).
    """

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def build_resnet50(pretrained: bool = True, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.4) -> nn.Module:
    """
    ResNet50 architecture with custom classification head:
    Dense(2048 -> 512) -> ReLU -> Dropout(0.4) -> Dense(512 -> num_classes)
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, num_classes),
    )
    return model


def build_efficientnet_b0(pretrained: bool = True, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.4) -> nn.Module:
    """
    EfficientNet-B0 architecture with custom classification head:
    Dropout(0.4) -> Dense(1280 -> 512) -> ReLU -> Dropout(0.4) -> Dense(512 -> num_classes)
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, num_classes),
    )
    return model


def get_risk_category(predicted_class: str) -> str:
    """Return risk category according to diagnostic malignancy level."""
    return RISK_MAPPING.get(predicted_class, "MODERATE")
