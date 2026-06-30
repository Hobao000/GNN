from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view(-1)
        logits = logits.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - p_t).pow(self.gamma) * bce).mean()


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long().view(-1)
    neg = int((labels == 0).sum().item())
    pos = int((labels == 1).sum().item())
    total = max(1, neg + pos)
    return torch.tensor([total / (2 * max(1, neg)), total / (2 * max(1, pos))], dtype=torch.float32)


def make_loss(loss_name: str, train_labels: torch.Tensor, focal_alpha: float = 0.75, focal_gamma: float = 2.0) -> nn.Module:
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=compute_class_weights(train_labels))
    if loss_name == "focal":
        return BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    if loss_name == "bce":
        labels = train_labels.long().view(-1)
        neg = max(1, int((labels == 0).sum().item()))
        pos = max(1, int((labels == 1).sum().item()))
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32))
    raise ValueError(f"Unsupported loss: {loss_name}")


def masked_or_seed_loss(outputs: torch.Tensor, labels: torch.Tensor, criterion: nn.Module, loss_name: str) -> torch.Tensor:
    if loss_name == "weighted_ce":
        return criterion(outputs, labels.long())
    return criterion(outputs.view(-1), labels.float().view(-1))


def positive_probabilities(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim == 2 and outputs.size(1) == 2:
        return torch.softmax(outputs, dim=1)[:, 1]
    return torch.sigmoid(outputs.view(-1))
