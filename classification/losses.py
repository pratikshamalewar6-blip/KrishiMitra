"""
KrishiMitra - Classification Loss Functions

Implements Class-Weighted CrossEntropy and Focal Loss.
Calculates weights dynamically based on class frequencies.

Author:
    Antigravity AI
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance in multiclass classification.
    Formula: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ) -> None:
        """
        Parameters
        ----------
        alpha : torch.Tensor
            A tensor of shape (num_classes,) containing weights for each class.
        gamma : float
            Focusing parameter. Higher values reduce loss for easy/correctly-classified samples.
        reduction : str
            Specifies reduction: 'none' | 'mean' | 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute cross-entropy loss (unreduced)
        # Note: If alpha weights are provided, they are applied here
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        
        # Calculate probability pt
        pt = torch.exp(-ce_loss)
        
        # Apply Focal Loss scaling factor
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def calculate_class_weights(class_counts: dict[int, int], num_classes: int, beta: float = 0.99) -> torch.Tensor:
    """
    Calculates class weights using the inverse-frequency or Effective Number of Samples strategy.
    
    Parameters
    ----------
    class_counts : dict[int, int]
        Dictionary mapping class index to count of samples.
    num_classes : int
        Total number of target classes.
    beta : float
        Hyperparameter for effective number of samples. Set to 0.0 for pure inverse-frequency.
    """
    weights = np.ones(num_classes, dtype=np.float32)
    
    # Fill counts
    counts = np.zeros(num_classes, dtype=np.float32)
    for idx, count in class_counts.items():
        if 0 <= idx < num_classes:
            counts[idx] = max(1.0, float(count))
            
    # If a class is completely missing, give it a count of 1.0
    counts[counts == 0.0] = 1.0

    if beta > 0.0:
        # Effective number of samples: (1 - beta) / (1 - beta^n)
        effective_num = (1.0 - beta) / (1.0 - np.power(beta, counts))
        weights = effective_num / np.sum(effective_num) * num_classes
    else:
        # Standard inverse frequency: total_samples / (num_classes * class_samples)
        total_samples = np.sum(counts)
        weights = total_samples / (num_classes * counts)

    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights)


def get_loss_criterion(
    loss_name: str,
    class_counts: dict[int, int],
    num_classes: int,
    device: str,
    gamma: float = 2.0,
    beta: float = 0.99
) -> nn.Module:
    """
    Factory function to return the configured loss criterion with calculated weights.
    """
    # 1. Calculate weights
    weights = calculate_class_weights(class_counts, num_classes, beta).to(device)
    
    # 2. Select Loss
    if loss_name.lower() in ["weightedcrossentropy", "weighted_ce"]:
        return nn.CrossEntropyLoss(weight=weights)
    elif loss_name.lower() in ["focalloss", "focal_loss"]:
        return FocalLoss(alpha=weights, gamma=gamma)
    else:
        # Standard unweighted CrossEntropy
        return nn.CrossEntropyLoss()
