"""
KrishiMitra - Disease Classifier Model

Wraps the pre-trained EfficientNet-B0 architecture with a custom classification head.

Author:
    Antigravity AI
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
from common.logger import LoggerManager
from classification.config import ClassificationConfig


class DiseaseClassifier(nn.Module):
    """
    Disease Classifier using EfficientNet-B0.
    """

    def __init__(self, config: ClassificationConfig | None = None) -> None:
        super().__init__()
        self.logger = LoggerManager.get_logger("DiseaseClassifier")
        self.config = config or ClassificationConfig()

        self.logger.info(f"Building EfficientNet-B0 classifier for {self.config.NUM_CLASSES} classes...")

        # Load pre-trained base model
        if self.config.PRETRAINED:
            try:
                # Try the newer torchvision weights API
                from torchvision.models import EfficientNet_B0_Weights
                self.base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                self.logger.info("Loaded pre-trained EfficientNet-B0 weights (EfficientNet_B0_Weights.DEFAULT)")
            except ImportError:
                # Fallback to the older pretrained parameter
                self.base_model = models.efficientnet_b0(pretrained=True)
                self.logger.info("Loaded pre-trained EfficientNet-B0 weights (pretrained=True)")
        else:
            self.base_model = models.efficientnet_b0(weights=None)
            self.logger.info("Loaded un-initialized EfficientNet-B0 architecture")

        # Replace classifier head mapping to self.config.NUM_CLASSES
        in_features = self.base_model.classifier[1].in_features
        
        # We replace the final linear layer (index 1) in self.base_model.classifier sequence
        self.base_model.classifier[1] = nn.Linear(in_features, self.config.NUM_CLASSES)
        self.logger.info(f"Custom classifier head initialized: nn.Linear(in_features={in_features}, out_features={self.config.NUM_CLASSES})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input batch of images, shape: (batch_size, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            Logits, shape: (batch_size, num_classes)
        """
        return self.base_model(x)
