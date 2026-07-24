"""
KrishiMitra - Disease Classifier Model Unit Test

Tests model initialization, forward pass, output shapes, and backpropagation.

Author:
    Antigravity AI
"""

from __future__ import annotations

import torch
import torch.nn as nn
from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier


def test_classifier_initialization_and_forward() -> None:
    print("=" * 60)
    print("Testing Disease Classifier Model Architecture")
    print("=" * 60)

    # 1. Initialize configuration and model
    # Set NUM_CLASSES to 38 for testing
    config = ClassificationConfig()
    model = DiseaseClassifier(config)
    
    # Verify properties
    print(f"Loaded classifier: {config.ARCHITECTURE}")
    print(f"Configured target classes: {config.NUM_CLASSES}")
    assert model is not None
    assert isinstance(model, nn.Module)

    # 2. Run mock forward pass
    # Input shape: (Batch Size = 4, Channels = 3, Height = 224, Width = 224)
    batch_size = 4
    mock_input = torch.randn(batch_size, 3, config.INPUT_SIZE, config.INPUT_SIZE)
    print(f"Mock input tensor shape: {mock_input.shape}")

    print("Running forward pass...")
    logits = model(mock_input)
    print(f"Forward logits tensor shape: {logits.shape}")

    # Assert correct dimensions (batch_size, num_classes)
    assert logits.shape == (batch_size, config.NUM_CLASSES), (
        f"Expected shape {(batch_size, config.NUM_CLASSES)}, got {logits.shape}"
    )

    # 3. Verify gradient backpropagation
    print("Running backward pass check...")
    criterion = nn.CrossEntropyLoss()
    mock_targets = torch.randint(0, config.NUM_CLASSES, (batch_size,))
    loss = criterion(logits, mock_targets)
    
    # Zero gradients, run backward, check that parameters receive gradients
    model.zero_grad()
    loss.backward()

    # Find the linear layer we replaced
    linear_layer = model.base_model.classifier[1]
    assert linear_layer.weight.grad is not None, "Linear layer weight did not receive gradients."
    print("Backpropagation gradient check passed!")
    print("=" * 60)
    print("Disease Classifier Model Test Passed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_classifier_initialization_and_forward()
