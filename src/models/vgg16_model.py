"""VGG16 model module for deepfake detection.

This module provides a function to create a VGG16 model adapted for binary
classification (real vs. fake). The model uses a pretrained VGG16 backbone
from ImageNet, with a modified classifier head that includes dropout for
regularization.

The classifier structure is:
- Fully connected layer (4096 units)
- Dropout
- Output layer (2 units for binary classification)

Example usage::

    from src.models.vgg16_model import create_model

    model = create_model(num_classes=2, freeze_features=True)
    # Model is ready for training with frozen backbone

"""

import torch
import torch.nn as nn
import torchvision.models as models


def create_model(
    num_classes: int = 2,
    freeze_features: bool = True,
) -> models.VGG:
    """Create a VGG16 model for binary classification.

    Loads a pretrained VGG16 model, replaces the classifier head with a
    custom sequence suitable for binary classification, and optionally
    freezes the convolutional backbone.

    Args:
        num_classes: number of output classes (default 2 for binary).
        freeze_features: if True, freeze the convolutional layers to train
            only the classifier head initially.

    Returns:
        A PyTorch VGG16 model with modified classifier, ready for training.
    """

    # Load pretrained VGG16
    model = models.vgg16(pretrained=True)

    # Replace the classifier with a custom head
    # Original classifier input features: 25088
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),  # Fully connected layer
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),       # Dropout for regularization
        nn.Linear(4096, num_classes),  # Output layer
    )

    # Optionally freeze the feature extractor (convolutional layers)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    return model


# end of file