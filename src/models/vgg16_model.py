"""VGG16 model module for deepfake detection."""

import torch.nn as nn
import torchvision.models as models


def create_model(
    num_classes: int = 2,
    freeze_features: bool = True,
    use_pretrained: bool = True,
) -> models.VGG:
    """Create a VGG16 model for binary classification."""

    if use_pretrained:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        model = models.vgg16(weights=None)

    # Disable in-place ReLU everywhere to avoid Grad-CAM backward hook errors
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.5),
        nn.Linear(4096, num_classes),
    )

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    return model