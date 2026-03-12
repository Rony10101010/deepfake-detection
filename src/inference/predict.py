"""Inference utilities for the deepfake detection project.

This module provides a simple API for loading a trained checkpoint,
preprocessing an image, and running predictions with optional Grad-CAM
visualizations. The functions are designed for reuse in both Colab notebooks
and the desktop demo application; GUI code is intentionally omitted.
"""

from __future__ import annotations

import argparse
from typing import Tuple, Union

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.explainability import gradcam


CLASS_NAMES = ["fake", "real"]


def load_trained_model(
    checkpoint_path: str,
    num_classes: int = 2,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a VGG16 model initialized with checkpoint weights."""
    return gradcam.load_model(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
    )


def _build_inference_transform(
    size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """Build the preprocessing pipeline used for inference."""
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_image(
    image: Union[str, Image.Image],
    size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Open and preprocess an image for model input."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = _build_inference_transform(size=size)
    return transform(image)


def predict_image(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device | None = None,
) -> Tuple[int, float]:
    """Run a forward pass and return predicted class and confidence."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
    tensor = tensor.to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return int(pred.item()), float(conf.item())


def predict_with_gradcam(
    model: torch.nn.Module,
    image: Union[str, Image.Image],
    device: torch.device | None = None,
    save_dir: str | None = None,
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """Perform inference with Grad-CAM visualization."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = _build_inference_transform()

    return gradcam.predict_with_gradcam(
        model=model,
        pil_image=image,
        transform=transform,
        device=device,
        save_dir=save_dir,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory in which to save Grad-CAM outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model = load_trained_model(args.checkpoint)

    pred_class, confidence, heatmap, overlay = predict_with_gradcam(
        model=model,
        image=args.image,
        save_dir=args.save_dir,
    )

    class_name = (
        CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else str(pred_class)
    )

    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.4f}")

    if args.save_dir:
        print(f"Heatmap and overlay saved in: {args.save_dir}")