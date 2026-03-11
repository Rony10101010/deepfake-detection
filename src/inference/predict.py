"""Inference utilities for the deepfake deepfake detection project.

This module provides a simple API for loading a trained checkpoint,
preprocessing an image, and running predictions with optional Grad-CAM
visualizations.  The functions are designed for reuse in both Colab notebooks
and the desktop demo application; GUI code is intentionally omitted.

Key functions
-------------

``load_trained_model``
    Shortcut for instantiating a VGG16 and loading weights from disk.
``preprocess_image``
    Read an image file and convert it to a tensor suitable for the model.
``predict_image``
    Run a forward pass to obtain class probabilities.
``predict_with_gradcam``
    Wrapper around :func:`src.explainability.gradcam.predict_with_gradcam`
    that handles preprocessing and returns prediction data.

The module also exposes a small command-line interface for quick testing.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple, Union

from PIL import Image
import torch
from torchvision import transforms

from src.explainability import gradcam


# ---------------------------------------------------------------------------
# public helpers
# ---------------------------------------------------------------------------


def load_trained_model(
    checkpoint_path: str,
    num_classes: int = 2,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a VGG16 model initialized with checkpoint weights.

    This is a thin wrapper around :func:`gradcam.load_model` so that callers in
    the inference package do not need to import from the explainability
    submodule directly.
    """

    return gradcam.load_model(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        device=device,
    )


def preprocess_image(
    image: Union[str, Image.Image],
    size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Open and preprocess an image for model input.

    Parameters
    ----------
    image : str or PIL.Image.Image
        Path to an image file or a PIL image object.
    size : tuple
        Desired output size (H, W).  Defaults to ``(224,224)`` to match VGG16.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(C, H, W)`` with normalization applied.
    """

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


def predict_image(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device | None = None,
) -> Tuple[int, float]:
    """Run a forward pass and return predicted class and confidence.

    Parameters
    ----------
    model : torch.nn.Module
        Evaluation model.
    input_tensor : torch.Tensor
        Preprocessed tensor of shape ``(C, H, W)`` or ``(1, C, H, W)``.
    device : torch.device or None
        Device to use; ``cuda`` if available when ``None``.

    Returns
    -------
    tuple
        ``(predicted_class, confidence)`` where ``confidence`` is a float.
    """

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
    """Perform inference with Grad-CAM visualization.

    This convenience function combines preprocessing, prediction, and
    visualization by delegating to :func:`gradcam.predict_with_gradcam`.

    Parameters
    ----------
    model : torch.nn.Module
        Evaluation model (already in ``eval`` mode).
    image : str or PIL.Image.Image
        Input image.
    device : torch.device or None
        Computation device.
    save_dir : str or None
        If provided, heatmap and overlay will be saved there.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return gradcam.predict_with_gradcam(
        model=model,
        image=image,
        transform=transform,
        device=device,
        save_dir=save_dir,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


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
        help="Directory in which to save gradcam outputs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import numpy as np

    args = _parse_args()
    model = load_trained_model(args.checkpoint)

    pred_class, confidence, heatmap, overlay = predict_with_gradcam(
        model=model, image=args.image, save_dir=args.save_dir
    )

    class_name = str(pred_class)
    print(f"Predicted class: {class_name} (confidence {confidence:.4f})")

    if args.save_dir:
        print(f"Heatmap and overlay saved in {args.save_dir}")
