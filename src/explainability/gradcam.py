"""Grad-CAM explainability utilities for VGG16 deepfake detector.

This module implements the standard Grad-CAM algorithm using the last
convolutional layer of a VGG16 backbone.  It provides helpers for loading a
trained checkpoint, generating a heatmap for a given input, overlaying the
heatmap on the original image, and performing a complete prediction with
visualizations.

The design is intentionally lightweight so that the functions can be imported
from Google Colab notebooks or the desktop demo without pulling in extraneous
dependencies.

Functions
---------

``load_model``
    Instantiate a VGG16 model and load weights from disk.
``generate_gradcam``
    Produce a normalized heatmap for a particular class prediction.
``overlay_heatmap``
    Superimpose a heatmap on top of an image and return the result.
``predict_with_gradcam``
    Convenience wrapper that runs inference, computes Grad-CAM, and returns
    prediction, score, heatmap and overlay; optionally saves visualizations.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.vgg16_model import create_model


# ---------------------------------------------------------------------------
# helper utilities
# ---------------------------------------------------------------------------


def _select_target_layer(model: torch.nn.Module) -> str:
    """Return the name of the last ``nn.Conv2d`` module in ``model``.

    Walking the named modules in reverse order ensures the deepest convolution
    is chosen; this is the layer typically used for Grad-CAM.
    """

    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return name
    raise ValueError("no convolutional layer found in model")


# ---------------------------------------------------------------------------
# public interface
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Instantiate a VGG16 and load a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pth`` file produced by the training routine.
    num_classes : int
        Number of classes used during training (default ``2``).
    device : torch.device or None
        Device on which to place the model.  If ``None`` the function will
        select ``cuda`` if available or ``cpu`` otherwise.

    Returns
    -------
    torch.nn.Module
        Model ready for inference (``.eval()`` has been called).
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=num_classes, freeze_features=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    target_layer: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single input.

    Parameters
    ----------
    model : torch.nn.Module
        A VGG16 model in evaluation mode.
    input_tensor : torch.Tensor
        Preprocessed tensor of shape ``(1, C, H, W)``.
    target_class : int or None
        Index of the class for which to compute Grad-CAM.  If ``None`` the
        model's predicted class is used.
    target_layer : str or None
        Dot-separated name of the convolutional layer to hook.  If ``None``
        the last ``nn.Conv2d`` layer in the model is selected automatically.
    device : torch.device or None
        Device for computation; inferred from ``input_tensor`` if ``None``.

    Returns
    -------
    numpy.ndarray
        Heatmap values in ``[0, 1]`` with shape ``(H, W)``.
    """

    if device is None:
        device = input_tensor.device
    model.to(device)
    input_tensor = input_tensor.to(device)

    if target_layer is None:
        target_layer = _select_target_layer(model)

    # storage for activation and gradient
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _forward_hook(module, inp, outp):
        activations["value"] = outp

    def _backward_hook(module, grad_in, grad_out):
        # grad_out is a tuple; for conv layers there's a single element
        gradients["value"] = grad_out[0]

    # attach hooks
    layer = dict(model.named_modules())[target_layer]
    handle_f = layer.register_forward_hook(_forward_hook)
    handle_b = layer.register_backward_hook(_backward_hook)

    # forward pass
    outputs = model(input_tensor)
    if target_class is None:
        target_class = int(outputs.argmax(dim=1).item())

    score = outputs[0, target_class]

    model.zero_grad()
    score.backward(retain_graph=True)

    # remove hooks
    handle_f.remove()
    handle_b.remove()

    # compute weights: global average pooling of gradients
    grads = gradients["value"][0]  # shape: (C, H, W)
    pooled = grads.mean(dim=(1, 2))  # shape: (C,)
    acts = activations["value"][0]  # shape: (C, H, W)

    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
    for i, w in enumerate(pooled):
        cam += w * acts[i]

    cam = torch.relu(cam)
    cam = cam.cpu().detach().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def overlay_heatmap(
    img: Union[np.ndarray, str],
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a heatmap onto an image.

    Parameters
    ----------
    img : ndarray or str
        Either a HxWx3 numpy array in RGB order, or a path to an image file.
    heatmap : ndarray
        2D array of values in ``[0, 1]``.
    alpha : float
        Opacity factor for the heatmap overlay.

    Returns
    -------
    ndarray
        RGB image with the heatmap superimposed (dtype ``uint8``).
    """

    if isinstance(img, str):
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    heat = cv2.resize(heatmap, (w, h))
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heat_color + (1 - alpha) * img)
    return overlay


def predict_with_gradcam(
    model: torch.nn.Module,
    pil_image,
    transform: transforms.Compose,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """Run a prediction and produce Grad-CAM visualizations.

    Parameters
    ----------
    model : torch.nn.Module
        Inference model (should already be in ``eval`` mode).
    pil_image : PIL.Image.Image or ndarray or str
        Input image; a PIL object, numpy array, or file path are accepted.
    transform : torchvision.transforms.Compose
        Preprocessing pipeline to apply to the image before inference.
    device : torch.device or None
        Device for computation; defaults to ``cuda`` if available.
    save_dir : str or None
        If provided, the heatmap and overlay images will be written to this
        directory under names ``heatmap.png`` and ``overlay.png``.

    Returns
    -------
    tuple
        ``(predicted_class, confidence, heatmap, overlay)`` where
        ``heatmap`` and ``overlay`` are numpy arrays.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if isinstance(pil_image, str):
        pil_image = Image.open(pil_image).convert("RGB")

    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    heatmap = generate_gradcam(
        model, input_tensor, target_class=int(pred.item()), device=device
    )
    overlay = overlay_heatmap(pil_image if isinstance(pil_image, str) else np.array(pil_image), heatmap)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        heat_path = os.path.join(save_dir, "heatmap.png")
        ov_path = os.path.join(save_dir, "overlay.png")
        plt.imsave(heat_path, heatmap, cmap="jet")
        plt.imsave(ov_path, overlay)

    return int(pred.item()), float(conf.item()), heatmap, overlay
