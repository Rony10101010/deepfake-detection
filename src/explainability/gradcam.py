"""Grad-CAM explainability utilities for the VGG16 deepfake detector.

This module implements Grad-CAM using the last convolutional layer of a VGG16
backbone. It provides helpers for loading a trained checkpoint, generating a
heatmap for a given input, overlaying the heatmap on the original image, and
performing a complete prediction with visualization outputs.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.vgg16_model import create_model


def _select_target_layer(model: torch.nn.Module) -> str:
    """Return the name of the last convolutional layer in the model."""
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            return name
    raise ValueError("No convolutional layer found in model.")


def load_model(
    checkpoint_path: str,
    num_classes: int = 2,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Instantiate a VGG16 model and load weights from a checkpoint, using passed device for consistency with Colab.
    
    For local inference, use_pretrained=False to prevent downloading ImageNet weights and avoid hangs.
    The trained checkpoint contains all necessary weights.
    """
    if device is None:
        device = torch.device("cpu")

    model = create_model(num_classes=num_classes, freeze_features=False, use_pretrained=False)
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
    """Generate a Grad-CAM heatmap for a single input tensor."""
    if device is None:
        device = input_tensor.device

    model.to(device)
    model.eval()
    input_tensor = input_tensor.to(device)

    if target_layer is None:
        target_layer = _select_target_layer(model)

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def _forward_hook(module, inputs, output):
        activations["value"] = output

    def _backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    layer = dict(model.named_modules())[target_layer]
    handle_f = layer.register_forward_hook(_forward_hook)
    handle_b = layer.register_full_backward_hook(_backward_hook)

    outputs = model(input_tensor)

    if target_class is None:
        target_class = int(outputs.argmax(dim=1).item())

    score = outputs[0, target_class]

    model.zero_grad()
    score.backward()

    handle_f.remove()
    handle_b.remove()

    grads = gradients["value"][0]
    acts = activations["value"][0]

    weights = grads.mean(dim=(1, 2))
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = torch.relu(cam)
    cam = cam.detach().cpu().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def overlay_heatmap(
    img: np.ndarray | str,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a heatmap on top of an RGB image, using high-quality interpolation for consistency."""
    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    heat = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heat_color + (1 - alpha) * img)
    return overlay


def predict_with_gradcam(
    model: torch.nn.Module,
    image: Image.Image | np.ndarray | str,
    transform: transforms.Compose,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Run prediction and Grad-CAM on a single image.

    Returns:
        (predicted_class, confidence, heatmap, overlay)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    original_np = np.array(pil_image)

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    predicted_class = int(pred.item())
    confidence = float(conf.item())

    heatmap = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_class=predicted_class,
        device=device,
    )

    overlay = overlay_heatmap(original_np, heatmap)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        heatmap_path = os.path.join(save_dir, "heatmap.png")
        overlay_path = os.path.join(save_dir, "overlay.png")

        plt.imsave(heatmap_path, heatmap, cmap="jet")
        plt.imsave(overlay_path, overlay)

    return predicted_class, confidence, heatmap, overlay