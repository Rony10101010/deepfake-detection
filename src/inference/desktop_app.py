"""Desktop demo application for deepfake detection with Grad-CAM visualization.

This module provides a simple Tkinter-based GUI application that allows users to:
1. Select an image from disk
2. Display the image
3. Run deepfake detection (real or fake)
4. View prediction with confidence score
5. Visualize Grad-CAM heatmap and overlay

The application reuses inference utilities from:
- src.inference.predict
- src.explainability.gradcam
- src.models.vgg16_model

Design goals:
- Simple and intuitive user interface
- Clear visualization of results
- Compatible with PyInstaller packaging
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import torch

from src.inference import predict


# Constants
CLASS_NAMES = ["fake", "real"]
WINDOW_TITLE = "Deepfake Detection Demo"
IMG_DISPLAY_SIZE = (400, 400)
DEFAULT_CHECKPOINT = "outputs/models/vgg16_checkpoint.pth"


class DeepfakeDetectorApp:
    """Tkinter-based desktop application for deepfake detection."""

    def __init__(self, root: tk.Tk, checkpoint_path: Optional[str] = None):
        """Initialize the application.

        Args:
            root: The Tkinter root window.
            checkpoint_path: Path to the trained model checkpoint.
                If None, defaults to outputs/models/vgg16_checkpoint.pth
        """
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1000x800")

        # Checkpoint path
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT

        # State
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_image_path: Optional[str] = None
        self.current_image_pil: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None

        # Build UI
        self._build_ui()
        self._load_model()

    def _build_ui(self) -> None:
        """Build the user interface components."""
        # Top control panel
        control_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            control_frame,
            text="Deepfake Detection Demo",
            font=("Arial", 14, "bold"),
            bg="#f0f0f0",
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="Select Image",
            command=self._on_select_image,
            width=15,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="Run Detection",
            command=self._on_run_detection,
            width=15,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame,
            text="Clear",
            command=self._on_clear,
            width=10,
            bg="#f44336",
            fg="white",
            font=("Arial", 10),
        ).pack(side=tk.LEFT, padx=5)

        # Main content area
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Image display
        left_frame = tk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(
            left_frame,
            text="Input Image",
            font=("Arial", 11, "bold"),
        ).pack(anchor=tk.W, pady=(0, 5))

        self.image_label = tk.Label(
            left_frame,
            bg="lightgray",
            width=50,
            height=20,
            text="No image selected",
            font=("Arial", 10),
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # File path label
        self.filepath_label = tk.Label(
            left_frame,
            text="",
            font=("Arial", 8),
            fg="gray",
            wraplength=400,
        )
        self.filepath_label.pack(anchor=tk.W, pady=(5, 0))

        # Right: Results panel
        right_frame = tk.Frame(content_frame, bg="white", relief=tk.SUNKEN, bd=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        tk.Label(
            right_frame,
            text="Prediction Results",
            font=("Arial", 11, "bold"),
            bg="white",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Prediction result
        self.prediction_label = tk.Label(
            right_frame,
            text="Status: Ready",
            font=("Arial", 10),
            bg="white",
            justify=tk.LEFT,
        )
        self.prediction_label.pack(anchor=tk.W, padx=10, pady=5)

        # Separator
        tk.Frame(right_frame, height=1, bg="gray").pack(
            fill=tk.X, padx=10, pady=10
        )

        # Grad-CAM visualization
        tk.Label(
            right_frame,
            text="Grad-CAM Visualization",
            font=("Arial", 10, "bold"),
            bg="white",
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Heatmap and overlay display
        self.gradcam_frame = tk.Frame(right_frame, bg="white")
        self.gradcam_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.heatmap_label = tk.Label(
            self.gradcam_frame,
            bg="lightgray",
            width=35,
            height=12,
            text="Heatmap",
            font=("Arial", 9),
        )
        self.heatmap_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.overlay_label = tk.Label(
            self.gradcam_frame,
            bg="lightgray",
            width=35,
            height=12,
            text="Overlay",
            font=("Arial", 9),
        )
        self.overlay_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Status bar
        status_frame = tk.Frame(self.root, bg="#e0e0e0", padx=10, pady=5)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(
            status_frame,
            text="Model loaded. Ready to detect.",
            font=("Arial", 9),
            bg="#e0e0e0",
            justify=tk.LEFT,
        )
        self.status_label.pack(anchor=tk.W)

    def _load_model(self) -> None:
        """Load the trained model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            messagebox.showwarning(
                "Model Not Found",
                f"Checkpoint not found at {self.checkpoint_path}\n\n"
                "Please train the model first or provide a valid checkpoint path.",
            )
            self.status_label.config(
                text=f"Error: Checkpoint not found at {self.checkpoint_path}"
            )
            return

        try:
            self.status_label.config(text="Loading model...")
            self.root.update()

            self.model = predict.load_trained_model(
                self.checkpoint_path, device=self.device
            )

            self.status_label.config(text="Model loaded successfully.")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n{e}")
            self.status_label.config(text=f"Error loading model: {e}")

    def _on_select_image(self) -> None:
        """Handle image selection dialog."""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
            ("All files", "*.*"),
        ]
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=filetypes,
        )

        if not file_path:
            return

        self.current_image_path = file_path
        self._display_image(file_path)
        self.filepath_label.config(text=f"File: {file_path}")

    def _display_image(self, image_path: str) -> None:
        """Display the selected image in the GUI.

        Args:
            image_path: Path to the image file.
        """
        try:
            self.current_image_pil = Image.open(image_path).convert("RGB")
            img_resized = self.current_image_pil.copy()
            img_resized.thumbnail(IMG_DISPLAY_SIZE, Image.Resampling.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(img_resized)
            self.image_label.config(image=self.photo_image, text="")
            self.image_label.image = self.photo_image

            self._clear_results()
            self.status_label.config(text="Image loaded. Click 'Run Detection' to analyze.")
        except Exception as e:
            messagebox.showerror("Image Loading Error", f"Failed to load image:\n{e}")
            self.status_label.config(text=f"Error loading image: {e}")

    def _on_run_detection(self) -> None:
        """Handle detection button click."""
        if self.model is None:
            messagebox.showwarning("Model Not Loaded", "Please load a model first.")
            return

        if self.current_image_path is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return

        try:
            self.status_label.config(text="Running detection...")
            self.root.update()

            pred_class, confidence, heatmap, overlay = predict.predict_with_gradcam(
                model=self.model,
                image=self.current_image_path,
                device=self.device,
            )

            self._display_results(pred_class, confidence, heatmap, overlay)
            self.status_label.config(text="Detection complete.")

        except Exception as e:
            messagebox.showerror("Detection Error", f"Failed to run detection:\n{e}")
            self.status_label.config(text=f"Error during detection: {e}")

    def _display_results(
        self,
        pred_class: int,
        confidence: float,
        heatmap: np.ndarray,
        overlay: np.ndarray,
    ) -> None:
        """Display prediction results and visualizations.

        Args:
            pred_class: Predicted class index (0=fake, 1=real).
            confidence: Confidence score (0-1).
            heatmap: Grad-CAM heatmap array.
            overlay: Heatmap overlaid on original image.
        """
        # Update prediction label
        class_name = (
            CLASS_NAMES[pred_class]
            if pred_class < len(CLASS_NAMES)
            else f"Class {pred_class}"
        )
        result_text = f"Prediction: {class_name.upper()}\nConfidence: {confidence:.2%}"
        self.prediction_label.config(text=result_text)

        # Display heatmap
        self._display_array_as_image(heatmap, self.heatmap_label, cmap="jet")

        # Display overlay
        self._display_array_as_image(overlay, self.overlay_label, cmap=None)

    def _display_array_as_image(
        self,
        array: np.ndarray,
        label: tk.Label,
        cmap: Optional[str] = None,
    ) -> None:
        """Convert a numpy array to an image and display it in a label.

        Args:
            array: Numpy array (H × W or H × W × 3).
            label: Tkinter label to display the image.
            cmap: Colormap name (if None, treat as RGB).
        """
        try:
            if cmap:
                # Apply colormap
                colored = plt.cm.get_cmap(cmap)(array)
                img_array = (colored[:, :, :3] * 255).astype(np.uint8)
            else:
                # Assume RGB
                img_array = (array * 255).astype(np.uint8) if array.max() <= 1.0 else array.astype(np.uint8)

            img = Image.fromarray(img_array)
            img.thumbnail((200, 150), Image.Resampling.LANCZOS)

            photo = ImageTk.PhotoImage(img)
            label.config(image=photo, text="")
            label.image = photo
        except Exception as e:
            label.config(text=f"Error: {e}")

    def _clear_results(self) -> None:
        """Clear all result displays."""
        self.prediction_label.config(text="Status: Ready")
        self.heatmap_label.config(image="", text="Heatmap")
        self.overlay_label.config(image="", text="Overlay")

    def _on_clear(self) -> None:
        """Handle clear button click."""
        self.current_image_path = None
        self.current_image_pil = None
        self.photo_image = None

        self.image_label.config(image="", text="No image selected")
        self.image_label.image = None
        self.filepath_label.config(text="")
        self._clear_results()
        self.status_label.config(text="Cleared. Ready for new image.")


def main(checkpoint_path: Optional[str] = None) -> None:
    """Launch the desktop application.

    Args:
        checkpoint_path: Optional path to a trained model checkpoint.
            If not provided, defaults to outputs/models/vgg16_checkpoint.pth
    """
    root = tk.Tk()
    app = DeepfakeDetectorApp(root, checkpoint_path=checkpoint_path)
    root.mainloop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the deepfake detection desktop application."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model checkpoint (.pth)",
    )
    args = parser.parse_args()

    main(checkpoint_path=args.checkpoint)
