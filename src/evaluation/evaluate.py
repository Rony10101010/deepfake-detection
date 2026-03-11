"""Evaluation utilities for the deepfake detection model.

This module provides functions to load a trained checkpoint, run inference on a
dataset, and compute standard classification metrics. A simple command-line
interface is also exposed so that the evaluation pipeline may be executed from
scripts or notebooks.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.models.vgg16_model import create_model


def load_model(
    checkpoint_path: str,
    num_classes: int = 2,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Instantiate a VGG16 model and load weights from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=num_classes, freeze_features=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Run inference on a dataloader and compute evaluation metrics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Iterable[str],
    output_path: str,
    normalize: bool = False,
    cmap=plt.cm.Blues,
) -> None:
    """Plot and save a confusion matrix image."""
    if normalize:
        cm_to_plot = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_to_plot = cm

    plt.figure(figsize=(6, 6))
    plt.imshow(cm_to_plot, interpolation="nearest", cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()

    class_names = list(classes)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm_to_plot.max() / 2.0

    for i, j in np.ndindex(cm_to_plot.shape):
        plt.text(
            j,
            i,
            format(cm_to_plot[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm_to_plot[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_metrics(results: dict[str, object], output_path: str) -> None:
    """Save scalar evaluation metrics to a text file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall:    {results['recall']:.4f}\n")
        f.write(f"F1 Score:  {results['f1']:.4f}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing train/valid/test splits",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation dataloader",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )
    parser.add_argument(
        "--output-reports",
        type=str,
        default="outputs/reports",
        help="Directory where textual reports will be written",
    )
    parser.add_argument(
        "--output-figures",
        type=str,
        default="outputs/figures",
        help="Directory where figures are saved",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from src.data.dataset_loader import build_transforms, make_datasets

    args = _parse_args()

    _, val_tfms = build_transforms()
    _, _, test_ds = make_datasets(
        root=args.data_root,
        train_transform=val_tfms,
        valid_transform=val_tfms,
        test_transform=val_tfms,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = load_model(args.checkpoint)
    results = evaluate_model(model, test_loader)

    os.makedirs(args.output_reports, exist_ok=True)
    os.makedirs(args.output_figures, exist_ok=True)

    report_path = os.path.join(args.output_reports, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(results["classification_report"])

    metrics_path = os.path.join(args.output_reports, "metrics.txt")
    save_metrics(results, metrics_path)

    print("Evaluation Metrics:")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}\n")
    print(results["classification_report"])

    fig_path = os.path.join(args.output_figures, "confusion_matrix.png")
    plot_confusion_matrix(results["confusion_matrix"], test_ds.classes, fig_path)

    print(f"Saved classification report to {report_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {fig_path}")