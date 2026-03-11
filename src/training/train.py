"""Training pipeline for the deepfake detection model.

This module provides functions to train a VGG16 model on the deepfake
dataset. It includes training and validation loops, metric tracking, and
checkpoint saving.

Example usage::

    from src.training.train import train_model
    from src.models.vgg16_model import create_model
    from src.data.dataset_loader import (
        build_transforms, make_datasets, make_dataloaders
    )

    # Prepare data
    train_tfms, val_tfms = build_transforms()
    train_ds, val_ds, _ = make_datasets(
        root="/content/real_vs_fake/real-vs-fake/",
        train_transform=train_tfms,
        valid_transform=val_tfms,
        test_transform=val_tfms,
    )
    train_loader, val_loader, _ = make_dataloaders(
        train_ds, val_ds, None, batch_size=32
    )

    # Create model
    model = create_model(num_classes=2, freeze_features=True)

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=20,
        lr=1e-4,
        checkpoint_path="outputs/models/best_model.pth"
    )

"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch of training.

    Args:
        model: the PyTorch model to train.
        dataloader: DataLoader for training data.
        criterion: loss function.
        optimizer: optimizer for updating model parameters.
        device: device to run on (CPU or GPU).

    Returns:
        (average_loss, accuracy) for the epoch.
    """

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch of validation.

    Args:
        model: the PyTorch model to validate.
        dataloader: DataLoader for validation data.
        criterion: loss function.
        device: device to run on (CPU or GPU).

    Returns:
        (average_loss, accuracy) for the epoch.
    """

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 32,
    checkpoint_path: str = "outputs/models/best_model.pth",
) -> dict[str, list]:
    """Train the model with training and validation loops.

    Args:
        model: the PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: number of training epochs.
        lr: learning rate for Adam optimizer.
        batch_size: batch size (for logging, not used in loops).
        checkpoint_path: path to save the best model checkpoint.

    Returns:
        Dictionary with training history: 'train_loss', 'train_acc',
        'val_loss', 'val_acc'.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        # Track metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")

    return history


# end of file