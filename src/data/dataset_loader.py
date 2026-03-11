"""Module for building datasets and dataloaders for the deepfake detection project.

This file defines helpers to create torchvision transforms, datasets and
DataLoader objects for the train/valid/test splits.  It is designed to be
imported from Colab and other scripts; configuration values are passed as
arguments rather than hardcoded.

Example usage::

    from src.data.dataset_loader import (
        build_transforms, make_datasets, make_dataloaders
    )

    train_tfms, val_tfms = build_transforms()
    train_ds, val_ds, test_ds = make_datasets(
        root="/content/real_vs_fake/real-vs-fake/",
        train_transform=train_tfms,
        valid_transform=val_tfms,
        test_transform=val_tfms,
    )
    train_loader, valid_loader, test_loader = make_dataloaders(
        train_ds, val_ds, test_ds, batch_size=64, num_workers=4
    )

"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(
    size: Tuple[int, int] = (224, 224),
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return a pair of torchvision transform pipelines for training and
    validation/test.

    The training pipeline includes basic augmentation (horizontal flip,
    small rotation) followed by normalization to ImageNet statistics.  The
    validation/test pipeline only resizes and normalizes.

    Args:
        size: output image size as (H, W); defaults to (224,224) for
            compatibility with VGG16.

    Returns:
        ``(train_transform, valid_transform)``
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_transform, valid_transform


# ---------------------------------------------------------------------------
# datasets & dataloaders
# ---------------------------------------------------------------------------


def _check_split_dir(root: str, split: str) -> str:
    """Return the path to a split subdirectory, raising if missing."""

    path = os.path.join(root, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Expected {split} directory at {path}")
    return path


def make_datasets(
    root: str,
    train_transform: transforms.Compose,
    valid_transform: transforms.Compose,
    test_transform: transforms.Compose,
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """Instantiate ``ImageFolder`` datasets for train/valid/test splits.

    Args:
        root: root directory containing ``train/``, ``valid/`` and
            ``test/`` folders with subdirectories for each class.
        train_transform: transform pipeline applied to training samples.
        valid_transform: transform pipeline applied to validation samples.
        test_transform: transform pipeline applied to test samples.

    Returns:
        ``(train_ds, valid_ds, test_ds)``

    The returned datasets are ready to be passed to ``make_dataloaders``.
    """

    train_dir = _check_split_dir(root, "train")
    valid_dir = _check_split_dir(root, "valid")
    test_dir = _check_split_dir(root, "test")

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_ds = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)

    return train_ds, valid_ds, test_ds


def make_dataloaders(
    train_ds: datasets.ImageFolder,
    valid_ds: datasets.ImageFolder,
    test_ds: datasets.ImageFolder,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create ``DataLoader`` objects for each dataset.

    Args:
        train_ds: training dataset created by :func:`make_datasets`.
        valid_ds: validation dataset.
        test_ds: test dataset.
        batch_size: number of samples per batch.
        num_workers: number of subprocesses for data loading.
        pin_memory: whether to pin memory (recommended for GPU training).

    Returns:
        ``(train_loader, valid_loader, test_loader)``.

    The train loader is shuffled while the others are not.
    """

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, test_loader


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def dataset_info(ds: datasets.ImageFolder) -> dict[str, object]:
    """Return summary information for an ImageFolder dataset.

    The dictionary contains:

    - ``size``: number of samples
    - ``classes``: list of class names
    - ``class_to_idx``: mapping from class name to index

    Args:
        ds: instance of ``torchvision.datasets.ImageFolder``
    """

    return {
        "size": len(ds),
        "classes": ds.classes,
        "class_to_idx": ds.class_to_idx,
    }


# end of file
