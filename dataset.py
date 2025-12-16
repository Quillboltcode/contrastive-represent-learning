"""Dataset construction and dataloader helpers."""
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms


def train_val_split(dataset: Dataset, val_fraction: float = 0.2, seed: Optional[int] = None) -> Tuple[Subset, Subset]:
    """Split a dataset into train and validation subsets.

    Args:
        dataset: a torch `Dataset` instance.
        val_fraction: fraction of the dataset to reserve for validation (0.0-1.0).
        seed: optional random seed for reproducibility.

    Returns:
        (train_subset, val_subset)
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0)")

    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty")

    val_len = int(n * val_fraction)
    train_len = n - val_len

    # Use torch.utils.data.random_split for deterministic splitting when seed provided
    if seed is None:
        generator = None
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_subset, val_subset = torch.utils.data.random_split(dataset, [train_len, val_len], generator=generator)
    return train_subset, val_subset


def getdataset_from_imagefolder(root: str, transform: Optional[transforms.Compose] = None) -> Dataset:
    """Create a dataset from an image folder using torchvision.datasets.ImageFolder.

    Args:
        root: path to the root directory with class-subfolders.
        transform: optional torchvision transforms to apply to images.

    Returns:
        An instance of `torchvision.datasets.ImageFolder`.
    """
    if transform is None:
        # Default transform: convert to tensor only
        transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=root, transform=transform)
    return dataset


def get_dataloader(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    sampler: Optional[torch.utils.data.Sampler] = None,
    collate_fn: Optional[callable] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create DataLoaders for train/val and optional test datasets.

    If a `sampler` is provided it will be used for the training loader and
    `shuffle` will be ignored for the training loader (set to False).

    Returns:
        (train_loader, val_loader) or (train_loader, val_loader, test_loader)
    """
    train_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }

    if sampler is not None:
        train_kwargs["sampler"] = sampler
        train_kwargs["shuffle"] = False
    else:
        train_kwargs["shuffle"] = shuffle

    train_loader = DataLoader(train_dataset, **{k: v for k, v in train_kwargs.items() if v is not None})

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    if test_loader is None:
        return train_loader, val_loader
    return train_loader, val_loader, test_loader
