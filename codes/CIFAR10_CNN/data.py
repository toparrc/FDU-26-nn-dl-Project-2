from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_ROOT = Path(__file__).resolve().parent / "data"
SHARED_DATA_ROOT = Path(__file__).resolve().parents[1] / "VGG_BatchNorm" / "data"


def get_cifar10_loaders(data_dir = None, batch_size = 128, val_size = 5000, num_workers = 4, seed = 42):
    root = Path(data_dir) if data_dir is not None else DATA_ROOT
    if data_dir is None and SHARED_DATA_ROOT.exists():
        root = SHARED_DATA_ROOT

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=root, train=True, download=True, transform=eval_transform
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=True, transform=eval_transform
    )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
