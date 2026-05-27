"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision.datasets as datasets


DATA_ROOT = Path(__file__).resolve().parent


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def get_cifar_loader(root=DATA_ROOT, batch_size=128, val_size=5000, train=True, shuffle=True, num_workers=4, n_items=-1, seed=42):

    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    )
    train_data_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_data_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    if train:
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=train_data_transforms)
        val_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=eval_data_transforms)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(train_dataset), generator=generator).tolist()
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        if n_items > 0:
            train_indices = train_indices[:n_items]
            val_indices = val_indices[: min(n_items, len(val_indices))]

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader

    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_data_transforms)
    if n_items > 0:
        test_dataset = PartialDataset(test_dataset, n_items)

    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

if __name__ == '__main__':
    train_loader, val_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        plt.imshow(np.clip(img * std + mean, 0, 1))
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break
