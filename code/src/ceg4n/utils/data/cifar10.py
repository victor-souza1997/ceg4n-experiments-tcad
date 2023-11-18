from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


def cifar10_dataloaders(
    train_batch_size: int = 100, test_batch_data: int = 100
) -> Dict[str, DataLoader]:
    train_data, test_data = cifar10_data()
    loaders = {
        "train": DataLoader(
            train_data,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=1,
        ),
        "test": DataLoader(
            test_data,
            batch_size=test_batch_data,
            shuffle=True,
            num_workers=1,
        ),
    }
    return loaders


def cifar10_data() -> Tuple[datasets.CIFAR10, datasets.CIFAR10]:
    transform_ = Compose(
        [
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_data = datasets.CIFAR10(
        root="/tmp",
        train=True,
        transform=transform_,
        download=True,
    )

    test_data = datasets.CIFAR10(
        root="/tmp",
        train=False,
        transform=transform_,
        download=True,
    )

    return train_data, test_data
