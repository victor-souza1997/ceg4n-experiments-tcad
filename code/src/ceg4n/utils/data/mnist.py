from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor

NETWORKS_FOLDER = Path.cwd().joinpath("data", "networks", "original")
PROPERTIES_FOLDER = Path.cwd().joinpath("data", "properties")


def mnist_data(size: Tuple[int, int] = None) -> Tuple[datasets.MNIST, datasets.MNIST]:
    #print(datasets.MNIST)
    transforms = (
        [ToTensor()]
        if size is None
        else [
            Resize(size=size),
            ToTensor(),
        ]
    )
    train_data = datasets.MNIST(
        root="/tmp",
        train=True,
        transform=Compose(transforms=transforms),
        download=True,
    )

    test_data = datasets.MNIST(
        root="/tmp",
        train=False,
        transform=Compose(transforms=transforms),
        download=True,
    )

    return train_data, test_data


def mnist_dataloaders(
    train_batch_size: int = 100,
    test_batch_data: int = 100,
    img_size: Tuple[int, int] = None,
) -> Dict[str, DataLoader]:
    train_data, test_data = mnist_data(size=img_size)

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
