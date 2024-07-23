import imp

from .acasxu import acasxu_dataloaders
from .cifar10 import cifar10_dataloaders
from .iris import iris_dataloaders
from .mnist import mnist_dataloaders
from .seeds import seeds_dataloaders


def data_provider(benchmark: str, batch_size: int):
    print(f"in data_provider {benchmark} is called")
    if "iris" in benchmark.lower():
        return iris_dataloaders()
    if "seeds" in benchmark.lower():
        return seeds_dataloaders()
    if "mnist_64" in benchmark.lower() or "mnist_8x8" in benchmark.lower():
        return mnist_dataloaders(
            img_size=(8, 8), train_batch_size=batch_size, test_batch_data=batch_size
        )
    if "mnist_784" in benchmark.lower() or "mnist_28x28" in benchmark.lower():
        return mnist_dataloaders(
            img_size=(28, 28), train_batch_size=batch_size, test_batch_data=batch_size
        )
    if "cifar" in benchmark.lower():
        return cifar10_dataloaders(
            train_batch_size=batch_size, test_batch_data=batch_size
        )
    if "mnist-net":
        return mnist_dataloaders(img_size=(28, 28))
    if "acasxu" in benchmark.lower():
        return acasxu_dataloaders()
    raise RuntimeError()
