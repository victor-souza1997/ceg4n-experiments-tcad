from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime
import torch

# import torchvision
import ujson
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, Resize, ToTensor

from ceg4n.nn import load_model_parameters
from ceg4n.utils.data import mnist_dataloaders
from properties.prop import main

NETWORKS_FOLDER = Path.cwd().parent.joinpath("data", "networks")
PROPERTIES_FOLDER = Path.cwd().parent.joinpath("data", "properties")


def _main(
    model_name: str,
    equivalence_type: str = "top",
    input_epsilon: float = 0.03,
    input_min_value: float = 0.0,
    input_max_value: float = 1.0,
    output_epsilon: float = None,
    output_top: int = 1,
    max_properties_count: int = 1,
    size: Tuple = None,
):
    torch.manual_seed(7777)
    loaders = mnist_dataloaders(test_batch_data=1, img_size=size, shape=(28 * 28, 1))

    model_path = str(NETWORKS_FOLDER.joinpath(model_name))
    print(f"Model Name: {model_path}")

    ort_session = onnxruntime.InferenceSession(model_path)
    model = load_model_parameters(model_path, suffix=None)
    model = model.model.eval()
    main(
        model_name,
        model,
        ort_session,
        loaders,
        max_properties_count,
        equivalence_type,
        input_epsilon,
        input_min_value,
        input_max_value,
        output_epsilon,
        output_top,
    )


if __name__ == "__main__":
    train_options = [
        ((28, 28), [], "mnist-net_256x2"),
        ((28, 28), [], "mnist-net_256x4"),
        ((28, 28), [], "mnist-net_256x6"),
    ]
    for size, _, model_name in train_options:
        _main(f"{model_name}.onnx", size=size)
