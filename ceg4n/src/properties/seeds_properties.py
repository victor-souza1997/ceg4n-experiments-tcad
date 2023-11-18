from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime
import torch

# import torchvision
import ujson
from prop import main
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Lambda, Resize, ToTensor

from ceg4n.nn import load_model_parameters
from ceg4n.utils.data import seeds_dataloaders

NETWORKS_FOLDER = Path.cwd().joinpath("data", "networks", "original")
PROPERTIES_FOLDER = Path.cwd().joinpath("data", "properties")


def to_numpy(x_t):
    return x_t.clone().detach().cpu().numpy()


def predict(ort_session, x_t):
    ort_inputs = {ort_session.get_inputs()[0].name: x_t}
    ort_outs = ort_session.run(None, ort_inputs)
    return np.argmax(ort_outs[0]), ort_outs[0]


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
    loaders = seeds_dataloaders()

    model_path = str(NETWORKS_FOLDER.joinpath(model_name))
    print(f"Model Name: {model_path}")

    model = load_model_parameters(model_path).model.eval()
    ort_session = onnxruntime.InferenceSession(model_path)
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
        ((7,), [15], "seeds_15x1"),
        ((7,), [10], "seeds_10x1"),
    ]
    for size, _, model_name in train_options:
        _main(f"{model_name}.onnx", size=size)
        # main(f"{model_name}_float64.onnx", size=size)
