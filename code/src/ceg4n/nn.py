from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from nnenum.onnx_network import load_onnx_network
from onnx2pytorch import ConvertModel
from onnx2torch import convert

from ceg4n.constants import MODELS_FOLDER, ORIGINAL_FOLDER, QUANTIZED_FOLDER


def predict(model_file, x_t):
    ort_session = onnxruntime.InferenceSession(model_file)
    ort_inputs = {ort_session.get_inputs()[0].name: x_t}
    ort_outs = ort_session.run(None, ort_inputs)
    return np.argmax(ort_outs[0]), ort_outs[0]


class ModelParams:
    def __init__(self, model, suffix):
        self._model = model
        self._suffix = ""

    @property
    def params(self) -> OrderedDict:
        return OrderedDict(
            [
                (self._new_module_name(idx), layer)
                for idx, (name, layer) in enumerate(self.model.named_children())
            ]
        )

    def _new_module_name(self, idx):
        suffix = f"_{self._suffix}" if self._suffix else ""
        return f"node_{idx}{suffix}"

    @property
    def quantizable_params(self) -> OrderedDict:
        return OrderedDict(
            [
                (name, layer)
                for name, layer in self.params.items()
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)
            ]
        )

    def index_of(self, layer_id: str) -> int:
        layers_ids = [i for i, _ in self.quantizable_params]
        return layers_ids.index(layer_id)

    @property
    def model(self, prefix=None) -> nn.Module:
        return self._model

    def to_onnx(self, filename: str, shape: List[int]):
        self.model.eval()
        x = torch.randn(shape, requires_grad=True)
        y0 = self.model(x).detach().cpu().numpy().astype(np.float32)
        torch.onnx.export(
            self.model,
            x,
            filename,
        )

        # # Check if model was saved correctly
        # y = (
        #     load_onnx_network(filename)
        #     .execute(x.detach().cpu().numpy().flatten())
        #     .astype(np.float32)
        # )
        # assert np.isclose(
        #     y0, y, atol=0.01
        # ).all(), "Something went wrong saving the network."


def map_layer(layer) -> nn.Module:
    # if isinstance(layer, Flatten):
    #     return nn.Flatten()
    return layer


def load_model_parameters(model_name: str, suffix: str = None) -> ModelParams:
    model = (
        ConvertModel(onnx.load(model_name))
        if "acasxu" in model_name.lower()
        else convert(model_name)
    )
    return ModelParams(model, suffix=suffix)


class _ParamProvider:
    def __init__(self, path: Path, suffix: str = None):
        self.path = path
        self.suffix = suffix

    def model_file(self, benchmark: str) -> str:
        model_file = self.path.joinpath(f"{benchmark}.onnx")
        model_file = str(model_file)
        return model_file

    def __call__(self, benchmark: str) -> ModelParams:
        model_file = self.model_file(benchmark)
        return load_model_parameters(model_name=model_file, suffix=self.suffix)


model_provider = _ParamProvider(path=MODELS_FOLDER)
#print("model_provider", model_provider)
original_model_provider = _ParamProvider(path=ORIGINAL_FOLDER)
quantized_model_provider = _ParamProvider(path=QUANTIZED_FOLDER, suffix="quantized")
