from copy import copy
from typing import List, OrderedDict, Tuple, Union

import torch
import torch.nn as nn

from ceg4n.nn import ModelParams, model_provider
from ceg4n.quantization.utils import calc_scaling_factor


class Quantizer:
    def __init__(self):
        self._float_type = torch.float32
        self._integer_type = torch.int32

    def _round(self, x: torch.Tensor) -> torch.Tensor:
        x_r = x.clone()
        x_r = torch.where(x_r < 0, x_r - 0.5, x_r + 0.5)
        x_r = x_r.to(dtype=self._integer_type)
        return x_r

    def quantize(self, x: torch.Tensor, scaling_factor: float) -> torch.Tensor:
        x_q = x.clone()
        x_q = x_q / scaling_factor
        return self._round(x_q)

    def dequantize(self, x_q: torch.Tensor, scaling_factor: float) -> torch.Tensor:
        x = x_q.clone()
        x = x * scaling_factor
        return x.to(dtype=self._float_type)


quantizer = Quantizer()


class QuantizationStrategy:
    def __init__(self):
        self._quantizer = quantizer

    def _quantize_fn(self, x, scaling_factor, minv, maxv):
        x_q = self._quantizer.quantize(x, scaling_factor).clamp(minv, maxv)
        q_x = self._quantizer.dequantize(x_q, scaling_factor)
        return q_x

    def quantize(self, benchmark: str, bits_sequence: List[int]) -> ModelParams:
        model_params = model_provider(benchmark=benchmark)
        layers = [layer for layer_id, layer in model_params.quantizable_params.items()]

        assert len(layers) == len(
            bits_sequence
        ), f"Bits sequence size {len(bits_sequence)} and layers {len(layers)} does not have same size."
        for bits, layer in zip(bits_sequence, layers):
            self.quantize_weights(layer, bits)

        return model_params

    def quantize_weights(
        self, layer: Union[nn.Linear, nn.Conv2d], bits: int
    ) -> Union[nn.Linear, nn.Conv2d]:
        weight = layer.weight.data.clone()
        bias = layer.bias.data.clone()

        weight_max = weight.abs().max().item()
        bias_max = bias.abs().max().item()

        beta = max(weight_max, bias_max)
        scaling_factor = calc_scaling_factor(bits=bits, alpha=-beta, beta=beta)

        maxv = 2 ** (bits - 1) - 1
        minv = -1 * (maxv + 1)

        weight = self._quantize_fn(weight, scaling_factor, minv, maxv)
        bias = self._quantize_fn(bias, scaling_factor, minv, maxv)
        layer.weight.data = weight.clone()
        layer.bias.data = bias.clone()
        return layer


quantization_strategy = QuantizationStrategy()
