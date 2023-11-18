from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScalingFactorContext:
    alpha: float
    beta: float
    bits: int
    scaling_factor: float


@dataclass(frozen=True)
class Precision:
    integer_type: torch.dtype
    float_type: torch.dtype


# SINGLE = Precision(integer_type=torch.int32, float_type=torch.float32)
# DOUBLE = Precision(integer_type=torch.int64, float_type=torch.float64)


def calc_scaling_factor(bits: int, alpha: float, beta: float) -> float:
    scale = (2**bits) - 1
    return (beta - alpha) / scale


def get_scaling_factor_context(x: torch.Tensor, bits: int) -> ScalingFactorContext:
    beta = x.abs().max().item()
    alpha = -beta
    scaling_factor = calc_scaling_factor(bits=bits, alpha=alpha, beta=beta)
    return ScalingFactorContext(
        alpha=alpha, beta=beta, bits=bits, scaling_factor=scaling_factor
    )
