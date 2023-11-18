from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import ujson

from ceg4n.constants import PROPERTIES_FOLDER


@dataclass
class InputConstraint:
    x: List[float]
    min_value: float
    max_value: float
    shape: List[int]


@dataclass
class OutputConstraint:
    y: Union[int, float]
    shape: List[int]


@dataclass
class EquivalenceConstraints:
    input_constraint: InputConstraint
    output_constraint: OutputConstraint


def to_equivalence_constraints(data: dict):
    def input_constraint(input_data: dict):
        x = np.array(input_data["x"]).astype(np.float32)
        return InputConstraint(
            x=x,
            min_value=0.0,
            max_value=1.0,
            shape=x.shape,
        )

    def output_constraint(output_data: dict):
        y = np.array(output_data["y"]).astype(np.float32)
        return OutputConstraint(
            y=np.argmax(y),
            shape=y.shape,
        )

    return EquivalenceConstraints(
        input_constraint=input_constraint(data),
        output_constraint=output_constraint(data),
    )


class _PropertyProvider:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, benchmark: str) -> List[dict]:
        prop_file = self.path.joinpath(f"{benchmark}.json")
        prop_file = str(prop_file)
        with open(prop_file, "r") as fp:
            return ujson.load(fp)


properties_provider = _PropertyProvider(path=PROPERTIES_FOLDER)
