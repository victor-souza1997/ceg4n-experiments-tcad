from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Union
from xmlrpc.client import Boolean

import numpy as np
import torch
import torch.nn as nn

from ceg4n.constants import device
from ceg4n.nn import model_provider
from ceg4n.quantization.static import quantization_strategy


@dataclass(frozen=True)
class CounterExample:
    x: List[float]
    y: int
    shape: tuple
    output_size: int = 10

    def __hash__(self) -> int:
        return hash(tuple(np.array(self.x).flatten().tolist()))


class ObjectiveFn:
    def __init__(
        self,
        benchmark: str,
        counter_examples: List[CounterExample],
        neurons_profile: OrderedDict,
        history: set,
        top: int,
        epsilon: float,
    ):

        self.neurons_profile = neurons_profile
        self.benchmark = benchmark
        self.counter_examples = counter_examples
        if top:
            self.loss_fn = nn.CrossEntropyLoss()
        elif epsilon:
            self.loss_fn = nn.MSELoss()
        else:
            raise RuntimeError("Invalid equivalence relation.")
        self.top = top
        self.epsilon = epsilon
        self.softmax = nn.Softmax(dim=1)
        self.counter = 0
        self.ub = 32
        self.lb = 2
        self.history = history

    @torch.no_grad()
    def _predict(self, model):
        for ce in self.counter_examples:
            x = torch.tensor(ce.x, dtype=torch.float32)
            yield model(x.to(device=device))

    def _equivalence_cost(
        self,
        predictions,
        quant_predictions,
    ):
        if self.top:
            predictions = torch.stack([pred.argmax(dim=1) for pred in predictions])
            quant_predictions = torch.stack(
                [pred.argmax(dim=1) for pred in quant_predictions]
            )
            return int(predictions.eq(quant_predictions).all().item())
        elif self.epsilon:
            equivalence = [
                (abs(a - b).abs() <= self.epsilon).all()
                for a, b in zip(predictions, quant_predictions)
            ]
            return int(torch.stack(equivalence).all())
        raise RecursionError("Invalid equivalence relation.")

    def _solution_cost(self, solution):
        return int(np.sum(solution))

    def _prepare_solution(self, solution):
        return solution

    def __call__(self, solution):
        self.counter += 1
        solution = self._prepare_solution(solution)
        self.history.add(tuple(solution))
        return self.objective(solution)

    def objective(self, solution):
        model = model_provider(self.benchmark).model.eval()
        model.to(device)

        quant_model = quantization_strategy.quantize(
            self.benchmark, bits_sequence=list(solution)
        ).model.eval()
        quant_model.to(device)

        pred = list(self._predict(model))
        quant_pred = list(self._predict(quant_model))

        eq = self._equivalence_cost(pred, quant_pred)
        sol = self._solution_cost(solution)
        sol = (
            sol
            if eq == len(self.counter_examples)
            else sol + len(self.counter_examples)
        )
        return [sol,], [
            eq,
        ]


class WeightedObjectiveFn(ObjectiveFn):
    def _solution_cost(self, solution):
        weights = torch.tensor(list(self.neurons_profile.values()), dtype=torch.int64)
        solution = torch.tensor(solution, dtype=torch.int64)

        solution *= weights
        return solution.sum().item()


class SingleObjectiveFn(ObjectiveFn):
    def _prepare_solution(self, solution):
        solution = solution * len(self.neurons_profile)
        return solution
