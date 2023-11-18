from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from choicesenum import ChoicesEnum
from platypus import GDE3, NSGAII, InjectedPopulation, Integer, Problem, Solution

from ceg4n.nn import ModelParams, model_provider, quantized_model_provider
from ceg4n.optimization.objective import (
    CounterExample,
    ObjectiveFn,
    SingleObjectiveFn,
    WeightedObjectiveFn,
)
from ceg4n.quantization.static import QuantizationStrategy


class OptimizationType(ChoicesEnum):
    NSGAII = "NSGAII", "nsgaii"
    GDE3 = "GDE3", "gde3"


@dataclass
class OptimizationInstance:
    benchmark: str
    params_bounds: List[Tuple[int, int]]
    counter_examples: List[CounterExample]
    top: int
    epsilon: float


class QuantizationBitsOptimizer:
    def __init__(
        self,
        optimization_type: OptimizationType,
        neurons_profile: OrderedDict,
        single: bool,
    ):
        self._optimization_type = optimization_type
        self._single = single
        self._neurons_profile = neurons_profile
        self.history = set()

    def clear_history(self):
        self.history = set()

    @property
    def single(self) -> bool:
        return self._single

    @property
    def neurons_profile(self):
        return self._neurons_profile

    @property
    def optimiation_type(self) -> OptimizationType:
        return self._optimization_type

    def _problem_setup(self, variables: List[Integer], fn: ObjectiveFn) -> Problem:
        problem = Problem(len(variables), 1, 1)
        problem.types[:] = variables
        problem.constraints[:] = [">0"]
        problem.function = fn
        problem.directions[:] = Problem.MINIMIZE
        return problem

    def _objective_fn(self, instance: OptimizationInstance) -> ObjectiveFn:
        if self.single == "single":
            builder = SingleObjectiveFn
        elif self.single == "multi":
            builder = ObjectiveFn
        elif self.single == "weighted":
            builder = WeightedObjectiveFn
        else:
            raise RuntimeError()

        fn = builder(
            benchmark=instance.benchmark,
            counter_examples=instance.counter_examples,
            neurons_profile=self.neurons_profile,
            history=self.history,
            top=instance.top,
            epsilon=instance.epsilon,
        )
        return fn

    def _build_variables(self, instance: OptimizationInstance):
        lower, upper = instance.params_bounds
        params_number = 1 if self.single == "single" else len(self.neurons_profile)
        variables = [Integer(lower, upper) for _ in range(params_number)]
        return variables

    def _get_algorithm(self, problem: Problem):
        if self.optimization_type == OptimizationType.NSGAII:
            return NSGAII(problem=problem)
        elif self.optimization_type == OptimizationType.GDE3:
            return GDE3(problem=problem)
        raise RuntimeError("Invalid Algorithm type.")

    def optimize(
        self, instance: OptimizationInstance, generations: int = 1000
    ) -> List[int]:
        if self.single == "single":
            generations = 30
        variables = self._build_variables(instance)
        fn = self._objective_fn(instance)
        problem = self._problem_setup(variables, fn)
        algorithm = NSGAII(problem, population_size=1)
        algorithm.run(generations)

        solutions = self._solutions(fn, algorithm.result, variables)

        if len(solutions) == 0:
            return []
        solution = solutions[0][0]
        if self.single == "single":
            solution = solution * len(self.neurons_profile)
        return solution

    def _solutions(self, fn, result, variables):
        solutions = []
        feasible_solutions = [s for s in result if s.feasible]
        for solution in feasible_solutions:
            solutions.append(
                [
                    variable.decode(gene)
                    for variable, gene in zip(variables, solution.variables)
                ]
            )
        fitness = [fn(solution) for solution in solutions]
        return [c for c in zip(solutions, fitness)]
