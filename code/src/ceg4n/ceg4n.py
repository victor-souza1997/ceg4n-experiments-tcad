from collections import OrderedDict
from dataclasses import dataclass
from typing import List
from ceg4n.verifier.base import (
    EquivalenceInstance,
    EquivalenceSpec,
    Verifier,
    VerifierType,
)
from ceg4n.verifier.verifier import verifier_provider

import numpy as np
import torch
import torch.nn.functional as F
import ujson
from choicesenum import ChoicesEnum
from datetime import datetime

# from ceg4n.abstractions import generate_abstractions
from ceg4n.constants import OUTPUT_FOLDER
from ceg4n.nn import (
    model_provider,
    original_model_provider,
    predict,
    quantized_model_provider,
)
from ceg4n.optimizer import (
    CounterExample,
    OptimizationInstance,
    OptimizationType,
    QuantizationBitsOptimizer,
)
from ceg4n.property import properties_provider
from ceg4n.quantization.static import quantization_strategy
from ceg4n.utils.data import data_provider


# class BallZise(ChoicesEnum):
#     R0_01 = "0.01"
#     R0_03 = "0.03"
#     R0_05 = "0.05"
#     R0_1 = "0.1"
#     R0_3 = "0.3"
#     R0_5 = "0.5"

TIMEOUTS = {VerifierType.ESBMC: 25 * 60, VerifierType.NNEQUIV: 20 * 60}


class EquivalenceType(ChoicesEnum):
    EPSILON = "EPS"
    TOP = "TOP"


@dataclass
class ExperimentInstance:
    optimizer: OptimizationType
    verifier: VerifierType
    equivalence: EquivalenceType
    benchmark: str
    strategy: str
    objective: str
    size: float

    @property
    def top(self) -> int:
        return 1 if self.equivalence == EquivalenceType.TOP else None

    @property
    def epsilon(self) -> float:
        return 0.05 if self.equivalence == EquivalenceType.EPSILON else None


@dataclass
class RunInstance:
    benchmark: str
    strategy: str
    properties: List[dict]
    objective: str
    ball_size: float
    top: int
    epsilon: float


class _Iteration:
    def __init__(
        self,
        iteration_id: int,
        counter_examples: List[CounterExample],
        verification_counter_examples: List[CounterExample],
        run_instance: RunInstance,
        optimizer: QuantizationBitsOptimizer,
        verifier: Verifier,
    ):
        self._iteration_id = iteration_id
        self._counter_examples = counter_examples
        self._run_instance = run_instance
        self._verification_counter_examples = verification_counter_examples
        self._optimizer = optimizer
        self._verifier = verifier
        print("Interaction is called")

    @property
    def verifier(self):
        return self._verifier

    @property
    def verification_counter_examples(self):
        return self._verification_counter_examples

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def single(self) -> bool:
        return self._run_instance.objective

    @property
    def iteration_id(self) -> int:
        return self._iteration_id

    @property
    def counter_examples(self) -> List[CounterExample]:
        return self._counter_examples

    @property
    def benchmark(self) -> str:
        return self._run_instance.benchmark

    @property
    def strategy(self) -> str:
        return self._run_instance.strategy

    @property
    def specs(self) -> List[EquivalenceSpec]:
        return [
            EquivalenceSpec(
                spec_id=prop["id"],
                input_ball_size=prop["ball_size"],
                input_lower_bounds=prop["lower_bounds"],
                input_upper_bounds=prop["upper_bounds"],
                input_shape=prop["input_shape"],
                x=prop["x"],
                y=prop["y"],
                top=self._run_instance.top,
                epsilon=self._run_instance.epsilon,
            )
            for prop in self._run_instance.properties
        ]

    def run(self) -> List[CounterExample]:
        start_date = datetime.utcnow()
        optimization_instance = OptimizationInstance(
            benchmark=self.benchmark,
            params_bounds=(2, 32),
            counter_examples=list(self.counter_examples)
            + list(self.verification_counter_examples),
            top=self._run_instance.top,
            epsilon=self._run_instance.epsilon,
        )

        bits_sequence = self.optimizer.optimize(optimization_instance)
        if len(bits_sequence) == 0:
            self._save(bits_sequence, [], [], 0, start_date, datetime.utcnow())
            return ([], []), bits_sequence, 0

        print(f"\nBits found: {bits_sequence}")

        equivalence_instances = [
            EquivalenceInstance(
                bits_sequence=bits_sequence,
                benchmark=self.benchmark,
                strategy=self.strategy,
                spec=spec,
                timeout=TIMEOUTS[self.verifier.verifier_type],
            )
            for spec in self.specs
        ]

        print(f"equivalence_instances{equivalence_instances}")

        (valid_ces, invalid_ces), bits_sequence, failure = self.loop(
            equivalence_instances, bits_sequence
        )
        end_date = datetime.utcnow()
        self._save(bits_sequence, valid_ces, invalid_ces, failure, start_date, end_date)
        return (valid_ces, invalid_ces), bits_sequence, failure

    def loop(
        self, equivalence_instances: List[EquivalenceInstance], bits_sequence: List[int]
    ):
        valid_ces = []
        invalid_ces = []
        for i, equivalence_instance in enumerate(equivalence_instances):
            print(
                f"\n[{equivalence_instance.benchmark}]:\nVerifying property: {equivalence_instance.spec.spec_id}"
            )

            verification_status = self.verifier.verify(equivalence_instance)
            (counterexample, timeout, failure) = (
                verification_status.counterexample,
                verification_status.timeout,
                verification_status.failure,
            )
           # print("counter example", counter_example)

            if timeout is not None:
                print("timeout")
                return ([], []), bits_sequence, 1

            if failure is not None:
                print("failure", failure)
                return ([], []), bits_sequence, 2

            if counterexample is None:
                continue

            x = (
                np.reshape(counterexample, equivalence_instance.spec.input_shape)
                .astype(np.float32)
                .tolist()
            )
            y = equivalence_instance.spec.y
            counter_example = CounterExample(
                x=x, y=y, shape=equivalence_instance.spec.input_shape
            )
            if self._is_valid(counter_example):
                valid_ces.append(counter_example)
            else:
                invalid_ces.append(counter_example)
        return (valid_ces, invalid_ces), bits_sequence, 0

    def _is_valid(self, counter_example: CounterExample):
        cex, y = counter_example.x, counter_example.y
        pred, quant_pred = _make_prediction(self._run_instance, cex, torch.float32)
        print("pred, quant_pred", pred, quant_pred)
        if self._run_instance.top:
            pred_y = np.argmax(pred)
            quant_pred_y = np.argmax(quant_pred)
            is_ce = pred_y != quant_pred_y
            valid = "Valid counterexample" if is_ce else "Invalid counterexample"
            print(
                f"\n[{valid}]: Label: {y}, Sanity Check: ({pred_y}, {quant_pred_y})\n"
            )
            return is_ce
        elif self._run_instance.epsilon:
            sanity = np.abs(pred - quant_pred)
            is_ce = (sanity > self._run_instance.epsilon).any()
            valid = "Valid counterexample" if is_ce else "Invalid counterexample"
            print(f"\n[{valid}]: Sanity Check: {sanity}\n")
        else:
            raise RuntimeError

    def _save(
        self, bits_sequence, valid_ces, invalid_ces, failure, start_date, end_date
    ):
        OUTPUT_FOLDER.mkdir(exist_ok=True)
        history_folder = (
            f"{self.optimizer.optimiation_type}-{self.verifier.verifier_type}"
        )
        history_folder = OUTPUT_FOLDER.joinpath(
            f"history-{history_folder}-{self._run_instance.objective}-{self._run_instance.strategy}".lower()
        )

        history_folder.mkdir(exist_ok=True)

        equivalence = (
            f"top-{str(self._run_instance.top)}"
            if self._run_instance.top
            else f"epsilon-{str(self._run_instance.epsilon).replace('.', '')}"
        )

        prop = self._run_instance.top
        prop = prop if prop else self._run_instance.epsilon
        prop = str(prop).replace(".", "")

        radious = f"radious-{str(self._run_instance.ball_size).replace('.', '')}"
        filename = history_folder.joinpath(
            f"{self._run_instance.benchmark}-{radious}-{equivalence}-{prop}-iteration-{self.iteration_id}.json".lower()
        )
        with open(str(filename), "w") as fp:
            data = {
                "iteration": self.iteration_id,
                "benchmark": self._run_instance.benchmark,
                "bits_sequence": bits_sequence,
                "valid_counter_examples": [{"x": ce.x, "y": ce.y} for ce in valid_ces],
                "invalid_counter_examples": [
                    {"x": ce.x, "y": ce.y} for ce in invalid_ces
                ],
                "failure": failure,
                "start_date": start_date.strftime("%y-%m-%d, %H:%M:%S"),
                "end_date": end_date.strftime("%y-%m-%d, %H:%M:%S"),
            }
            ujson.dump(data, fp, indent=4)


@torch.no_grad()
def _make_prediction(run_instance, cex, dtype):
    model = original_model_provider(run_instance.benchmark).model.eval().to(dtype=dtype)
    quant_model = (
        quantized_model_provider(run_instance.benchmark).model.eval().to(dtype=dtype)
    )

    x = torch.tensor(cex, dtype=dtype)
    p = model(x).detach().cpu().numpy()
    q_p = quant_model(x).detach().cpu().numpy()
    return p, q_p


def _make_prediction_onnx(run_instance: RunInstance, cex):
    cex = np.array(cex).astype(np.float32)
    p = predict(
        original_model_provider.model_file(run_instance.benchmark),
        cex,
    )
    q_p = predict(
        quantized_model_provider.model_file(run_instance.benchmark),
        cex,
    )
    return p, q_p


class ExperimentRunner:
    def __init__(
        self,
        optimizer: QuantizationBitsOptimizer,
        verifier: Verifier,
    ):
        self._optimizer = optimizer
        self._verifier = verifier

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def verifier(self):
        return self._verifier

    def run(self, run_instance: RunInstance):
        iteration_counter = 0
        counter_examples = [
            CounterExample(
                x=c["x"],
                y=c["y"],
                shape=c["input_shape"],
            )
            for c in run_instance.properties
        ]
        counter_examples = {hash(b): b for b in counter_examples}
        verification_counter_examples = {}
        bits_sequences = list()
        self.optimizer.clear_history()
        start_date = datetime.utcnow()
        while True:
            iteration = _Iteration(
                iteration_id=iteration_counter,
                counter_examples=counter_examples.values(),
                verification_counter_examples=verification_counter_examples.values(),
                run_instance=run_instance,
                optimizer=self.optimizer,
                verifier=self.verifier,
            )

            (valid_ces, invalid_ces), bits_sequence, failure = iteration.run()

            valid_counter_examples = {
                hash(b): b
                for b in valid_ces
                if hash(b) not in verification_counter_examples
            }
            repeated_counter_examples = {
                hash(b): b
                for b in valid_ces
                if hash(b) in verification_counter_examples
            }

            print(f"\n[{run_instance.benchmark}]:")
            print(f"Valid counterexamples: {len(valid_counter_examples)}")
            print(f"Invalid counterexamples: {len(repeated_counter_examples)}")

            if len(bits_sequence) == 0:
                print("\nNo bits where found.")
                print("\nStopping")
            if len(valid_counter_examples) == 0:
                print(f"\nQuantization bits sequence: {bits_sequence}")
                print("\nUnnable to find any new counter-examples.")
                print("\nStopping.")
                bits_sequences.append(bits_sequence)
                break

            # verification_counter_examples =
            a = verification_counter_examples.update(valid_counter_examples)
            if failure != 0:
                print("\nVerification failed!")
                print("Stopping.")
                bits_sequences.append(bits_sequence)
                break
            iteration_counter += 1
        end_date = datetime.utcnow()
        self._save_stats(
            run_instance,
            bits_sequences,
            verification_counter_examples.values(),
            failure,
            iteration_counter,
            start_date,
            end_date,
        )

    def _save_stats(
        self,
        run_instance: RunInstance,
        bits_sequences,
        counter_examples,
        failure,
        iteration_counter,
        start_date,
        end_date,
    ):
        ces = [{"x": ce.x, "y": ce.y} for ce in counter_examples]
        if len(bits_sequences[-1]) > 0:
            original_model = original_model_provider(
                run_instance.benchmark
            ).model.eval()
            original_model_accuracy, original_model_loss = self._collect_stats(
                original_model,
                run_instance.benchmark,
                is_acasxu="acasxu" in run_instance.benchmark.lower()
                or "cifar" in run_instance.benchmark.lower(),
            )
            quantized_model = quantization_strategy.quantize(
                run_instance.benchmark, bits_sequences[-1]
            ).model.eval()

            quantized_model_accuracy, quantized_model_loss = self._collect_stats(
                quantized_model,
                run_instance.benchmark,
                is_acasxu="acasxu" in run_instance.benchmark.lower()
                or "cifar" in run_instance.benchmark.lower(),
            )
        else:
            original_model_accuracy, original_model_loss = -1, -1
            quantized_model_accuracy, quantized_model_loss = -1, -1
        OUTPUT_FOLDER.mkdir(exist_ok=True)
        stats_folder = (
            f"{self.optimizer.optimiation_type}-{self.verifier.verifier_type}"
        )
        stats_folder = OUTPUT_FOLDER.joinpath(
            f"stats-{stats_folder}-{run_instance.objective}-{run_instance.strategy}".lower()
        )
        stats_folder.mkdir(exist_ok=True)
        equivalence = (
            f"top-{str(run_instance.top)}"
            if run_instance.top
            else f"epsilon-{str(run_instance.epsilon).replace('.', '')}"
        )

        prop = run_instance.top
        prop = prop if prop else run_instance.epsilon
        prop = str(prop).replace(".", "")

        ballsize = f"radious-{str(run_instance.ball_size).replace('.', '')}"
        filename = stats_folder.joinpath(
            f"{run_instance.benchmark}-{ballsize}-{equivalence}-{prop}.json".lower()
        )

        with open(str(filename), "w") as fp:
            data = {
                "iteration": iteration_counter,
                "benchmark": run_instance.benchmark,
                "bits_sequences": bits_sequences,
                "counter_examples": ces,
                "original_model_accuracy": original_model_accuracy,
                "original_model_loss": original_model_loss,
                "quantized_model_accuracy": quantized_model_accuracy,
                "quantized_model_loss": quantized_model_loss,
                "failure": failure,
                "start_date": start_date.strftime("%y-%m-%d, %H:%M:%S"),
                "end_date": end_date.strftime("%y-%m-%d, %H:%M:%S"),
            }
            ujson.dump(data, fp, indent=4)

    @torch.no_grad()
    def _collect_stats(self, model, benchmark, is_acasxu):
        if is_acasxu:
            return -1, -1
        accuracy = 0
        loss = 0
        print("data_provider is called")
        test_loader = data_provider(benchmark, 32)["test"]
        for batch_id, (x, y) in enumerate(test_loader):
            pred = model(x)
            loss += F.cross_entropy(pred, y).item()
            accuracy += pred.argmax(dim=1, keepdim=False).eq(y).sum().item()
        return accuracy / len(test_loader), loss / len(test_loader)


def get_run_instance(experiment_instance: ExperimentInstance):
    properties = properties_provider(experiment_instance.benchmark)
    properties = [p for p in properties if p["ball_size"] == experiment_instance.size]
    return RunInstance(
        benchmark=experiment_instance.benchmark,
        objective=experiment_instance.objective,
        strategy=experiment_instance.strategy,
        properties=properties,
        top=experiment_instance.top,
        epsilon=experiment_instance.epsilon,
        ball_size=experiment_instance.size,
    )


def _profile_model(benchmark: str):
    def neurons(layer):
        weight_shape = torch.tensor(layer.weight.data.shape).prod()
        bias_shape = torch.tensor(layer.bias.data.shape).prod()
        print(f"in profile model {benchmark} {weight_shape} {bias_shape}")
        return (weight_shape + bias_shape).int().item()

    a = [
        (name, neurons(layer))
        for name, layer in model_provider(benchmark).quantizable_params.items()
    ]
    return OrderedDict(a)


def run(experiment_instance: ExperimentInstance):
    # Profile model
    neurons_profile = _profile_model(experiment_instance.benchmark)

    optimizer = QuantizationBitsOptimizer(
        neurons_profile=neurons_profile,
        single=experiment_instance.objective,
        optimization_type=experiment_instance.optimizer,
    )

    #
    verifier = verifier_provider(experiment_instance.verifier)
    print("verifier", verifier)
    #
    run_instance = get_run_instance(experiment_instance)
    runner = ExperimentRunner(optimizer=optimizer, verifier=verifier)

    #
    runner.run(run_instance=run_instance)
