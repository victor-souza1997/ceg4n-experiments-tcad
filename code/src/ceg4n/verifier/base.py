import signal
from dataclasses import dataclass
from socket import timeout
from subprocess import TimeoutExpired
from typing import List, Union

import numpy as np
from choicesenum import ChoicesEnum

from ceg4n.nn import model_provider, original_model_provider, quantized_model_provider
from ceg4n.quantization.static import quantization_strategy


@dataclass
class EquivalenceSpec:
    spec_id: str
    input_ball_size: float
    input_lower_bounds: List[float]
    input_upper_bounds: List[float]
    input_shape: List[int]
    x: List[float]
    y: List[float]
    top: int = None
    epsilon: float = None


class VerifierType(ChoicesEnum):
    NNEQUIV = "NNEQUIV"
    ESBMC = "ESBMC"
    CBMC = "CBMC"


@dataclass
class EquivalenceInstance:
    benchmark: str
    strategy: str
    bits_sequence: List[int]
    spec: EquivalenceSpec
    timeout: int


@dataclass
class VerificationOutput:
    counterexample: Union[List[int], np.ndarray]
    timeout: int
    failure: Exception


class Verifier:
    def __init__(self, verifier_type: VerifierType):
        self.verifier_type = verifier_type

    def _handler(self, signum, frame):
        raise TimeoutError("End of time")

    def prepare_benchmarks(self, equivalence_instance: EquivalenceInstance):
        filename = original_model_provider.model_file(equivalence_instance.benchmark)
        model_provider(equivalence_instance.benchmark).to_onnx(
            filename, equivalence_instance.spec.input_shape
        )

        filename = quantized_model_provider.model_file(equivalence_instance.benchmark)
        quantization_strategy.quantize(
            equivalence_instance.benchmark, equivalence_instance.bits_sequence
        ).to_onnx(filename, equivalence_instance.spec.input_shape)

    def verify(self, equivalence_instance: EquivalenceInstance) -> VerificationOutput:
        self.prepare_benchmarks(equivalence_instance)
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(equivalence_instance.timeout)
        counterexample = None
        timeout_ = None
        exception = None

        try:
            counterexample = self.verify_inner(equivalence_instance)
        except TimeoutError as ex:
            timeout_ = ex
        except TimeoutExpired as ex:
            timeout_ = TimeoutError()
        except Exception as ex:
            # raise ex
            exception = ex
        finally:
            signal.alarm(0)
        return VerificationOutput(
            counterexample=counterexample, timeout=timeout_, failure=exception
        )

    def verify_inner(
        selff, equivalence_instance: EquivalenceInstance
    ) -> VerificationOutput:
        raise RuntimeError("Should not be called!")
