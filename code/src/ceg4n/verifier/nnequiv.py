from enum import Enum
from typing import Tuple
import signal
import numpy as np
from nnenum.network import NeuralNetwork
from nnenum.onnx_network import load_onnx_network
from nnenum.settings import Settings
from nnenum.timerutil import Timers
from nnenum.zonotope import Zonotope
from nnequiv.state_manager import StateManager
from nnequiv.zono_state import ZonoState, status_update
from nnequiv.equivalence_properties import EpsilonEquivalence
from nnequiv.equivalence_properties.top1 import Top1Equivalence

from ceg4n.nn import original_model_provider, quantized_model_provider

from ceg4n.verifier.base import EquivalenceInstance, Verifier, VerifierType
from nnenum.timerutil import Timers
from nnenum.settings import Settings


class GracefulKiller:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\nEXITING...")
        if Timers.enabled:
            Timers.tocRec()
            Timers.print_stats()
        self.kill_now = True


class _EquivalenceStatus(Enum):
    NOT_EQUIVALENT = 0
    EQUIVALENT = 1


class _StateLoop:
    def __init__(self, manager: StateManager, killer: GracefulKiller):
        self.manager = manager
        self.killer = killer
        self.cur_state = None
        self.equivalence_status = _EquivalenceStatus.EQUIVALENT

    def __call__(self) -> Tuple[Tuple[int, np.array], Tuple[int, int]]:
        self._main_loop()
        if self.equivalence_status == _EquivalenceStatus.EQUIVALENT:
            return
        (k, counter_example), (r1, r2) = self._get_counter_example()
        return counter_example

    def _main_loop(self):
        counter = 0
        while not self.manager.done() and not self.killer.kill_now:
            self.cur_state = self.manager.pop()
            if not self.cur_state.state.active:
                continue
            if self.cur_state.is_finished(self.manager.get_networks()):
                if not self.cur_state.state.check_feasible(None, None):
                    continue
                if not self.manager.check(self.cur_state):
                    # print(f"NETWORKS NOT EQUIVALENT")
                    self.equivalence_status = _EquivalenceStatus.NOT_EQUIVALENT
                    break
            else:
                newStackEl = self.cur_state.advance_zono(self.manager.get_networks())
                if newStackEl is not None and newStackEl.state.active:
                    self.manager.push(newStackEl)
                if self.cur_state.state.active:
                    self.manager.push(self.cur_state)
            counter += 1
            if counter % 5000 == 1:
                status_update()
        status_update()

    def _get_counter_example(self) -> Tuple[Tuple[int, np.array], Tuple[int, int]]:
        input_size = self.manager.networks[0].get_num_inputs()
        r1 = (
            self.manager.networks[0]
            .execute(self.manager.counter_example[1][:input_size].astype(np.float32))
            .astype(np.float32)
        )
        r2 = (
            self.manager.networks[1]
            .execute(self.manager.counter_example[1][:input_size].astype(np.float32))
            .astype(np.float32)
        )
        return self.manager.counter_example, (r1, r2)


class Checker:
    def make_init_zs(self, init, networks):
        zono_state = ZonoState(len(networks))
        zono_state.from_init_zono(init)

        zono_state.propagate_up_to_split(networks)

        return zono_state

    def __call__(
        self, network1: NeuralNetwork, network2: NeuralNetwork, input: Zonotope, equiv
    ) -> Tuple[Tuple[int, np.array], Tuple[int, int]]:
        if not Settings.TIMING_STATS:
            Timers.disable()

        Timers.tic("network_equivalence")
        assert (
            network1.get_input_shape() == network2.get_input_shape()
        ), "Networks must have same input shape"
        assert (
            network1.get_output_shape() == network2.get_output_shape()
        ), "Networks must have same output shape"
        network1.check_io()
        network2.check_io()
        networks = [network1, network2]
        init = self.make_init_zs(input, networks)

        main_loop = _StateLoop(
            manager=StateManager(init, equiv, networks), killer=GracefulKiller()
        )
        checker_output = main_loop()
        Timers.toc("network_equivalence")
        return checker_output


class NNEquivVerifier(Verifier):
    def __init__(
        self, verifier_type: VerifierType, is_timing_stats_enabled: bool = False
    ):
        super(NNEquivVerifier, self).__init__(verifier_type)
        Settings.TIMING_STATS = is_timing_stats_enabled

    def verify_inner(
        self, equivalence_instance: EquivalenceInstance
    ) -> Tuple[Tuple[int, np.array], Tuple[int, int]]:
        if not Settings.TIMING_STATS:
            Timers.disable()
        Timers.reset()
        Timers.tic("main")
        Timers.tic("main_init")
        Settings.TIMING_STATS = True
        # TODO(steuber): Check out implications of this setting
        Settings.SPLIT_TOLERANCE = 1e-8
        Settings.PARALLEL_ROOT_LP = False
        Settings.NUM_PROCESSES = 1
        _check_equivalence_instance(equivalence_instance=equivalence_instance)
        Settings.EQUIV_OVERAPPROX_STRAT = (
            "CEGAR"
            if equivalence_instance.strategy == "CEGAR_OPTIMAL"
            else equivalence_instance.strategy
        )
        Settings.EQUIV_OVERAPPROX_STRAT_REFINE_UNTIL = (
            equivalence_instance.strategy.startswith("REFINE_UNTIL")
        )

        Timers.toc("main_init")
        Timers.tic("net_load")
        networks = _load_networks(equivalence_instance=equivalence_instance)
        Timers.toc("net_load")
        Timers.tic("property_create")
        equivalence_property = _get_equivalence_property(
            equivalence_instance=equivalence_instance, networks=networks
        )
        Timers.toc("property_create")
        Timers.tic("generate_box")
        input_zonotope = _generate_box(equivalence_instance)
        Timers.toc("generate_box")
        Timers.tic("checker_run")
        checker = Checker()
        equivalence_output = checker.check_equivalence(
            network1=networks[0],
            network2=networks[1],
            input=input_zonotope,
            equiv=equivalence_property,
        )

        Timers.toc("checker_run")
        main_time = Timers.toc("main")

        if Timers.enabled:
            print(f"\n[MAIN_TIME] {main_time}")
            print("")
            Timers.print_stats()
            print("")

        return equivalence_output


def _check_equivalence_instance(equivalence_instance: EquivalenceInstance):
    is_strategy_known = (
        equivalence_instance.strategy in Settings.EQUIV_STRATEGIES
        or equivalence_instance.strategy == "CEGAR_OPTIMAL"
    )
    assert is_strategy_known, f"ERROR: Strategy {equivalence_instance.strategy} unknown"


def _load_networks(
    equivalence_instance: EquivalenceInstance,
) -> Tuple[NeuralNetwork, NeuralNetwork]:
    original_model = load_onnx_network(
        original_model_provider.model_file(equivalence_instance.benchmark)
    )
    quantized_model = load_onnx_network(
        quantized_model_provider.model_file(equivalence_instance.benchmark)
    )
    return original_model, quantized_model


def _get_equivalence_property(
    equivalence_instance: EquivalenceInstance,
    networks: Tuple[NeuralNetwork, NeuralNetwork],
):
    if equivalence_instance.spec.top:
        return Top1Equivalence()
    elif equivalence_instance.spec.epsilon:
        return EpsilonEquivalence(
            epsilon=equivalence_instance.spec.epsilon,
            networks=list(networks),
        )
    else:
        raise RuntimeError("Unsuported Equivalence Property")


def _generate_box(equivalence_instance: EquivalenceInstance):
    lower_bounds = np.array(
        equivalence_instance.spec.input_lower_bounds, dtype=np.float32
    ).flatten()
    upper_bounds = np.array(
        equivalence_instance.spec.input_upper_bounds, dtype=np.float32
    ).flatten()
    assert lower_bounds.shape == upper_bounds.shape
    inshape = np.prod(upper_bounds.shape)
    generator = np.identity(inshape, dtype=np.float32)
    bias = np.zeros(inshape, dtype=np.float32)

    bounds = [p for p in zip(lower_bounds.tolist(), upper_bounds.tolist())]
    return Zonotope(bias, generator, init_bounds=bounds)
