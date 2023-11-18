from pydoc import cli

import click

from ceg4n.ceg4n import EquivalenceType, ExperimentInstance, run
from ceg4n.constants import BENCHMARKS_OPTIONS
from ceg4n.optimizer import OptimizationType
from ceg4n.verifier.verifier import VerifierType


@click.command()
@click.option("--benchmark")
@click.option("--optimizer")
@click.option("--verifier")
@click.option("--equivalence")
@click.option("--size", type=float)
def main(benchmark: str, optimizer: str, verifier: str, equivalence: str, size: float):
    strategy = "DONT"
    objective = "weighted"
    optimizer = OptimizationType(optimizer)
    verifier = VerifierType(verifier)
    equivalence = EquivalenceType(equivalence)

    if equivalence == EquivalenceType.EPSILON:
        assert "acasxu" in benchmark.lower()
        # epsilon, top = 0.05, None
    elif equivalence == EquivalenceType.TOP:
        assert "acasxu" not in benchmark.lower()
        # top, epsilon = 1, None
    else:
        raise RuntimeError()

    if "acasxu" in benchmark.lower():
        bso = [0.1, 0.3, 0.5]
    else:
        bso = [0.01, 0.03, 0.05]
    assert size in bso

    # top = None if "acasxu" in benchmark.lower() else 1
    # epsilon = 0.05 if "acasxu" in benchmark.lower() else None

    experiment_instance = ExperimentInstance(
        optimizer=optimizer,
        verifier=verifier,
        equivalence=equivalence,
        benchmark=benchmark,
        objective=objective,
        strategy=strategy,
        size=size,
    )
    run(experiment_instance)

main()
