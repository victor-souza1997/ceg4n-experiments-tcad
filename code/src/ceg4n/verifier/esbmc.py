import os
import signal
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from typing import List, Tuple

import numpy as np

from ceg4n.abstractions.esbmc import save_abstraction
from ceg4n.constants import ESBMC_ABSTRACTIONS_PATH

from ceg4n.verifier.base import EquivalenceInstance, VerificationOutput, Verifier
from ceg4n.constants import CMD_TEMPLATE
import psutil


class ESBMCChecker:
    def __call__(self, benchmark_path: str, property_name, timeout: int) -> List[float]:
        esbmc = "esbmc"
        filename = benchmark_path.joinpath(property_name)
        headers = str(benchmark_path.joinpath("."))
        #print(headers)
        print("ESBMC was called")
        cmd = (
            CMD_TEMPLATE.replace("@benchmark", str(filename))
            .replace("@esbmc", esbmc)
            .replace("@HEADERS", f"-I{headers}")
        )
        # cmd = cmd.split(" ")
        
        std_output_file = benchmark_path.joinpath(f"{property_name}-output.txt")
        std_error_file = benchmark_path.joinpath(f"{property_name}-error.txt")
        #print(f"std_error {std_error_file}")
        with open(std_output_file, "w") as stdout_fp, open(
            std_error_file, "w"
        ) as stderr_fp:
            esbmc_process = Popen(
                cmd,
                shell=True,
                bufsize=1,
                stdout=stdout_fp,
                stderr=stderr_fp,
                universal_newlines=True,
                preexec_fn=os.setsid,
            )
            try:
                esbmc_process.communicate(timeout=timeout)
                stdout_fp.flush()
                stderr_fp.flush()
            except TimeoutExpired as tex:
                self.kill_process(esbmc_process)
                stdout_fp.flush()
                stderr_fp.flush()
                raise tex
        has_errors = self._has_errors(std_error_file)
        if has_errors:
            raise RuntimeError("Verification failed with errors.")
        ces = self.get_counter_examples(std_output_file)
        return ces

    def kill_process(self, process):
        # process.kill()
        proc_pid = process.pid
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            proc.kill()
        os.killpg(
            os.getpgid(proc_pid), signal.SIGTERM
        )  # Send the signal to all the process groups

    def _has_errors(self, filename):
        with open(filename, "r") as fp:
            lines = fp.readlines()
        lines = [line for line in lines if line.strip() != ""]
        return len(lines) > 0

    def get_counter_examples(self, filename: str) -> Tuple:
        with open(filename, "r") as fp:
            lines = (
                fp.readlines()
            )  # [line for line in fp.readlines() if self._check(line)]
        is_successfull = self._is_successfull(lines)
        if is_successfull:
            return None
        status = not self._verification_status(lines)
        if status:
            raise RuntimeError("Verification failed. Is neither SAT or UNSAT")
        return self._process_output(lines)

    def _is_successfull(self, lines):
        lines = [line for line in lines if "VERIFICATION SUCCESSFUL" in line]
        return len(lines) == 1

    def _verification_status(self, lines):
        property_violated = False
        counter_example = False
        not_equivalent = False
        verification_failed = False

        for line in lines:
            property_violated = property_violated or "Violated property" in line
            counter_example = counter_example or "Counterexample" in line
            not_equivalent = not_equivalent or "Networks not equivalent." in line
            verification_failed = verification_failed or "VERIFICATION FAILED" in line

        return (
            property_violated
            and counter_example
            and not_equivalent
            and verification_failed
        )

    def _process_output(self, output) -> np.ndarray:
        lines = [line.strip() for line in output]
        values = []
        for line in lines:
            line = line
            if (
                "(input + (signed long int)i" not in line
                and "input[(signed long int)i]" not in line
            ):
                continue
            value = line.split("=")[1].split(" ")[1].replace("f", "")
            value = float(value)
            # value = Decimal(value)
            values.append(value)
        return np.array(values)

    def _has_counter_example(self, lines: List[str]) -> str:
        for line in lines:
            if "Violated property" in line:  # and not ("Invalid" in line):
                return "VALID"
        return "NONE"

    def _check(self, output: str):
        symbolic_line = "(*(input + (signed long int)b))"
        return symbolic_line in output or (
            self._has_counter_example(lines=[output]) != "NONE"
        )


check_equivalence = ESBMCChecker()


class ESBMCVerifier(Verifier):
    def verify_inner(
        self, equivalence_instance: EquivalenceInstance
    ) -> VerificationOutput:
        prop_id = equivalence_instance.spec.spec_id.replace(".", "_")
        benchmark_path = self._get_benchmark_path(equivalence_instance)

        timeout = max(0, equivalence_instance.timeout - (5 * 60))
        return check_equivalence(benchmark_path, f"main_{prop_id}.c", timeout)

    def _get_benchmark_path(self, equivalence_instance: EquivalenceInstance) -> Path:
        benchmark = (
            equivalence_instance.benchmark.lower().replace("-", "_").replace(".", "_")
        )
        benchmark_path = ESBMC_ABSTRACTIONS_PATH.joinpath(benchmark)
        benchmark_path.mkdir(exist_ok=True)
        return benchmark_path

    def prepare_benchmarks(self, equivalence_instance: EquivalenceInstance):
        super().prepare_benchmarks(equivalence_instance)
        abstraction_path = self._get_benchmark_path(equivalence_instance)
        save_abstraction(
            equivalence_instance.benchmark, abstraction_path, equivalence_instance.spec
        )
        pass
