"""
Benchmark Configurations for Serial Quantum Simulation Baseline.

Defines benchmark cases across multiple scaling axes:
1. Qubit Scaling: 5, 10, 20, 50, 100 qubits at fixed depth.
2. Depth Scaling: 20, 50, 100, 200, 500 layers at fixed qubit count.
3. Shot Scaling: 1, 10, 100, 500, 1000 shots.
4. Workload Variety: Canonical (Bell, GHZ), Random Clifford, Measurement-Heavy, QEC.
"""

from typing import Callable, List
from dataclasses import dataclass
from simulator.circuit import Circuit
from circuits.basic import create_bell_state, create_ghz_state
from circuits.random_clifford import create_random_clifford_circuit
from circuits.measurement_heavy import create_measurement_heavy_circuit
from circuits.qec import create_repetition_code_syndrome_circuit


@dataclass
class BenchmarkCase:
    """Specification of an individual benchmark experiment."""
    name: str
    category: str
    builder: Callable[[], Circuit]
    shots: int = 1
    repetitions: int = 1


def get_default_benchmarks() -> List[BenchmarkCase]:
    """Returns the comprehensive benchmark suite."""
    benchmarks: List[BenchmarkCase] = []

    # -------------------------------------------------------------
    # 1. Canonical Reference Circuits
    # -------------------------------------------------------------
    benchmarks.append(
        BenchmarkCase(
            name="Bell_State",
            category="canonical",
            builder=lambda: create_bell_state(),
            shots=1000,
            repetitions=3
        )
    )
    benchmarks.append(
        BenchmarkCase(
            name="GHZ_10Q",
            category="canonical",
            builder=lambda: create_ghz_state(num_qubits=10),
            shots=500,
            repetitions=3
        )
    )
    benchmarks.append(
        BenchmarkCase(
            name="GHZ_50Q",
            category="canonical",
            builder=lambda: create_ghz_state(num_qubits=50),
            shots=100,
            repetitions=2
        )
    )

    # -------------------------------------------------------------
    # 2. Qubit Scaling Suite (Depth fixed at 50, shots = 5)
    # -------------------------------------------------------------
    for n_qubits in [5, 10, 20, 50, 100]:
        benchmarks.append(
            BenchmarkCase(
                name=f"QubitScaling_{n_qubits}Q_D50",
                category="qubit_scaling",
                builder=lambda n=n_qubits: create_random_clifford_circuit(
                    num_qubits=n, depth=50, seed=100 + n
                ),
                shots=5,
                repetitions=2
            )
        )

    # -------------------------------------------------------------
    # 3. Depth Scaling Suite (20 Qubits fixed, shots = 5)
    # -------------------------------------------------------------
    for depth in [20, 50, 100, 200, 500]:
        benchmarks.append(
            BenchmarkCase(
                name=f"DepthScaling_20Q_D{depth}",
                category="depth_scaling",
                builder=lambda d=depth: create_random_clifford_circuit(
                    num_qubits=20, depth=d, seed=200 + d
                ),
                shots=5,
                repetitions=2
            )
        )

    # -------------------------------------------------------------
    # 4. Shot Scaling Suite (GHZ 10Q and Random 10Q D20)
    # -------------------------------------------------------------
    for shots in [1, 10, 50, 100, 500, 1000]:
        benchmarks.append(
            BenchmarkCase(
                name=f"ShotScaling_GHZ10Q_S{shots}",
                category="shot_scaling",
                builder=lambda: create_ghz_state(num_qubits=10),
                shots=shots,
                repetitions=1
            )
        )

    # -------------------------------------------------------------
    # 5. Workload Comparison (Gate-Heavy vs Measurement-Heavy vs QEC)
    # -------------------------------------------------------------
    benchmarks.append(
        BenchmarkCase(
            name="Workload_GateHeavy_30Q_D100",
            category="workload_type",
            builder=lambda: create_random_clifford_circuit(
                num_qubits=30, depth=100, seed=42, include_measurements=True
            ),
            shots=5,
            repetitions=2
        )
    )
    benchmarks.append(
        BenchmarkCase(
            name="Workload_MeasurementHeavy_20Q_20Cycles",
            category="workload_type",
            builder=lambda: create_measurement_heavy_circuit(
                num_qubits=20, cycles=20, gates_per_cycle=6, measure_fraction=0.5, seed=42
            ),
            shots=5,
            repetitions=2
        )
    )
    benchmarks.append(
        BenchmarkCase(
            name="Workload_QEC_Repetition_10Data_5Rounds",
            category="workload_type",
            builder=lambda: create_repetition_code_syndrome_circuit(
                num_data_qubits=10, rounds=5
            ),
            shots=20,
            repetitions=2
        )
    )

    return benchmarks


BENCHMARK_CONFIGS = get_default_benchmarks()
