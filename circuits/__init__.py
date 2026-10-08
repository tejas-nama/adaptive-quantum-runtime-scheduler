"""
Test and Benchmark Circuits for Clifford / Stabilizer Simulation.

Contains standard canonical circuits (Bell, GHZ), configurable random
Clifford circuits, measurement-heavy workloads, and small quantum error
correction (QEC) stabilizer circuits.
"""

from circuits.basic import (
    create_single_qubit_h,
    create_bell_state,
    create_ghz_state,
)
from circuits.random_clifford import create_random_clifford_circuit
from circuits.measurement_heavy import create_measurement_heavy_circuit
from circuits.qec import (
    create_bit_flip_code_circuit,
    create_repetition_code_syndrome_circuit,
)

__all__ = [
    "create_single_qubit_h",
    "create_bell_state",
    "create_ghz_state",
    "create_random_clifford_circuit",
    "create_measurement_heavy_circuit",
    "create_bit_flip_code_circuit",
    "create_repetition_code_syndrome_circuit",
]
