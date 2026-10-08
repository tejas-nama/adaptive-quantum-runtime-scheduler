"""
Measurement-Heavy Quantum Workloads.

Theoretical Reference:
    QuaSARQ: GPU-accelerated stabilizer simulation (arXiv:2603.14641)

Simulates active quantum workloads featuring interleaved Clifford operations
and mid-circuit measurements. Such workloads trigger frequent Gaussian elimination
and row-reduction updates, testing the measurement computational hotspot.
"""

from typing import Optional
import random
from simulator.circuit import Circuit


def create_measurement_heavy_circuit(
    num_qubits: int,
    cycles: int,
    gates_per_cycle: int = 5,
    measure_fraction: float = 0.4,
    seed: Optional[int] = None
) -> Circuit:
    """
    Constructs a circuit with frequent interleaved mid-circuit measurements.
    
    Args:
        num_qubits: Total number of qubits in the circuit.
        cycles: Number of measurement-and-gate cycles.
        gates_per_cycle: Number of Clifford gates per cycle before measurement.
        measure_fraction: Fraction of qubits to measure in each cycle.
        seed: Random seed for reproducibility.
        
    Returns:
        Circuit object with interleaved gates and measurements.
    """
    if num_qubits < 2:
        raise ValueError(f"Measurement-heavy circuit requires at least 2 qubits, got {num_qubits}")
    if cycles <= 0:
        raise ValueError(f"cycles must be positive, got {cycles}")

    rng = random.Random(seed)
    circuit = Circuit(num_qubits)
    num_to_measure = max(1, int(num_qubits * measure_fraction))

    clbit_counter = 0

    for cycle in range(cycles):
        # Apply a burst of Clifford gates to create entanglement / superposition
        for _ in range(gates_per_cycle):
            gate_type = rng.choice(["H", "S", "CNOT"])
            if gate_type == "H":
                q = rng.randrange(num_qubits)
                circuit.h(q)
            elif gate_type == "S":
                q = rng.randrange(num_qubits)
                circuit.s(q)
            elif gate_type == "CNOT":
                c = rng.randrange(num_qubits)
                t = rng.randrange(num_qubits)
                while t == c:
                    t = rng.randrange(num_qubits)
                circuit.cnot(c, t)

        # Mid-circuit measurement burst: randomly sample qubits to measure
        measured_qubits = rng.sample(range(num_qubits), num_to_measure)
        for q in measured_qubits:
            circuit.measure(q, clbit=clbit_counter)
            clbit_counter += 1

    return circuit
