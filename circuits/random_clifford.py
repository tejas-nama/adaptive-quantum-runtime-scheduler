"""
Random Clifford Circuit Generator for Benchmarking and Profiling.

Generates reproducible random Clifford circuits parameterized by:
- num_qubits: Register width
- depth: Number of gate cycles/layers
- seed: Random seed for exact reproducibility
- include_measurements: Whether to measure all qubits at the circuit end
"""

from typing import Optional, List
import random
from simulator.circuit import Circuit


def create_random_clifford_circuit(
    num_qubits: int,
    depth: int,
    seed: Optional[int] = None,
    include_measurements: bool = True,
    gate_set: str = "standard"  # 'standard' (H, S, CNOT) or 'extended' (+ X, Y, Z, CZ)
) -> Circuit:
    """
    Constructs a random Clifford circuit with configurable depth and width.
    
    In each layer/cycle:
    1. Single-qubit Clifford gates are randomly selected and applied.
    2. Random disjoint pairs of qubits are selected for two-qubit CNOT gates.
    
    Args:
        num_qubits: Number of qubits.
        depth: Number of layers.
        seed: Random seed for reproducibility.
        include_measurements: If True, appends computational basis measurements
                              on all qubits at the end of the circuit.
        gate_set: 'standard' uses {H, S, CNOT}; 'extended' also includes {X, Y, Z, CZ}.
        
    Returns:
        Circuit instance containing the generated operations.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be positive, got {num_qubits}")
    if depth < 0:
        raise ValueError(f"depth cannot be negative, got {depth}")

    rng = random.Random(seed)
    circuit = Circuit(num_qubits)

    single_gates = ["H", "S"]
    if gate_set == "extended":
        single_gates.extend(["X", "Y", "Z"])

    for _ in range(depth):
        # Layer step 1: Single qubit gates
        for q in range(num_qubits):
            choice = rng.random()
            if choice < 0.6:  # 60% chance to apply a single qubit gate
                gate = rng.choice(single_gates)
                if gate == "H":
                    circuit.h(q)
                elif gate == "S":
                    circuit.s(q)
                elif gate == "X":
                    circuit.x(q)
                elif gate == "Y":
                    circuit.y(q)
                elif gate == "Z":
                    circuit.z(q)

        # Layer step 2: Two-qubit entangling gates
        if num_qubits >= 2:
            qubit_pool = list(range(num_qubits))
            rng.shuffle(qubit_pool)
            
            # Pair adjacent qubits from shuffled pool
            num_pairs = len(qubit_pool) // 2
            for i in range(num_pairs):
                q1 = qubit_pool[2 * i]
                q2 = qubit_pool[2 * i + 1]
                
                if rng.random() < 0.7:  # 70% chance to entangle each pair
                    if gate_set == "extended" and rng.random() < 0.3:
                        circuit.cz(q1, q2)
                    else:
                        # Randomly assign control and target
                        ctrl, tgt = (q1, q2) if rng.random() < 0.5 else (q2, q1)
                        circuit.cnot(ctrl, tgt)

    # Optional terminal measurements
    if include_measurements:
        for q in range(num_qubits):
            circuit.measure(q, clbit=q)

    return circuit
