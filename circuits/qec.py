"""
Quantum Error Correction (QEC) Stabilizer Benchmark Workloads.

Implements small representative stabilizer error-correction circuits:
1. 3-qubit bit-flip code syndrome extraction circuit (3 data qubits + 2 ancilla qubits).
2. Multi-round repetition code stabilizer measurement cycle.

Theoretical Reference:
    Gottesman, D., 1997. "Stabilizer codes and quantum error correction."
"""

from typing import Optional
from simulator.circuit import Circuit


def create_bit_flip_code_circuit(error_qubit: Optional[int] = 1) -> Circuit:
    """
    Creates a 3-qubit bit-flip code with 2 syndrome ancillas (total 5 qubits).
    
    Qubit assignments:
        q0, q1, q2: Data qubits (d0, d1, d2)
        q3: Ancilla for stabilizer S1 = Z0 * Z1
        q4: Ancilla for stabilizer S2 = Z1 * Z2
        
    Workflow:
        1. State preparation: Encodes |0> into |000> via CNOTs.
        2. Error injection: Optional Pauli-X bit flip on error_qubit (0, 1, or 2).
        3. Syndrome extraction:
           - S1 (Z0*Z1): CNOT(d0, a0), CNOT(d1, a0), measure a0 -> c0
           - S2 (Z1*Z2): CNOT(d1, a1), CNOT(d2, a1), measure a1 -> c1
           
    Expected syndromes (c0, c1):
        - No error (None):  (0, 0)
        - Error on q0:      (1, 0)
        - Error on q1:      (1, 1)
        - Error on q2:      (0, 1)
    """
    circuit = Circuit(num_qubits=5)
    d0, d1, d2 = 0, 1, 2
    a0, a1 = 3, 4

    # 1. Encoding |0> -> |000>
    circuit.cnot(d0, d1)
    circuit.cnot(d0, d2)

    # 2. Error injection (if specified)
    if error_qubit is not None:
        if error_qubit not in (0, 1, 2):
            raise ValueError(f"error_qubit must be 0, 1, or 2, got {error_qubit}")
        circuit.x(error_qubit)

    # 3. Syndrome measurement for S1 = Z0 * Z1
    # Standard syndrome extraction into ancilla a0 in computational basis:
    circuit.cnot(d0, a0)
    circuit.cnot(d1, a0)
    circuit.measure(a0, clbit=0)

    # 4. Syndrome measurement for S2 = Z1 * Z2
    circuit.cnot(d1, a1)
    circuit.cnot(d2, a1)
    circuit.measure(a1, clbit=1)

    return circuit


def create_repetition_code_syndrome_circuit(
    num_data_qubits: int = 3,
    rounds: int = 2
) -> Circuit:
    """
    Creates a multi-round repetition code syndrome extraction circuit.
    
    Args:
        num_data_qubits: Number of data qubits (e.g. 3, 5).
        rounds: Number of syndrome extraction rounds.
        
    Total qubits = num_data_qubits + (num_data_qubits - 1) ancillas.
    """
    if num_data_qubits < 2:
        raise ValueError("Must have at least 2 data qubits")
    if rounds < 1:
        raise ValueError("Must have at least 1 syndrome round")

    num_ancillas = num_data_qubits - 1
    total_qubits = num_data_qubits + num_ancillas
    circuit = Circuit(num_qubits=total_qubits)

    # Encode: CNOT chain across data qubits
    for d in range(num_data_qubits - 1):
        circuit.cnot(d, d + 1)

    clbit_idx = 0
    for r in range(rounds):
        # Measure nearest-neighbor Z_d * Z_{d+1} checks
        for a in range(num_ancillas):
            d1 = a
            d2 = a + 1
            anc = num_data_qubits + a
            
            circuit.cnot(d1, anc)
            circuit.cnot(d2, anc)
            circuit.measure(anc, clbit=clbit_idx)
            clbit_idx += 1

    return circuit
