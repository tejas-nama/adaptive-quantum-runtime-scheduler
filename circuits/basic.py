"""
Basic Canonical Quantum Test Circuits.

Implements standard benchmark tests:
- Test 1: Single-qubit Hadamard superposition (0 -> 50%, 1 -> 50%)
- Test 2: Two-qubit Bell State (|Phi+> = (|00> + |11>)/sqrt(2))
- Test 3: n-qubit Greenberger-Horne-Zeilinger (GHZ) state (|00...0> + |11...1>)/sqrt(2)
"""

from simulator.circuit import Circuit


def create_single_qubit_h(qubit: int = 0) -> Circuit:
    """
    Creates a single-qubit Hadamard test circuit.
    
    Circuit:
        q0: --[H]--[M]--
        
    Expected outcome distribution:
        '0': ~50%
        '1': ~50%
    """
    circuit = Circuit(num_qubits=max(1, qubit + 1))
    circuit.h(qubit)
    circuit.measure(qubit, clbit=0)
    return circuit


def create_bell_state() -> Circuit:
    """
    Creates a 2-qubit Bell state circuit (|Phi+>).
    
    Circuit:
        q0: --[H]--*----[M0]--
                   |
        q1: -------X----[M1]--
        
    Expected outcome distribution:
        '00': ~50%
        '11': ~50%
        '01': 0%
        '10': 0%
    """
    circuit = Circuit(num_qubits=2)
    circuit.h(0)
    circuit.cnot(0, 1)
    circuit.measure(0, clbit=0)
    circuit.measure(1, clbit=1)
    return circuit


def create_ghz_state(num_qubits: int = 5) -> Circuit:
    """
    Creates an n-qubit Greenberger-Horne-Zeilinger (GHZ) state circuit.
    
    Circuit:
        q0:     --[H]--*-------------- ... --[M]--
                       |
        q1:     -------X--*----------- ... --[M]--
                          |
        ...               ...
        q(n-1): -------------------X-- ... --[M]--
        
    Expected outcome distribution:
        '00...0': ~50%
        '11...1': ~50%
        All other bitstrings: 0%
    """
    if num_qubits < 2:
        raise ValueError(f"GHZ state requires at least 2 qubits, got {num_qubits}")

    circuit = Circuit(num_qubits=num_qubits)
    circuit.h(0)
    for q in range(num_qubits - 1):
        circuit.cnot(q, q + 1)
    for q in range(num_qubits):
        circuit.measure(q, clbit=q)
    return circuit
