"""
Clifford Gate Implementations for Stabilizer Tableau Simulation.

Theoretical Reference:
    Aaronson, S. and Gottesman, D., 2004.
    "Improved simulation of stabilizer circuits."
    Physical Review A, 70(5), p.052328.

All Clifford gates conjugate the stabilizer and destabilizer Pauli operators.
Because Clifford operations map Pauli operators to Pauli operators, each gate
updates the binary coordinates (x[i, q], z[i, q]) and phase vector r[i]
across all rows i in {0, ..., 2n-1}.

SERIAL IMPLEMENTATION NOTE:
Each gate iterates through rows sequentially. No multi-threading or GPU kernel
launch is used here to ensure a clean profiling baseline.
"""

from typing import Callable, Dict
from simulator.tableau import StabilizerTableau


def apply_h(tableau: StabilizerTableau, qubit: int) -> None:
    """
    Applies Hadamard gate H to the specified qubit.
    
    Transformations on Pauli basis:
        H X H^dag = Z
        H Z H^dag = X
        H Y H^dag = -Y
    
    Tableau update rule for each row i in {0, ..., 2n - 1}:
        r[i]   <- r[i] ^ (x[i, qubit] * z[i, qubit])
        x[i, qubit], z[i, qubit] <- z[i, qubit], x[i, qubit]
    
    # TODO (Optimization):
    # This row loop is embarrassingly parallel across the 2n rows.
    # In GPU acceleration (QuaSARQ) or OpenMP CPU, this can be mapped to
    # 2n threads updating their respective row elements concurrently.
    """
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError(f"Qubit index {qubit} out of range [0, {tableau.num_qubits - 1}]")

    num_rows = 2 * tableau.num_qubits
    for i in range(num_rows):
        xi = tableau.x[i, qubit]
        zi = tableau.z[i, qubit]
        tableau.r[i] ^= (xi * zi) & 1
        tableau.x[i, qubit] = zi
        tableau.z[i, qubit] = xi


def apply_s(tableau: StabilizerTableau, qubit: int) -> None:
    """
    Applies Phase gate S = diag(1, i) to the specified qubit.
    
    Transformations on Pauli basis:
        S X S^dag = Y = i * X * Z
        S Z S^dag = Z
        S Y S^dag = -X
    
    Tableau update rule for each row i in {0, ..., 2n - 1}:
        r[i]         <- r[i] ^ (x[i, qubit] * z[i, qubit])
        z[i, qubit]  <- z[i, qubit] ^ x[i, qubit]
    
    # TODO (Optimization):
    # Parallelizable across all 2n rows on GPU threads or SIMD vector lanes.
    """
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError(f"Qubit index {qubit} out of range [0, {tableau.num_qubits - 1}]")

    num_rows = 2 * tableau.num_qubits
    for i in range(num_rows):
        xi = tableau.x[i, qubit]
        zi = tableau.z[i, qubit]
        tableau.r[i] ^= (xi * zi) & 1
        tableau.z[i, qubit] ^= xi


def apply_cnot(tableau: StabilizerTableau, control: int, target: int) -> None:
    """
    Applies Controlled-NOT (CNOT / CX) gate between control and target qubits.
    
    Transformations on Pauli basis:
        CNOT (X (x) I) CNOT = X (x) X
        CNOT (I (x) X) CNOT = I (x) X
        CNOT (Z (x) I) CNOT = Z (x) I
        CNOT (I (x) Z) CNOT = Z (x) Z
        CNOT (Y (x) Y) CNOT = - X (x) Z
    
    Tableau update rule for each row i in {0, ..., 2n - 1}:
        r[i]        <- r[i] ^ (x[i, c] * z[i, t] * (x[i, t] ^ z[i, c] ^ 1))
        x[i, target]  <- x[i, target] ^ x[i, control]
        z[i, control] <- z[i, control] ^ z[i, target]
    
    # TODO (Optimization):
    # In multi-qubit systems, two-qubit gates are candidate hotspots.
    # The 2n row updates are completely independent and can be executed
    # concurrently across CPU threads or GPU thread blocks.
    """
    if control == target:
        raise ValueError(f"Control and target cannot be the same qubit ({control})")
    if control < 0 or control >= tableau.num_qubits:
        raise IndexError(f"Control qubit {control} out of range [0, {tableau.num_qubits - 1}]")
    if target < 0 or target >= tableau.num_qubits:
        raise IndexError(f"Target qubit {target} out of range [0, {tableau.num_qubits - 1}]")

    num_rows = 2 * tableau.num_qubits
    for i in range(num_rows):
        xc = tableau.x[i, control]
        zc = tableau.z[i, control]
        xt = tableau.x[i, target]
        zt = tableau.z[i, target]

        tableau.r[i] ^= (xc * zt * (xt ^ zc ^ 1)) & 1
        tableau.x[i, target] ^= xc
        tableau.z[i, control] ^= zt


def apply_x(tableau: StabilizerTableau, qubit: int) -> None:
    """
    Applies Pauli-X (Bit Flip) gate to the specified qubit.
    
    Transformations:
        X X X = X,  X Z X = -Z,  X Y X = -Y
    
    Tableau update rule:
        r[i] <- r[i] ^ z[i, qubit]
    """
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError(f"Qubit index {qubit} out of range")
    num_rows = 2 * tableau.num_qubits
    for i in range(num_rows):
        tableau.r[i] ^= tableau.z[i, qubit]


def apply_y(tableau: StabilizerTableau, qubit: int) -> None:
    """
    Applies Pauli-Y gate to the specified qubit.
    
    Transformations:
        Y X Y = -X,  Y Z Y = -Z,  Y Y Y = Y
    
    Tableau update rule:
        r[i] <- r[i] ^ (x[i, qubit] ^ z[i, qubit])
    """
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError(f"Qubit index {qubit} out of range")
    num_rows = 2 * tableau.num_qubits
    for i in range(num_rows):
        tableau.r[i] ^= (tableau.x[i, qubit] ^ tableau.z[i, qubit])


def apply_z(tableau: StabilizerTableau, qubit: int) -> None:
    """
    Applies Pauli-Z (Phase Flip) gate to the specified qubit.
    
    Transformations:
        Z X Z = -X,  Z Z Z = Z,  Z Y Z = -Y
    
    Tableau update rule:
        r[i] <- r[i] ^ x[i, qubit]
    """
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError(f"Qubit index {qubit} out of range")
    num_rows = 2 * tableau.num_qubits
    for i in range(num_rows):
        tableau.r[i] ^= tableau.x[i, qubit]


def apply_cz(tableau: StabilizerTableau, control: int, target: int) -> None:
    """
    Applies Controlled-Z (CZ) gate between two qubits.
    
    Transformations:
        CZ (X (x) I) CZ = X (x) Z
        CZ (I (x) X) CZ = Z (x) X
        CZ (Z (x) I) CZ = Z (x) I
        CZ (I (x) Z) CZ = I (x) Z
        CZ (X (x) Y) CZ = - Y (x) X
        CZ (Y (x) X) CZ = - X (x) Y
    
    Tableau update rule:
        r[i]        <- r[i] ^ (x[i, c] * x[i, t] * (z[i, c] ^ z[i, t]))
        z[i, control] <- z[i, control] ^ x[i, target]
        z[i, target]  <- z[i, target] ^ x[i, control]
    """
    if control == target:
        raise ValueError(f"Control and target cannot be the same qubit ({control})")
    if control < 0 or control >= tableau.num_qubits:
        raise IndexError(f"Control qubit {control} out of range")
    if target < 0 or target >= tableau.num_qubits:
        raise IndexError(f"Target qubit {target} out of range")

    num_rows = 2 * tableau.num_qubits
    for i in range(num_rows):
        xc = tableau.x[i, control]
        zc = tableau.z[i, control]
        xt = tableau.x[i, target]
        zt = tableau.z[i, target]

        tableau.r[i] ^= (xc * xt * (zc ^ zt)) & 1
        tableau.z[i, control] ^= xt
        tableau.z[i, target] ^= xc


# Dispatch dictionary for Clifford gates
CLIFFORD_GATES: Dict[str, Callable] = {
    "H": apply_h,
    "S": apply_s,
    "CNOT": apply_cnot,
    "CX": apply_cnot,
    "X": apply_x,
    "Y": apply_y,
    "Z": apply_z,
    "CZ": apply_cz,
}
