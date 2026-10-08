"""
Stabilizer Tableau Representation for n-Qubit Clifford Quantum Simulation.

Theoretical Reference:
    Aaronson, S. and Gottesman, D., 2004.
    "Improved simulation of stabilizer circuits."
    Physical Review A, 70(5), p.052328.

Tableau Structure:
------------------
For an n-qubit system, the stabilizer state is uniquely specified by:
- n destabilizer generators: rows 0 to n - 1 (denoted R_0, ..., R_{n-1})
- n stabilizer generators: rows n to 2n - 1 (denoted R_n, ..., R_{2n-1})
- 1 scratch row: row 2n (used to evaluate deterministic measurement phases)

Each row i represents an n-qubit Pauli operator with sign:
    R_i = (-1)^{r_i} * prod_{j=0}^{n-1} P_{i, j}

Where the single-qubit Pauli operator P_{i, j} on qubit j is encoded by:
    (x_{i, j}, z_{i, j}) = (0, 0) -> Identity (I)
    (x_{i, j}, z_{i, j}) = (1, 0) -> Pauli-X   (X)
    (x_{i, j}, z_{i, j}) = (0, 1) -> Pauli-Z   (Z)
    (x_{i, j}, z_{i, j}) = (1, 1) -> Pauli-Y   (Y = i*X*Z)

Phase Representation:
---------------------
- r[i] in {0, 1} represents the overall scalar sign factor (-1)^{r[i]}.
- r[i] == 0 denotes +1
- r[i] == 1 denotes -1

Initial State (|00...0>):
-------------------------
At initialization, the system represents the computational basis state |00...0>:
- Destabilizers are initialized to single-qubit X operators:
    R_i = X_i  for 0 <= i < n   (i.e., x[i, i] = 1, all others 0, r[i] = 0)
- Stabilizers are initialized to single-qubit Z operators:
    R_{n+i} = Z_i for 0 <= i < n (i.e., z[n+i, i] = 1, all others 0, r[n+i] = 0)
"""

from typing import List, Tuple
import numpy as np


class StabilizerTableau:
    """
    Serial Stabilizer Tableau data structure.
    
    Maintains a (2n + 1) x (2n + 1) binary tableau representation for an n-qubit
    stabilizer state. All operations are strictly serial to provide an accurate
    profiling baseline.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes an n-qubit stabilizer tableau in the ground state |0...0>.

        Args:
            num_qubits: Number of qubits in the quantum register.
        """
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        
        self.num_qubits: int = num_qubits
        
        # Binary matrices of size (2n + 1, n)
        # Rows 0 .. n-1: destabilizers
        # Rows n .. 2n-1: stabilizers
        # Row 2n: scratch row for measurement calculations
        self.x: np.ndarray = np.zeros((2 * num_qubits + 1, num_qubits), dtype=np.uint8)
        self.z: np.ndarray = np.zeros((2 * num_qubits + 1, num_qubits), dtype=np.uint8)
        
        # Binary phase vector of size (2n + 1)
        self.r: np.ndarray = np.zeros(2 * num_qubits + 1, dtype=np.uint8)
        
        self.reset()

    def reset(self) -> None:
        """
        Resets the tableau to the ground state |00...0>.
        Destabilizers become X_i, stabilizers become Z_i, phases become 0 (+1).
        """
        self.x.fill(0)
        self.z.fill(0)
        self.r.fill(0)
        
        # Initial destabilizers: R_i = X_i for i in [0, n - 1]
        for i in range(self.num_qubits):
            self.x[i, i] = 1
            
        # Initial stabilizers: R_{n+i} = Z_i for i in [0, n - 1]
        for i in range(self.num_qubits):
            self.z[self.num_qubits + i, i] = 1

    def copy(self) -> "StabilizerTableau":
        """
        Creates a deep copy of the current stabilizer tableau.
        
        Useful for multi-shot simulations or state branching.
        """
        new_tab = StabilizerTableau(self.num_qubits)
        new_tab.x[:] = self.x[:]
        new_tab.z[:] = self.z[:]
        new_tab.r[:] = self.r[:]
        return new_tab

    @staticmethod
    def _g(x1: int, z1: int, x2: int, z2: int) -> int:
        """
        Aaronson-Gottesman (2004) g function.
        
        Returns the exponent of i in the single-qubit Pauli product:
            (X^{x1} Z^{z1}) * (X^{x2} Z^{z2}) = i^g * (-1)^... * X^{x1 ^ x2} Z^{z1 ^ z2}
        
        Values of g(x1, z1, x2, z2) in {-1, 0, 1}.
        """
        if x1 == 0 and z1 == 0:
            return 0
        if x1 == 1 and z1 == 1:
            return int(z2) - int(x2)
        if x1 == 1 and z1 == 0:
            return int(z2) * (2 * int(x2) - 1)
        if x1 == 0 and z1 == 1:
            return int(x2) * (1 - 2 * int(z2))
        return 0

    def row_mult(self, h: int, i: int) -> None:
        """
        Performs in-place Pauli row multiplication: R_h <- R_h * R_i.
        
        Multiplies the Pauli operator in row h by the Pauli operator in row i,
        updating the phase bit r[h] and the Pauli components (x[h], z[h]).
        
        SERIAL IMPLEMENTATION:
        Processes one qubit column at a time sequentially.
        
        # TODO (Optimization):
        # 1. Parallel CPU (OpenMP): Row-level SIMD bit-packing (64 qubits per uint64)
        #    reduces this loop to n/64 operations and bitwise XORs.
        # 2. GPU Kernel (CUDA/OpenCL): Warp-level reductions for computing the
        #    exponent sum, or parallel row elimination across multiple rows.
        """
        # Sum single-qubit phase contributions across all qubits
        sum_g = 0
        for j in range(self.num_qubits):
            sum_g += self._g(
                int(self.x[h, j]),
                int(self.z[h, j]),
                int(self.x[i, j]),
                int(self.z[i, j])
            )

        # Total exponent of i in the product:
        # 2*r[h] + 2*r[i] + sum_g  (since (-1)^r = i^{2r})
        # For commuting or anti-commuting Pauli products in valid states,
        # exponent mod 4 must be 0 (phase +1) or 2 (phase -1).
        exponent = (2 * int(self.r[h]) + 2 * int(self.r[i]) + sum_g) % 4
        
        # If exponent is 2 (mod 4), sign is -1 (r[h] = 1); if 0, sign is +1 (r[h] = 0)
        self.r[h] = 1 if exponent == 2 else 0

        # Update Pauli bits via bitwise XOR
        for j in range(self.num_qubits):
            self.x[h, j] ^= self.x[i, j]
            self.z[h, j] ^= self.z[i, j]

    def get_pauli_row_string(self, row: int) -> str:
        """
        Returns a human-readable Pauli string for the specified row.
        Example: '+ XIXZ' or '- ZIZI'.
        """
        sign_char = '-' if self.r[row] == 1 else '+'
        paulis: List[str] = []
        for j in range(self.num_qubits):
            xj = self.x[row, j]
            zj = self.z[row, j]
            if xj == 0 and zj == 0:
                paulis.append('I')
            elif xj == 1 and zj == 0:
                paulis.append('X')
            elif xj == 0 and zj == 1:
                paulis.append('Z')
            else:
                paulis.append('Y')
        return f"{sign_char} {''.join(paulis)}"

    def get_stabilizers(self) -> List[str]:
        """Returns the list of n stabilizer Pauli strings."""
        return [
            self.get_pauli_row_string(self.num_qubits + i)
            for i in range(self.num_qubits)
        ]

    def get_destabilizers(self) -> List[str]:
        """Returns the list of n destabilizer Pauli strings."""
        return [
            self.get_pauli_row_string(i)
            for i in range(self.num_qubits)
        ]

    def __str__(self) -> str:
        """Formatted string representation of the entire tableau."""
        lines = [f"StabilizerTableau({self.num_qubits} qubits):"]
        lines.append("  Destabilizers:")
        for i in range(self.num_qubits):
            lines.append(f"    R_{i:<3d}: {self.get_pauli_row_string(i)}")
        lines.append("  Stabilizers:")
        for i in range(self.num_qubits):
            row_idx = self.num_qubits + i
            lines.append(f"    R_{row_idx:<3d}: {self.get_pauli_row_string(row_idx)}")
        return "\n".join(lines)
