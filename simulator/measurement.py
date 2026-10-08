"""
Computational-Basis (Z-Basis) Quantum Measurement.

Theoretical Reference:
    Aaronson, S. and Gottesman, D., 2004.
    "Improved simulation of stabilizer circuits."
    Physical Review A, 70(5), p.052328.

Measurement of a single qubit in the computational basis decomposes into two cases:

Case 1: Random (Projective) Outcome:
    There exists at least one stabilizer generator R_p (for n <= p < 2n) that
    anti-commutes with Z_qubit (i.e., x[p, qubit] == 1).
    - The outcome is 0 or 1 with equal probability (50% / 50%).
    - Row operations (Gaussian elimination pivot) eliminate the anti-commuting X
      component from all other rows i != p where x[i, qubit] == 1.
    - Destabilizer R_{p - n} is updated with the previous stabilizer R_p.
    - Stabilizer R_p is replaced with the projector (+/-) Z_qubit according to the outcome.

Case 2: Deterministic Outcome:
    All stabilizer generators R_i (for n <= i < 2n) commute with Z_qubit
    (i.e., x[i, qubit] == 0 for all stabilizers).
    - The state is already an eigenstate of Z_qubit; the outcome is fixed (0 or 1).
    - The outcome phase is extracted by multiplying stabilizer generators corresponding
      to destabilizers that have x[i, qubit] == 1 into a scratch row.
    - The tableau state itself is preserved unchanged.

SERIAL IMPLEMENTATION NOTE:
All searches, Gaussian elimination row multiplications, and scratch reductions
are executed in serial loops to preserve natural hotspot profiling.
"""

from typing import Optional
import random
from simulator.tableau import StabilizerTableau


def find_anticommuting_stabilizer(tableau: StabilizerTableau, qubit: int) -> Optional[int]:
    """
    Searches for the first stabilizer generator that anti-commutes with Z_qubit.
    
    A stabilizer R_p (for n <= p < 2n) anti-commutes with Z_qubit iff x[p, qubit] == 1.
    
    Returns:
        Index p in [n, 2n - 1] if found, otherwise None.
    """
    n = tableau.num_qubits
    for p in range(n, 2 * n):
        if tableau.x[p, qubit] == 1:
            return p
    return None


def _measure_random(
    tableau: StabilizerTableau,
    qubit: int,
    pivot_p: int,
    rng: random.Random
) -> int:
    """
    Executes a random (projective) measurement update on the tableau.
    
    Outcome is 0 or 1 with 50/50 probability.
    
    # TODO (Optimization):
    # This function is a MAJOR computational hotspot in measurement-heavy circuits:
    # 1. Row multiplications (Gaussian elimination) across up to 2n rows: O(n^2) serial complexity.
    # 2. In QuaSARQ (GPU), this is reformulated as parallel row-elimination primitives.
    # 3. In Stim, inverse tableau techniques and 64-bit SIMD bitmasks accelerate this step.
    """
    n = tableau.num_qubits
    outcome = rng.randint(0, 1)

    # Eliminate X on qubit in all other rows (destabilizers and stabilizers)
    # using pivot row p
    for i in range(2 * n):
        if i != pivot_p and tableau.x[i, qubit] == 1:
            tableau.row_mult(i, pivot_p)

    # Update corresponding destabilizer (row p - n) to old stabilizer row p
    destab_idx = pivot_p - n
    tableau.x[destab_idx, :] = tableau.x[pivot_p, :]
    tableau.z[destab_idx, :] = tableau.z[pivot_p, :]
    tableau.r[destab_idx] = tableau.r[pivot_p]

    # Project stabilizer row p to (-1)^outcome * Z_qubit
    tableau.x[pivot_p, :].fill(0)
    tableau.z[pivot_p, :].fill(0)
    tableau.z[pivot_p, qubit] = 1
    tableau.r[pivot_p] = outcome

    return outcome


def _measure_deterministic(tableau: StabilizerTableau, qubit: int) -> int:
    """
    Computes the deterministic measurement outcome without modifying the state.
    
    Uses scratch row 2n to accumulate phase from stabilizers corresponding to
    destabilizers that have an X component on the measured qubit.
    
    # TODO (Optimization):
    # Serial accumulation over destabilizers can be computed using parallel tree
    # reduction or inverse-tableau lookups.
    """
    n = tableau.num_qubits
    scratch = 2 * n

    # Reset scratch row to identity (+ I)
    tableau.x[scratch, :].fill(0)
    tableau.z[scratch, :].fill(0)
    tableau.r[scratch] = 0

    # Accumulate stabilizers corresponding to destabilizers with x[i, qubit] == 1
    for i in range(n):
        if tableau.x[i, qubit] == 1:
            tableau.row_mult(scratch, n + i)

    # The accumulated sign (-1)^r determines the outcome (r in {0, 1})
    return int(tableau.r[scratch])


def measure_qubit(
    tableau: StabilizerTableau,
    qubit: int,
    rng: Optional[random.Random] = None
) -> int:
    """
    Measures the specified qubit in the computational (Z) basis.
    
    Args:
        tableau: The StabilizerTableau representing the quantum state.
        qubit: Index of the qubit to measure (0 <= qubit < n).
        rng: Optional random.Random instance for reproducible sampling.
             If None, the default random module is used.
             
    Returns:
        0 or 1: The measurement outcome bit.
    """
    if qubit < 0 or qubit >= tableau.num_qubits:
        raise IndexError(f"Qubit index {qubit} out of range [0, {tableau.num_qubits - 1}]")

    if rng is None:
        rng = random.Random()

    # Step 1: Check for anti-commuting stabilizer
    p = find_anticommuting_stabilizer(tableau, qubit)

    # Step 2: Branch to random or deterministic measurement
    if p is not None:
        return _measure_random(tableau, qubit, p, rng)
    else:
        return _measure_deterministic(tableau, qubit)
