"""
Test Suite for Greenberger-Horne-Zeilinger (GHZ) States.

Circuit:
    H(q0) -> CNOT(q0, q1) -> CNOT(q1, q2) -> ... -> CNOT(q_{n-2}, q_{n-1})
    followed by measurement of all qubits.

Expected Behavior:
- Exact multi-qubit entanglement across n qubits.
- Non-zero outcomes restricted strictly to '00...0' and '11...1'.
- Zero counts for all 2^n - 2 intermediate bitstrings.
- Balanced empirical distribution (~50% each).
"""

import pytest
from simulator.simulator import Simulator
from circuits.basic import create_ghz_state


class TestGHZState:

    @pytest.mark.parametrize("n_qubits", [3, 5, 10])
    def test_ghz_correlations(self, n_qubits: int):
        """Tests GHZ state generation for various qubit counts."""
        sim = Simulator(seed=42 + n_qubits)
        circuit = create_ghz_state(num_qubits=n_qubits)
        shots = 500

        result = sim.simulate(circuit, shots=shots)

        zero_str = "0" * n_qubits
        one_str = "1" * n_qubits

        # 1. Verify only all-zeros and all-ones strings were produced
        for bs in result.counts.keys():
            assert bs in (zero_str, one_str), f"Illegal intermediate state: |{bs}> observed in GHZ-{n_qubits}!"

        # 2. Check total counts
        count_zero = result.counts.get(zero_str, 0)
        count_one = result.counts.get(one_str, 0)
        assert count_zero + count_one == shots

        # 3. Check balanced distribution (within 3 sigma)
        assert 200 <= count_zero <= 300, f"Unbalanced zeros: {count_zero}/{shots}"
        assert 200 <= count_one <= 300, f"Unbalanced ones: {count_one}/{shots}"
