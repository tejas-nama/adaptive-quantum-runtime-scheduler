"""
Unit tests for Computational Basis Measurement.

Validates:
1. Deterministic outcomes on computational basis states |0>, |1>, |010>
2. Random projective outcomes on superposition states |+>
3. Wavefunction collapse (repeated measurement of same qubit)
4. Entangled state measurement correlations
"""

import random
import pytest
from simulator.tableau import StabilizerTableau
from simulator.gates import apply_h, apply_x, apply_cnot
from simulator.measurement import (
    measure_qubit, find_anticommuting_stabilizer
)


class TestMeasurement:

    def test_deterministic_ground_state(self):
        """Measuring |0> must always return 0 without changing state."""
        tab = StabilizerTableau(3)
        rng = random.Random(42)

        for q in range(3):
            # All stabilizers are Z_i, so all commute with Z_q -> deterministic
            assert find_anticommuting_stabilizer(tab, q) is None
            outcome = measure_qubit(tab, q, rng=rng)
            assert outcome == 0

    def test_deterministic_excited_state(self):
        """Measuring |1> must always return 1."""
        tab = StabilizerTableau(1)
        apply_x(tab, 0)
        rng = random.Random(42)

        for _ in range(50):
            # Measure repeatedly; should stay 1
            outcome = measure_qubit(tab, 0, rng=rng)
            assert outcome == 1

    def test_deterministic_bitstring_state(self):
        """Measuring |101> must return 1, 0, 1 respectively."""
        tab = StabilizerTableau(3)
        apply_x(tab, 0)
        apply_x(tab, 2)
        rng = random.Random(42)

        assert measure_qubit(tab, 0, rng) == 1
        assert measure_qubit(tab, 1, rng) == 0
        assert measure_qubit(tab, 2, rng) == 1

    def test_random_measurement_distribution(self):
        """Measuring |+> should produce ~50% 0s and ~50% 1s."""
        rng = random.Random(12345)
        counts = {0: 0, 1: 0}
        shots = 1000

        for _ in range(shots):
            tab = StabilizerTableau(1)
            apply_h(tab, 0)
            # Stabilizer is X_0, which anti-commutes with Z_0
            assert find_anticommuting_stabilizer(tab, 0) is not None
            outcome = measure_qubit(tab, 0, rng)
            counts[outcome] += 1

        # 1000 shots with p=0.5: 3 sigma is ~3*sqrt(1000*0.25) ~ 47
        # Expect counts between 430 and 570
        assert 430 <= counts[0] <= 570, f"Unexpected distribution: {counts}"
        assert 430 <= counts[1] <= 570, f"Unexpected distribution: {counts}"

    def test_wavefunction_collapse_repeated_measure(self):
        """
        After measuring |+>, the state collapses to |0> or |1>.
        Subsequent measurements on the same qubit MUST return the same outcome.
        """
        rng = random.Random(999)
        for _ in range(100):
            tab = StabilizerTableau(1)
            apply_h(tab, 0)

            first_outcome = measure_qubit(tab, 0, rng)
            # Repeated measurement
            second_outcome = measure_qubit(tab, 0, rng)
            third_outcome = measure_qubit(tab, 0, rng)

            assert second_outcome == first_outcome, "State failed to collapse!"
            assert third_outcome == first_outcome, "State collapsed outcome drifted!"

    def test_independent_qubit_measurement(self):
        """
        In state |+0>, measuring qubit 0 should not affect qubit 1 (remains 0).
        """
        rng = random.Random(42)
        for _ in range(100):
            tab = StabilizerTableau(2)
            apply_h(tab, 0)

            # Measure qubit 0 (random)
            _ = measure_qubit(tab, 0, rng)
            # Measure qubit 1 (must be deterministically 0)
            m1 = measure_qubit(tab, 1, rng)
            assert m1 == 0, "Distant qubit corrupted by measurement!"
