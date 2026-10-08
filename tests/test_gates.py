"""
Unit tests for Clifford Gates on Stabilizer Tableau.

Tests algebraic identities, conjugation rules, and basic gate combinations:
- Involutions: H^2 = I, X^2 = I, Y^2 = I, Z^2 = I, CNOT^2 = I, CZ^2 = I
- Phase order: S^4 = I, S^2 = Z
- Conjugations: H Z H = X, H X H = Z, S X S^dag = Y, S Z S^dag = Z
"""

import pytest
import numpy as np
from simulator.tableau import StabilizerTableau
from simulator.gates import (
    apply_h, apply_s, apply_cnot, apply_x, apply_y, apply_z, apply_cz
)


def assert_tableaux_equal(tab1: StabilizerTableau, tab2: StabilizerTableau):
    """Helper asserting identical tableau contents."""
    assert tab1.num_qubits == tab2.num_qubits
    assert np.array_equal(tab1.x[:2*tab1.num_qubits], tab2.x[:2*tab2.num_qubits]), "X matrices differ"
    assert np.array_equal(tab1.z[:2*tab1.num_qubits], tab2.z[:2*tab2.num_qubits]), "Z matrices differ"
    assert np.array_equal(tab1.r[:2*tab1.num_qubits], tab2.r[:2*tab2.num_qubits]), "Phase vectors differ"


class TestSingleQubitGates:

    def test_hadamard_involution(self):
        """H^2 = I."""
        tab = StabilizerTableau(2)
        initial = tab.copy()

        apply_h(tab, 0)
        # Verify it transformed state away from initial
        assert not np.array_equal(tab.x, initial.x)

        apply_h(tab, 0)
        # H^2 brings back to initial
        assert_tableaux_equal(tab, initial)

    def test_hadamard_conjugation(self):
        """H transforms Z into X and X into Z."""
        tab = StabilizerTableau(1)
        # Initial: Destab X_0, Stab Z_0
        apply_h(tab, 0)
        # After H: Destab Z_0, Stab X_0
        assert tab.get_stabilizers() == ["+ X"]
        assert tab.get_destabilizers() == ["+ Z"]

    def test_phase_gate_period(self):
        """S^4 = I and S^2 = Z."""
        tab = StabilizerTableau(1)
        initial = tab.copy()

        apply_s(tab, 0)
        apply_s(tab, 0)
        # S^2 should equal Z
        tab_z = initial.copy()
        apply_z(tab_z, 0)
        assert_tableaux_equal(tab, tab_z)

        # S^4 should return to initial ground state
        apply_s(tab, 0)
        apply_s(tab, 0)
        assert_tableaux_equal(tab, initial)

    def test_phase_conjugation(self):
        """S X S^dag = Y, where S^dag = S^3."""
        tab = StabilizerTableau(1)
        # Prepare |+> state so stabilizer is X
        apply_h(tab, 0)
        assert tab.get_stabilizers() == ["+ X"]

        # Apply S: X -> Y
        apply_s(tab, 0)
        assert tab.get_stabilizers() == ["+ Y"]

        # Apply S^3 (S^dag): Y -> X
        apply_s(tab, 0)
        apply_s(tab, 0)
        apply_s(tab, 0)
        assert tab.get_stabilizers() == ["+ X"]

    def test_pauli_involutions(self):
        """X^2 = Y^2 = Z^2 = I."""
        for gate_fn in [apply_x, apply_y, apply_z]:
            tab = StabilizerTableau(2)
            initial = tab.copy()
            gate_fn(tab, 0)
            gate_fn(tab, 0)
            assert_tableaux_equal(tab, initial)


class TestTwoQubitGates:

    def test_cnot_involution(self):
        """CNOT^2 = I."""
        tab = StabilizerTableau(3)
        initial = tab.copy()

        apply_cnot(tab, 0, 1)
        assert not np.array_equal(tab.x, initial.x)

        apply_cnot(tab, 0, 1)
        assert_tableaux_equal(tab, initial)

    def test_cnot_stabilizer_propagation(self):
        """
        Tests CNOT on |+0>:
        Initial state |+0>: Stabilizers are X_0 and Z_1.
        After CNOT(0, 1):
        CNOT (X_0) CNOT = X_0 X_1
        CNOT (Z_1) CNOT = Z_0 Z_1
        Stabilizers should be X_0 X_1 and Z_0 Z_1 (Bell state).
        """
        tab = StabilizerTableau(2)
        apply_h(tab, 0)
        apply_cnot(tab, 0, 1)

        stabs = tab.get_stabilizers()
        assert "+ XX" in stabs
        assert "+ ZZ" in stabs

    def test_cz_involution_and_symmetry(self):
        """CZ^2 = I and CZ is symmetric between control and target."""
        tab1 = StabilizerTableau(2)
        apply_h(tab1, 0)
        apply_h(tab1, 1)
        initial = tab1.copy()

        # CZ(0, 1)
        apply_cz(tab1, 0, 1)
        # CZ^2
        apply_cz(tab1, 0, 1)
        assert_tableaux_equal(tab1, initial)

        # Symmetry: CZ(0, 1) == CZ(1, 0)
        tab_sym1 = initial.copy()
        apply_cz(tab_sym1, 0, 1)

        tab_sym2 = initial.copy()
        apply_cz(tab_sym2, 1, 0)

        assert_tableaux_equal(tab_sym1, tab_sym2)

    def test_invalid_qubit_indices(self):
        """Verifies boundary index error handling."""
        tab = StabilizerTableau(2)
        with pytest.raises(IndexError):
            apply_h(tab, 2)
        with pytest.raises(IndexError):
            apply_s(tab, -1)
        with pytest.raises(ValueError):
            apply_cnot(tab, 0, 0)
        with pytest.raises(ValueError):
            apply_cz(tab, 1, 1)
