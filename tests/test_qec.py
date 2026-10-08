"""
Test Suite for Quantum Error Correction (QEC) Stabilizer Circuits.

Validates stabilizer syndrome extraction:
1. 3-qubit bit flip code:
   - No error -> syndrome '00'
   - Error on q0 -> syndrome '10'
   - Error on q1 -> syndrome '11'
   - Error on q2 -> syndrome '01'
2. Repetition code multi-round syndrome stability.
"""

import pytest
from simulator.simulator import Simulator
from circuits.qec import (
    create_bit_flip_code_circuit,
    create_repetition_code_syndrome_circuit
)


class TestQEC:

    def test_bit_flip_syndrome_no_error(self):
        """No error injected -> ancilla syndrome measurement must be '00'."""
        circuit = create_bit_flip_code_circuit(error_qubit=None)
        sim = Simulator(seed=42)
        res = sim.simulate(circuit, shots=50)

        assert len(res.counts) == 1
        assert "00" in res.counts
        assert res.counts["00"] == 50

    def test_bit_flip_syndrome_qubit_0(self):
        """Error on data qubit 0 -> syndrome must be '10'."""
        circuit = create_bit_flip_code_circuit(error_qubit=0)
        sim = Simulator(seed=42)
        res = sim.simulate(circuit, shots=50)

        assert len(res.counts) == 1
        assert "10" in res.counts
        assert res.counts["10"] == 50

    def test_bit_flip_syndrome_qubit_1(self):
        """Error on data qubit 1 -> syndrome must be '11'."""
        circuit = create_bit_flip_code_circuit(error_qubit=1)
        sim = Simulator(seed=42)
        res = sim.simulate(circuit, shots=50)

        assert len(res.counts) == 1
        assert "11" in res.counts
        assert res.counts["11"] == 50

    def test_bit_flip_syndrome_qubit_2(self):
        """Error on data qubit 2 -> syndrome must be '01'."""
        circuit = create_bit_flip_code_circuit(error_qubit=2)
        sim = Simulator(seed=42)
        res = sim.simulate(circuit, shots=50)

        assert len(res.counts) == 1
        assert "01" in res.counts
        assert res.counts["01"] == 50

    def test_repetition_code_rounds(self):
        """Repetition code syndrome extraction in ground state produces all zeros."""
        circuit = create_repetition_code_syndrome_circuit(num_data_qubits=4, rounds=3)
        sim = Simulator(seed=42)
        res = sim.simulate(circuit, shots=20)

        expected_syndrome_length = (4 - 1) * 3  # 9 syndrome bits
        all_zeros = "0" * expected_syndrome_length

        assert len(res.counts) == 1
        assert all_zeros in res.counts
        assert res.counts[all_zeros] == 20
