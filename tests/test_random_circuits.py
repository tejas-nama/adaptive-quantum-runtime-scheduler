"""
Test Suite for Random Clifford Circuits.

Validates:
1. Exact deterministic reproducibility with pseudorandom seeds.
2. Stability across varying qubit counts and circuit depths.
3. Correct execution of both standard and extended Clifford gate sets.
"""

import pytest
from simulator.simulator import Simulator
from circuits.random_clifford import create_random_clifford_circuit


class TestRandomCircuits:

    def test_seed_reproducibility(self):
        """Two simulator instances with the same seed must produce identical results."""
        circuit = create_random_clifford_circuit(num_qubits=12, depth=30, seed=777)

        sim1 = Simulator(seed=999)
        sim2 = Simulator(seed=999)

        res1 = sim1.simulate(circuit, shots=50)
        res2 = sim2.simulate(circuit, shots=50)

        assert res1.bitstrings == res2.bitstrings
        assert res1.counts == res2.counts

    @pytest.mark.parametrize("n_qubits,depth", [
        (4, 10),
        (8, 20),
        (16, 30),
        (32, 20)
    ])
    def test_scaling_dimensions(self, n_qubits: int, depth: int):
        """Verifies random circuits of different dimensions run cleanly."""
        circuit = create_random_clifford_circuit(
            num_qubits=n_qubits,
            depth=depth,
            seed=42,
            include_measurements=True
        )
        assert circuit.num_qubits == n_qubits
        assert circuit.num_measurements == n_qubits
        assert circuit.num_gates > 0

        sim = Simulator(seed=42)
        res = sim.simulate(circuit, shots=5)
        assert res.shots == 5
        assert len(res.bitstrings) == 5
        for bs in res.bitstrings:
            assert len(bs) == n_qubits

    def test_extended_gate_set(self):
        """Verifies circuit generation and simulation with extended gate set (X, Y, Z, CZ)."""
        circuit = create_random_clifford_circuit(
            num_qubits=10,
            depth=25,
            seed=101,
            gate_set="extended"
        )
        sim = Simulator(seed=101)
        res = sim.simulate(circuit, shots=10)
        assert res.shots == 10
        assert len(res.bitstrings) == 10
