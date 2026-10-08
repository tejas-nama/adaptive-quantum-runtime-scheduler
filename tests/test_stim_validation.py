"""
Validation Suite: Serial Python Simulator vs. Stim Reference Simulator.

Theoretical and Practical Reference:
    Gidney, C., 2021. "Stim: a fast stabilizer circuit simulator."
    Quantum 5, 497.

This module validates the correctness of our serial Python simulator against
the trusted C++ reference stabilizer simulator (Stim).
IMPORTANT: Stim is used strictly as an independent validation oracle, NOT
as an underlying execution engine.
"""

import pytest
import numpy as np
from simulator.simulator import Simulator
from simulator.circuit import Circuit
from circuits.basic import create_bell_state, create_ghz_state
from circuits.random_clifford import create_random_clifford_circuit


class TestStimValidation:

    @pytest.fixture(autouse=True)
    def check_stim(self):
        """Skip if stim is not installed."""
        pytest.importorskip("stim")

    def test_bell_state_matches_stim(self):
        """Validates that Bell state probabilities match Stim."""
        circuit = create_bell_state()
        stim_circuit = circuit.to_stim()

        # Run our simulator
        our_sim = Simulator(seed=42)
        our_res = our_sim.simulate(circuit, shots=1000)

        # Run Stim
        stim_sampler = stim_circuit.compile_sampler(seed=42)
        stim_samples = stim_sampler.sample(shots=1000)
        stim_counts = {}
        for row in stim_samples:
            bs = "".join(str(int(b)) for b in row)
            stim_counts[bs] = stim_counts.get(bs, 0) + 1

        # Both simulators should produce only '00' and '11'
        assert set(our_res.counts.keys()) == {"00", "11"}
        assert set(stim_counts.keys()) == {"00", "11"}

        # Probabilities should agree closely
        our_p00 = our_res.counts["00"] / 1000
        stim_p00 = stim_counts["00"] / 1000
        assert abs(our_p00 - stim_p00) < 0.08

    def test_ghz_state_matches_stim(self):
        """Validates that GHZ-5 state probabilities match Stim."""
        circuit = create_ghz_state(num_qubits=5)
        stim_circuit = circuit.to_stim()

        our_sim = Simulator(seed=123)
        our_res = our_sim.simulate(circuit, shots=1000)

        stim_sampler = stim_circuit.compile_sampler(seed=123)
        stim_samples = stim_sampler.sample(shots=1000)
        stim_counts = {}
        for row in stim_samples:
            bs = "".join(str(int(b)) for b in row)
            stim_counts[bs] = stim_counts.get(bs, 0) + 1

        expected_support = {"00000", "11111"}
        assert set(our_res.counts.keys()) == expected_support
        assert set(stim_counts.keys()) == expected_support

    def test_deterministic_clifford_circuits(self):
        """
        Generates deterministic Clifford circuits (pairs of gates and inverse operations)
        and verifies exact bit-for-bit equivalence between our simulator and Stim.
        """
        for seed in range(10):
            # Construct a circuit that starts in |0...0>, applies arbitrary Clifford operations,
            # then inverts them (or applies deterministic projections), ending in a known eigenstate.
            c = Circuit(num_qubits=4)
            if seed % 2 == 0:
                c.x(0)
            if seed % 3 == 0:
                c.x(1)
            # Apply some reversible Clifford gates
            c.h(2)
            c.s(2)
            c.s(2)
            c.s(2)
            c.s(2) # S^4 = I
            c.h(2) # H^2 = I
            c.cnot(0, 3)
            c.cnot(0, 3) # CNOT^2 = I
            c.cz(1, 2)
            c.cz(1, 2) # CZ^2 = I

            # Measure all qubits
            for q in range(4):
                c.measure(q, clbit=q)

            # Our simulator
            sim = Simulator(seed=seed)
            our_res = sim.simulate(c, shots=1)
            our_bs = our_res.bitstrings[0]

            # Stim simulator
            stim_c = c.to_stim()
            stim_sampler = stim_c.compile_sampler(seed=seed)
            stim_sample = stim_sampler.sample(shots=1)[0]
            stim_bs = "".join(str(int(b)) for b in stim_sample)

            assert our_bs == stim_bs, f"Mismatch on seed {seed}: Our={our_bs} vs Stim={stim_bs}"

    def test_random_clifford_support_and_marginal_distributions(self):
        """
        Runs a 6-qubit random Clifford circuit in both simulators.
        Verifies that every bitstring sampled by our simulator is physically valid
        according to Stim.
        """
        circuit = create_random_clifford_circuit(
            num_qubits=6,
            depth=15,
            seed=42,
            include_measurements=True
        )
        stim_circuit = circuit.to_stim()

        # Run Stim with a large number of shots to find valid state support
        stim_sampler = stim_circuit.compile_sampler(seed=42)
        stim_samples = stim_sampler.sample(shots=2000)
        stim_support = set("".join(str(int(b)) for b in row) for row in stim_samples)

        # Run our simulator
        our_sim = Simulator(seed=42)
        our_res = our_sim.simulate(circuit, shots=500)
        our_support = set(our_res.counts.keys())

        # Every outcome our simulator generates must be in Stim's valid subspace
        for bs in our_support:
            assert bs in stim_support, f"Our simulator generated invalid bitstring {bs} not in Stim support!"

        # Qubit marginals should agree within sampling tolerance
        our_marginals = np.zeros(6)
        for bs, count in our_res.counts.items():
            for i, bit in enumerate(bs):
                if bit == "1":
                    our_marginals[i] += count
        our_marginals /= 500

        stim_marginals = np.mean(stim_samples, axis=0)
        for q in range(6):
            diff = abs(our_marginals[q] - stim_marginals[q])
            assert diff < 0.10, f"Qubit {q} marginal probability differs by {diff:.4f}"
