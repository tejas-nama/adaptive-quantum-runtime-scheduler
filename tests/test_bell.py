"""
Test Suite for Two-Qubit Bell State (|Phi+>).

Circuit:
    q0: --[H]--*----[M]--
               |
    q1: -------X----[M]--

Expected Behavior:
- Exact quantum correlation: Outcomes must strictly be '00' or '11'.
- Exactly 0% probability for '01' and '10'.
- Approximately 50% probability for '00' and 50% for '11'.
"""

import pytest
from simulator.simulator import Simulator
from circuits.basic import create_bell_state
from simulator.circuit import Circuit


class TestBellState:

    def test_phi_plus_correlations(self):
        """Standard |Phi+> state: (|00> + |11>) / sqrt(2)."""
        sim = Simulator(seed=42)
        circuit = create_bell_state()
        shots = 1000

        result = sim.simulate(circuit, shots=shots)

        # 1. No orthogonal basis leakage
        assert "01" not in result.counts or result.counts["01"] == 0, "Leaked to |01>!"
        assert "10" not in result.counts or result.counts["10"] == 0, "Leaked to |10>!"

        # 2. Both valid states observed
        count_00 = result.counts.get("00", 0)
        count_11 = result.counts.get("11", 0)
        assert count_00 + count_11 == shots

        # 3. 50/50 distribution within 3-sigma binomial confidence
        assert 430 <= count_00 <= 570, f"Excessive imbalance: count_00={count_00}"
        assert 430 <= count_11 <= 570, f"Excessive imbalance: count_11={count_11}"

    def test_psi_plus_correlations(self):
        """
        |Psi+> state: (|01> + |10>) / sqrt(2).
        Created by applying X on qubit 1 before/after Bell circuit.
        Outcomes must strictly be '01' and '10'.
        """
        circuit = Circuit(2)
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.x(1)  # Bit-flip second qubit
        circuit.measure(0, clbit=0)
        circuit.measure(1, clbit=1)

        sim = Simulator(seed=123)
        result = sim.simulate(circuit, shots=500)

        assert "00" not in result.counts or result.counts["00"] == 0
        assert "11" not in result.counts or result.counts["11"] == 0

        count_01 = result.counts.get("01", 0)
        count_10 = result.counts.get("10", 0)
        assert count_01 + count_10 == 500
        assert 200 <= count_01 <= 300
        assert 200 <= count_10 <= 300
