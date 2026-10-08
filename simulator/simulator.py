"""
Serial Quantum Circuit Simulator for Stabilizer / Clifford Circuits.

Executes quantum circuits on the StabilizerTableau data structure.
All gate applications, measurements, and multi-shot iterations are strictly
SERIAL to provide a clean profiling baseline.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time
import random

from simulator.tableau import StabilizerTableau
from simulator.circuit import Circuit, GateOperation
from simulator.gates import CLIFFORD_GATES
from simulator.measurement import measure_qubit


@dataclass
class SimulationResult:
    """
    Encapsulates results from a multi-shot circuit simulation.
    
    Attributes:
        shots: Total number of shots executed.
        counts: Histogram of measured bitstrings (e.g., {'00': 48, '11': 52}).
        bitstrings: List of raw measured bitstrings, one per shot.
        measurements: List of dictionaries mapping classical bit index to outcome.
        total_time: Total elapsed simulation wall time in seconds.
        time_per_shot: Average simulation wall time per shot in seconds.
    """
    shots: int
    counts: Dict[str, int] = field(default_factory=dict)
    bitstrings: List[str] = field(default_factory=list)
    measurements: List[Dict[int, int]] = field(default_factory=list)
    total_time: float = 0.0
    time_per_shot: float = 0.0

    def get_probabilities(self) -> Dict[str, float]:
        """Returns empirical probability distribution for measured bitstrings."""
        if self.shots == 0:
            return {}
        return {bs: count / self.shots for bs, count in self.counts.items()}

    def __str__(self) -> str:
        probs = self.get_probabilities()
        sorted_items = sorted(self.counts.items(), key=lambda x: x[1], reverse=True)
        lines = [
            f"SimulationResult (shots={self.shots}, wall_time={self.total_time*1000:.3f} ms, "
            f"time/shot={self.time_per_shot*1000:.4f} ms):"
        ]
        for bs, count in sorted_items[:10]:
            p = probs.get(bs, 0.0)
            lines.append(f"  |{bs}> : {count:6d} ({p*100:5.1f}%)")
        if len(sorted_items) > 10:
            lines.append(f"  ... ({len(sorted_items) - 10} more bitstrings)")
        return "\n".join(lines)


class Simulator:
    """
    Serial Clifford Circuit Simulator.
    
    Executes circuits one operation at a time and one shot at a time.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the simulator.
        
        Args:
            seed: Optional integer seed for deterministic pseudo-random sampling.
        """
        self.seed: Optional[int] = seed
        self.rng: random.Random = random.Random(seed)

    def run_shot(self, circuit: Circuit, rng: Optional[random.Random] = None) -> Dict[int, int]:
        """
        Executes a single shot of the quantum circuit.
        
        Args:
            circuit: The Circuit to execute.
            rng: Optional random generator. If None, uses simulator's internal rng.
            
        Returns:
            Dictionary mapping classical bit indices to measured outcomes (0 or 1).
        """
        if rng is None:
            rng = self.rng

        # Initialize fresh tableau in ground state |0...0>
        tableau = StabilizerTableau(circuit.num_qubits)
        clbit_results: Dict[int, int] = {}

        # Serial execution: process one operation at a time sequentially
        for op in circuit.operations:
            if op.is_measurement():
                qubit = op.qubits[0]
                outcome = measure_qubit(tableau, qubit, rng=rng)
                clbit = op.clbit if op.clbit is not None else qubit
                clbit_results[clbit] = outcome
            else:
                gate_fn = CLIFFORD_GATES.get(op.name)
                if gate_fn is None:
                    raise ValueError(f"Unknown or unsupported Clifford gate: {op.name}")
                gate_fn(tableau, *op.qubits)

        return clbit_results

    def simulate(
        self,
        circuit: Circuit,
        shots: int = 1,
        seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Executes multi-shot simulation of the circuit sequentially.
        
        Args:
            circuit: The Circuit to simulate.
            shots: Number of independent simulation runs (default: 1).
            seed: Optional random seed overriding simulator default.
            
        Returns:
            SimulationResult containing counts, bitstrings, and timing metrics.
            
        # TODO (Optimization):
        # Multi-shot simulation is an embarrassingly parallel workload:
        # 1. CPU: Shots can be parallelized across multi-core processors using OpenMP/threads.
        # 2. GPU: QuaSARQ maps shots to thousands of CUDA threads/blocks running concurrently.
        # 3. Algorithmic (Stim): Reference-sample reuse avoids re-simulating deterministic
        #    parts, only tracking error Pauli frames across shots.
        """
        if shots <= 0:
            raise ValueError(f"shots must be positive, got {shots}")

        rng = random.Random(seed if seed is not None else self.seed)

        start_time = time.perf_counter()
        
        measurements: List[Dict[int, int]] = []
        bitstrings: List[str] = []
        counts: Dict[str, int] = {}

        # SERIAL SHOT LOOP: One shot at a time
        for _ in range(shots):
            shot_result = self.run_shot(circuit, rng=rng)
            measurements.append(shot_result)
            
            # Format measured classical bits into a bitstring (ordered by clbit index)
            if shot_result:
                sorted_clbits = sorted(shot_result.keys())
                bitstring = "".join(str(shot_result[k]) for k in sorted_clbits)
            else:
                bitstring = ""
            
            bitstrings.append(bitstring)
            counts[bitstring] = counts.get(bitstring, 0) + 1

        end_time = time.perf_counter()
        total_time = end_time - start_time
        time_per_shot = total_time / shots

        return SimulationResult(
            shots=shots,
            counts=counts,
            bitstrings=bitstrings,
            measurements=measurements,
            total_time=total_time,
            time_per_shot=time_per_shot,
        )
