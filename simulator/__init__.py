"""
Serial Stabilizer / Clifford Quantum Circuit Simulator.

This package provides a serial baseline implementation of the Aaronson-Gottesman
(2004) stabilizer tableau algorithm for Clifford quantum circuit simulation.

NOTE: This is the serial baseline implementation. No parallelization (OpenMP,
CUDA, OpenCL, multiprocessing) is included in this phase to allow pure profiling
and computational hotspot identification.
"""

from simulator.tableau import StabilizerTableau
from simulator.circuit import Circuit, GateOperation
from simulator.simulator import Simulator, SimulationResult
from simulator.measurement import measure_qubit

__all__ = [
    "StabilizerTableau",
    "Circuit",
    "GateOperation",
    "Simulator",
    "SimulationResult",
    "measure_qubit",
]
