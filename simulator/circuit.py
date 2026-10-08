"""
Quantum Circuit Representation for Stabilizer Simulation.

Defines the gate operation data structure and the Circuit container.
Supports sequential execution, gate inspection, depth calculation,
and optional export to Stim circuits for cross-validation.
"""

from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class GateOperation:
    """
    Represents an individual quantum gate or measurement operation.
    
    Attributes:
        name: Gate identifier (e.g. 'H', 'S', 'CNOT', 'X', 'Y', 'Z', 'CZ', 'M').
        qubits: Tuple of qubit indices acted on by this operation.
        clbit: Optional classical register index for measurement storage.
    """
    name: str
    qubits: Tuple[int, ...]
    clbit: Optional[int] = None

    def is_measurement(self) -> bool:
        """Returns True if the operation is a measurement."""
        return self.name in ("M", "MEASURE")

    def __str__(self) -> str:
        qubits_str = ", ".join(f"q{q}" for q in self.qubits)
        if self.is_measurement():
            clbit_str = f" -> c{self.clbit}" if self.clbit is not None else ""
            return f"M({qubits_str}){clbit_str}"
        return f"{self.name}({qubits_str})"


class Circuit:
    """
    Represents a quantum circuit consisting of Clifford gates and measurements.
    
    Provides a fluent builder interface and helper utilities for circuit analysis.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes an empty circuit on num_qubits.
        
        Args:
            num_qubits: Number of quantum bits in the circuit.
        """
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        self.num_qubits: int = num_qubits
        self.operations: List[GateOperation] = []

    def _validate_qubit(self, q: int) -> None:
        if q < 0 or q >= self.num_qubits:
            raise IndexError(f"Qubit index {q} out of bounds [0, {self.num_qubits - 1}]")

    def add_operation(self, op: GateOperation) -> "Circuit":
        """Appends a GateOperation to the circuit."""
        for q in op.qubits:
            self._validate_qubit(q)
        self.operations.append(op)
        return self

    def h(self, qubit: int) -> "Circuit":
        """Appends a Hadamard gate on qubit."""
        self._validate_qubit(qubit)
        self.operations.append(GateOperation("H", (qubit,)))
        return self

    def s(self, qubit: int) -> "Circuit":
        """Appends a Phase gate S on qubit."""
        self._validate_qubit(qubit)
        self.operations.append(GateOperation("S", (qubit,)))
        return self

    def cnot(self, control: int, target: int) -> "Circuit":
        """Appends a CNOT gate with control and target qubits."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError(f"Control and target must be distinct qubits ({control})")
        self.operations.append(GateOperation("CNOT", (control, target)))
        return self

    def cx(self, control: int, target: int) -> "Circuit":
        """Alias for cnot."""
        return self.cnot(control, target)

    def x(self, qubit: int) -> "Circuit":
        """Appends a Pauli-X gate on qubit."""
        self._validate_qubit(qubit)
        self.operations.append(GateOperation("X", (qubit,)))
        return self

    def y(self, qubit: int) -> "Circuit":
        """Appends a Pauli-Y gate on qubit."""
        self._validate_qubit(qubit)
        self.operations.append(GateOperation("Y", (qubit,)))
        return self

    def z(self, qubit: int) -> "Circuit":
        """Appends a Pauli-Z gate on qubit."""
        self._validate_qubit(qubit)
        self.operations.append(GateOperation("Z", (qubit,)))
        return self

    def cz(self, control: int, target: int) -> "Circuit":
        """Appends a Controlled-Z gate."""
        self._validate_qubit(control)
        self._validate_qubit(target)
        if control == target:
            raise ValueError(f"Control and target must be distinct qubits ({control})")
        self.operations.append(GateOperation("CZ", (control, target)))
        return self

    def measure(self, qubit: int, clbit: Optional[int] = None) -> "Circuit":
        """
        Appends a computational basis measurement on qubit.
        
        Args:
            qubit: Qubit index to measure.
            clbit: Optional classical bit index to record outcome. Defaults to qubit index.
        """
        self._validate_qubit(qubit)
        assigned_clbit = clbit if clbit is not None else qubit
        self.operations.append(GateOperation("M", (qubit,), clbit=assigned_clbit))
        return self

    @property
    def num_gates(self) -> int:
        """Returns the number of Clifford unitary gates (excluding measurements)."""
        return sum(1 for op in self.operations if not op.is_measurement())

    @property
    def num_measurements(self) -> int:
        """Returns the total number of measurement operations."""
        return sum(1 for op in self.operations if op.is_measurement())

    @property
    def depth(self) -> int:
        """
        Calculates the circuit depth (critical path length).
        
        Depth is the length of the longest path of non-overlapping gates.
        """
        qubit_depths = [0] * self.num_qubits
        for op in self.operations:
            current_max = max(qubit_depths[q] for q in op.qubits)
            new_depth = current_max + 1
            for q in op.qubits:
                qubit_depths[q] = new_depth
        return max(qubit_depths) if qubit_depths else 0

    def to_stim(self):
        """
        Converts this circuit into a Stim Circuit object for reference validation.
        
        Returns:
            stim.Circuit: An equivalent Stim circuit.
            
        Raises:
            ImportError: If the 'stim' package is not installed.
        """
        import stim
        stim_circuit = stim.Circuit()
        for op in self.operations:
            if op.name == "H":
                stim_circuit.append("H", op.qubits[0])
            elif op.name == "S":
                stim_circuit.append("S", op.qubits[0])
            elif op.name in ("CNOT", "CX"):
                stim_circuit.append("CX", [op.qubits[0], op.qubits[1]])
            elif op.name == "X":
                stim_circuit.append("X", op.qubits[0])
            elif op.name == "Y":
                stim_circuit.append("Y", op.qubits[0])
            elif op.name == "Z":
                stim_circuit.append("Z", op.qubits[0])
            elif op.name == "CZ":
                stim_circuit.append("CZ", [op.qubits[0], op.qubits[1]])
            elif op.is_measurement():
                stim_circuit.append("M", op.qubits[0])
            else:
                raise ValueError(f"Unsupported gate for Stim conversion: {op.name}")
        return stim_circuit

    def __len__(self) -> int:
        return len(self.operations)

    def __iter__(self) -> Iterator[GateOperation]:
        return iter(self.operations)

    def __str__(self) -> str:
        lines = [f"Circuit(qubits={self.num_qubits}, gates={self.num_gates}, measurements={self.num_measurements}, depth={self.depth}):"]
        for idx, op in enumerate(self.operations):
            lines.append(f"  [{idx:03d}] {op}")
        return "\n".join(lines)
