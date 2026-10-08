# Adaptive CPU–GPU Runtime Scheduling for Quantum Simulation
## Milestone 1: Base Serial Stabilizer / Clifford Quantum Circuit Simulator

> [!IMPORTANT]
> **Serial Baseline Milestone**: This repository currently contains the **serial baseline only**.
> All parallelization (OpenMP, CUDA, OpenCL, GPU execution), CPU–GPU adaptive scheduling, cost models, ML-based runtime selection, and hybrid execution will be implemented in subsequent phases **after rigorous empirical hotspot identification**.

---

## 1. Project Objective

Stabilizer (Clifford) circuits play a foundational role in quantum information science, forming the backbone of quantum error correction (QEC), randomized benchmarking, fault-tolerant magic state distillation, and large-scale quantum circuit validation. By the Gottesman–Knill theorem, Clifford circuits acting on $n$ qubits can be simulated classically in polynomial time $\mathcal{O}(n^3)$ (or $\mathcal{O}(n^2)$ per gate/measurement) on classical computers.

The overarching research objective of this HPC project is to develop an **Adaptive CPU–GPU Runtime Scheduler** that dynamically decides whether different quantum circuit operations, simulation stages, or shot batches should execute on:
- **CPU** (optimizing low-latency scalar operations, cache-locality, and branching),
- **GPU** (exploiting massive data-parallelism across qubits and shots), or
- **CPU + GPU (Hybrid)** (co-scheduling independent phases or pipelining multi-shot workloads).

The research evaluation will quantify:
1. **Execution Time & Speedup**
2. **Scheduling Overhead**
3. **Hardware Utilization** (CPU core usage vs. GPU compute/memory bandwidth)
4. **Crossover Points** where GPU acceleration overcomes PCIe host-device transfer overheads.

Before implementing any parallel or adaptive scheduling logic, the methodology demands building a **correct, transparent, and un-optimized serial baseline** to profile and empirically identify computational bottlenecks.

---

## 2. Theoretical Background: The Gottesman–Knill Theorem

A quantum state $|\psi\rangle$ of an $n$-qubit register is a **stabilizer state** if it is the unique simultaneous $+1$ eigenstate of an abelian subgroup $\mathcal{S} \subset \mathcal{P}_n$ of order $2^n$ of the $n$-qubit Pauli group $\mathcal{P}_n$, where $-I \notin \mathcal{S}$.

The Clifford group $\mathcal{C}_n$ is the normalizer of the Pauli group:
$$\mathcal{C}_n = \{ U \in \mathcal{U}(2^n) \mid U P U^\dagger \in \mathcal{P}_n \quad \forall P \in \mathcal{P}_n \}$$

Any Clifford circuit can be generated using three elementary gates:
- **Hadamard ($H$)**
- **Phase ($S$)**
- **Controlled-NOT ($CNOT$)**

Because Clifford gates map Pauli operators to Pauli operators under conjugation, the quantum state can be tracked entirely through the generators of its stabilizer group rather than maintaining a $2^n$-dimensional state vector.

---

## 3. Reference Frameworks

### 3.1 Stim (Fast CPU Stabilizer Simulator)
- **Reference**: Gidney, C., 2021. *"Stim: a fast stabilizer circuit simulator."* Quantum 5, p.497.
- **Role in Project**: High-performance CPU reference architecture. Stim demonstrates that CPU stabilizer simulation achieves billions of Pauli operations per second through:
  - Cache-friendly layouts and 64-bit/256-bit AVX2/AVX-512 SIMD bit-packing,
  - Inverse-tableau tracking,
  - Reference-sample reuse (Pauli frame tracking across many shots).
- **Milestone 1 Boundary**: Stim is utilized strictly as an **external verification oracle** in tests (`tests/test_stim_validation.py`) to guarantee mathematical correctness. We do not incorporate Stim's C++ code into our simulator.

### 3.2 QuaSARQ (GPU-Accelerated Stabilizer Simulation)
- **Reference**: *"QuaSARQ: GPU-accelerated stabilizer simulation."* arXiv:2603.14641.
- **Role in Project**: High-performance GPU reference architecture. QuaSARQ reformulates stabilizer simulation into data-parallel GPU primitives:
  - Reformulating Gaussian elimination into parallel row-elimination kernels,
  - Massively parallel multi-shot evaluation across thousands of CUDA threads,
  - Vectorized Pauli updates.
- **Milestone 1 Boundary**: QuaSARQ serves as our algorithmic reference for understanding how Gaussian elimination and measurement row reductions map to hardware. We deliberately avoid premature GPU implementation until profiling data is acquired.

---

## 4. Architecture of the Serial Simulator

```
adaptive-quantum-scheduler/
├── simulator/                      # Core simulation engine
│   ├── __init__.py                 # Package exports
│   ├── tableau.py                  # StabilizerTableau (Aaronson-Gottesman representation)
│   ├── gates.py                    # Clifford gates (H, S, CNOT, X, Y, Z, CZ)
│   ├── measurement.py              # Computational Z-basis measurement (random & deterministic)
│   ├── circuit.py                  # Quantum Circuit & GateOperation abstractions
│   └── simulator.py                # Serial Simulator & multi-shot runner
├── circuits/                       # Benchmark and test circuit generators
│   ├── __init__.py
│   ├── basic.py                    # Bell state, GHZ state, single-qubit H
│   ├── random_clifford.py          # Configurable depth/width random Clifford circuits
│   ├── measurement_heavy.py        # Interleaved gate and mid-circuit measurement cycles
│   └── qec.py                      # 3-qubit bit flip code & repetition code stabilizer circuits
├── benchmarks/                     # Benchmarking harness
│   ├── __init__.py
│   ├── configs.py                  # Scaling configurations (qubits, depth, shots)
│   └── benchmark.py                # High-resolution benchmark runner & CSV logger
├── profiling/                      # Computational hotspot profiling
│   ├── __init__.py
│   ├── profile.py                  # cProfile / pstats driver across 4 workload types
│   └── results/                    # Saved profiling tables and summary reports
├── tests/                          # Automated pytest suite
│   ├── __init__.py
│   ├── test_gates.py               # Gate algebraic identities and conjugation rules
│   ├── test_measurement.py         # Deterministic/random measurements & projective collapse
│   ├── test_bell.py                # Bell state 50/50 correlation validation
│   ├── test_ghz.py                 # Multi-qubit GHZ state correlation validation
│   ├── test_random_circuits.py     # Deterministic seed reproducibility & scaling
│   ├── test_qec.py                 # Stabilizer syndrome detection validation
│   └── test_stim_validation.py     # Direct cross-validation against Stim
├── results/                        # Benchmark execution logs
│   └── benchmark_results.csv
├── requirements.txt                # Python dependencies
└── README.md                       # Complete documentation
```

---

## 5. Core Data Structure: Aaronson–Gottesman Stabilizer Tableau

The simulator maintains the quantum state using the binary tableau representation introduced by Aaronson and Gottesman (2004):

$$\text{Tableau} = \begin{pmatrix} X & Z & r \end{pmatrix} \in \mathbb{F}_2^{(2n + 1) \times (2n + 1)}$$

### 5.1 Row Deconstruction
For an $n$-qubit register:
- **Rows $0$ to $n - 1$ (Destabilizers $R_0, \dots, R_{n-1}$)**: Pauli operators that generate an independent subgroup completing the symplectic basis.
- **Rows $n$ to $2n - 1$ (Stabilizers $R_n, \dots, R_{2n-1}$)**: The $n$ commuting Pauli operators that uniquely specify the stabilizer state $|\psi\rangle$ ($R_{n+i} |\psi\rangle = +|\psi\rangle$).
- **Row $2n$ (Scratch Space)**: Dedicated row used to accumulate stabilizer generators during deterministic measurement phase calculation without altering the active state.

### 5.2 Column Representation
Each row $i$ encodes a Pauli operator $R_i = (-1)^{r_i} \prod_{j=0}^{n-1} P_{i, j}$:
- **$X$ Component ($x_{i, j} \in \{0, 1\}$)**: 1 if $P_{i, j} \in \{X, Y\}$, 0 otherwise.
- **$Z$ Component ($z_{i, j} \in \{0, 1\}$)**: 1 if $P_{i, j} \in \{Z, Y\}$, 0 otherwise.

$$\begin{aligned}
(x_{i,j}=0, z_{i,j}=0) &\implies I \\
(x_{i,j}=1, z_{i,j}=0) &\implies X \\
(x_{i,j}=0, z_{i,j}=1) &\implies Z \\
(x_{i,j}=1, z_{i,j}=1) &\implies Y \quad (\text{with } Y = iXZ)
\end{aligned}$$

### 5.3 Phase Vector ($r_i \in \{0, 1\}$)
- $r_i = 0 \implies \text{Sign } +1$
- $r_i = 1 \implies \text{Sign } -1$

### 5.4 Exact Row Multiplication (`row_mult`)
To compute $R_h \leftarrow R_h \cdot R_i$, the phase exponent is calculated via the Aaronson–Gottesman $g$ function:
$$g(x_1, z_1, x_2, z_2) = \begin{cases} 
0 & \text{if } x_1=0, z_1=0 \\
z_2 - x_2 & \text{if } x_1=1, z_1=1 \\
z_2(2x_2 - 1) & \text{if } x_1=1, z_1=0 \\
x_2(1 - 2z_2) & \text{if } x_1=0, z_1=1
\end{cases}$$

The net exponent of $i$ is:
$$E = \left(2r_h + 2r_i + \sum_{j=0}^{n-1} g(x_{hj}, z_{hj}, x_{ij}, z_{ij})\right) \pmod 4$$
Since the stabilizer product is Hermitian ($\pm 1$), $E \in \{0, 2\}$. If $E = 2$, $r_h \leftarrow 1$; else $r_h \leftarrow 0$. The Pauli coordinates update via bitwise XOR: $x_{h, j} \leftarrow x_{h, j} \oplus x_{i, j}$ and $z_{h, j} \leftarrow z_{h, j} \oplus z_{i, j}$.

---

## 6. Supported Clifford Gates

All gate operations act via unitary conjugation on the stabilizer and destabilizer generators across all $2n$ rows $i \in \{0, \dots, 2n-1\}$:

| Gate | Unitary Matrix | Action on Generators | Tableau Update Rule ($\forall i \in [0, 2n-1]$) |
|---|---|---|---|
| **$H$** (Hadamard) | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | $X \to Z$, $Z \to X$, $Y \to -Y$ | $r_i \leftarrow r_i \oplus (x_{iq} z_{iq})$, $\text{swap}(x_{iq}, z_{iq})$ |
| **$S$** (Phase) | $\begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ | $X \to Y$, $Z \to Z$, $Y \to -X$ | $r_i \leftarrow r_i \oplus (x_{iq} z_{iq})$, $z_{iq} \leftarrow z_{iq} \oplus x_{iq}$ |
| **$CNOT$** ($CX$) | $\begin{pmatrix} I & 0 \\ 0 & X \end{pmatrix}$ | $X_c \to X_c X_t$, $Z_t \to Z_c Z_t$, $Y_c Y_t \to -X_c Z_t$ | $r_i \leftarrow r_i \oplus (x_{ic} z_{it} (x_{it} \oplus z_{ic} \oplus 1))$, $x_{it} \leftarrow x_{it} \oplus x_{ic}$, $z_{ic} \leftarrow z_{ic} \oplus z_{it}$ |
| **$X$** (Bit Flip) | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | $Z \to -Z$, $Y \to -Y$ | $r_i \leftarrow r_i \oplus z_{iq}$ |
| **$Y$** | $\begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$ | $X \to -X$, $Z \to -Z$ | $r_i \leftarrow r_i \oplus (x_{iq} \oplus z_{iq})$ |
| **$Z$** (Phase Flip) | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | $X \to -X$, $Y \to -Y$ | $r_i \leftarrow r_i \oplus x_{iq}$ |
| **$CZ$** | $\text{diag}(1, 1, 1, -1)$ | $X_c \to X_c Z_t$, $X_t \to Z_c X_t$ | $r_i \leftarrow r_i \oplus (x_{ic} x_{it} (z_{ic} \oplus z_{it}))$, $z_{ic} \leftarrow z_{ic} \oplus x_{it}$, $z_{it} \leftarrow z_{it} \oplus x_{ic}$ |

---

## 7. Computational Basis Measurement Algorithm

Measuring qubit $q$ in the computational ($Z$) basis strictly follows the Aaronson–Gottesman procedure:

1. **Anti-Commutation Search**:
   Scan stabilizer rows $p \in \{n, \dots, 2n-1\}$ to check if any generator anti-commutes with $Z_q$ (i.e., $x_{pq} == 1$).

2. **Branch A: Random (Projective) Outcome**:
   - Condition: $\exists p \in \{n, \dots, 2n-1\}$ such that $x_{pq} == 1$.
   - Outcome $v \in \{0, 1\}$ is non-deterministic ($p(0) = p(1) = 0.5$).
   - **Gaussian Elimination**: For all other rows $i \in \{0, \dots, 2n-1\}$ where $i \ne p$ and $x_{iq} == 1$, eliminate the $X$ component by calling `row_mult(i, p)`.
   - Update destabilizer: Set row $p - n \leftarrow \text{old row } p$.
   - Project stabilizer: Set row $p \leftarrow (-1)^v Z_q$.
   - Return $v$.

3. **Branch B: Deterministic Outcome**:
   - Condition: $\forall i \in \{n, \dots, 2n-1\}$, $x_{iq} == 0$.
   - Outcome is deterministic; state is already an eigenstate of $Z_q$.
   - Initialize scratch row $2n \leftarrow +I$.
   - For each destabilizer $i \in \{0, \dots, n-1\}$ where $x_{iq} == 1$, call `row_mult(2n, n + i)`.
   - The measured bit is $v = r_{2n}$.
   - The active tableau state is preserved without change.
   - Return $v$.

---

## 8. Verification and Test Suite

The test suite covers 35 unit and integration tests across 7 test modules:

```bash
pytest -v
```

### Test Categories:
1. `test_gates.py`: Single-qubit and two-qubit identities ($H^2=I$, $S^4=I$, $CNOT^2=I$, $CZ^2=I$, conjugation transformations).
2. `test_measurement.py`: Deterministic states ($|0\rangle, |1\rangle, |101\rangle$), random superposition ($|+\rangle$), projective collapse (repeated measurement of the same qubit yields identical outcome).
3. `test_bell.py`: Bell state $|\Phi^+\rangle$ distribution across 1,000 shots. Strictly verifies $0\%$ probability for $|01\rangle$ and $|10\rangle$, and $\sim 50\%$ for $|00\rangle$ and $|11\rangle$.
4. `test_ghz.py`: Multi-qubit GHZ state ($N = 3, 5, 10$) across 500 shots. Validates zero leakage into intermediate bitstrings.
5. `test_random_circuits.py`: Deterministic seed reproducibility across variable depths (up to 30) and qubits (up to 32).
6. `test_qec.py`: 3-qubit bit flip code syndrome detection ($Z_0 Z_1$ and $Z_1 Z_2$ ancilla measurements) diagnosing single-qubit errors deterministically, plus multi-round repetition code checks.
7. `test_stim_validation.py`: Direct cross-validation against the Stim reference simulator:
   - Identical Bell/GHZ state distributions.
   - Exact bit-for-bit equivalence on deterministic Clifford circuits.
   - Subspace support and marginal probability agreement on random Clifford circuits.

---

## 9. Benchmark Methodology & Results

The benchmark framework measures performance scaling along three primary axes:
1. **Qubit Scaling** ($n \in \{5, 10, 20, 50, 100\}$ at fixed depth 50)
2. **Depth Scaling** ($d \in \{20, 50, 100, 200, 500\}$ at fixed $n = 20$)
3. **Shot Scaling** ($s \in \{1, 10, 50, 100, 500, 1000\}$)
4. **Workload Variety** (Gate-heavy vs. Measurement-heavy vs. QEC repetition code)

### Running Benchmarks:
```bash
python -m benchmarks.benchmark
```
*Results are automatically written to `results/benchmark_results.csv`.*

---

## 10. Profiling Methodology & Hotspot Identification

### Running Profiling:
```bash
python -m profiling.profile --workload all
```
*Individual reports are saved to `profiling/results/`.*

### Profiling Workloads:
1. **Gate-Heavy Workload**: 40 Qubits, Depth 100 random Clifford circuit (unitary gate dominant).
2. **Measurement-Heavy Workload**: 25 Qubits, 25 syndrome-extraction cycles (Gaussian elimination dominant).
3. **Many-Shot Workload**: 10 Qubit GHZ state across 1,500 shots (multi-shot overhead).
4. **High-Qubit Scaling Workload**: 80 Qubits, Depth 40 random Clifford circuit ($O(n^2)$ scaling).

### Top Computational Hotspots Identified:

| Workload Type | Top Hotspot Function | % of Execution Time | Underlying Cause |
|---|---|---|---|
| **Gate-Heavy** | `apply_cnot` | **37.6%** | Iterating across $2n$ rows updating bitwise parity for two qubit columns. |
| **Gate-Heavy** | `row_mult` | **20.8%** | Row updates during terminal measurements. |
| **Gate-Heavy** | `apply_s`, `apply_h` | **36.7%** (combined) | Iterating across $2n$ rows performing single-column swaps and phase accumulation. |
| **Measurement-Heavy** | `row_mult` | **62.8%** | Repeated row elimination during random and deterministic measurements ($O(n^2)$ complexity). |
| **Measurement-Heavy** | `_measure_deterministic` | **61.2%** (cumulative) | Accumulating stabilizer generators into scratch row for syndrome checks. |
| **Many-Shot** | `run_shot` / Tableau Init | **99.8%** (cumulative) | Repeated per-shot allocation, resets, and serial circuit iteration. |

---

## 11. Current Limitations

1. **Serial Execution Only**: All row updates and shot iterations execute sequentially on a single CPU thread.
2. **No Bit-Packing**: Binary arrays are stored as `np.uint8` rather than bit-packed 64-bit words (`uint64`), increasing memory footprint and preventing SIMD bitwise parallelism.
3. **No Reference-Sample Reuse**: Every shot re-simulates the entire circuit from scratch without caching deterministic frames.
4. **No GPU Acceleration**: No CUDA/OpenCL kernels or GPU-friendly data-parallel primitives are active in this phase.

---

## 12. Future Parallelization & Adaptive Scheduling Plan

Having identified the exact computational bottlenecks, subsequent project phases will proceed according to the research roadmap:

```
[ PHASE 1: Serial Baseline (Current) ]
                 ↓
[ PHASE 2: Parallel CPU Implementation ]
  - Row-level OpenMP multi-threading for apply_cnot, apply_h, apply_s
  - 64-bit SIMD bit-packing for row_mult
  - Multi-threaded shot distribution
                 ↓
[ PHASE 3: Parallel GPU Implementation (QuaSARQ-inspired) ]
  - Data-parallel CUDA kernels for Clifford gate row updates
  - GPU-accelerated parallel Gaussian elimination for measurement updates
  - Massive multi-shot batching across thousands of CUDA threads
                 ↓
[ PHASE 4: Adaptive CPU–GPU Runtime Scheduler ]
  - Cost models for CPU vs. GPU crossover points
  - Dynamic scheduling based on: (qubits, depth, measurement density, shot count)
  - Evaluation of speedup, scheduling overhead, and hardware utilization
```
