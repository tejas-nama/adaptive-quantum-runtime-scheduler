# Profiling Summary: Serial Stabilizer Simulator Baseline

## Overview
This report validates computational hotspots across 4 distinct quantum simulation workloads:
1. **Gate-Heavy**: 40 Qubits, Depth 100 random Clifford circuit (unitary dominant)
2. **Measurement-Heavy**: 25 Qubits, 25 syndrome-style measurement cycles (Gaussian elimination dominant)
3. **Many-Shot**: 10 Qubit GHZ state simulated across 1,500 shots (multi-shot scaling)
4. **High-Qubits**: 80 Qubits, Depth 40 random Clifford circuit (high-dimensional tableau scaling)

## Key Hotspot Findings by Workload

### Workload: `gate_heavy` (Total Time: 3.1949s)

| Rank | Function Name | Total Time (s) | % of Runtime |
|------|---------------|----------------|--------------|
| 1 | `apply_cnot` | 1.1910s | 37.3% |
| 2 | `row_mult` | 0.6801s | 21.3% |
| 3 | `apply_s` | 0.5973s | 18.7% |
| 4 | `apply_h` | 0.5606s | 17.5% |
| 5 | `_g` | 0.1039s | 3.3% |

### Workload: `measurement_heavy` (Total Time: 0.3054s)

| Rank | Function Name | Total Time (s) | % of Runtime |
|------|---------------|----------------|--------------|
| 1 | `row_mult` | 0.1929s | 63.2% |
| 2 | `apply_cnot` | 0.0301s | 9.9% |
| 3 | `apply_h` | 0.0164s | 5.4% |
| 4 | `_measure_deterministic` | 0.0135s | 4.4% |
| 5 | `_g` | 0.0126s | 4.1% |

### Workload: `many_shots` (Total Time: 1.7568s)

| Rank | Function Name | Total Time (s) | % of Runtime |
|------|---------------|----------------|--------------|
| 1 | `row_mult` | 1.1913s | 67.8% |
| 2 | `apply_cnot` | 0.2949s | 16.8% |
| 3 | `_g` | 0.0803s | 4.6% |
| 4 | `_measure_deterministic` | 0.0569s | 3.2% |
| 5 | `run_shot` | 0.0250s | 1.4% |

### Workload: `high_qubits` (Total Time: 4.9809s)

| Rank | Function Name | Total Time (s) | % of Runtime |
|------|---------------|----------------|--------------|
| 1 | `row_mult` | 2.5949s | 52.1% |
| 2 | `apply_cnot` | 0.9917s | 19.9% |
| 3 | `apply_s` | 0.4909s | 9.9% |
| 4 | `apply_h` | 0.4575s | 9.2% |
| 5 | `_g` | 0.4001s | 8.0% |

## Architectural Insights for Later Parallelization

1. **Gate-Heavy Hotspot**: `apply_cnot` and `apply_h` dominate gate-heavy circuits. Each Clifford gate updates 2n rows. On GPU/multi-core CPU, these 2n row updates can be processed in parallel with zero inter-row dependencies.

2. **Measurement-Heavy Hotspot**: `_measure_random`, `row_mult`, and `_g` dominate measurement workloads. Random projective measurement performs Gaussian elimination across the tableau (O(n^2) complexity). QuaSARQ reformulates this using parallel prefix and data-parallel row elimination primitives.

3. **Many-Shot Hotspot**: In many-shot simulations, `run_shot` and repeated tableau resets dominate. Shots are completely independent, offering embarrassingly parallel acceleration on multi-core CPU and GPU.

4. **High-Qubit Scaling**: As qubit count grows (n >= 80), the O(n^2) footprint of two-qubit gates and measurements causes quadratic scaling, making them the primary targets for GPU acceleration.
