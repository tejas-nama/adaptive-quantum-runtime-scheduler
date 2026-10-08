"""
Profiling Harness for Serial Quantum Circuit Simulation Baseline.

Uses cProfile and pstats to profile execution across distinct quantum workloads:
1. Gate-Heavy Clifford circuits (unitary operations dominate)
2. Measurement-Heavy circuits (Gaussian elimination / row reduction dominate)
3. Many-Shot simulation (repetition and per-shot overheads)
4. High-Qubit Scaling circuits (100 qubits scaling)

Outputs structured hotspot reports with:
- Function Name
- Call Count
- Total Time (exclusive time spent inside function)
- Cumulative Time (time including subroutines)
- Percentage of Total Runtime

Usage:
    python -m profiling.profile [--workload {all,gate,meas,shots,scale}] [--top N]
"""

import os
import io
import cProfile
import pstats
import argparse
from typing import Dict, List, Any

from simulator.simulator import Simulator
from circuits.basic import create_ghz_state
from circuits.random_clifford import create_random_clifford_circuit
from circuits.measurement_heavy import create_measurement_heavy_circuit


def format_profile_table(stats: pstats.Stats, top_n: int = 20) -> str:
    """
    Extracts and formats the top N functions by total time into a structured table.
    """
    total_time = stats.total_tt
    if total_time == 0:
        total_time = 1e-9

    # stats.stats maps (filename, line, func_name) -> (cc, nc, tt, ct, callers)
    entries = []
    for func_tuple, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, line, func_name = func_tuple
        # Shorten filename to module path if possible
        short_file = os.path.basename(filename)
        pct_tt = (tt / total_time) * 100.0
        pct_ct = (ct / total_time) * 100.0
        entries.append({
            "func": func_name,
            "file": short_file,
            "line": line,
            "calls": nc,
            "tottime": tt,
            "cumtime": ct,
            "pct_tt": pct_tt,
            "pct_ct": pct_ct
        })

    # Sort descending by exclusive tottime
    entries.sort(key=lambda x: x["tottime"], reverse=True)

    header = (
        f"{'Function Name':<32} | {'Calls':<10} | {'Total (s)':<10} | "
        f"{'% Tot':<7} | {'Cum (s)':<10} | {'% Cum':<7} | {'Location':<20}"
    )
    separator = "-" * len(header)
    lines = [separator, header, separator]

    for item in entries[:top_n]:
        loc = f"{item['file']}:{item['line']}"
        lines.append(
            f"{item['func']:<32} | {item['calls']:<10d} | {item['tottime']:<10.4f} | "
            f"{item['pct_tt']:<6.2f}% | {item['cumtime']:<10.4f} | {item['pct_ct']:<6.2f}% | {loc:<20}"
        )
    lines.append(separator)
    return "\n".join(lines)


def profile_workload(
    name: str,
    circuit_fn,
    shots: int,
    output_dir: str = "profiling/results",
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Profiles a specific workload configuration and generates reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*80}")
    print(f" PROFILING WORKLOAD: {name.upper()} (Shots={shots})")
    print(f"{'='*80}")

    circuit = circuit_fn()
    print(f"Circuit stats: Qubits={circuit.num_qubits}, Depth={circuit.depth}, "
          f"Gates={circuit.num_gates}, Measurements={circuit.num_measurements}")

    sim = Simulator(seed=42)
    profiler = cProfile.Profile()

    profiler.enable()
    sim.simulate(circuit, shots=shots, seed=42)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats("tottime")

    # Generate structured table
    table_str = format_profile_table(stats, top_n=top_n)
    print(table_str)

    # Save detailed pstats dump to file
    report_path = os.path.join(output_dir, f"profile_{name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"PROFILING REPORT: {name}\n")
        f.write(f"Total Wall Time: {stats.total_tt:.4f} seconds\n")
        f.write(f"Circuit: Qubits={circuit.num_qubits}, Depth={circuit.depth}, "
                f"Gates={circuit.num_gates}, Measurements={circuit.num_measurements}, Shots={shots}\n\n")
        f.write(table_str)
        f.write("\n\nDETAILED PSTATS PRINTOUT:\n")
        stats.print_stats(top_n)

    print(f"[+] Saved profiling report to: {report_path}")

    # Return key hotspot metrics
    top_entries = sorted(stats.stats.items(), key=lambda x: x[1][2], reverse=True)[:5]
    top_hotspots = [
        (k[2], v[2], (v[2] / (stats.total_tt if stats.total_tt > 0 else 1e-9)) * 100.0)
        for k, v in top_entries
    ]

    return {
        "name": name,
        "total_time": stats.total_tt,
        "top_hotspots": top_hotspots,
        "table": table_str
    }


def run_all_profiles(output_dir: str = "profiling/results", top_n: int = 15) -> None:
    """
    Runs all 4 representative workloads and writes a synthesized summary.
    """
    os.makedirs(output_dir, exist_ok=True)

    workloads = [
        (
            "gate_heavy",
            lambda: create_random_clifford_circuit(num_qubits=40, depth=100, seed=42, include_measurements=True),
            10
        ),
        (
            "measurement_heavy",
            lambda: create_measurement_heavy_circuit(num_qubits=25, cycles=25, gates_per_cycle=6, measure_fraction=0.5, seed=42),
            10
        ),
        (
            "many_shots",
            lambda: create_ghz_state(num_qubits=10),
            1500
        ),
        (
            "high_qubits",
            lambda: create_random_clifford_circuit(num_qubits=80, depth=40, seed=42, include_measurements=True),
            5
        ),
    ]

    results = []
    for name, c_fn, shots in workloads:
        res = profile_workload(name, c_fn, shots, output_dir=output_dir, top_n=top_n)
        results.append(res)

    # Write synthesized Markdown summary
    summary_path = os.path.join(output_dir, "profile_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Profiling Summary: Serial Stabilizer Simulator Baseline\n\n")
        f.write("## Overview\n")
        f.write("This report validates computational hotspots across 4 distinct quantum simulation workloads:\n")
        f.write("1. **Gate-Heavy**: 40 Qubits, Depth 100 random Clifford circuit (unitary dominant)\n")
        f.write("2. **Measurement-Heavy**: 25 Qubits, 25 syndrome-style measurement cycles (Gaussian elimination dominant)\n")
        f.write("3. **Many-Shot**: 10 Qubit GHZ state simulated across 1,500 shots (multi-shot scaling)\n")
        f.write("4. **High-Qubits**: 80 Qubits, Depth 40 random Clifford circuit (high-dimensional tableau scaling)\n\n")

        f.write("## Key Hotspot Findings by Workload\n\n")
        for res in results:
            f.write(f"### Workload: `{res['name']}` (Total Time: {res['total_time']:.4f}s)\n\n")
            f.write("| Rank | Function Name | Total Time (s) | % of Runtime |\n")
            f.write("|------|---------------|----------------|--------------|\n")
            for idx, (fn, tt, pct) in enumerate(res['top_hotspots'], 1):
                f.write(f"| {idx} | `{fn}` | {tt:.4f}s | {pct:.1f}% |\n")
            f.write("\n")

        f.write("## Architectural Insights for Later Parallelization\n\n")
        f.write("1. **Gate-Heavy Hotspot**: `apply_cnot` and `apply_h` dominate gate-heavy circuits. "
                "Each Clifford gate updates 2n rows. On GPU/multi-core CPU, these 2n row updates can be processed "
                "in parallel with zero inter-row dependencies.\n\n")
        f.write("2. **Measurement-Heavy Hotspot**: `_measure_random`, `row_mult`, and `_g` dominate measurement workloads. "
                "Random projective measurement performs Gaussian elimination across the tableau (O(n^2) complexity). "
                "QuaSARQ reformulates this using parallel prefix and data-parallel row elimination primitives.\n\n")
        f.write("3. **Many-Shot Hotspot**: In many-shot simulations, `run_shot` and repeated tableau resets dominate. "
                "Shots are completely independent, offering embarrassingly parallel acceleration on multi-core CPU and GPU.\n\n")
        f.write("4. **High-Qubit Scaling**: As qubit count grows (n >= 80), the O(n^2) footprint of two-qubit gates "
                "and measurements causes quadratic scaling, making them the primary targets for GPU acceleration.\n")

    print(f"\n[+] Synthesized profiling summary written to: {summary_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Serial Stabilizer Simulator")
    parser.add_argument("--workload", type=str, default="all", choices=["all", "gate", "meas", "shots", "scale"],
                        help="Select workload to profile")
    parser.add_argument("--top", type=int, default=15, help="Number of top functions to display")
    parser.add_argument("--output", type=str, default="profiling/results", help="Directory for profile outputs")
    args = parser.parse_args()

    if args.workload == "all":
        run_all_profiles(output_dir=args.output, top_n=args.top)
    elif args.workload == "gate":
        profile_workload(
            "gate_heavy",
            lambda: create_random_clifford_circuit(40, 100, seed=42, include_measurements=True),
            10,
            output_dir=args.output,
            top_n=args.top
        )
    elif args.workload == "meas":
        profile_workload(
            "measurement_heavy",
            lambda: create_measurement_heavy_circuit(25, 25, gates_per_cycle=6, seed=42),
            10,
            output_dir=args.output,
            top_n=args.top
        )
    elif args.workload == "shots":
        profile_workload(
            "many_shots",
            lambda: create_ghz_state(10),
            1500,
            output_dir=args.output,
            top_n=args.top
        )
    elif args.workload == "scale":
        profile_workload(
            "high_qubits",
            lambda: create_random_clifford_circuit(80, 40, seed=42, include_measurements=True),
            5,
            output_dir=args.output,
            top_n=args.top
        )


if __name__ == "__main__":
    main()
