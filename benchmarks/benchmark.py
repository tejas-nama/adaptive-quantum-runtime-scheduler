"""
Benchmark Runner for Serial Stabilizer Simulation Baseline.

Measures execution time across varying qubit counts, depths, and shot counts.
Records results with high-resolution timers (time.perf_counter) and outputs
both a formatted console table and a CSV log for subsequent analysis.

Usage:
    python -m benchmarks.benchmark [--category CATEGORY] [--output PATH] [--smoke]
"""

import sys
import os
import time
import csv
import argparse
from typing import List, Dict, Any

from simulator.simulator import Simulator
from benchmarks.configs import BENCHMARK_CONFIGS, BenchmarkCase


def run_benchmarks(
    configs: List[BenchmarkCase],
    output_csv_path: str = "results/benchmark_results.csv"
) -> List[Dict[str, Any]]:
    """
    Executes benchmark cases and writes results to CSV.
    
    Args:
        configs: List of BenchmarkCase definitions to run.
        output_csv_path: Relative or absolute path to write CSV results.
        
    Returns:
        List of recorded benchmark result records.
    """
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    sim = Simulator(seed=42)

    results: List[Dict[str, Any]] = []

    header = (
        f"{'Benchmark':<36} | {'Qubits':<6} | {'Depth':<6} | {'Gates':<6} | "
        f"{'Meas':<5} | {'Shots':<6} | {'Total Time (s)':<14} | {'Time/Shot (s)':<13}"
    )
    separator = "-" * len(header)
    print("\n" + "=" * len(header))
    print(" SERIAL STABILIZER SIMULATOR BENCHMARK SUITE")
    print("=" * len(header))
    print(header)
    print(separator)

    for case in configs:
        circuit = case.builder()
        num_qubits = circuit.num_qubits
        depth = circuit.depth
        num_gates = circuit.num_gates
        num_meas = circuit.num_measurements

        # Run repetitions and compute average time
        rep_times: List[float] = []
        for _ in range(case.repetitions):
            t0 = time.perf_counter()
            sim.simulate(circuit, shots=case.shots, seed=42)
            t1 = time.perf_counter()
            rep_times.append(t1 - t0)

        avg_total_time = sum(rep_times) / len(rep_times)
        avg_time_per_shot = avg_total_time / case.shots

        record = {
            "Benchmark": case.name,
            "Category": case.category,
            "Qubits": num_qubits,
            "Depth": depth,
            "Gates": num_gates,
            "Measurements": num_meas,
            "Shots": case.shots,
            "Repetitions": case.repetitions,
            "Total Time (s)": round(avg_total_time, 6),
            "Time/Shot (s)": round(avg_time_per_shot, 7),
        }
        results.append(record)

        print(
            f"{case.name:<36} | {num_qubits:<6d} | {depth:<6d} | {num_gates:<6d} | "
            f"{num_meas:<5d} | {case.shots:<6d} | {avg_total_time:<14.5f} | {avg_time_per_shot:<13.6f}"
        )

    print(separator)

    # Save to CSV
    fieldnames = [
        "Benchmark", "Category", "Qubits", "Depth", "Gates",
        "Measurements", "Shots", "Repetitions", "Total Time (s)", "Time/Shot (s)"
    ]
    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[+] Benchmark results successfully saved to: {output_csv_path}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Serial Quantum Simulator Benchmarks")
    parser.add_argument("--category", type=str, default=None, help="Filter benchmarks by category")
    parser.add_argument("--output", type=str, default="results/benchmark_results.csv", help="CSV output file path")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test subset")
    args = parser.parse_args()

    configs = BENCHMARK_CONFIGS
    if args.category:
        configs = [c for c in configs if c.category == args.category]

    if args.smoke:
        # Select one from each category with small counts
        configs = [
            c for c in configs
            if c.name in ("Bell_State", "GHZ_10Q", "QubitScaling_10Q_D50", "DepthScaling_20Q_D20", "ShotScaling_GHZ10Q_S10")
        ]

    run_benchmarks(configs, output_csv_path=args.output)


if __name__ == "__main__":
    main()
