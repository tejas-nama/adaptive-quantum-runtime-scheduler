"""
Benchmarking Suite for Serial Stabilizer Simulation.

Contains configuration suites and benchmark runner recording timing,
qubit scaling, depth scaling, and shot scaling to CSV.
"""

from benchmarks.configs import BENCHMARK_CONFIGS, BenchmarkCase

__all__ = ["BENCHMARK_CONFIGS", "BenchmarkCase"]
