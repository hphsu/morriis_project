#!/usr/bin/env python3
"""
Large System Benchmarks for SSH-Hubbard Model

Tests L=6 and L=8 systems with multiple parameter regimes across all ansätze.
"""

import numpy as np
import time
from typing import Dict, Tuple
import warnings

# Qiskit imports
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit import QuantumCircuit

try:
    from qiskit.primitives import StatevectorEstimator as Estimator
except ImportError:
    from qiskit.primitives import Estimator

try:
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import L_BFGS_B
except ImportError:
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import L_BFGS_B

# Import from our implementations
import sys
sys.path.insert(0, '/home/user/morriis_project')

from ssh_hubbard_vqe import (
    ssh_hubbard_hamiltonian,
    build_ansatz_hea,
    build_ansatz_hva_sshh,
    build_ansatz_topo_sshh,
    build_ansatz_topo_rn_sshh,
    build_ansatz_dqap_sshh,
    build_ansatz_np_hva_sshh,
    prepare_half_filling_state,
)

from ssh_hubbard_tn_vqe import (
    build_ansatz_tn_mps_sshh,
    build_ansatz_tn_mps_np_sshh,
)

warnings.filterwarnings('ignore', category=DeprecationWarning)


def exact_diagonalization(H: SparsePauliOp) -> Tuple[float, np.ndarray]:
    """Compute exact ground state energy."""
    H_matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    return eigenvalues[0], eigenvectors[:, 0]


def run_vqe(ansatz: QuantumCircuit, H: SparsePauliOp, maxiter: int = 200) -> Dict:
    """Run VQE optimization."""
    estimator = Estimator()
    optimizer = L_BFGS_B(maxiter=maxiter)

    np.random.seed(42)
    initial_point = 0.01 * np.random.randn(ansatz.num_parameters)

    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)

    start_time = time.time()
    result = vqe.compute_minimum_eigenvalue(H)
    runtime = time.time() - start_time

    return {
        'energy': result.eigenvalue.real,
        'evaluations': result.cost_function_evals,
        'runtime': runtime,
        'optimal_params': result.optimal_point,
    }


def benchmark_system(L: int, t1: float, t2: float, U: float, reps: int = 2, maxiter: int = 200):
    """
    Benchmark all ansätze for a given system size and parameters.
    """
    N = 2 * L
    delta = (t1 - t2) / (t1 + t2)

    print("=" * 80)
    print(f"BENCHMARK: L={L} sites ({N} qubits), δ={delta:.3f}, U={U:.2f}")
    print("=" * 80)

    # Build Hamiltonian
    print("\nBuilding Hamiltonian...")
    H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    print(f"  Pauli terms: {len(H)}")

    # Exact diagonalization
    print("\nComputing exact ground state...")
    E_exact, _ = exact_diagonalization(H)
    print(f"  E_exact = {E_exact:.10f}")
    print(f"  E/site  = {E_exact/L:.10f}")

    # Define all ansätze (including TN)
    ansatz_configs = [
        ('hea', lambda: build_ansatz_hea(N, reps), False),
        ('hva', lambda: build_ansatz_hva_sshh(L, reps, t1, t2, include_U=True), True),
        ('topoinsp', lambda: build_ansatz_topo_sshh(L, reps, use_edge_link=True), False),
        ('topo_rn', lambda: build_ansatz_topo_rn_sshh(L, reps, use_edge_link=True), False),
        ('dqap', lambda: build_ansatz_dqap_sshh(L, reps, include_U=True), True),
        ('np_hva', lambda: build_ansatz_np_hva_sshh(L, reps), True),
        ('tn_mps', lambda: build_ansatz_tn_mps_sshh(L, reps), True),  # Fixed: needs initial state to avoid vacuum trap
        ('tn_mps_np', lambda: build_ansatz_tn_mps_np_sshh(L, reps), True),
    ]

    results = {}

    print("\n" + "-" * 80)
    print("Running VQE for all ansätze...")
    print("-" * 80)

    for ansatz_name, ansatz_builder, needs_initial_state in ansatz_configs:
        print(f"\n[{ansatz_name.upper()}]")

        try:
            # Build ansatz
            ansatz = ansatz_builder()

            # Add initial state for number-conserving ansätze
            if needs_initial_state:
                initial_state = prepare_half_filling_state(L)
                full_circuit = QuantumCircuit(N)
                full_circuit.compose(initial_state, inplace=True)
                full_circuit.compose(ansatz, inplace=True)
                ansatz = full_circuit

            print(f"  Circuit: {ansatz.num_parameters} params, depth {ansatz.depth()}")

            # Run VQE
            print(f"  Running VQE (maxiter={maxiter})...")
            vqe_result = run_vqe(ansatz, H, maxiter=maxiter)

            # Compute errors
            energy = vqe_result['energy']
            abs_error = abs(energy - E_exact)
            rel_error = 100 * abs_error / abs(E_exact) if E_exact != 0 else 0

            results[ansatz_name] = {
                'energy': energy,
                'abs_error': abs_error,
                'rel_error': rel_error,
                'num_params': ansatz.num_parameters,
                'depth': ansatz.depth(),
                'evaluations': vqe_result['evaluations'],
                'runtime': vqe_result['runtime'],
            }

            print(f"  ✓ Energy:      {energy:.10f}")
            print(f"  ✓ Error:       {abs_error:.3e} ({rel_error:.2f}%)")
            print(f"  ✓ Evaluations: {vqe_result['evaluations']}")
            print(f"  ✓ Runtime:     {vqe_result['runtime']:.2f}s")

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            results[ansatz_name] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid_results = [(name, res) for name, res in results.items() if 'error' not in res]

    if valid_results:
        # Sort by accuracy
        sorted_by_accuracy = sorted(valid_results, key=lambda x: x[1]['abs_error'])

        print("\nRanked by Accuracy:")
        print(f"  {'Rank':<6} {'Ansatz':<12} {'Rel. Error':<12} {'Abs. Error':<12} {'Params':<8}")
        print("  " + "-" * 60)
        for i, (name, res) in enumerate(sorted_by_accuracy, 1):
            print(f"  {i:<6} {name:<12} {res['rel_error']:>10.2f}% "
                  f"{res['abs_error']:>11.3e} {res['num_params']:>7}")

        # Best performers
        best_accuracy = sorted_by_accuracy[0]
        fastest = min(valid_results, key=lambda x: x[1]['runtime'])
        most_efficient = min(valid_results, key=lambda x: x[1]['abs_error'] / x[1]['num_params'])

        print("\nBest Performers:")
        print(f"  Most Accurate:       {best_accuracy[0]:<12} ({best_accuracy[1]['rel_error']:.2f}% error)")
        print(f"  Fastest:             {fastest[0]:<12} ({fastest[1]['runtime']:.2f}s)")
        print(f"  Most Efficient:      {most_efficient[0]:<12} "
              f"({most_efficient[1]['abs_error']/most_efficient[1]['num_params']:.3e} error/param)")

    return {
        'L': L,
        't1': t1,
        't2': t2,
        'U': U,
        'delta': delta,
        'E_exact': E_exact,
        'results': results
    }


def main():
    """Run benchmarks for L=6 and L=8 systems."""
    print("#" * 80)
    print("# LARGE SYSTEM BENCHMARKS: L=6 and L=8")
    print("#" * 80)

    all_benchmarks = []

    # Test configurations
    configs = [
        # L=6 tests
        (6, 1.0, 0.5, 2.0, 2, 200, "L=6, Standard (δ=0.33, U=2.0)"),
        (6, 1.0, 0.8, 2.0, 2, 200, "L=6, Weak SSH (δ=0.11, U=2.0)"),
        (6, 1.0, 0.2, 2.0, 2, 200, "L=6, Strong SSH (δ=0.67, U=2.0)"),

        # L=8 tests
        (8, 1.0, 0.5, 2.0, 2, 200, "L=8, Standard (δ=0.33, U=2.0)"),
        (8, 1.0, 0.8, 2.0, 2, 200, "L=8, Weak SSH (δ=0.11, U=2.0)"),
        (8, 1.0, 0.2, 2.0, 2, 200, "L=8, Strong SSH (δ=0.67, U=2.0)"),
    ]

    for L, t1, t2, U, reps, maxiter, description in configs:
        print(f"\n\n{'#' * 80}")
        print(f"# TEST: {description}")
        print(f"{'#' * 80}\n")

        result = benchmark_system(L, t1, t2, U, reps=reps, maxiter=maxiter)
        all_benchmarks.append((description, result))

    # Final comparison table
    print("\n\n" + "#" * 80)
    print("# FINAL COMPARISON TABLE")
    print("#" * 80)

    print("\n" + "=" * 110)
    print(f"{'Test':<30} | {'Ansatz':<12} | {'Params':<8} | {'Error':<12} | {'Rel%':<8} | {'Runtime':<10}")
    print("=" * 110)

    for description, benchmark in all_benchmarks:
        test_name = description.split(',')[0]  # Get L=6 or L=8 part
        for ansatz_name, res in benchmark['results'].items():
            if 'error' not in res:
                print(f"{test_name:<30} | {ansatz_name:<12} | {res['num_params']:<8} | "
                      f"{res['abs_error']:<12.3e} | {res['rel_error']:<8.2f} | "
                      f"{res['runtime']:<10.2f}")

    print("=" * 110)

    print("\n" + "#" * 80)
    print("# BENCHMARKS COMPLETE")
    print("#" * 80)


if __name__ == "__main__":
    main()
