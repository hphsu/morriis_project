#!/usr/bin/env python3
"""
Focused L=8 Test for SSH-Hubbard Model

Tests L=8 system with selected ansätze (excluding slow TN ansätze).
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


def benchmark_L8():
    """
    Benchmark L=8 system with selected ansätze (excluding TN ansätze).
    """
    L = 8
    N = 2 * L

    # Test parameters: standard regime
    t1 = 1.0
    t2 = 0.5
    U = 2.0
    delta = (t1 - t2) / (t1 + t2)
    reps = 2
    maxiter = 200

    print("=" * 80)
    print(f"L=8 FOCUSED TEST")
    print("=" * 80)
    print(f"System: L={L} sites ({N} qubits), δ={delta:.3f}, U={U:.2f}")
    print(f"Parameters: t1={t1}, t2={t2}, U={U}, reps={reps}, maxiter={maxiter}")
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

    # Define ansätze (excluding slow TN ansätze)
    ansatz_configs = [
        ('hea', lambda: build_ansatz_hea(N, reps), False),
        ('hva', lambda: build_ansatz_hva_sshh(L, reps, t1, t2, include_U=True), True),
        ('topoinsp', lambda: build_ansatz_topo_sshh(L, reps, use_edge_link=True), False),
        ('topo_rn', lambda: build_ansatz_topo_rn_sshh(L, reps, use_edge_link=True), False),
        ('dqap', lambda: build_ansatz_dqap_sshh(L, reps, include_U=True), True),
        ('np_hva', lambda: build_ansatz_np_hva_sshh(L, reps), True),
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
    print("L=8 RESULTS SUMMARY")
    print("=" * 80)

    valid_results = [(name, res) for name, res in results.items() if 'error' not in res]

    if valid_results:
        # Sort by accuracy
        sorted_by_accuracy = sorted(valid_results, key=lambda x: x[1]['abs_error'])

        print("\nRanked by Accuracy:")
        print(f"  {'Rank':<6} {'Ansatz':<12} {'Rel. Error':<12} {'Abs. Error':<12} {'Params':<8} {'Runtime':<10}")
        print("  " + "-" * 70)
        for i, (name, res) in enumerate(sorted_by_accuracy, 1):
            print(f"  {i:<6} {name:<12} {res['rel_error']:>10.2f}% "
                  f"{res['abs_error']:>11.3e} {res['num_params']:>7} {res['runtime']:>9.2f}s")

        # Best performers
        best_accuracy = sorted_by_accuracy[0]
        fastest = min(valid_results, key=lambda x: x[1]['runtime'])
        most_efficient = min(valid_results, key=lambda x: x[1]['abs_error'] / x[1]['num_params'])

        print("\nBest Performers:")
        print(f"  Most Accurate:       {best_accuracy[0]:<12} ({best_accuracy[1]['rel_error']:.2f}% error)")
        print(f"  Fastest:             {fastest[0]:<12} ({fastest[1]['runtime']:.2f}s)")
        print(f"  Most Efficient:      {most_efficient[0]:<12} "
              f"({most_efficient[1]['abs_error']/most_efficient[1]['num_params']:.3e} error/param)")

    print("\n" + "=" * 80)
    print("L=8 TEST COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    benchmark_L8()
