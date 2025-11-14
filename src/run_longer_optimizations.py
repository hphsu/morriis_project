#!/usr/bin/env python3
"""
Longer VQE Optimizations for Best Performing Ansätze

Runs extended VQE optimizations (maxiter=500-1000) on the best ansätze
to see how close we can get to exact ground state energies.
"""

import numpy as np
import time
import warnings

from qiskit.quantum_info import SparsePauliOp
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

from ssh_hubbard_vqe import (
    ssh_hubbard_hamiltonian,
    build_ansatz_np_hva_sshh,
    prepare_half_filling_state,
)

from ssh_hubbard_tn_vqe import (
    build_ansatz_tn_mps_sshh,
)

warnings.filterwarnings('ignore', category=DeprecationWarning)


def exact_diagonalization(H: SparsePauliOp):
    """Compute exact ground state energy."""
    H_matrix = H.to_matrix()
    eigenvalues = np.linalg.eigh(H_matrix)[0]
    return eigenvalues[0]


def run_vqe_extended(ansatz, H, maxiter=500, seed=42):
    """Run VQE with extended optimization."""
    estimator = Estimator()
    optimizer = L_BFGS_B(maxiter=maxiter)

    np.random.seed(seed)
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


def test_ansatz(ansatz_name, ansatz_builder, needs_initial_state, L, H, E_exact, maxiter_list):
    """Test an ansatz with multiple maxiter values."""
    N = 2 * L

    print(f"\n{'=' * 80}")
    print(f"ANSATZ: {ansatz_name.upper()}")
    print(f"{'=' * 80}")

    results = []

    for maxiter in maxiter_list:
        print(f"\n  Running with maxiter={maxiter}...")

        # Build ansatz
        ansatz = ansatz_builder()

        # Add initial state if needed
        if needs_initial_state:
            initial_state = prepare_half_filling_state(L)
            full_circuit = QuantumCircuit(N)
            full_circuit.compose(initial_state, inplace=True)
            full_circuit.compose(ansatz, inplace=True)
            ansatz = full_circuit

        # Run VQE
        result = run_vqe_extended(ansatz, H, maxiter=maxiter)

        # Compute errors
        energy = result['energy']
        abs_error = abs(energy - E_exact)
        rel_error = 100 * abs_error / abs(E_exact)

        results.append({
            'maxiter': maxiter,
            'energy': energy,
            'abs_error': abs_error,
            'rel_error': rel_error,
            'evaluations': result['evaluations'],
            'runtime': result['runtime'],
        })

        print(f"    Energy:       {energy:.10f}")
        print(f"    Error:        {abs_error:.6f} ({rel_error:.3f}%)")
        print(f"    Evaluations:  {result['evaluations']}")
        print(f"    Runtime:      {result['runtime']:.2f}s")

    # Summary for this ansatz
    print(f"\n  Summary for {ansatz_name}:")
    print(f"    {'maxiter':<10} {'Error %':<12} {'Runtime (s)':<12}")
    print(f"    {'-' * 35}")
    for r in results:
        print(f"    {r['maxiter']:<10} {r['rel_error']:<12.3f} {r['runtime']:<12.2f}")

    return results


def main():
    """Run longer optimizations."""
    print("#" * 80)
    print("# LONGER VQE OPTIMIZATIONS")
    print("#" * 80)

    # Test system - L=6 standard parameters where we know NP_HVA does well
    L = 6
    t1 = 1.0
    t2 = 0.5
    U = 2.0
    N = 2 * L
    delta = (t1 - t2) / (t1 + t2)

    print(f"\nSystem Parameters:")
    print(f"  L = {L} sites ({N} qubits)")
    print(f"  δ = {delta:.3f}")
    print(f"  U = {U:.2f}")

    # Build Hamiltonian and get exact result
    print(f"\nBuilding Hamiltonian...")
    H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)
    print(f"  Pauli terms: {len(H)}")

    print(f"\nComputing exact ground state...")
    E_exact = exact_diagonalization(H)
    print(f"  E_exact = {E_exact:.10f}")

    # Test configurations: (name, builder, needs_initial_state)
    ansatz_configs = [
        ('np_hva', lambda: build_ansatz_np_hva_sshh(L, reps=2), True),
        ('tn_mps_fixed', lambda: build_ansatz_tn_mps_sshh(L, reps=2), True),  # With initial state
    ]

    # Test with increasing maxiter
    maxiter_list = [200, 500, 1000]

    all_results = {}

    for ansatz_name, ansatz_builder, needs_initial_state in ansatz_configs:
        results = test_ansatz(
            ansatz_name,
            ansatz_builder,
            needs_initial_state,
            L, H, E_exact,
            maxiter_list
        )
        all_results[ansatz_name] = results

    # Final comparison
    print(f"\n\n{'#' * 80}")
    print("# FINAL COMPARISON")
    print(f"{'#' * 80}")

    print(f"\n{'Ansatz':<15} {'maxiter':<10} {'Error %':<12} {'Abs Error':<14} {'Runtime (s)':<12}")
    print("-" * 75)

    for ansatz_name, results in all_results.items():
        for r in results:
            print(f"{ansatz_name:<15} {r['maxiter']:<10} {r['rel_error']:<12.3f} "
                  f"{r['abs_error']:<14.6e} {r['runtime']:<12.2f}")

    # Best results
    print(f"\n\nBest Results:")
    for ansatz_name, results in all_results.items():
        best = min(results, key=lambda x: x['abs_error'])
        print(f"  {ansatz_name}:")
        print(f"    Best error:   {best['rel_error']:.3f}% (maxiter={best['maxiter']})")
        print(f"    Energy:       {best['energy']:.10f}")
        print(f"    Runtime:      {best['runtime']:.2f}s")

    print(f"\n{'#' * 80}")


if __name__ == "__main__":
    main()
