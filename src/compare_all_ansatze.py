#!/usr/bin/env python3
"""
Comprehensive Ansatz Comparison for SSH-Hubbard Model

Benchmarks all available VQE ansätze against exact diagonalization:
- 6 ansätze from ssh_hubbard_vqe.py: hea, hva, topoinsp, topo_rn, dqap, np_hva
- Exact diagonalization reference
- Multiple system sizes and parameter regimes
- Performance metrics: energy error, convergence speed, parameter efficiency

Usage:
    python compare_all_ansatze.py
"""

import numpy as np
import time
from typing import Dict, List, Tuple
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
    from qiskit_algorithms.optimizers import L_BFGS_B, COBYLA
except ImportError:
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import L_BFGS_B, COBYLA

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


# ============================================================================
# EXACT DIAGONALIZATION REFERENCE
# ============================================================================

def exact_diagonalization(H: SparsePauliOp) -> Tuple[float, np.ndarray]:
    """
    Compute exact ground state energy and state via full diagonalization.

    Parameters
    ----------
    H : SparsePauliOp
        The Hamiltonian

    Returns
    -------
    E0 : float
        Ground state energy
    psi0 : np.ndarray
        Ground state vector
    """
    H_matrix = H.to_matrix()
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    return E0, psi0


# ============================================================================
# VQE RUNNER
# ============================================================================

class VQERunner:
    """Run VQE with consistent settings across all ansätze."""

    def __init__(self, maxiter: int = 100, optimizer_name: str = 'L_BFGS_B'):
        """
        Initialize VQE runner.

        Parameters
        ----------
        maxiter : int
            Maximum optimizer iterations
        optimizer_name : str
            Optimizer to use ('L_BFGS_B' or 'COBYLA')
        """
        self.maxiter = maxiter
        self.optimizer_name = optimizer_name
        self.energy_history = []
        self.eval_count = 0

    def callback(self, eval_count, params, mean, std):
        """Track optimization progress."""
        self.eval_count = eval_count
        self.energy_history.append(float(mean))

    def run(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp,
            initial_point: np.ndarray = None) -> Dict:
        """
        Run VQE optimization.

        Parameters
        ----------
        ansatz : QuantumCircuit
            The variational ansatz
        hamiltonian : SparsePauliOp
            The Hamiltonian to minimize
        initial_point : np.ndarray, optional
            Initial parameter values

        Returns
        -------
        result : dict
            VQE results including energy, parameters, convergence info
        """
        # Reset history
        self.energy_history = []
        self.eval_count = 0

        # Initialize optimizer
        if self.optimizer_name == 'L_BFGS_B':
            optimizer = L_BFGS_B(maxiter=self.maxiter)
        elif self.optimizer_name == 'COBYLA':
            optimizer = COBYLA(maxiter=self.maxiter)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # Initial parameters
        if initial_point is None:
            np.random.seed(42)
            initial_point = 0.01 * np.random.randn(ansatz.num_parameters)

        # Setup VQE
        estimator = Estimator()
        vqe = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=estimator,
            initial_point=initial_point,
            callback=self.callback
        )

        # Run optimization
        start_time = time.time()
        vqe_result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)
        runtime = time.time() - start_time

        # Extract results
        energy = float(vqe_result.eigenvalue.real)

        if hasattr(vqe_result, 'optimal_point'):
            optimal_params = vqe_result.optimal_point
        elif hasattr(vqe_result, 'optimal_parameters'):
            optimal_params = np.array(list(vqe_result.optimal_parameters.values()))
        else:
            optimal_params = initial_point

        return {
            'energy': energy,
            'optimal_params': optimal_params,
            'evaluations': self.eval_count,
            'runtime': runtime,
            'energy_history': self.energy_history.copy()
        }


# ============================================================================
# ANSATZ COMPARISON
# ============================================================================

def compare_ansatze(L: int, t1: float, t2: float, U: float,
                    reps: int = 2, maxiter: int = 200,
                    verbose: bool = True) -> Dict:
    """
    Compare all available ansätze on a single parameter point.

    Parameters
    ----------
    L : int
        Number of lattice sites
    t1, t2 : float
        SSH hopping amplitudes
    U : float
        Hubbard interaction strength
    reps : int
        Ansatz repetitions/depth
    maxiter : int
        Maximum VQE iterations
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Comparison results for all ansätze
    """
    N = 2 * L  # Total qubits
    delta = (t1 - t2) / (t1 + t2)

    if verbose:
        print("=" * 70)
        print(f"ANSATZ COMPARISON: L={L}, δ={delta:.3f}, U={U:.2f}")
        print("=" * 70)

    # Build Hamiltonian
    H = ssh_hubbard_hamiltonian(L, t1, t2, U, periodic=False)

    # Exact diagonalization
    if verbose:
        print("\n[Reference] Computing exact diagonalization...")
    E_exact, psi_exact = exact_diagonalization(H)
    E_exact_per_site = E_exact / L

    if verbose:
        print(f"  E_exact:          {E_exact:.10f}")
        print(f"  E_exact/site:     {E_exact_per_site:.10f}")

    # Ansatz definitions
    ansatz_configs = [
        ('hea', lambda: build_ansatz_hea(N, reps), False),
        ('hva', lambda: build_ansatz_hva_sshh(L, reps, t1, t2, include_U=True), True),
        ('topoinsp', lambda: build_ansatz_topo_sshh(L, reps, use_edge_link=True), False),
        ('topo_rn', lambda: build_ansatz_topo_rn_sshh(L, reps, use_edge_link=True), False),
        ('dqap', lambda: build_ansatz_dqap_sshh(L, reps, include_U=True), True),
        ('np_hva', lambda: build_ansatz_np_hva_sshh(L, reps), True),
    ]

    results = {
        'system': {'L': L, 't1': t1, 't2': t2, 'U': U, 'delta': delta},
        'exact': {'energy': E_exact, 'energy_per_site': E_exact_per_site},
        'ansatze': {}
    }

    # Run VQE for each ansatz
    runner = VQERunner(maxiter=maxiter, optimizer_name='L_BFGS_B')

    for ansatz_name, ansatz_builder, needs_initial_state in ansatz_configs:
        if verbose:
            print(f"\n[{ansatz_name.upper()}] Running VQE...")

        try:
            # Build ansatz
            ansatz = ansatz_builder()

            # Add initial state preparation for number-conserving ansätze
            if needs_initial_state:
                initial_state = prepare_half_filling_state(L)
                full_circuit = QuantumCircuit(N)
                full_circuit.compose(initial_state, inplace=True)
                full_circuit.compose(ansatz, inplace=True)
                ansatz = full_circuit

            # Run VQE
            vqe_result = runner.run(ansatz, H)

            # Compute errors
            energy = vqe_result['energy']
            abs_error = abs(energy - E_exact)
            rel_error = 100 * abs_error / abs(E_exact) if E_exact != 0 else 0

            # Store results
            results['ansatze'][ansatz_name] = {
                'energy': energy,
                'energy_per_site': energy / L,
                'abs_error': abs_error,
                'rel_error': rel_error,
                'num_params': ansatz.num_parameters,
                'depth': ansatz.depth(),
                'evaluations': vqe_result['evaluations'],
                'runtime': vqe_result['runtime'],
                'convergence': len(vqe_result['energy_history'])
            }

            if verbose:
                print(f"  Energy:           {energy:.10f}")
                print(f"  Error:            {abs_error:.3e} ({rel_error:.2f}%)")
                print(f"  Parameters:       {ansatz.num_parameters}")
                print(f"  Evaluations:      {vqe_result['evaluations']}")
                print(f"  Runtime:          {vqe_result['runtime']:.2f}s")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {str(e)}")
            results['ansatze'][ansatz_name] = {'error': str(e)}

    return results


# ============================================================================
# COMPREHENSIVE BENCHMARK
# ============================================================================

def comprehensive_benchmark():
    """
    Run comprehensive benchmark across multiple parameter regimes.
    """
    print("#" * 70)
    print("# COMPREHENSIVE ANSATZ BENCHMARK")
    print("#" * 70)

    all_results = []

    # Test configurations
    test_configs = [
        # (L, t1, t2, U, reps, maxiter, description)
        (4, 1.0, 0.5, 2.0, 2, 200, "Standard (L=4, δ=0.33, U=2.0)"),
        (4, 1.0, 0.8, 2.0, 2, 200, "Weak SSH (L=4, δ=0.11, U=2.0)"),
        (4, 1.0, 0.2, 2.0, 2, 200, "Strong SSH (L=4, δ=0.67, U=2.0)"),
        (4, 1.0, 0.5, 0.0, 2, 200, "Non-interacting (L=4, δ=0.33, U=0.0)"),
        (4, 1.0, 0.5, 4.0, 2, 200, "Strong U (L=4, δ=0.33, U=4.0)"),
        (6, 1.0, 0.5, 2.0, 2, 200, "Larger system (L=6, δ=0.33, U=2.0)"),
    ]

    for L, t1, t2, U, reps, maxiter, description in test_configs:
        print(f"\n{'=' * 70}")
        print(f"TEST: {description}")
        print(f"{'=' * 70}")

        result = compare_ansatze(L, t1, t2, U, reps=reps, maxiter=maxiter, verbose=True)
        all_results.append((description, result))

    # Print summary table
    print("\n\n" + "#" * 70)
    print("# SUMMARY TABLE")
    print("#" * 70)

    print("\n" + "=" * 120)
    print(f"{'Test':<35} | {'Ansatz':<10} | {'Params':<7} | {'Energy Error':<12} | {'Rel%':<8} | {'Runtime':<8}")
    print("=" * 120)

    for description, result in all_results:
        test_name = description.split('(')[0].strip()
        for ansatz_name, ansatz_result in result['ansatze'].items():
            if 'error' not in ansatz_result:
                print(f"{test_name:<35} | {ansatz_name:<10} | "
                      f"{ansatz_result['num_params']:<7} | "
                      f"{ansatz_result['abs_error']:<12.3e} | "
                      f"{ansatz_result['rel_error']:<8.2f} | "
                      f"{ansatz_result['runtime']:<8.2f}")

    print("=" * 120)

    # Best performers analysis
    print("\n\n" + "#" * 70)
    print("# BEST PERFORMERS BY METRIC")
    print("#" * 70)

    for description, result in all_results:
        print(f"\n{description}:")

        ansatze = [(name, res) for name, res in result['ansatze'].items()
                   if 'error' not in res]

        # Best energy accuracy
        best_energy = min(ansatze, key=lambda x: x[1]['abs_error'])
        print(f"  Best accuracy:    {best_energy[0]:<10} "
              f"({best_energy[1]['rel_error']:.2f}% error)")

        # Most parameter-efficient (lowest error per parameter)
        param_eff = [(name, res['abs_error'] / res['num_params'])
                     for name, res in ansatze]
        best_eff = min(param_eff, key=lambda x: x[1])
        print(f"  Best efficiency:  {best_eff[0]:<10} "
              f"({best_eff[1]:.3e} error/param)")

        # Fastest
        fastest = min(ansatze, key=lambda x: x[1]['runtime'])
        print(f"  Fastest:          {fastest[0]:<10} "
              f"({fastest[1]['runtime']:.2f}s)")

    print("\n" + "#" * 70)
    print("# BENCHMARK COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    comprehensive_benchmark()
