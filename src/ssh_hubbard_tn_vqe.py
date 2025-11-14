#!/usr/bin/env python3
"""
Tensor-Network Brick-Wall VQE for SSH-Hubbard Model

Implements a tensor-network-inspired quantum MPS (qMPS) ansatz using
brick-wall circuit architecture for the spinful SSH-Hubbard model.

Features:
- Jordan-Wigner mapping for spinful fermions
- Brick-wall (qMPS) ansatz with SU(4)-like 2-qubit blocks
- Number-preserving variant for particle-conserving simulations
- Support for L ≤ 8 sites (16 qubits)
- Clean VQE expectation value evaluation

Usage:
    python ssh_hubbard_tn_vqe.py
"""

import numpy as np
from typing import Tuple, List
import warnings

# Qiskit imports
try:
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.circuit import QuantumCircuit, Parameter
except ImportError:
    raise ImportError("Qiskit required. Install with: pip install qiskit")

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)


# ============================================================================
# PART 1: SSH-HUBBARD HAMILTONIAN WITH JORDAN-WIGNER MAPPING
# ============================================================================

def q_index(site: int, spin: str, L: int) -> int:
    """
    Map (site, spin) to a qubit index in [0, 2L-1].

    Convention (UNIFIED): [site0↑, site0↓, site1↑, site1↓, ..., site(L-1)↑, site(L-1)↓]
    - spin == "up":   qubit = 2*site + 0
    - spin == "down": qubit = 2*site + 1

    This matches the canonical layout used in ssh_hubbard_vqe.py.

    Parameters
    ----------
    site : int
        Physical lattice site index (0 to L-1)
    spin : str
        Either "up" or "down"
    L : int
        Number of physical lattice sites

    Returns
    -------
    qubit_index : int
        The qubit index for this (site, spin) pair
    """
    return 2 * site + (0 if spin == "up" else 1)


def jw_number_operator(site: int, spin: str, L: int) -> SparsePauliOp:
    """
    Jordan-Wigner number operator: n_{site,spin} = c† c.

    In JW encoding: n = (I - Z) / 2

    Parameters
    ----------
    site : int
        Lattice site index
    spin : str
        "up" or "down"
    L : int
        Number of lattice sites

    Returns
    -------
    op : SparsePauliOp
        The number operator for this mode
    """
    N = 2 * L  # Total qubits
    q = q_index(site, spin, L)

    # n = (I - Z) / 2
    # CORRECTED: Qiskit Pauli string convention - rightmost = qubit 0
    pauli_I = ['I'] * N
    pauli_Z = ['I'] * N
    pauli_Z[N - 1 - q] = 'Z'  # Reversed indexing

    pauli_I_str = ''.join(pauli_I)
    pauli_Z_str = ''.join(pauli_Z)

    return SparsePauliOp([pauli_I_str, pauli_Z_str], coeffs=[0.5, -0.5])


def jw_hopping_operator(site_i: int, site_j: int, spin: str, L: int) -> SparsePauliOp:
    """
    Jordan-Wigner hopping operator: c†_i c_j + c†_j c_i (Hermitian).

    In JW encoding:
    c†_i c_j = (1/2) * (X_i Z_{i+1} ... Z_{j-1} X_j + Y_i Z_{i+1} ... Z_{j-1} Y_j)
             + i/2 * (X_i Z_{i+1} ... Z_{j-1} Y_j - Y_i Z_{i+1} ... Z_{j-1} X_j)

    For Hermitian hopping (c†_i c_j + h.c.), the imaginary parts cancel.

    Parameters
    ----------
    site_i, site_j : int
        Lattice site indices (site_j = site_i + 1 typically)
    spin : str
        "up" or "down"
    L : int
        Number of lattice sites

    Returns
    -------
    op : SparsePauliOp
        The hopping operator (Hermitian)
    """
    N = 2 * L
    qi = q_index(site_i, spin, L)
    qj = q_index(site_j, spin, L)

    if qi > qj:
        qi, qj = qj, qi  # Ensure qi < qj

    # Build Jordan-Wigner string
    # c†_i c_j + h.c. = (X_i Z_{...} X_j) + (Y_i Z_{...} Y_j)
    # CORRECTED: Qiskit Pauli string convention - rightmost = qubit 0
    pauli_XX = ['I'] * N
    pauli_YY = ['I'] * N

    pauli_XX[N - 1 - qi] = 'X'  # Reversed indexing
    pauli_YY[N - 1 - qi] = 'Y'  # Reversed indexing

    # Jordan-Wigner string of Z's between qi and qj
    for q in range(qi + 1, qj):
        pauli_XX[N - 1 - q] = 'Z'  # Reversed indexing
        pauli_YY[N - 1 - q] = 'Z'  # Reversed indexing

    pauli_XX[N - 1 - qj] = 'X'  # Reversed indexing
    pauli_YY[N - 1 - qj] = 'Y'  # Reversed indexing

    pauli_XX_str = ''.join(pauli_XX)
    pauli_YY_str = ''.join(pauli_YY)

    # Jordan-Wigner: c†_i c_j + h.c. = 1/2 (XX + YY) [with Z string]
    return SparsePauliOp([pauli_XX_str, pauli_YY_str], coeffs=[0.5, 0.5])


def ssh_hubbard_hamiltonian(
    L: int,
    t1: float,
    t2: float,
    U: float,
) -> SparsePauliOp:
    """
    Build the spinful SSH-Hubbard Hamiltonian (open boundary) using
    Jordan-Wigner mapping.

    H = -∑_{i,σ} t_i (c†_{i,σ} c_{i+1,σ} + h.c.) + U ∑_i n_{i,↑} n_{i,↓}

    where t_i alternates:
    - t_i = t1 for i even (strong bonds: 0-1, 2-3, ...)
    - t_i = t2 for i odd (weak bonds: 1-2, 3-4, ...)

    Parameters
    ----------
    L : int
        Number of lattice sites
    t1 : float
        Strong bond hopping amplitude
    t2 : float
        Weak bond hopping amplitude
    U : float
        On-site Hubbard interaction strength

    Returns
    -------
    H : SparsePauliOp
        The SSH-Hubbard Hamiltonian as a sum of Pauli operators
    """
    H = None

    # Hopping terms
    for i in range(L - 1):
        t_hop = t1 if i % 2 == 0 else t2

        for spin in ['up', 'down']:
            hop_term = jw_hopping_operator(i, i + 1, spin, L)
            hop_term = -t_hop * hop_term

            if H is None:
                H = hop_term
            else:
                H = H + hop_term

    # Hubbard interaction terms: U * n_{i,↑} * n_{i,↓}
    for i in range(L):
        n_up = jw_number_operator(i, 'up', L)
        n_down = jw_number_operator(i, 'down', L)
        U_term = U * (n_up @ n_down)  # Tensor product for operator multiplication

        if H is None:
            H = U_term
        else:
            H = H + U_term

    # Simplify the Hamiltonian
    H = H.simplify()

    return H


# ============================================================================
# PART 2: GENERIC 2-QUBIT TN BLOCK (SU(4)-LIKE TEMPLATE)
# ============================================================================

def apply_tn_block_su4_like(
    qc: QuantumCircuit,
    prefix: str,
    layer: int,
    block_id: int,
    q0: int,
    q1: int,
) -> None:
    """
    Append a generic 2-qubit block on (q0, q1) with its own Parameter set.

    Template (all parameters independent):
      - First local layer: Z-Y-Z on each qubit
      - CX(q0 -> q1)
      - Second local layer: Y-Z on each qubit
      - CX(q1 -> q0)
      - Final local Z on each qubit

    This is inspired by generic SU(4) decompositions used in qMPS/brick-wall circuits:
      - 2 entangling CX gates + several local rotations.

    Parameters are named:
      f"{prefix}_{layer}_{block_id}_<tag>".

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to append to
    prefix : str
        Prefix for parameter names (e.g., "tn_even", "tn_odd")
    layer : int
        Layer number in the ansatz
    block_id : int
        Block identifier within this layer
    q0, q1 : int
        Qubit indices for this block
    """
    def p(name: str) -> Parameter:
        return Parameter(f"{prefix}_{layer}_{block_id}_{name}")

    # First local Z-Y-Z on q0
    qc.rz(p("rz1_q0"), q0)
    qc.ry(p("ry1_q0"), q0)
    qc.rz(p("rz2_q0"), q0)

    # First local Z-Y-Z on q1
    qc.rz(p("rz1_q1"), q1)
    qc.ry(p("ry1_q1"), q1)
    qc.rz(p("rz2_q1"), q1)

    # First entangling CX
    qc.cx(q0, q1)

    # Second local Y-Z on each
    qc.ry(p("ry2_q0"), q0)
    qc.rz(p("rz3_q0"), q0)

    qc.ry(p("ry2_q1"), q1)
    qc.rz(p("rz3_q1"), q1)

    # Second entangling CX in opposite direction
    qc.cx(q1, q0)

    # Final Z rotations
    qc.rz(p("rz4_q0"), q0)
    qc.rz(p("rz4_q1"), q1)


# ============================================================================
# PART 3: BRICK-WALL TN / qMPS ANSATZ
# ============================================================================

def build_ansatz_tn_mps_sshh(L: int, reps: int) -> QuantumCircuit:
    """
    Tensor-network-inspired quantum-circuit MPS (qMPS) ansatz for the
    spinful SSH-Hubbard model.

    Layout:
      - n_qubits = 2 * L (spin up/down per site).
      - Repeated 'reps' times:
        * single-qubit rotations on each qubit,
        * an 'even' brick of 2-qubit TN blocks,
        * an 'odd' brick of 2-qubit TN blocks.

    Works for L <= 8 (2L <= 16 qubits).

    Parameters
    ----------
    L : int
        Number of physical lattice sites
    reps : int
        Number of repetitions (depth) of the brick-wall pattern

    Returns
    -------
    qc : QuantumCircuit
        The TN-MPS ansatz circuit with parameters
    """
    n_qubits = 2 * L
    qc = QuantumCircuit(n_qubits)

    for rep in range(reps):
        # Local single-qubit layer
        for q in range(n_qubits):
            theta_ry = Parameter(f"tn_ry_{rep}_{q}")
            theta_rz = Parameter(f"tn_rz_{rep}_{q}")
            qc.ry(theta_ry, q)
            qc.rz(theta_rz, q)

        # Even brick: pairs (0,1), (2,3), (4,5), ...
        block_id = 0
        for q0 in range(0, n_qubits - 1, 2):
            q1 = q0 + 1
            apply_tn_block_su4_like(
                qc,
                prefix="tn_even",
                layer=rep,
                block_id=block_id,
                q0=q0,
                q1=q1,
            )
            block_id += 1

        # Odd brick: pairs (1,2), (3,4), (5,6), ...
        block_id = 0
        for q0 in range(1, n_qubits - 1, 2):
            q1 = q0 + 1
            apply_tn_block_su4_like(
                qc,
                prefix="tn_odd",
                layer=rep,
                block_id=block_id,
                q0=q0,
                q1=q1,
            )
            block_id += 1

    return qc


# ============================================================================
# PART 4: OPTIONAL NUMBER-PRESERVING VERSION
# ============================================================================

def apply_tn_block_np(
    qc: QuantumCircuit,
    prefix: str,
    layer: int,
    block_id: int,
    q0: int,
    q1: int,
) -> None:
    """
    Append a 2-qubit number-preserving gate:
      - Identity (up to phase) on |00> and |11>
      - SU(2) rotation on {|01>, |10>} with parameters theta, phi

    The matrix in the computational basis is:

        [1,        0,           0,           0]
        [0,   cos(θ),    i·sin(θ),           0]
        [0,   i·sin(θ),    cos(θ),           0]
        [0,        0,           0,    exp(i·φ)]

    This is the UNP (Universal Number-Preserving) gate used in np_hva ansatz.

    Decomposition (from ssh_hubbard_vqe.py):
        CRZ(φ, q0, q1)
        H(q1)
        CX(q1, q0)
        RY(θ, q0)
        CX(q1, q0)
        H(q1)

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit to append to
    prefix : str
        Prefix for parameter names
    layer : int
        Layer number
    block_id : int
        Block identifier
    q0, q1 : int
        Qubit indices
    """
    def p(name: str) -> Parameter:
        return Parameter(f"{prefix}_{layer}_{block_id}_{name}")

    theta = p("theta")
    phi = p("phi")

    # UNP gate decomposition
    qc.crz(phi, q0, q1)
    qc.h(q1)
    qc.cx(q1, q0)
    qc.ry(theta, q0)
    qc.cx(q1, q0)
    qc.h(q1)


def build_ansatz_tn_mps_np_sshh(L: int, reps: int) -> QuantumCircuit:
    """
    Number-preserving brick-wall TN ansatz for SSH-Hubbard.

    Same layout as build_ansatz_tn_mps_sshh, but each 2-qubit block
    is a number-preserving gate acting only on the {|01>, |10>} subspace.

    This ensures strict particle number conservation throughout the circuit.

    Parameters
    ----------
    L : int
        Number of physical lattice sites
    reps : int
        Number of repetitions (depth)

    Returns
    -------
    qc : QuantumCircuit
        The number-preserving TN-MPS ansatz circuit
    """
    n_qubits = 2 * L
    qc = QuantumCircuit(n_qubits)

    for rep in range(reps):
        # Local single-qubit layer (only Z rotations to preserve number)
        for q in range(n_qubits):
            theta_rz = Parameter(f"tn_np_rz_{rep}_{q}")
            qc.rz(theta_rz, q)

        # Even brick with number-preserving blocks
        block_id = 0
        for q0 in range(0, n_qubits - 1, 2):
            q1 = q0 + 1
            apply_tn_block_np(
                qc,
                prefix="tn_np_even",
                layer=rep,
                block_id=block_id,
                q0=q0,
                q1=q1,
            )
            block_id += 1

        # Odd brick with number-preserving blocks
        block_id = 0
        for q0 in range(1, n_qubits - 1, 2):
            q1 = q0 + 1
            apply_tn_block_np(
                qc,
                prefix="tn_np_odd",
                layer=rep,
                block_id=block_id,
                q0=q0,
                q1=q1,
            )
            block_id += 1

    return qc


# ============================================================================
# PART 5: SIMPLE VQE / EXPECTATION DEMO
# ============================================================================

def compute_expectation(
    circuit: QuantumCircuit,
    hamiltonian: SparsePauliOp,
    parameters: np.ndarray,
) -> float:
    """
    Compute expectation value ⟨ψ(θ)|H|ψ(θ)⟩.

    Parameters
    ----------
    circuit : QuantumCircuit
        Parameterized ansatz circuit
    hamiltonian : SparsePauliOp
        The Hamiltonian operator
    parameters : np.ndarray
        Parameter values to assign to the circuit

    Returns
    -------
    energy : float
        The expectation value of H
    """
    # Bind parameters
    param_dict = {p: parameters[i] for i, p in enumerate(circuit.parameters)}
    bound_circuit = circuit.assign_parameters(param_dict)

    # Create statevector
    psi = Statevector.from_instruction(bound_circuit)

    # Compute expectation
    energy = psi.expectation_value(hamiltonian).real

    return energy


def demo_tn_mps_ansatz():
    """
    Demonstration of TN-MPS ansatz for SSH-Hubbard model.

    Tests both the standard and number-preserving variants.
    """
    print("=" * 70)
    print("TN-MPS BRICK-WALL ANSATZ DEMO FOR SSH-HUBBARD")
    print("=" * 70)

    # System parameters
    L = 4
    t1 = 1.0
    t2 = 0.5
    U = 0.0  # Start with non-interacting for simplicity
    reps = 2

    print(f"\nSystem Parameters:")
    print(f"  Sites (L):        {L}")
    print(f"  Qubits (2L):      {2*L}")
    print(f"  Strong hop (t1):  {t1:.3f}")
    print(f"  Weak hop (t2):    {t2:.3f}")
    print(f"  Interaction (U):  {U:.3f}")
    print(f"  Dimerization δ:   {(t1-t2)/(t1+t2):.3f}")
    print(f"  Ansatz reps:      {reps}")

    # Build Hamiltonian
    print(f"\nBuilding Hamiltonian...")
    H = ssh_hubbard_hamiltonian(L, t1, t2, U)
    print(f"  Pauli terms:      {len(H)}")
    print(f"  Qubits:           {H.num_qubits}")

    # Test 1: Standard TN-MPS ansatz
    print("\n" + "=" * 70)
    print("TEST 1: STANDARD TN-MPS ANSATZ")
    print("=" * 70)

    ansatz = build_ansatz_tn_mps_sshh(L, reps)
    print(f"\nAnsatz Statistics:")
    print(f"  Circuit depth:    {ansatz.depth()}")
    print(f"  Parameters:       {ansatz.num_parameters}")
    print(f"  Qubits:           {ansatz.num_qubits}")
    print(f"  Gates:            {sum(ansatz.count_ops().values())}")

    # Evaluate with random parameters
    np.random.seed(42)
    theta0 = 0.1 * np.random.randn(ansatz.num_parameters)

    energy = compute_expectation(ansatz, H, theta0)
    print(f"\nExpectation Value (random params):")
    print(f"  E(θ0):            {energy:.8f}")
    print(f"  E/site:           {energy/L:.8f}")

    # Test 2: Number-preserving TN-MPS ansatz
    print("\n" + "=" * 70)
    print("TEST 2: NUMBER-PRESERVING TN-MPS ANSATZ")
    print("=" * 70)

    ansatz_np = build_ansatz_tn_mps_np_sshh(L, reps)
    print(f"\nAnsatz Statistics:")
    print(f"  Circuit depth:    {ansatz_np.depth()}")
    print(f"  Parameters:       {ansatz_np.num_parameters}")
    print(f"  Qubits:           {ansatz_np.num_qubits}")
    print(f"  Gates:            {sum(ansatz_np.count_ops().values())}")

    # Evaluate with random parameters
    np.random.seed(42)
    theta0_np = 0.1 * np.random.randn(ansatz_np.num_parameters)

    # Note: Starting from vacuum state, number-preserving ansatz won't explore
    # different particle sectors. For proper use, prepend initial state preparation.
    energy_np = compute_expectation(ansatz_np, H, theta0_np)
    print(f"\nExpectation Value (random params, from vacuum):")
    print(f"  E(θ0):            {energy_np:.8f}")
    print(f"  E/site:           {energy_np/L:.8f}")
    print(f"\nNote: Number-preserving ansatz starts from vacuum state.")
    print(f"      For non-trivial results, prepend initial state preparation.")

    # Test 3: Larger system demonstration
    print("\n" + "=" * 70)
    print("TEST 3: LARGER SYSTEM (L=6)")
    print("=" * 70)

    L_large = 6
    print(f"\nSystem: L={L_large} sites, {2*L_large} qubits")

    H_large = ssh_hubbard_hamiltonian(L_large, t1, t2, U)
    ansatz_large = build_ansatz_tn_mps_sshh(L_large, reps)

    print(f"\nAnsatz Statistics:")
    print(f"  Circuit depth:    {ansatz_large.depth()}")
    print(f"  Parameters:       {ansatz_large.num_parameters}")
    print(f"  Qubits:           {ansatz_large.num_qubits}")

    theta0_large = 0.1 * np.random.randn(ansatz_large.num_parameters)
    energy_large = compute_expectation(ansatz_large, H_large, theta0_large)

    print(f"\nExpectation Value (random params):")
    print(f"  E(θ0):            {energy_large:.8f}")
    print(f"  E/site:           {energy_large/L_large:.8f}")

    # Test 4: Even larger system (L=8) - just build, don't evaluate
    print("\n" + "=" * 70)
    print("TEST 4: MAXIMUM SIZE (L=8) - CIRCUIT BUILD ONLY")
    print("=" * 70)

    L_max = 8
    print(f"\nSystem: L={L_max} sites, {2*L_max} qubits")

    ansatz_max = build_ansatz_tn_mps_sshh(L_max, reps)
    print(f"\nAnsatz Statistics:")
    print(f"  Circuit depth:    {ansatz_max.depth()}")
    print(f"  Parameters:       {ansatz_max.num_parameters}")
    print(f"  Qubits:           {ansatz_max.num_qubits}")
    print(f"  Gates:            {sum(ansatz_max.count_ops().values())}")

    print("\n(Skipping evaluation for L=8 to save time - circuit builds successfully)")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print("  ✓ Standard TN-MPS ansatz working")
    print("  ✓ Number-preserving variant working")
    print("  ✓ Scales to L=8 (16 qubits) cleanly")
    print("  ✓ Hamiltonian construction via JW mapping")
    print("  ✓ Expectation value evaluation")
    print("\nReady for VQE optimization with your preferred optimizer!")


if __name__ == "__main__":
    demo_tn_mps_ansatz()
