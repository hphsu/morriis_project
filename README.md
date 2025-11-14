# SSH-Hubbard Lattice: DMRG and VQE Implementations

This project implements two complementary approaches for studying the SSH-Hubbard model on 1D lattices:

1. **DMRG (Density Matrix Renormalization Group)**: Classical exact diagonalization for up to 8 sites
2. **VQE (Variational Quantum Eigensolver)**: Quantum-inspired algorithm with topology-aware ansätze

## Repository Structure

```
morriis_project/
├── src/                          # Source code
│   ├── ssh_hubbard_vqe.py       # Main VQE implementation
│   ├── ssh_hubbard_tn_vqe.py    # Tensor network VQE
│   ├── ssh_hubbard_tenpy_dmrg_fixed.py  # DMRG solver
│   ├── compare_all_ansatze.py   # Ansatz comparison script
│   ├── benchmark_large_systems.py  # System benchmarking
│   └── run_longer_optimizations.py  # Extended optimization runs
├── results/                      # Output files (plots, CSVs)
├── docs/                         # Documentation
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── DMRG_STATUS.md
│   ├── REPOSITORY_FULL_CONTEXT.txt
│   └── dmrg_test_fixed.txt
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Physics Background

### SSH Model (Su-Schrieffer-Heeger)
The SSH model is a 1D tight-binding model with alternating hopping amplitudes:
- Strong bonds (t1) connect sites: 0-1, 2-3, 4-5, 6-7
- Weak bonds (t2) connect sites: 1-2, 3-4, 5-6

This dimerization leads to topological edge states when t1 > t2.

### Hubbard Model
The Hubbard model adds on-site electron-electron interaction:
- U: interaction energy when two electrons (opposite spins) occupy the same site

### Combined SSH-Hubbard Hamiltonian
```
H = -∑_<i,j> t_ij (c†_iσ c_jσ + h.c.) + U ∑_i n_i↑ n_i↓
```

where:
- c†_iσ, c_iσ: creation/annihilation operators for electron with spin σ at site i
- t_ij: hopping amplitude (alternates between t1 and t2)
- n_iσ: number operator for spin σ at site i
- U: on-site interaction strength

## Implementation Details

### Key Features

1. **Exact Diagonalization**: For 8 sites (Hilbert space dimension 4^8 = 65,536), the code uses exact diagonalization to find the ground state.

2. **Operator Construction**: Implements fermionic creation/annihilation operators with proper anticommutation relations.

3. **Observable Calculation**: Computes:
   - Ground state energy
   - Site occupancies (spin-up, spin-down)
   - Double occupancy (both spins on same site)

### Code Structure

- `SpinOperators`: Defines single-site operators (creation, annihilation, number operators)
- `SSHHubbardHamiltonian`: Constructs the full Hamiltonian matrix
- `DMRG`: Main DMRG class (uses exact diagonalization for L ≤ 8)
- `main()`: Runs the simulation and displays results

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- qiskit >= 1.0.0 (for VQE)
- qiskit-algorithms >= 0.3.0 (for VQE)
- qiskit-aer >= 0.14.0 (for VQE)

---

# VQE Implementation with Topology-Aware Ansätze

## Overview

The `ssh_hubbard_vqe.py` script implements a Variational Quantum Eigensolver for the SSH-Hubbard model with three ansatz options inspired by topological quantum computing and Hamiltonian structure.

### Three Ansatz Options

1. **HEA (Hardware-Efficient Ansatz)**
   - Standard `EfficientSU2` circuit with Ry, Rz gates and linear entanglement
   - General-purpose, good baseline performance
   - Default option for compatibility

2. **HVA (Hamiltonian-Variational Ansatz)**
   - Layers structured like the Hamiltonian terms
   - Even bond hopping layer (t1), odd bond hopping layer (t2), on-site U layer
   - Uses number-conserving XX+YY gates for hopping, ZZ for interactions
   - Physics-informed, preserves particle number and Sz symmetry

3. **TopoInspired (Topological/Problem-Inspired)**
   - Designed for SSH topology: dimer pattern + edge links
   - Local Ry rotations followed by strong/weak bond layers
   - Special edge-to-edge entangler to seed topological edge correlations
   - Best for studying topological phase transitions

## Usage

All scripts should be run from the `src/` directory:

```bash
cd src/
```

### Basic Single-Point Calculations

```bash
# Hardware-efficient ansatz (default)
python ssh_hubbard_vqe.py --ansatz hea --reps 3

# Hamiltonian-variational ansatz
python ssh_hubbard_vqe.py --ansatz hva --reps 2

# Topological ansatz
python ssh_hubbard_vqe.py --ansatz topoinsp --reps 3
```

### Custom Parameters

```bash
# Customize system size, interaction, and hopping
python ssh_hubbard_vqe.py --ansatz topoinsp --L 6 --U 2.0 --t1 1.0 --t2 0.8 --reps 3

# Use periodic boundary conditions
python ssh_hubbard_vqe.py --ansatz hva --periodic --reps 2

# Adjust optimizer iterations
python ssh_hubbard_vqe.py --ansatz hea --maxiter 500
```

### Warm-Start Delta Sweep

Sweep through dimerization parameter δ = (t1-t2)/(t1+t2) with parameter warm-starting:

```bash
# Sweep from δ=-0.6 to δ=+0.6 with 13 points
python ssh_hubbard_vqe.py --ansatz topoinsp --delta-sweep -0.6 0.6 13

# Results saved to: ../results/L{L}_{ansatz}_delta_sweep.csv
# Plot saved to: ../results/L{L}_{ansatz}_delta_error.png
```

The warm-start feature uses the optimal parameters from each δ point as the initial parameters for the next, significantly reducing optimization time and improving convergence.

## Features

### Exact Diagonalization Benchmarking

Every VQE run includes exact diagonalization for validation:
- Ground state energy comparison
- Observable-by-observable error analysis
- Convergence tracking against ED baseline

### Comprehensive Observables

**Z-basis (single measurement circuit)**:
- Energy per site and variance
- Double occupancy: ⟨n↑n↓⟩
- Nearest-neighbor spin correlations: ⟨Sz_i Sz_j⟩
- Structure factor S^zz(π) (optional)

**XY-basis (multiple circuits)**:
- Bond order parameters for each bond
- Dimer order parameter (alternating sum)

**Edge diagnostics (open boundaries)**:
- Edge particle densities
- Edge-to-edge spin correlation

**Topological diagnostics (statevector mode)**:
- **Edge concurrence**: Wootters concurrence between edge qubits (averaged over spins)
- **Bond purity**: Tr(ρ²) for strong vs weak bonds, measures entanglement asymmetry

### Output Files

All results are saved to the `results/` directory (located at `../results/` relative to `src/`):
- Circuit diagrams: `L{L}_{ansatz}_circuit.png`
- Energy convergence: `L{L}_{ansatz}_energy_convergence.png`
- Error convergence (log scale): `L{L}_{ansatz}_error_log.png`
- Delta sweep data: `L{L}_{ansatz}_delta_sweep.csv`
- Delta sweep plot: `L{L}_{ansatz}_delta_error.png`

## Example Output

```
======================================================================
SSH-HUBBARD VQE SIMULATION
======================================================================
System: L=6 sites, N=12 qubits
Ansatz: TOPOINSP, Reps=3
Parameters: t1=1.0, t2=0.8, U=2.0, δ=0.1111
Boundary: Open
======================================================================

Hamiltonian: 156 Pauli terms on 12 qubits

--- Exact Diagonalization ---
Hilbert space dimension: 4096
Ground state energy: -4.8234567890
Energy per site:     -0.8039094648

--- Building TOPOINSP Ansatz ---
Circuit depth:  42
Parameters:     78
Qubits:         12
✓ Circuit saved to ../results/L6_topoinsp_circuit.png

--- Running VQE Optimization ---

======================================================================
ENERGY COMPARISON
======================================================================
ED energy:      -4.8234567890
VQE energy:     -4.8234512345
Absolute error: 5.555e-06
Relative error: 0.0001%
Parameters:     78
Evaluations:    145

======================================================================
OBSERVABLE RESULTS
======================================================================

--- Z-basis Observables ---
  Energy/site:        -0.80390946
  Energy variance:    2.34e-08
  Double occupancy:   0.142567
  <S^z S^z>_NN:       0.089234

--- XY-basis Observables ---
  Dimer order D:      0.234567
  Bond orders:
    Bond 0 (strong):  0.567890
    Bond 1 (weak):    0.345678
    ...

--- Edge Diagnostics (Open BC) ---
  Edge density:       1.234567
  Edge S^z corr:      0.123456

--- Topological Diagnostics ---
  Edge concurrence:   0.345678
  Strong bond purity: 0.678901
  Weak bond purity:   0.456789

======================================================================
VQE vs ED COMPARISON
======================================================================
Observable           VQE           ED        Error      Rel%
----------------------------------------------------------------------
Energy/site          -0.803909    -0.803909    1.15e-06   0.00%
Double occ            0.142567     0.142568    1.23e-06   0.00%
<SzSz>_NN             0.089234     0.089235    8.90e-07   0.00%
D_dimer               0.234567     0.234568    1.45e-06   0.00%
```

## Physics Insights

### Topological Phase Transition

The SSH model exhibits a topological phase transition at δ=0:
- **δ > 0 (t1 > t2)**: Topologically non-trivial phase with edge states
- **δ < 0 (t1 < t2)**: Trivial phase without edge states
- **δ = 0**: Critical point

**Observables to watch**:
- Edge concurrence increases in topological phase
- Dimer order parameter changes sign across transition
- Edge densities show enhancement for δ > 0
- Bond purity shows strong/weak asymmetry

### Interaction Effects (Hubbard U)

- **U = 0**: Pure SSH, non-interacting fermions
- **U ~ 2-4**: Moderate correlations, interesting competition with topology
- **U >> t**: Mott insulator regime, strong localization, suppressed double occupancy

### Ansatz Comparison

**HEA**: Fast convergence, general-purpose, but may miss topology-specific features

**HVA**: Best for capturing interaction physics (U term), preserves symmetries, moderate parameter count

**TopoInspired**: Optimized for SSH topology, explicit edge links capture topological features, best for δ sweeps across phase transition

## Advanced Usage

### Comparing Ansätze

Run all three ansätze on the same system from the `src/` directory:

```bash
cd src/
for ansatz in hea hva topoinsp; do
    python ssh_hubbard_vqe.py --ansatz $ansatz --L 6 --reps 3 --U 2.0
done
```

Or use the comparison script:

```bash
cd src/
python compare_all_ansatze.py
```

Compare results in the `results/` directory.

### Topological Phase Diagram

Scan δ with the topological ansatz to map the phase transition:

```bash
python ssh_hubbard_vqe.py --ansatz topoinsp --delta-sweep -0.8 0.8 17 --L 6 --U 2.0
```

Plot edge concurrence and dimer order from the CSV to identify the transition.

### Interaction Strength Scan

Fix δ, scan U to study Mott physics:

```python
# Modify script or run multiple times with different --U values
for U in 0.0 1.0 2.0 4.0 8.0; do
    python ssh_hubbard_vqe.py --ansatz hva --U $U --L 6 --reps 3
done
```

## Implementation Details

### Jordan-Wigner Mapping

Fermions mapped to qubits with Jordan-Wigner transformation:
- Each lattice site has 2 qubits: spin-up and spin-down
- Qubit ordering: [site0↑, site0↓, site1↑, site1↓, ...]
- Hopping terms become XX+YY with Z string for anticommutation
- Number operators: n = (I - Z)/2

### Number Conservation

HVA and TopoInspired ansätze preserve total particle number:
- No single-qubit X/Y gates (would change occupation)
- Only number-conserving 2-qubit gates: XX+YY (hopping-like) and ZZ (interaction-like)
- Reduces Hilbert space exploration, improves convergence for fixed particle number

### Measurement Grouping

Observables grouped by Pauli basis:
- **Z-only group**: All measured simultaneously in computational basis (1 circuit)
- **XY group**: Each bond order requires separate circuit preparation (~L circuits)
- **Topology group**: Post-processed from statevector (no additional measurements)

Total measurement count: ~(1 + L) circuits for full observable set

---

# DMRG Implementation

## Usage

Run the DMRG simulation from the `src/` directory:

```bash
cd src/
python ssh_hubbard_tenpy_dmrg_fixed.py
```

### Customizing Parameters

Edit the `main()` function in `src/ssh_hubbard_tenpy_dmrg_fixed.py`:

```python
L = 8          # Number of sites
t1 = 1.0       # Strong hopping amplitude
t2 = 0.5       # Weak hopping amplitude
U = 2.0        # Hubbard interaction strength
max_states = 64  # Maximum DMRG states (not used for L ≤ 8)
```

### Example Parameters

**Topological regime** (t1 > t2):
```python
t1 = 1.0
t2 = 0.5
```

**Trivial regime** (t1 < t2):
```python
t1 = 0.5
t2 = 1.0
```

**Interaction strength**:
- U = 0: Non-interacting (pure SSH)
- U ~ 2-4: Moderate interaction
- U >> t: Strong interaction (Mott insulator regime)

## Output

The simulation outputs:

1. **Ground State Energy**: Total energy and energy per site
2. **Site Occupancies**: Electron density for each spin at each site
3. **Double Occupancy**: Probability of two electrons on same site
4. **Total Particle Numbers**: Total spin-up, spin-down, and total particles

### Sample Output

```
==============================================================
DMRG Simulation: 8-Site SSH-Hubbard Lattice
==============================================================

Starting DMRG for 8-site SSH-Hubbard lattice
Parameters: t1=1.0, t2=0.5, U=2.0
Max states kept: 64
------------------------------------------------------------
System small enough for exact diagonalization
Hilbert space dimension: 65536
Diagonalizing Hamiltonian...

Ground state energy: -5.2345678900

Calculating observables...

==============================================================
RESULTS
==============================================================

Ground State Energy: -5.2345678900
Energy per site: -0.6543209862

Site Occupancies:
------------------------------------------------------------
Site  |  n_up  |  n_down  |  n_total  |  double_occ
------------------------------------------------------------
  0   | 0.5123 | 0.5123  | 1.0246   | 0.2341
  1   | 0.4987 | 0.4987  | 0.9974   | 0.2289
  ...

Total particles (up):    4.0000
Total particles (down):  4.0000
Total particles:         8.0000
Total double occupancy:  1.8456
```

## Physics Insights

### What to Look For

1. **Edge States**: In topological regime (t1 > t2), expect enhanced density at edges (sites 0 and 7)

2. **Dimerization Pattern**: Density should show modulation matching the hopping pattern

3. **Interaction Effects**:
   - Weak U: Delocalized electrons
   - Strong U: Electrons localize to minimize double occupancy

4. **Half-Filling**: With 8 sites and 8 electrons, system is at half-filling (1 electron per site on average)

## Mathematical Details

### Hilbert Space

Each site has 4 states:
- |0⟩: empty
- |↑⟩: spin-up electron
- |↓⟩: spin-down electron
- |↑↓⟩: doubly occupied

Total Hilbert space dimension: 4^L = 4^8 = 65,536

### Fermionic Anticommutation

The code implements proper fermionic anticommutation:
```
{c_i, c†_j} = δ_ij
{c_i, c_j} = 0
```

Sign factors are handled in the operator definitions.

## Future Extensions

1. **Full DMRG**: Implement sweeping algorithm for L > 8
2. **Correlation Functions**: Calculate spin-spin and density-density correlations
3. **Entanglement Entropy**: Measure bipartite entanglement
4. **Time Evolution**: Implement time-dependent DMRG (t-DMRG)
5. **Visualization**: Plot density profiles, correlation functions

## References

1. White, S. R. (1992). "Density matrix formulation for quantum renormalization groups". Physical Review Letters.
2. Su, W. P., Schrieffer, J. R., & Heeger, A. J. (1979). "Solitons in polyacetylene". Physical Review Letters.
3. Hubbard, J. (1963). "Electron correlations in narrow energy bands". Proceedings of the Royal Society A.

## License

MIT License

## Author

Generated for quantum many-body physics research.
