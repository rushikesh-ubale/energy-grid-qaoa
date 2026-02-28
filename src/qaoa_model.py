"""
qaoa_model.py
-------------
QAOA circuit, angle optimiser, and result analyser.

Stack: Qiskit statevector simulator → optimised (γ, β) → sampling → approximation ratio.
Optimised angles are exported as JSON so rigetti_runner.py can load them directly.

Two entry points:
  run_qaoa(G, model, p)  — full pipeline, returns QAOAResult
  python -m src.qaoa_model  — CLI smoke-test on 8-node hw subgraph
"""

import json
import time
import numpy as np
import networkx as nx
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.qubo_builder import IsingModel, build_ising, ising_to_qaoa_coefficients


# ── Result Container ───────────────────────────────────────────────────────────

@dataclass
class QAOAResult:
    p                : int
    optimal_gamma    : list
    optimal_beta     : list
    best_cut_value   : float
    approx_ratio     : Optional[float]
    energy_history   : list  = field(default_factory=list)
    runtime_s        : float = 0.0
    shots            : int   = 0
    bitstring_counts : dict  = field(default_factory=dict)
    notes            : str   = ""

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f'  saved → {path}')


# ── Circuit ────────────────────────────────────────────────────────────────────

def build_qaoa_circuit(zz_coeffs: dict, z_coeffs: dict,
                       n_qubits: int, gamma: list, beta: list, p: int):
    """
    QAOA circuit in Qiskit.
    Cost layer  : RZZ(2γ·J_ij) per coupling, RZ(2γ·h_i) per local field.
    Mixer layer : RX(2β) per qubit (standard X-mixer).
    Initial state: uniform superposition via H gates.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    for layer in range(p):
        g = gamma[layer]
        b = beta[layer]

        # cost layer
        for (i, j), Jval in zz_coeffs.items():
            qc.rzz(2 * g * Jval, i, j)
        for i, hval in z_coeffs.items():
            if abs(hval) > 1e-10:
                qc.rz(2 * g * hval, i)

        # mixer layer
        for i in range(n_qubits):
            qc.rx(2 * b, i)

    return qc


def _expectation_sv(qc, zz_coeffs: dict, z_coeffs: dict, offset: float) -> float:
    """
    Exact expectation ⟨H⟩ via statevector simulation (Qiskit).
    We minimise H so lower = better (= higher cut value).
    """
    from qiskit.quantum_info import Statevector

    probs = Statevector(qc).probabilities()
    n     = qc.num_qubits
    exp   = offset

    for state_idx, prob in enumerate(probs):
        if prob < 1e-15:
            continue
        # Qiskit convention: bit 0 of state_idx = qubit 0
        spins = [1 - 2 * ((state_idx >> i) & 1) for i in range(n)]
        e     = 0.0
        for (i, j), Jval in zz_coeffs.items():
            e += Jval * spins[i] * spins[j]
        for i, hval in z_coeffs.items():
            e += hval * spins[i]
        exp += prob * e

    return exp


# ── Angle Optimisation ─────────────────────────────────────────────────────────

def _grid_search_p1(zz_coeffs, z_coeffs, n_qubits, offset,
                    n_grid: int = 20) -> tuple:
    """
    Coarse grid search over (γ, β) for p=1.
    Returns (gamma_list, beta_list, best_energy, landscape_array).
    """
    gammas    = np.linspace(0, np.pi,     n_grid)
    betas     = np.linspace(0, np.pi / 2, n_grid)
    landscape = np.zeros((n_grid, n_grid))
    best_e    = np.inf
    best_g, best_b = 0.0, 0.0

    for gi, g in enumerate(gammas):
        for bi, b in enumerate(betas):
            qc = build_qaoa_circuit(zz_coeffs, z_coeffs, n_qubits, [g], [b], p=1)
            e  = _expectation_sv(qc, zz_coeffs, z_coeffs, offset)
            landscape[gi, bi] = e
            if e < best_e:
                best_e, best_g, best_b = e, g, b

    return [best_g], [best_b], best_e, landscape


def _cobyla_optimise(zz_coeffs, z_coeffs, n_qubits, offset,
                     p: int, n_restarts: int = 5,
                     seed: int = 42) -> tuple:
    """
    COBYLA multi-start optimisation.
    Returns (gamma, beta, best_energy, energy_history).
    """
    from scipy.optimize import minimize

    rng            = np.random.default_rng(seed)
    energy_history = []
    best_e         = np.inf
    best_params    = None

    def objective(params):
        g  = params[:p].tolist()
        b  = params[p:].tolist()
        qc = build_qaoa_circuit(zz_coeffs, z_coeffs, n_qubits, g, b, p)
        e  = _expectation_sv(qc, zz_coeffs, z_coeffs, offset)
        energy_history.append(e)
        return e

    for _ in range(n_restarts):
        g0  = rng.uniform(0,       np.pi,     p)
        b0  = rng.uniform(0, np.pi / 2,       p)
        x0  = np.concatenate([g0, b0])
        res = minimize(objective, x0, method='COBYLA',
                       options={'maxiter': 1000, 'rhobeg': 0.5})
        if res.fun < best_e:
            best_e, best_params = res.fun, res.x

    return (best_params[:p].tolist(), best_params[p:].tolist(),
            best_e, energy_history)


# ── Shot Sampling ──────────────────────────────────────────────────────────────

def _sample(qc, shots: int = 2048, seed: int = 42) -> dict:
    """Shot-based simulation via Qiskit Aer. Returns {bitstring: count}."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    qc_m = qc.copy()
    qc_m.measure_all()
    sim  = AerSimulator(method='statevector', seed_simulator=seed)
    job  = sim.run(transpile(qc_m, sim), shots=shots)
    return job.result().get_counts()


def _best_cut_from_counts(counts: dict, nodes: list, G: nx.Graph) -> tuple:
    """Return (best_cut_value, best_partition) from shot counts."""
    best_cut, best_part = -np.inf, {}
    for bitstring in counts:
        bits  = bitstring[::-1]           # Qiskit: rightmost bit = qubit 0
        spins = {nodes[i]: (1 if bits[i] == '0' else -1)
                 for i in range(len(nodes))}
        cut   = sum(d['weight'] for u, v, d in G.edges(data=True)
                    if spins[u] != spins[v]) 
        if cut > best_cut:
            best_cut, best_part = cut, spins
    return best_cut, best_part


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def run_qaoa(G: nx.Graph,
             ising_model    : IsingModel,
             p              : int            = 1,
             exact_optimum  : Optional[float] = None,
             shots          : int            = 2048,
             n_restarts     : int            = 5,
             save_path      : Optional[str]  = None) -> QAOAResult:
    """
    Full QAOA pipeline:
      1. Extract ZZ/Z coefficients from IsingModel
      2. Grid search (p=1) or warm-start (p>1)
      3. COBYLA multi-start optimisation
      4. Sample best circuit
      5. Compute approximation ratio

    Returns QAOAResult; optionally saves JSON to save_path.
    """
    t0                   = time.perf_counter()
    zz, z                = ising_to_qaoa_coefficients(ising_model)
    n_qubits             = ising_model.num_qubits()
    offset               = ising_model.offset
    nodes                = ising_model.nodes

    print(f'  QAOA p={p}  |  {n_qubits} qubits  |  {len(zz)} ZZ terms')

    # Step 1: angle initialisation
    if p == 1:
        print('  grid search (p=1)...', end=' ', flush=True)
        g0, b0, e0, landscape = _grid_search_p1(zz, z, n_qubits, offset)
        print(f'best_E = {e0:.4f}')
    else:
        g0, b0 = [0.3] * p, [0.3] * p

    # Step 2: COBYLA refinement
    print(f'  COBYLA ({n_restarts} restarts)...', end=' ', flush=True)
    best_gamma, best_beta, best_e, e_hist = _cobyla_optimise(
        zz, z, n_qubits, offset, p=p, n_restarts=n_restarts
    )
    print(f'best_E = {best_e:.4f}')

    # Step 3: sample
    print(f'  sampling ({shots} shots)...', end=' ', flush=True)
    best_qc            = build_qaoa_circuit(zz, z, n_qubits, best_gamma, best_beta, p)
    counts             = _sample(best_qc, shots=shots)
    best_cut, best_part = _best_cut_from_counts(counts, nodes, G)
    print(f'best cut = {best_cut:.4f}')

    approx_ratio = (best_cut / exact_optimum) if exact_optimum else None

    result = QAOAResult(
        p=p,
        optimal_gamma=best_gamma,
        optimal_beta=best_beta,
        best_cut_value=best_cut,
        approx_ratio=approx_ratio,
        energy_history=e_hist,
        runtime_s=time.perf_counter() - t0,
        shots=shots,
        bitstring_counts={k: int(v) for k, v in counts.items()},
        notes=f'statevector sim · COBYLA · {n_restarts} restarts'
    )

    if save_path:
        result.save(save_path)

    return result


# ── Plotting Helpers ───────────────────────────────────────────────────────────

def plot_angle_landscape(landscape: np.ndarray, n_grid: int = 20,
                         save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    gammas = np.linspace(0, np.pi,     n_grid)
    betas  = np.linspace(0, np.pi / 2, n_grid)
    fig, ax = plt.subplots(figsize=(7, 5))
    c = ax.contourf(betas, gammas, landscape, levels=30, cmap='viridis_r')
    plt.colorbar(c, ax=ax, label='⟨H⟩ (lower = better)')
    ax.set_xlabel('β'); ax.set_ylabel('γ')
    ax.set_title('QAOA p=1 energy landscape', fontsize=13)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_approx_ratio_vs_p(results: list, exact_optimum: float,
                            classical_baseline: float,
                            save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    ps      = [r.p for r in results]
    ratios  = [r.approx_ratio for r in results]
    cl_r    = classical_baseline / exact_optimum

    fig, ax = plt.subplots(figsize=(7, 4))
    bars    = ax.bar([f'QAOA p={p}' for p in ps], ratios,
                     color='steelblue', alpha=0.85)
    ax.axhline(cl_r, color='darkorange', linestyle='--', linewidth=1.8,
               label=f'Greedy one_exchange ({cl_r:.3f})')
    ax.axhline(1.0, color='green', linestyle=':', linewidth=1.5,
               label='Exact optimum (1.0)')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Approximation ratio  r = cut / MPES')
    ax.set_title('QAOA approximation ratio vs circuit depth p', fontsize=13)
    ax.legend(fontsize=9)
    for bar, r in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{r:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_shot_distribution(counts: dict, top_k: int = 15,
                           save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    top     = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    labels, freqs = zip(*top)
    total   = sum(counts.values())
    probs   = [f / total for f in freqs]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(labels)), probs, color='steelblue', edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Probability')
    ax.set_title(f'QAOA shot distribution — top {top_k} bitstrings', fontsize=13)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from src.graph_utils import load_graph, bfs_subgraph
    from src.classical_solvers import exact_brute_force
    from src.qubo_builder import QUBOParams

    GA   = load_graph('data/problemA.parquet')
    hwG  = bfs_subgraph(GA, target_size=8)
    opt  = exact_brute_force(hwG).cut_value
    print(f'Exact hw subgraph cut: {opt:.4f}\n')

    model = build_ising(hwG, QUBOParams.pure_maxcut())

    all_results = []
    for p_val in [1, 2]:
        print(f'{"="*50}\nRunning QAOA p={p_val}')
        r = run_qaoa(hwG, model, p=p_val,
                     exact_optimum=opt, shots=2048, n_restarts=5,
                     save_path=f'results/simulator_logs/qaoa_p{p_val}_hw8_pure.json')
        all_results.append(r)
        print(f'  approx ratio: {r.approx_ratio:.4f}  |  time: {r.runtime_s:.1f}s')

    from src.classical_solvers import greedy_one_exchange
    greedy_cut = greedy_one_exchange(hwG).cut_value
    plot_approx_ratio_vs_p(all_results, opt, greedy_cut,
                           save_path='results/plots/qaoa_approx_ratio_vs_p.png')
