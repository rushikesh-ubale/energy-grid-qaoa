"""
rigetti_runner.py
-----------------
Translates optimised QAOA angles into a pyQuil program and submits to Rigetti QCS.

Native gate set (Ankaa-3):  iSWAP,  RZ(θ),  RX(π/2)

Workflow:
  1. Load (γ, β) from simulator JSON   [results/simulator_logs/]
  2. Decompose RZZ → iSWAP + RZ + RX
  3. Build pyQuil Program
  4. Estimate gate count — verify within budget BEFORE submitting
  5. Dry-run prints the program; set dry_run=False to hit real hardware
  6. Parse QCS results, compute cut values, save hardware log

Usage:
  python -m src.rigetti_runner            # dry run
  python -m src.rigetti_runner --submit   # real hardware submission
"""

import json
import time
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from typing import Optional

from src.qubo_builder import IsingModel, ising_to_qaoa_coefficients


# ── Native Gate Decompositions ─────────────────────────────────────────────────

def _rzz_native(prog, qi: int, qj: int, angle: float):
    """
    Decompose RZZ(θ) = exp(−iθ/2 · ZZ) into Ankaa-3 native gates.

    Decomposition (iSWAP-based):
        RZZ(θ) ≡  RX(π/2)_i · iSWAP(i,j) · RZ(θ)_j · RX(π/2)_i
                  · iSWAP(i,j) · RX(−π/2)_j

    Two iSWAP gates per RZZ → 2 two-qubit gates per logical edge per layer.
    """
    from pyquil.gates import RZ, RX, ISWAP
    prog += RX(np.pi / 2, qi)
    prog += ISWAP(qi, qj)
    prog += RZ(angle, qj)
    prog += RX(np.pi / 2, qi)
    prog += ISWAP(qi, qj)
    prog += RX(-np.pi / 2, qj)
    return prog


def _rx_native(prog, qubit: int, angle: float):
    """
    Express RX(θ) using only RX(π/2) and RZ:
        RX(θ) = RZ(−π/2) · RX(π/2) · RZ(π − θ) · RX(π/2) · RZ(−π/2)
    """
    from pyquil.gates import RZ, RX
    prog += RZ(-np.pi / 2, qubit)
    prog += RX( np.pi / 2, qubit)
    prog += RZ( np.pi - angle, qubit)
    prog += RX( np.pi / 2, qubit)
    prog += RZ(-np.pi / 2, qubit)
    return prog


def _h_native(prog, qubit: int):
    """Hadamard via native gates:  H = RZ(π/2) · RX(π/2) · RZ(π/2)"""
    from pyquil.gates import RZ, RX
    prog += RZ(np.pi / 2, qubit)
    prog += RX(np.pi / 2, qubit)
    prog += RZ(np.pi / 2, qubit)
    return prog


# ── Gate Count Estimator ───────────────────────────────────────────────────────

def estimate_native_gate_count(zz_coeffs: dict, n_qubits: int, p: int) -> dict:
    """
    Count native gates before submission.  Call this to verify feasibility.

    RZZ → 2 iSWAP + 4 RZ/RX  (6 native gates, 2 of which are 2Q)
    RX(2β) mixer → 5 RZ/RX per qubit  (all 1Q)
    State prep (H) → 3 RZ/RX per qubit  (all 1Q)
    """
    n_rzz         = len(zz_coeffs)
    two_q_per_rzz = 2                               # 2 iSWAP per RZZ
    one_q_per_rzz = 4                               # 4 single-qubit per RZZ
    two_q_total   = p * n_rzz * two_q_per_rzz
    one_q_total   = (p * n_rzz * one_q_per_rzz     # cost layer 1Q
                     + p * n_qubits * 5             # mixer
                     + n_qubits * 3)                # state prep

    return {
        'two_qubit_gates'  : two_q_total,
        'single_qubit_gates': one_q_total,
        'total_gates'      : two_q_total + one_q_total,
        'within_budget'    : two_q_total <= 100,
    }


# ── Program Builder ────────────────────────────────────────────────────────────

def build_pyquil_program(zz_coeffs: dict, z_coeffs: dict,
                          n_qubits: int, gamma: list, beta: list, p: int,
                          qubit_mapping: Optional[dict] = None):
    """
    Build a pyQuil Program using only Ankaa-3 native gates.

    qubit_mapping : {logical_idx: physical_qubit_id}
                    Defaults to identity (0,1,...,n−1).
                    Verify adjacency against device topology before submitting.
    """
    try:
        from pyquil import Program
        from pyquil.gates import MEASURE
    except ImportError:
        raise ImportError('pyquil not installed — run: pip install pyquil')

    if qubit_mapping is None:
        qubit_mapping = {i: i for i in range(n_qubits)}

    prog = Program()
    ro   = prog.declare('ro', 'BIT', n_qubits)

    # State prep: H on all qubits
    for i in range(n_qubits):
        prog = _h_native(prog, qubit_mapping[i])

    for layer in range(p):
        g = gamma[layer]
        b = beta[layer]

        # Cost layer — ZZ rotations
        for (i, j), Jval in zz_coeffs.items():
            qi, qj = qubit_mapping[i], qubit_mapping[j]
            prog   = _rzz_native(prog, qi, qj, 2 * g * Jval)

        # Local Z fields (non-zero only with balance penalty)
        for i, hval in z_coeffs.items():
            if abs(hval) > 1e-10:
                from pyquil.gates import RZ
                prog += RZ(2 * g * hval, qubit_mapping[i])

        # Mixer layer — RX(2β)
        for i in range(n_qubits):
            prog = _rx_native(prog, qubit_mapping[i], 2 * b)

    # Measurement
    for i in range(n_qubits):
        prog += MEASURE(qubit_mapping[i], ro[i])

    return prog


# ── QCS Submission ─────────────────────────────────────────────────────────────

def submit_to_qcs(prog, processor_id: str = 'Ankaa-3',
                  shots: int = 1000) -> dict:
    """
    Compile and run on Rigetti QCS.
    Requires active QCS credentials: https://qcs.rigetti.com
    """
    try:
        from pyquil import get_qc
    except ImportError:
        raise ImportError('pyquil required — pip install pyquil')

    print(f'  connecting to {processor_id}...')
    qc         = get_qc(processor_id)
    executable = qc.compile(prog)

    print(f'  running {shots} shots...')
    t0     = time.perf_counter()
    result = qc.run(executable)
    rt     = time.perf_counter() - t0

    bitstrings = result.get_register_map().get('ro', [])
    counts     = {}
    for bits in bitstrings:
        bs           = ''.join(str(b) for b in bits)
        counts[bs]   = counts.get(bs, 0) + 1

    print(f'  done — {len(bitstrings)} shots in {rt:.1f}s')
    return {'counts': counts, 'shots': len(bitstrings),
            'runtime_s': rt, 'processor': processor_id}


# ── Full Hardware Pipeline ─────────────────────────────────────────────────────

def run_hardware(G: nx.Graph,
                 ising_model    : IsingModel,
                 gamma          : list,
                 beta           : list,
                 p              : int            = 1,
                 exact_optimum  : Optional[float] = None,
                 shots          : int            = 1000,
                 processor_id   : str            = 'Ankaa-3',
                 save_dir       : str            = 'results/hardware_logs',
                 dry_run        : bool           = True) -> dict:
    """
    End-to-end hardware pipeline.

    dry_run=True  → build and inspect program without submitting (default: safe)
    dry_run=False → submit to QCS (requires active credentials)
    """
    zz, z    = ising_to_qaoa_coefficients(ising_model)
    n_qubits = ising_model.num_qubits()
    nodes    = ising_model.nodes

    # Gate budget check
    est = estimate_native_gate_count(zz, n_qubits, p)
    print('\n=== Native gate count estimate ===')
    print(f'  Two-qubit gates : {est["two_qubit_gates"]}  '
          f'({"✅ within budget" if est["within_budget"] else "❌ EXCEEDS 100 — reduce p or subgraph"})')
    print(f'  Single-qubit    : {est["single_qubit_gates"]}')
    print(f'  Total           : {est["total_gates"]}')

    if not est['within_budget'] and not dry_run:
        raise ValueError(
            f'Two-qubit gate count {est["two_qubit_gates"]} exceeds ~100. '
            f'Reduce p or subgraph size before submitting to hardware.'
        )

    prog = build_pyquil_program(zz, z, n_qubits, gamma, beta, p)
    print(f'\n  Program built — {len(prog.instructions)} instructions')

    if dry_run:
        print('\n  DRY RUN — not submitted.  First 25 instructions:')
        for inst in list(prog.instructions)[:25]:
            print(f'    {inst}')
        return {'dry_run': True, 'gate_estimate': est,
                'n_instructions': len(prog.instructions)}

    # Live submission
    hw = submit_to_qcs(prog, processor_id, shots)

    # Parse results
    best_cut, best_part = -np.inf, {}
    for bitstring, count in hw['counts'].items():
        if len(bitstring) != n_qubits:
            continue
        spins = {nodes[i]: (1 if bitstring[i] == '0' else -1)
                 for i in range(n_qubits)}
        cut   = sum(d['weight'] for u, v, d in G.edges(data=True)
                    if spins[u] != spins[v]) / 2.0
        if cut > best_cut:
            best_cut, best_part = cut, spins

    approx_ratio = (best_cut / exact_optimum) if exact_optimum else None

    log = {
        'processor'       : processor_id,
        'p'               : p,
        'gamma'           : gamma,
        'beta'            : beta,
        'shots'           : hw['shots'],
        'runtime_s'       : hw['runtime_s'],
        'best_cut_value'  : best_cut,
        'approx_ratio'    : approx_ratio,
        'exact_optimum'   : exact_optimum,
        'gate_estimate'   : est,
        'bitstring_counts': hw['counts'],
        'timestamp'       : time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(save_dir) / f'rigetti_job_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f'\n  best cut : {best_cut:.4f}')
    print(f'  approx r : {approx_ratio}')
    print(f'  log saved: {log_path}')
    return log


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from src.graph_utils import load_graph, bfs_subgraph
    from src.qubo_builder import build_ising, QUBOParams

    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true',
                        help='Submit to real Rigetti hardware (default: dry run)')
    parser.add_argument('--shots', type=int, default=1000)
    parser.add_argument('--p', type=int, default=1)
    args = parser.parse_args()

    # Load optimised angles from simulator
    sim_path = f'results/simulator_logs/qaoa_p{args.p}_hw8_pure.json'
    try:
        with open(sim_path) as f:
            sim = json.load(f)
        gamma, beta = sim['optimal_gamma'], sim['optimal_beta']
        print(f'Loaded angles from {sim_path}')
        print(f'  γ = {[f"{v:.4f}" for v in gamma]}')
        print(f'  β = {[f"{v:.4f}" for v in beta]}')
    except FileNotFoundError:
        print(f'No simulator result at {sim_path} — using default angles.')
        gamma, beta = [0.5] * args.p, [0.3] * args.p

    GA    = load_graph('data/problemA.parquet')
    hwG   = bfs_subgraph(GA, target_size=8)
    model = build_ising(hwG, QUBOParams.pure_maxcut())

    run_hardware(
        hwG, model,
        gamma=gamma, beta=beta, p=args.p,
        shots=args.shots,
        dry_run=not args.submit
    )
