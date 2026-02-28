"""
classical_solvers.py
--------------------
Classical MaxCut solvers used as benchmarks against QAOA.

All solvers return cut_value = sum of weights of cut edges
(each undirected edge counted once).
"""

import time
import csv
import numpy as np
import networkx as nx
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    solver: str
    cut_value: float
    partition: dict
    runtime_s: float
    approx_ratio: Optional[float] = None
    notes: str = ""

    def to_dict(self):
        d = asdict(self)
        d["partition"] = {str(k): v for k, v in self.partition.items()}
        return d


# ─────────────────────────────────────────────────────────────
# 1. Exact Brute Force
# ─────────────────────────────────────────────────────────────

def exact_brute_force(G: nx.Graph) -> SolverResult:
    nodes = list(G.nodes())
    n = len(nodes)
    assert n <= 22, f"Brute force infeasible for n={n}"

    node_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v], data['weight'])
             for u, v, data in G.edges(data=True)]

    t0 = time.perf_counter()
    best_val = -np.inf
    best_mask = 0

    for bitmask in range(1 << n):
        cut = 0.0
        for i, j, w in edges:
            if ((bitmask >> i) & 1) != ((bitmask >> j) & 1):
                cut += w
        if cut > best_val:
            best_val = cut
            best_mask = bitmask

    partition = {nodes[i]: (1 if (best_mask >> i) & 1 else -1)
                 for i in range(n)}

    return SolverResult(
        solver="exact_brute_force",
        cut_value=best_val,
        partition=partition,
        runtime_s=time.perf_counter() - t0,
        approx_ratio=1.0
    )


# ─────────────────────────────────────────────────────────────
# 2. Greedy One Exchange
# ─────────────────────────────────────────────────────────────

def greedy_one_exchange(G: nx.Graph, seed: int = 42) -> SolverResult:
    from networkx.algorithms.approximation import one_exchange

    t0 = time.perf_counter()
    cut_size, part_set = one_exchange(G, weight="weight", seed=seed)

    partition = {node: (1 if node in part_set else -1)
                 for node in G.nodes()}

    return SolverResult(
        solver="greedy_one_exchange",
        cut_value=cut_size,
        partition=partition,
        runtime_s=time.perf_counter() - t0
    )


# ─────────────────────────────────────────────────────────────
# 3. Simulated Annealing (FIXED)
# ─────────────────────────────────────────────────────────────

def simulated_annealing(
    G: nx.Graph,
    T_start: float = 10.0,
    T_end: float = 0.01,
    cooling: float = 0.995,
    max_iter: int = 50000,
    seed: int = 42
) -> SolverResult:

    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = len(nodes)

    spins = {node: int(rng.choice([-1, 1])) for node in nodes}

    # correct full cut (NO /2)
    def full_cut(s):
        return sum(d["weight"] for u, v, d in G.edges(data=True)
                   if s[u] != s[v])

    # correct delta (NO /2)
    def delta_flip(node, s):
        d = 0.0
        for nb in G.neighbors(node):
            w = G[node][nb]["weight"]
            d += w if s[node] == s[nb] else -w
        return d

    t0 = time.perf_counter()

    current_cut = full_cut(spins)
    best_cut = current_cut
    best_spins = spins.copy()
    T = T_start

    for _ in range(max_iter):
        node = nodes[int(rng.integers(n))]
        delta = delta_flip(node, spins)

        if delta > 0 or rng.random() < np.exp(delta / T):
            spins[node] *= -1
            current_cut += delta

            if current_cut > best_cut:
                best_cut = current_cut
                best_spins = spins.copy()

        T = max(T * cooling, T_end)

    return SolverResult(
        solver="simulated_annealing",
        cut_value=best_cut,
        partition=best_spins,
        runtime_s=time.perf_counter() - t0,
        notes=f"T_start={T_start}, cooling={cooling}, iter={max_iter}"
    )


# ─────────────────────────────────────────────────────────────
# 4. Goemans–Williamson (FIXED)
# ─────────────────────────────────────────────────────────────

def goemans_williamson(
    G: nx.Graph,
    seed: int = 42,
    n_rounding_trials: int = 200
) -> SolverResult:

    try:
        import cvxpy as cp
    except ImportError:
        raise ImportError("Install cvxpy: pip install cvxpy")

    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()

    Y = cp.Variable((n, n), symmetric=True)
    constraints = [Y >> 0] + [Y[i, i] == 1 for i in range(n)]
    obj_terms = [d["weight"] * (1 - Y[idx[u], idx[v]])
                 for u, v, d in G.edges(data=True)]

    prob = cp.Problem(cp.Maximize(0.25 * sum(obj_terms)), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    Y_val = (Y.value + Y.value.T) / 2 + np.eye(n) * 1e-8
    eigvals, eigvecs = np.linalg.eigh(Y_val)
    L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0)))

    best_cut = -np.inf
    best_part = {}

    for _ in range(n_rounding_trials):
        r = rng.standard_normal(n)
        r /= np.linalg.norm(r)
        signs = np.sign(L @ r)
        signs[signs == 0] = 1

        cut = sum(d["weight"] for u, v, d in G.edges(data=True)
                  if signs[idx[u]] != signs[idx[v]])

        if cut > best_cut:
            best_cut = cut
            best_part = {nodes[i]: int(signs[i]) for i in range(n)}

    return SolverResult(
        solver="goemans_williamson",
        cut_value=best_cut,
        partition=best_part,
        runtime_s=time.perf_counter() - t0,
        notes=f"SDP + {n_rounding_trials} rounding trials"
    )


# ─────────────────────────────────────────────────────────────
# 5. Runner
# ─────────────────────────────────────────────────────────────

def run_all_baselines(
    G: nx.Graph,
    label: str = "graph",
    exact_optimum: Optional[float] = None,
    results_dir: str = "results/tables",
    skip_exact: bool = False,
    skip_gw: bool = False
):

    n = G.number_of_nodes()
    solvers = []

    if not skip_exact and n <= 22:
        solvers.append(("exact_brute_force", lambda: exact_brute_force(G)))

    solvers.append(("greedy_one_exchange", lambda: greedy_one_exchange(G)))
    solvers.append(("simulated_annealing", lambda: simulated_annealing(G)))

    if not skip_gw:
        solvers.append(("goemans_williamson", lambda: goemans_williamson(G)))

    results = []
    print(f"\n{'Solver':<25} {'Cut':>12} {'Approx':>10} {'Time (s)':>10}")
    print("-" * 60)

    for name, fn in solvers:
        r = fn()
        if exact_optimum:
            r.approx_ratio = r.cut_value / exact_optimum
        results.append(r)

        ratio = f"{r.approx_ratio:.4f}" if r.approx_ratio else "-"
        print(f"{name:<25} {r.cut_value:>12.4f} {ratio:>10} {r.runtime_s:>10.3f}")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(results_dir) / f"classical_baselines_{label}.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["solver", "cut_value", "approx_ratio", "runtime_s", "notes"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "solver": r.solver,
                "cut_value": r.cut_value,
                "approx_ratio": r.approx_ratio,
                "runtime_s": r.runtime_s,
                "notes": r.notes
            })

    print(f"\nSaved → {out_path}")
    return results