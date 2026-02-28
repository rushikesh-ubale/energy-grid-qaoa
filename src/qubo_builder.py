"""
qubo_builder.py
---------------
Multi-objective QUBO / Ising Hamiltonian builder.

Objective (maximise):
    F(z) = C_w̃(z)  −  P_bal(z)  −  P_frag(z)

    C_w̃(z)   = ½ Σ_{(i,j)∈E} w̃_ij (1 − z_i z_j)       admittance-weighted MaxCut
    w̃_ij     = w_ij · (1 + β · C_ij)                     criticality-adjusted weight
    P_bal(z)  = λ_bal · (Σ_i z_i)²                        balance penalty  (2-local)
    P_frag(z) = λ_frag · Σ_{(i,j)∈E} α_ij (1 − z_i z_j)  fragmentation surrogate (2-local)

Setting λ_bal = λ_frag = β = 0 recovers pure MaxCut — use this for hackathon scoring.
"""

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional


# ── Hyperparameters ────────────────────────────────────────────────────────────

@dataclass
class QUBOParams:
    lambda_bal  : float = 0.10   # balance penalty weight
    lambda_frag : float = 0.05   # fragmentation surrogate weight
    beta        : float = 0.50   # criticality emphasis (edge betweenness scaling)

    @classmethod
    def pure_maxcut(cls) -> 'QUBOParams':
        """Zero all penalties → recovers standard MaxCut (hackathon mode)."""
        return cls(lambda_bal=0.0, lambda_frag=0.0, beta=0.0)


# ── Ising Model ────────────────────────────────────────────────────────────────

@dataclass
class IsingModel:
    """
    H = Σ_{i<j} J_ij z_i z_j + Σ_i h_i z_i + offset   (minimisation form)

    We store H = −F so that minimising H maximises the cut objective.
    """
    nodes  : list
    J      : dict = field(default_factory=dict)   # {(node_i, node_j): coupling}
    h      : dict = field(default_factory=dict)   # {node_i: local field}
    offset : float = 0.0

    def num_qubits(self) -> int:
        return len(self.nodes)

    def node_index(self, node) -> int:
        return self.nodes.index(node)

    def to_matrix(self) -> np.ndarray:
        """Full n×n coupling matrix (symmetric, zero diagonal)."""
        n   = len(self.nodes)
        mat = np.zeros((n, n))
        for (ni, nj), val in self.J.items():
            i, j = self.node_index(ni), self.node_index(nj)
            mat[i, j] = mat[j, i] = val
        return mat

    def evaluate(self, z: dict) -> float:
        """Ising energy H(z) for spin assignment {node: ±1}."""
        e = self.offset
        for (ni, nj), Jval in self.J.items():
            e += Jval * z[ni] * z[nj]
        for ni, hval in self.h.items():
            e += hval * z[ni]
        return e

    def cut_from_spins(self, G: nx.Graph, z: dict) -> float:
        """Recover original C(z) cut value from a spin assignment."""
        val = 0.0
        for u, v, data in G.edges(data=True):
            val += data['weight'] * (1 - z[u] * z[v])
        return val / 2.0

    def summary(self):
        print(f"IsingModel  |  {self.num_qubits()} qubits  "
              f"|  {len(self.J)} couplings  |  {len(self.h)} local fields")
        if self.J:
            print(f"  max |J|  = {max(abs(v) for v in self.J.values()):.6f}")
        if self.h:
            print(f"  max |h|  = {max(abs(v) for v in self.h.values()):.6f}")
        print(f"  offset   = {self.offset:.6f}")


# ── Helper Functions ───────────────────────────────────────────────────────────

def compute_edge_criticality(G: nx.Graph) -> dict:
    """
    Normalised edge betweenness centrality → C_ij ∈ [0, 1].
    Higher = more critical backbone edge.
    """
    ebc   = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
    max_v = max(ebc.values()) if ebc else 1.0
    return {edge: val / max_v for edge, val in ebc.items()}


def compute_fragmentation_weights(G: nx.Graph) -> dict:
    """
    α_ij = 1 / (deg(i) + deg(j)).
    Penalises cuts that isolate low-degree (vulnerable) nodes.
    """
    deg   = dict(G.degree())
    return {(u, v): 1.0 / max(deg[u] + deg[v], 1) for u, v in G.edges()}


# ── Core Builder ───────────────────────────────────────────────────────────────

def build_ising(G: nx.Graph,
                params: Optional[QUBOParams] = None) -> IsingModel:
    """
    Build the multi-objective Ising Hamiltonian H = −F(z).

    All penalty terms are 2-local so the result is directly Ising/QUBO-compatible.
    Pass QUBOParams.pure_maxcut() to recover standard MaxCut with no penalties.
    """
    if params is None:
        params = QUBOParams()

    nodes = sorted(G.nodes())
    n     = len(nodes)

    # ── adjusted weights w̃_ij ─────────────────────────────────────────────────
    criticality = compute_edge_criticality(G)
    w_tilde = {}
    for u, v, data in G.edges(data=True):
        cij            = criticality.get((u, v), criticality.get((v, u), 0.0))
        w_tilde[(u, v)] = data['weight'] * (1.0 + params.beta * cij)

    # ── fragmentation weights α_ij ─────────────────────────────────────────────
    alpha = compute_fragmentation_weights(G)

    # ── Assemble J and h from H = −F ──────────────────────────────────────────
    #
    # F(z) = C_w̃ − P_bal − P_frag
    #
    # C_w̃  = ½ Σ w̃_ij − ½ Σ w̃_ij z_i z_j
    #   → constant ½Σw̃_ij   and   J^cut_{ij} = −½ w̃_ij  (for maximisation)
    #   → in H = −F:  J^cut_{ij} = +½ w̃_ij
    #
    # P_bal = λ_bal (Σ z_i)²
    #       = λ_bal n  +  2λ_bal Σ_{i<j} z_i z_j
    #   → constant λ_bal·n   and   J^bal_{ij} = +2λ_bal  for all pairs
    #   → in H = −F:  J^bal sign flips → −2λ_bal,
    #                 but P_bal enters H with + sign → +2λ_bal
    #
    # P_frag = λ_frag Σ α_ij − λ_frag Σ α_ij z_i z_j
    #   → in H = −F:  J^frag_{ij} = +λ_frag · α_ij

    J      : dict = {}
    h      : dict = {}
    offset : float = 0.0

    # Constants
    offset += 0.5  * sum(w_tilde.values())          # from C_w̃
    offset += params.lambda_frag * sum(alpha.values())  # from P_frag
    offset += params.lambda_bal  * n                    # from P_bal (z_i²=1)

    # Edge-local couplings
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        wt  = w_tilde[(u, v)]
        a   = alpha.get((u, v), alpha.get((v, u), 0.0))
        J[key] = J.get(key, 0.0) + 0.5 * wt + params.lambda_frag * a

    # Dense balance penalty — all pairs
    for i in range(n):
        for j in range(i + 1, n):
            key    = (nodes[i], nodes[j])
            J[key] = J.get(key, 0.0) + 2.0 * params.lambda_bal

    return IsingModel(nodes=nodes, J=J, h=h, offset=offset)


# ── QAOA Coefficient Extraction ────────────────────────────────────────────────

def ising_to_qaoa_coefficients(model: IsingModel) -> tuple:
    """
    Extract QAOA-ready coefficients from an IsingModel.

    Returns:
        zz_coeffs : {(qubit_i, qubit_j): gamma_coeff}   for RZZ cost-layer gates
        z_coeffs  : {qubit_i: gamma_coeff}               for RZ gates (local fields)

    Indices are positional into model.nodes.
    """
    zz_coeffs : dict = {}
    z_coeffs  : dict = {i: 0.0 for i in range(model.num_qubits())}

    for (ni, nj), Jval in model.J.items():
        i   = model.node_index(ni)
        j   = model.node_index(nj)
        key = (min(i, j), max(i, j))
        zz_coeffs[key] = zz_coeffs.get(key, 0.0) + Jval

    for ni, hval in model.h.items():
        z_coeffs[model.node_index(ni)] += hval

    return zz_coeffs, z_coeffs


# ── Verification ───────────────────────────────────────────────────────────────

def verify_ising(G: nx.Graph, model: IsingModel,
                 exact_optimum: Optional[float] = None) -> dict:
    """
    Enumerate all 2^n spin assignments (n ≤ 20) and confirm the Ising minimum
    corresponds to the maximum cut.  Returns a result dict.
    """
    nodes = model.nodes
    n     = len(nodes)
    assert n <= 20, f"n={n} too large for enumeration — use random sampling instead"

    best_ising = np.inf
    best_cut   = -np.inf
    best_z     = None

    for bitmask in range(1 << n):
        z = {nodes[i]: (1 if (bitmask >> i) & 1 else -1) for i in range(n)}
        e = model.evaluate(z)
        c = model.cut_from_spins(G, z)
        if e < best_ising:
            best_ising = e
            best_z     = z
        best_cut = max(best_cut, c)

    recovered = model.cut_from_spins(G, best_z)
    match     = abs(recovered - best_cut) < 1e-4 if exact_optimum is None \
                else abs(recovered - exact_optimum) < 1e-4

    return {
        'best_ising_energy' : best_ising,
        'recovered_cut'     : recovered,
        'enum_best_cut'     : best_cut,
        'exact_optimum'     : exact_optimum,
        'verified'          : match,
    }


# ── CLI smoke-test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from src.graph_utils import load_graph, bfs_subgraph

    G    = load_graph('data/problemA.parquet')
    subG = bfs_subgraph(G, target_size=8)

    # Pure MaxCut
    model_pure = build_ising(subG, QUBOParams.pure_maxcut())
    model_pure.summary()

    # Multi-objective
    model_multi = build_ising(subG, QUBOParams(lambda_bal=0.1, lambda_frag=0.05, beta=0.5))
    model_multi.summary()

    zz, z = ising_to_qaoa_coefficients(model_pure)
    print(f"\nQAOA ZZ terms: {len(zz)}  |  Z terms: {sum(1 for v in z.values() if abs(v) > 1e-9)}")

    res = verify_ising(subG, model_pure)
    print(f"Verification: recovered_cut={res['recovered_cut']:.4f}  verified={res['verified']}")
