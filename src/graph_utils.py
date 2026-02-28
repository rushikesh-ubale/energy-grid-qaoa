"""
graph_utils.py
--------------
Graph loading, subgraph extraction, and visualization utilities.
All functions return NetworkX graphs with integer nodes and 'weight' edge attributes.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Optional


# ── Loading ────────────────────────────────────────────────────────────────────

def load_graph(path: str | Path) -> nx.Graph:
    """
    Load a weighted graph from a .parquet edge list.
    Expects columns: node_1, node_2, weight.
    """
    df = pd.read_parquet(path)
    assert {'node_1', 'node_2', 'weight'}.issubset(df.columns), \
        f"Expected node_1, node_2, weight — got {list(df.columns)}"
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(int(row['node_1']), int(row['node_2']), weight=float(row['weight']))
    return G


def graph_summary(G: nx.Graph, label: str = "") -> dict:
    """Return a dict of key topology metrics."""
    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    stats = {
        'label'           : label,
        'nodes'           : G.number_of_nodes(),
        'edges'           : G.number_of_edges(),
        'density'         : nx.density(G),
        'connected'       : nx.is_connected(G),
        'num_components'  : nx.number_connected_components(G),
        'avg_clustering'  : nx.average_clustering(G),
        'weight_mean'     : float(np.mean(weights)),
        'weight_std'      : float(np.std(weights)),
        'weight_min'      : float(np.min(weights)),
        'weight_max'      : float(np.max(weights)),
        'weight_cv'       : float(np.std(weights) / np.mean(weights)),
        'total_admittance': float(np.sum(weights)),
    }
    if nx.is_connected(G):
        stats['diameter']          = nx.diameter(G)
        stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
    return stats


# ── Subgraph Extraction ────────────────────────────────────────────────────────

def bfs_subgraph(G: nx.Graph, target_size: int = 8,
                 seed_node: Optional[int] = None) -> nx.Graph:
    """
    Extract a connected subgraph of exactly `target_size` nodes via BFS.
    Seeds from the highest weighted-degree node unless seed_node is provided.
    Neighbours are visited in descending edge-weight order so the extracted
    subgraph captures the densest local structure.
    """
    if seed_node is None:
        seed_node = max(dict(G.degree(weight='weight')).items(),
                        key=lambda x: x[1])[0]

    visited  = [seed_node]
    frontier = [seed_node]

    while len(visited) < target_size and frontier:
        current = frontier.pop(0)
        neighbours = sorted(
            G.neighbors(current),
            key=lambda nb: G[current][nb]['weight'],
            reverse=True
        )
        for nb in neighbours:
            if nb not in visited:
                visited.append(nb)
                frontier.append(nb)
            if len(visited) >= target_size:
                break

    subG = G.subgraph(visited).copy()
    assert nx.is_connected(subG), \
        "BFS subgraph is disconnected — increase target_size or change seed_node"
    return subG


def community_subgraphs(G: nx.Graph, max_size: int = 10) -> list:
    """
    Decompose G using greedy modularity communities.
    Communities exceeding max_size are recursively split via BFS chunks.
    Returns a list of connected subgraphs each with <= max_size nodes.
    """
    from networkx.algorithms.community import greedy_modularity_communities

    communities = list(greedy_modularity_communities(G, weight='weight'))
    subgraphs   = []

    for comm in communities:
        subG = G.subgraph(comm).copy()
        if not nx.is_connected(subG):
            largest = max(nx.connected_components(subG), key=len)
            subG    = subG.subgraph(largest).copy()

        if subG.number_of_nodes() <= max_size:
            subgraphs.append(subG)
        else:
            remaining = list(subG.nodes())
            while len(remaining) >= 4:
                chunk_size = min(max_size, len(remaining))
                try:
                    chunk = bfs_subgraph(G.subgraph(remaining).copy(),
                                         target_size=chunk_size,
                                         seed_node=remaining[0])
                    subgraphs.append(chunk)
                    remaining = [n for n in remaining if n not in chunk.nodes()]
                except Exception:
                    break

    return subgraphs


# ── Cut Utilities ──────────────────────────────────────────────────────────────

def cut_value(G: nx.Graph, partition: dict) -> float:
    """
    Compute C(z) = 0.5 * Σ_{(i,j)∈E} w_ij * (1 - z_i * z_j)
    where partition maps node → ±1.
    """
    val = 0.0
    for u, v, data in G.edges(data=True):
        val += data['weight'] * (1 - partition[u] * partition[v])
    return val / 2.0


def approximation_ratio(achieved: float, optimal: float) -> float:
    return achieved / optimal if optimal > 0 else 0.0


def bitstring_to_partition(bitstring: str, nodes: list) -> dict:
    """'0101...' → {node: ±1}  (0 → +1, 1 → −1)"""
    return {nodes[i]: (1 if b == '0' else -1) for i, b in enumerate(bitstring)}


# ── Visualisation ──────────────────────────────────────────────────────────────

_PALETTE = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A', '#F4A261', '#264653']


def _node_sizes(G: nx.Graph, base: int = 350, scale: int = 650) -> list:
    w = dict(G.degree(weight='weight'))
    mx = max(w.values()) if w else 1
    return [base + scale * (w[n] / mx) for n in G.nodes()]


def plot_graph_communities(G: nx.Graph, communities, title: str = "",
                           save_path: Optional[str] = None):
    """Graph coloured by community, edge width ∝ admittance weight."""
    node_comm = {}
    for idx, comm in enumerate(communities):
        for n in comm:
            node_comm[n] = idx

    node_colors = [_PALETTE[node_comm.get(n, 0) % len(_PALETTE)] for n in G.nodes()]
    weights     = [G[u][v]['weight'] for u, v in G.edges()]
    max_w       = max(weights) if weights else 1
    edge_widths = [0.5 + 3.5 * (w / max_w) for w in weights]

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=_node_sizes(G), alpha=0.92, ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.45,
                           edge_color='gray', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white',
                            font_weight='bold', ax=ax)

    legend = [mpatches.Patch(facecolor=_PALETTE[i % len(_PALETTE)],
                             label=f'Community {i+1} ({len(list(communities)[i])} nodes)')
              for i in range(len(communities))]
    ax.legend(handles=legend, loc='upper left', fontsize=9)
    ax.set_title(title or f'{G.number_of_nodes()} nodes · {G.number_of_edges()} edges',
                 fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_optimal_cut(G: nx.Graph, partition: dict, cut_val: float,
                     title: str = "", save_path: Optional[str] = None):
    """Visualise a MaxCut partition — cut edges in gold, non-cut edges dashed."""
    pos         = nx.spring_layout(G, seed=42)
    node_colors = ['#E63946' if partition.get(n, 1) == 1 else '#457B9D'
                   for n in G.nodes()]
    cut_edges   = [(u, v) for u, v in G.edges()
                   if partition.get(u) != partition.get(v)]
    non_cut     = [(u, v) for u, v in G.edges()
                   if partition.get(u) == partition.get(v)]

    fig, ax = plt.subplots(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=_node_sizes(G), alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges,
                           edge_color='gold', width=2.5, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=non_cut,
                           edge_color='gray', width=0.8,
                           style='dashed', alpha=0.4, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white',
                            font_weight='bold', ax=ax)

    legend = [
        mpatches.Patch(facecolor='#E63946', label='Partition +1'),
        mpatches.Patch(facecolor='#457B9D', label='Partition −1'),
        Line2D([0], [0], color='gold', linewidth=2.5,
               label=f'Cut edges ({len(cut_edges)})'),
    ]
    ax.legend(handles=legend, loc='upper left', fontsize=10)
    ax.set_title(title or f'MaxCut partition  |  cut = {cut_val:.4f}', fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
