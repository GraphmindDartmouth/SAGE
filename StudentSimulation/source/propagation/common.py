"""Share aligned graph matrices, distance weights, and virtual Beta updates."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import scipy.sparse as sp

from source.scoring.beta import add_evidence, knowledge


# Func: Build sparse undirected adjacency in the supplied concept node order.
# Input: Line graph and ordered node IDs.
# Output: Float64 CSR adjacency with no self-loops.
def adjacency(graph: nx.Graph, node_ids: list[str]) -> sp.csr_matrix:
    matrix = sp.csr_matrix(nx.to_scipy_sparse_array(graph, nodelist=node_ids, format="csr", dtype=np.float64))
    matrix.setdiag(0)
    matrix.eliminate_zeros()
    return matrix


# Func: Construct the signed knowledge vector from current Beta parameters.
# Input: Belief table and ordered node IDs.
# Output: Float64 vector of signed knowledge values.
def knowledge_vector(beliefs: dict[str, dict[str, float]], node_ids: list[str]) -> np.ndarray:
    return np.asarray([knowledge(beliefs[node]["alpha"], beliefs[node]["beta"]) for node in node_ids], dtype=np.float64)


# Func: Weight nodes by inverse shortest distance to directly tested concepts.
# Input: Graph, tested node IDs, and full node order.
# Output: Weight vector equal to 1/(distance+1), or zero if unreachable.
def distance_weights(graph: nx.Graph, tested: set[str], node_ids: list[str]) -> np.ndarray:
    sources = [node for node in tested if node in graph]
    if not sources:
        return np.zeros(len(node_ids), dtype=np.float64)
    distances = nx.multi_source_dijkstra_path_length(graph, sources, weight=None)
    return np.asarray([1.0 / (distances[node] + 1.0) if node in distances else 0.0 for node in node_ids])


# Func: Add signed virtual evidence to nodes outside the directly tested set.
# Input: Beliefs, graph, tested nodes, node order, propagated signal, and strength.
# Output: Count of nodes whose Beta parameters changed.
def apply_virtual_evidence(
    beliefs: dict[str, dict[str, float]], graph: nx.Graph, tested: set[str],
    node_ids: list[str], signal: np.ndarray, strength: float,
) -> int:
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("propagation.strength must be finite and nonnegative")
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(values) != len(node_ids) or not np.isfinite(values).all():
        raise ValueError("Propagator returned a non-finite or misaligned signal")
    weights = distance_weights(graph, tested, node_ids)
    changed = 0
    for index, node in enumerate(node_ids):
        if node in tested:
            continue
        increment = strength * values[index] * weights[index]
        if increment != 0.0:
            add_evidence(beliefs[node], increment)
            changed += 1
    return changed
