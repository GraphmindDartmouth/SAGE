"""Diffuse signed knowledge with the PPR/APPNP random-walk resolvent."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .common import adjacency


# Func: Factor the paper's PPR/APPNP linear system once per graph.
# Input: Undirected line graph, node order, and restart probability.
# Output: Sparse LU factorization and restart weight.
def build_operator(graph: nx.Graph, node_ids: list[str], restart: float) -> tuple[object, float]:
    if not 0 < restart <= 1:
        raise ValueError("propagation.restart must be in (0, 1]")
    matrix = adjacency(graph, node_ids)
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    degrees[degrees == 0] = 1.0
    walk = sp.diags(1.0 / degrees) @ matrix
    system = sp.eye(len(node_ids), format="csc") - (1.0 - restart) * walk.T.tocsc()
    return spla.splu(system), restart


# Func: Apply anchored PPR/APPNP diffusion to signed knowledge.
# Input: Factored operator and current knowledge vector.
# Output: Diffused signal in the original node order.
def propagate(operator: tuple[object, float], values: np.ndarray) -> np.ndarray:
    factorization, restart = operator
    return np.asarray(factorization.solve(restart * values), dtype=np.float64)
