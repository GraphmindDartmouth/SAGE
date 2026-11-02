"""Diffuse signed knowledge with the heat kernel of the graph Laplacian."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .common import adjacency


# Func: Build the sparse graph Laplacian for heat diffusion.
# Input: Undirected line graph, node order, and diffusion time.
# Output: Negative time-scaled Laplacian.
def build_operator(graph: nx.Graph, node_ids: list[str], time: float) -> sp.csr_matrix:
    if not math.isfinite(time) or time < 0:
        raise ValueError("propagation.time must be finite and nonnegative")
    matrix = adjacency(graph, node_ids)
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    return (-time * (sp.diags(degrees) - matrix)).tocsr()


# Func: Apply the heat-kernel matrix exponential to signed knowledge.
# Input: Time-scaled Laplacian and current knowledge vector.
# Output: Diffused signal in the original node order.
def propagate(operator: sp.csr_matrix, values: np.ndarray) -> np.ndarray:
    return np.asarray(spla.expm_multiply(operator, values), dtype=np.float64)
