"""Diffuse signed knowledge through a finite-order Chebyshev graph filter."""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .common import adjacency


# Func: Compute the Jackson damping weights used by the SAGE benchmark filter.
# Input: Nonnegative Chebyshev polynomial order.
# Output: One damping coefficient per polynomial term.
def jackson_weights(order: int) -> np.ndarray:
    if order == 0:
        return np.ones(1, dtype=np.float64)
    indices = np.arange(order + 1, dtype=np.float64)
    angle = np.pi / (order + 2.0)
    return ((order - indices + 1.0) * np.cos(indices * angle) + np.sin(indices * angle) / np.tan(angle)) / (order + 2.0)


# Func: Scale the graph Laplacian and prepare K filter coefficients.
# Input: Graph, node order, number of terms, optional coefficients, and damping switch.
# Output: Scaled sparse Laplacian and effective coefficients.
def build_operator(
    graph: nx.Graph, node_ids: list[str], terms: int,
    coefficients: list[float] | None, damping: bool,
) -> tuple[sp.csr_matrix, np.ndarray]:
    if terms < 1:
        raise ValueError("propagation.terms must be positive")
    matrix = adjacency(graph, node_ids)
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    laplacian = (sp.diags(degrees) - matrix).tocsr()
    if len(node_ids) > 1 and laplacian.nnz:
        largest = float(spla.eigsh(laplacian, k=1, which="LM", return_eigenvectors=False)[0])
    else:
        largest = 0.0
    largest = max(largest, 1e-8)
    scaled = ((2.0 / largest) * laplacian - sp.eye(len(node_ids), format="csr")).tocsr()
    coeffs = np.asarray(coefficients if coefficients is not None else [1.0 / terms] * terms, dtype=np.float64)
    if len(coeffs) != terms or not np.isfinite(coeffs).all():
        raise ValueError("Chebyshev coefficients must have one finite value per term")
    if damping:
        coeffs = coeffs * jackson_weights(terms - 1)
    return scaled, coeffs


# Func: Apply the Chebyshev recurrence to a signed knowledge vector.
# Input: Scaled Laplacian, coefficients, and current knowledge vector.
# Output: Finite-hop diffused signal in the original node order.
def propagate(operator: tuple[sp.csr_matrix, np.ndarray], values: np.ndarray) -> np.ndarray:
    scaled, coefficients = operator
    result = coefficients[0] * values
    if len(coefficients) == 1:
        return result
    previous = values
    current = scaled @ values
    result = result + coefficients[1] * current
    for index in range(2, len(coefficients)):
        following = 2.0 * (scaled @ current) - previous
        result = result + coefficients[index] * following
        previous, current = current, following
    return np.asarray(result, dtype=np.float64)
