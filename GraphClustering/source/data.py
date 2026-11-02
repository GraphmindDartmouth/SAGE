"""Load and validate a GraphConstruction line graph and aligned SAGE matrix."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp


@dataclass(frozen=True)
class GraphData:
    node_ids: list[str]
    graph: nx.Graph
    adjacency: sp.csr_matrix
    features: np.ndarray


# Func: Convert a MATLAB cell or character value into one node identifier.
# Input: A scalar string or one-element NumPy array from a .mat file.
# Output: A Python string.
def matlab_string(value: object) -> str:
    while isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Matrix node_ids must contain one string per row")
        value = value.item()
    return str(value)


# Func: Load the graph, node order, and matrix with strict alignment checks.
# Input: Paths to line_graph.mat, line_graph.json, and node_ids.json.
# Output: Validated graph, adjacency, and semantic feature matrix.
def load_graph_data(matrix_path: Path, graph_path: Path, node_ids_path: Path) -> GraphData:
    for path in (matrix_path, graph_path, node_ids_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required input does not exist: {path}")
    matrix = sio.loadmat(matrix_path)
    if "fea" not in matrix or "W" not in matrix:
        raise ValueError("The .mat file must contain fea and W")
    features = matrix["fea"]
    features = features.toarray() if sp.issparse(features) else np.asarray(features)
    features = np.asarray(features, dtype=np.float64)
    adjacency = sp.csr_matrix(matrix["W"], dtype=np.float64)
    node_ids = json.loads(node_ids_path.read_text(encoding="utf-8"))
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_ids = [str(node["id"]) for node in payload["nodes"]]
    if not isinstance(node_ids, list) or any(not isinstance(item, str) for item in node_ids):
        raise ValueError("node_ids.json must be a list of strings")
    if not node_ids or len(set(node_ids)) != len(node_ids) or node_ids != graph_ids:
        raise ValueError("node_ids.json must match the unique line_graph.json node order")
    if "node_ids" in matrix:
        matrix_ids = [matlab_string(item) for item in matrix["node_ids"].reshape(-1)]
        if matrix_ids != node_ids:
            raise ValueError("Matrix node_ids do not match node_ids.json")
    n_nodes = len(node_ids)
    if features.ndim != 2 or features.shape[0] != n_nodes or features.shape[1] == 0:
        raise ValueError("The fea rows must match graph nodes and have at least one feature")
    if not np.isfinite(features).all():
        raise ValueError("The fea matrix contains non-finite values")
    if adjacency.shape != (n_nodes, n_nodes) or not np.isfinite(adjacency.data).all():
        raise ValueError("The W matrix must be finite and aligned with graph nodes")
    if np.any(adjacency.data < 0) or (adjacency - adjacency.T).nnz:
        raise ValueError("The W matrix must be nonnegative and symmetric")
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for link in payload.get("links", []):
        source, target = str(link["source"]), str(link["target"])
        if source not in graph or target not in graph:
            raise ValueError("A graph link refers to an unknown node")
        if source != target:
            graph.add_edge(source, target)
    index = {node_id: row for row, node_id in enumerate(node_ids)}
    graph_edges = {tuple(sorted((index[u], index[v]))) for u, v in graph.edges()}
    matrix_edges = {tuple(sorted((int(u), int(v)))) for u, v in zip(*sp.triu(adjacency).nonzero())}
    if graph_edges != matrix_edges:
        raise ValueError("The W adjacency does not match line_graph.json links")
    return GraphData(node_ids, graph, adjacency, features)
