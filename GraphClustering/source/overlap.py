"""Share SAGE's boundary, duplication, search, and overlapping-cluster metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import networkx as nx
import numpy as np
import scipy.sparse as sp


# Func: Measure each node's distance advantage for its assigned centroid.
# Input: Node features and hard cluster labels.
# Output: One boundary score per node; smaller scores lie nearer a competing cluster.
def boundary_scores(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_clusters = int(labels.max()) + 1
    centroids = np.zeros((n_clusters, features.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_clusters)
    np.add.at(centroids, labels, features)
    present = counts > 0
    centroids[present] /= counts[present, None]
    distances = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
    distances[:, ~present] = np.inf
    primary = distances[np.arange(len(labels)), labels]
    distances[np.arange(len(labels)), labels] = np.inf
    secondary = distances.min(axis=1)
    return primary - secondary


# Func: Find nodes adjacent to at least two different hard clusters.
# Input: An undirected graph, ordered node IDs, and hard cluster labels.
# Output: A boolean structural-boundary mask in node order.
def structural_boundary(graph: nx.Graph, node_ids: list[str], labels: np.ndarray) -> np.ndarray:
    label_by_node = dict(zip(node_ids, labels.tolist()))
    return np.asarray(
        [len({label_by_node[neighbor] for neighbor in graph.neighbors(node_id)}) >= 2 for node_id in node_ids],
        dtype=bool,
    )


# Func: Add connected ego-neighborhood cluster labels for selected nodes.
# Input: Graph, ordered nodes, hard labels, and the selected split-node set.
# Output: One or more integer cluster memberships per node index.
def ego_memberships(
    graph: nx.Graph, node_ids: list[str], labels: np.ndarray, split_ids: set[str]
) -> dict[int, list[int]]:
    label_by_node = dict(zip(node_ids, labels.tolist()))
    memberships: dict[int, list[int]] = {}
    for index, node_id in enumerate(node_ids):
        assigned = {int(labels[index])}
        if node_id in split_ids:
            neighbors = list(graph.neighbors(node_id))
            for component in nx.connected_components(graph.subgraph(neighbors)):
                counts = Counter(label_by_node[neighbor] for neighbor in component)
                if counts:
                    assigned.add(min(counts, key=lambda label: (-counts[label], label)))
        memberships[index] = sorted(assigned)
    return memberships


# Func: Compute the fraction of nodes assigned to multiple clusters.
# Input: Membership lists and total node count.
# Output: A duplication rate between zero and one.
def duplication_rate(memberships: dict[int, list[int]], n_nodes: int) -> float:
    return sum(len(cluster_ids) > 1 for cluster_ids in memberships.values()) / n_nodes


# Func: Tune one monotonic soft-clustering parameter to a duplication target.
# Input: Rate function, target, parameter interval, direction, and search settings.
# Output: Parameter with the closest observed rate and that observed rate.
def search_parameter(
    rate: Callable[[float], float],
    target: float,
    lower: float,
    upper: float,
    increasing: bool,
    iterations: int,
    tolerance: float,
) -> tuple[float, float]:
    if not 0 <= target <= 1 or iterations < 1 or tolerance < 0 or lower > upper:
        raise ValueError("Invalid duplication target or parameter search settings")
    candidates = []
    for bound in (lower, upper):
        value = rate(bound)
        candidates.append((abs(value - target), bound, value))
    for _ in range(iterations):
        middle = (lower + upper) / 2.0
        value = rate(middle)
        candidates.append((abs(value - target), middle, value))
        if abs(value - target) <= tolerance or lower == upper:
            break
        if (value < target) == increasing:
            lower = middle
        else:
            upper = middle
    _, parameter, achieved = min(candidates, key=lambda item: (item[0], item[1]))
    return parameter, achieved


# Func: Summarize overlap and edge coverage without using placeholder labels.
# Input: Undirected adjacency and complete node memberships.
# Output: Overlapping modularity, mean conductance, coverage, and duplication.
def graph_metrics(adjacency: sp.csr_matrix, memberships: dict[int, list[int]]) -> dict[str, float]:
    n_nodes = adjacency.shape[0]
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    two_m = float(degrees.sum())
    if two_m == 0:
        return {"q_ov": 0.0, "conductance": 0.0, "coverage": 0.0, "duplication_rate": duplication_rate(memberships, n_nodes)}
    clusters: dict[int, set[int]] = {}
    for node, cluster_ids in memberships.items():
        for cluster_id in cluster_ids:
            clusters.setdefault(cluster_id, set()).add(node)
    overlap = np.asarray([len(memberships[node]) for node in range(n_nodes)], dtype=np.float64)
    edges = adjacency.tocoo()
    modularity = 0.0
    conductances: list[float] = []
    for nodes in clusters.values():
        indices = np.asarray(sorted(nodes), dtype=int)
        included = np.zeros(n_nodes, dtype=bool)
        included[indices] = True
        inside = included[edges.row] & included[edges.col]
        weighted_edges = np.sum(edges.data[inside] / (overlap[edges.row[inside]] * overlap[edges.col[inside]]))
        weighted_degree = np.sum(degrees[indices] / overlap[indices])
        modularity += float(weighted_edges / two_m - (weighted_degree / two_m) ** 2)
        degree_sum = float(degrees[indices].sum())
        internal = float(edges.data[inside].sum())
        if degree_sum > 0:
            conductances.append((degree_sum - internal) / degree_sum)
    covered = np.asarray(
        [bool(set(memberships[int(u)]) & set(memberships[int(v)])) for u, v in zip(edges.row, edges.col)],
        dtype=bool,
    )
    return {
        "q_ov": modularity,
        "conductance": float(np.mean(conductances)) if conductances else 0.0,
        "coverage": float(edges.data[covered].sum() / two_m),
        "duplication_rate": duplication_rate(memberships, n_nodes),
    }
