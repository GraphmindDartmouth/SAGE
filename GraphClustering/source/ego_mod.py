"""Filter Ego boundary concepts by SAGE's normalized modularity gain."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .ego import boundary_nodes
from .overlap import duplication_rate, ego_memberships, search_parameter


# Func: Score each selected boundary node for adding a secondary cluster.
# Input: Graph, node order, raw features, hard labels, and boundary IDs.
# Output: Normalized modularity gain for each eligible node.
def modularity_gains(
    graph: nx.Graph,
    node_ids: list[str],
    features: np.ndarray,
    labels: np.ndarray,
    selected: set[str],
) -> dict[str, float]:
    n_clusters = int(labels.max()) + 1
    centroids = np.zeros((n_clusters, features.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_clusters)
    np.add.at(centroids, labels, features)
    present = counts > 0
    centroids[present] /= counts[present, None]
    label_by_node = dict(zip(node_ids, labels.tolist()))
    cluster_degree = {cluster: 0 for cluster in range(n_clusters)}
    internal_edges = {cluster: 0 for cluster in range(n_clusters)}
    for node_id in node_ids:
        cluster_degree[label_by_node[node_id]] += graph.degree(node_id)
    for source, target in graph.edges():
        if label_by_node[source] == label_by_node[target]:
            internal_edges[label_by_node[source]] += 1
    total_edges = graph.number_of_edges()
    gains: dict[str, float] = {}
    for index, node_id in enumerate(node_ids):
        if node_id not in selected:
            continue
        degree = graph.degree(node_id)
        if total_edges == 0 or degree == 0:
            gains[node_id] = float("-inf")
            continue
        distances = np.linalg.norm(centroids - features[index], axis=1)
        distances[~present] = np.inf
        distances[labels[index]] = np.inf
        secondary = int(np.argmin(distances))
        edge_count = internal_edges[secondary]
        degree_sum = cluster_degree[secondary]
        neighbor_count = sum(label_by_node[neighbor] == secondary for neighbor in graph.neighbors(node_id))
        original = edge_count / total_edges - (degree_sum / (2.0 * total_edges)) ** 2
        added = (edge_count + neighbor_count) / total_edges - ((degree_sum + degree) / (2.0 * total_edges)) ** 2
        gains[node_id] = (added - original) / degree
    return gains


# Func: Tune EgoMod's gain cutoff to a target duplicated-node fraction.
# Input: Graph, node order, raw features, labels, and search settings.
# Output: Overlapping memberships, selected cutoff, and achieved rate.
def cluster(
    graph: nx.Graph,
    node_ids: list[str],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    fixed_percentile: float,
    target_duplication_rate: float,
    search_iterations: int,
    search_tolerance: float,
) -> tuple[dict[int, list[int]], float, float]:
    if not 0 <= fixed_percentile <= 100:
        raise ValueError("method.fixed_percentile must be between 0 and 100")
    selected = boundary_nodes(graph, node_ids, features, labels, fixed_percentile)
    gains = modularity_gains(graph, node_ids, features, labels, selected)
    finite = [value for value in gains.values() if np.isfinite(value)]
    lower, upper = (min(finite), max(finite)) if finite else (0.0, 0.0)

    # Func: Build EgoMod memberships at one modularity cutoff.
    # Input: A candidate cutoff.
    # Output: Cluster memberships for all concept nodes.
    def memberships_at(cutoff: float) -> dict[int, list[int]]:
        accepted = {node_id for node_id, gain in gains.items() if np.isfinite(gain) and gain >= cutoff}
        return ego_memberships(graph, node_ids, labels, accepted)

    # Func: Evaluate duplication at one modularity cutoff.
    # Input: A candidate cutoff.
    # Output: The duplicated-node fraction.
    def rate_at(cutoff: float) -> float:
        return duplication_rate(memberships_at(cutoff), len(node_ids))

    cutoff, rate = search_parameter(
        rate_at, target_duplication_rate, lower, upper, False, search_iterations, search_tolerance
    )
    return memberships_at(cutoff), cutoff, rate
