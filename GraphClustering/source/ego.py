"""Assign selected GCC boundary concepts to connected ego-neighborhood clusters."""

from __future__ import annotations

import networkx as nx
import numpy as np

from .overlap import boundary_scores, duplication_rate, ego_memberships, search_parameter, structural_boundary


# Func: Select structural and semantic boundary nodes below a percentile.
# Input: Graph, ordered node IDs, raw features, hard labels, and percentile.
# Output: Node IDs eligible for overlapping Ego memberships.
def boundary_nodes(
    graph: nx.Graph, node_ids: list[str], features: np.ndarray, labels: np.ndarray, percentile: float
) -> set[str]:
    mask = structural_boundary(graph, node_ids, labels)
    if not mask.any():
        return set()
    scores = boundary_scores(features, labels)
    threshold = np.percentile(scores[mask], percentile)
    return {node_ids[index] for index in np.where(mask & (scores <= threshold))[0]}


# Func: Tune Ego's percentile for a target duplicated-node fraction.
# Input: Graph, node order, raw features, labels, and search settings.
# Output: Overlapping memberships, selected percentile, and achieved rate.
def cluster(
    graph: nx.Graph,
    node_ids: list[str],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    target_duplication_rate: float,
    search_iterations: int,
    search_tolerance: float,
) -> tuple[dict[int, list[int]], float, float]:
    mask = structural_boundary(graph, node_ids, labels)
    scores = boundary_scores(features, labels) if mask.any() else np.empty(len(node_ids))

    # Func: Build Ego memberships at one boundary percentile.
    # Input: A candidate percentile.
    # Output: Cluster memberships for all concept nodes.
    def memberships_at(percentile: float) -> dict[int, list[int]]:
        if not mask.any():
            return ego_memberships(graph, node_ids, labels, set())
        threshold = np.percentile(scores[mask], percentile)
        selected = {node_ids[index] for index in np.where(mask & (scores <= threshold))[0]}
        return ego_memberships(graph, node_ids, labels, selected)

    # Func: Evaluate duplication at one Ego percentile.
    # Input: A candidate percentile.
    # Output: The duplicated-node fraction.
    def rate_at(percentile: float) -> float:
        return duplication_rate(memberships_at(percentile), len(node_ids))

    percentile, rate = search_parameter(
        rate_at, target_duplication_rate, 0.0, 100.0, True, search_iterations, search_tolerance
    )
    return memberships_at(percentile), percentile, rate
