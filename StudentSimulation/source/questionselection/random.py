"""Select an unqueried cluster uniformly, then one question within it."""

from __future__ import annotations

import random

import networkx as nx

from .common import candidates, reach


# Func: Draw one question from a uniformly selected unqueried cluster.
# Input: Cluster data, asked pairs, graph, hop count, and random generator.
# Output: Selected cluster, question index, tested concepts, and reached concepts.
def select(
    data: dict[str, dict], asked: set[tuple[str, int]], graph: nx.Graph,
    neighbor_hops: int, rng: random.Random,
) -> tuple[str, int, list[str], set[str]] | None:
    queried_clusters = {cluster_id for cluster_id, _ in asked}
    options = [item for item in candidates(data, asked) if item[0] not in queried_clusters]
    if not options:
        return None
    cluster_id = rng.choice(sorted({item[0] for item in options}))
    local = [item for item in options if item[0] == cluster_id]
    selected_cluster, index, tested = rng.choice(local)
    return selected_cluster, index, tested, reach(graph, tested, neighbor_hops)
