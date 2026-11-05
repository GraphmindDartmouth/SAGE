"""Select clusters uniformly within each round and never repeat a question."""

from __future__ import annotations

import random

import networkx as nx

from .common import candidates, reach


# Func: Draw a cluster with the fewest prior queries, then an unasked question.
# Input: Cluster data, asked pairs, graph, hop count, and random generator.
# Output: Selected cluster, question index, tested concepts, and reached concepts.
def select(
    data: dict[str, dict], asked: set[tuple[str, int]], graph: nx.Graph,
    neighbor_hops: int, rng: random.Random,
) -> tuple[str, int, list[str], set[str]] | None:
    options = candidates(data, asked)
    if not options:
        return None
    counts = {cluster_id: 0 for cluster_id, _, _ in options}
    for cluster_id, _ in asked:
        if cluster_id in counts:
            counts[cluster_id] += 1
    minimum = min(counts.values())
    cluster_id = rng.choice([cluster_id for cluster_id in counts if counts[cluster_id] == minimum])
    local = [item for item in options if item[0] == cluster_id]
    selected_cluster, index, tested = rng.choice(local)
    return selected_cluster, index, tested, reach(graph, tested, neighbor_hops)
