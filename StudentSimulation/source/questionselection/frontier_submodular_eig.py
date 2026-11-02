"""Prefer uncertain concepts at the frontier of previously covered graph regions."""

from __future__ import annotations

import random

import networkx as nx

from .common import choose, node_uncertainty


# Func: Measure newly reached nodes bordering previously covered nodes.
# Input: Graph, beliefs, newly reached nodes, and covered nodes.
# Output: Frontier uncertainty bonus before its configured multiplier.
def frontier_bonus(graph: nx.Graph, beliefs: dict[str, dict], new_nodes: set[str], covered: set[str]) -> float:
    return sum(
        node_uncertainty(beliefs, node)
        for node in new_nodes
        if covered and any(neighbor in covered for neighbor in graph.neighbors(node))
    )


# Func: Add a frontier expansion bonus to the base selection score.
# Input: Candidate data, beliefs, graph, asked set, coverage, settings, and RNG.
# Output: Selected question and its reached graph nodes.
def select(data: dict[str, dict], beliefs: dict[str, dict], graph: nx.Graph,
           asked: set[tuple[str, int]], covered: set[str], config: dict, rng: random.Random):
    # Func: Score newly reached concepts bordering covered graph regions.
    # Input: Newly reached concepts and unused candidate context.
    # Output: Weighted frontier bonus.
    def bonus(_tested: list[str], _reached: set[str], new_nodes: set[str]) -> float:
        return float(config["mu_frontier"]) * frontier_bonus(graph, beliefs, new_nodes, covered)
    return choose(data, beliefs, graph, asked, covered, config, bonus, rng)
