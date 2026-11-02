"""Prefer reached neighborhoods with many edges to the rest of the graph."""

from __future__ import annotations

import random

import networkx as nx

from .common import choose


# Func: Count graph edges leaving a candidate's reached neighborhood.
# Input: Graph and reached node set.
# Output: The paper's CutMass bonus before its multiplier.
def cutmass_bonus(graph: nx.Graph, reached: set[str]) -> float:
    return float(sum(neighbor not in reached for node in reached for neighbor in graph.neighbors(node)))


# Func: Add the outgoing-edge count to the base selection score.
# Input: Candidate data, beliefs, graph, asked set, coverage, settings, and RNG.
# Output: Selected question and its reached graph nodes.
def select(data: dict[str, dict], beliefs: dict[str, dict], graph: nx.Graph,
           asked: set[tuple[str, int]], covered: set[str], config: dict, rng: random.Random):
    # Func: Score one candidate's outgoing neighborhood edges.
    # Input: Reached nodes and unused candidate context.
    # Output: Weighted CutMass bonus.
    def bonus(_tested: list[str], reached: set[str], _new: set[str]) -> float:
        return float(config["mu_cut"]) * cutmass_bonus(graph, reached)
    return choose(data, beliefs, graph, asked, covered, config, bonus, rng)
