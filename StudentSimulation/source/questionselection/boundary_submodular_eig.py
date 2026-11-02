"""Prefer reached neighborhoods containing edges between opposite signed beliefs."""

from __future__ import annotations

import random

import networkx as nx

from source.scoring.beta import knowledge

from .common import choose


# Func: Count graph edges with opposite signed knowledge inside the reach.
# Input: Graph, Beta beliefs, and reached nodes.
# Output: The paper's Boundary bonus before its multiplier.
def boundary_bonus(graph: nx.Graph, beliefs: dict[str, dict], reached: set[str]) -> float:
    return float(sum(
        knowledge(beliefs[left]["alpha"], beliefs[left]["beta"])
        * knowledge(beliefs[right]["alpha"], beliefs[right]["beta"]) < 0
        for left, right in graph.subgraph(reached).edges()
    ))


# Func: Add the opposing-belief edge count to the base selection score.
# Input: Candidate data, beliefs, graph, asked set, coverage, settings, and RNG.
# Output: Selected question and its reached graph nodes.
def select(data: dict[str, dict], beliefs: dict[str, dict], graph: nx.Graph,
           asked: set[tuple[str, int]], covered: set[str], config: dict, rng: random.Random):
    # Func: Score one candidate's reached disagreement edges.
    # Input: Reached nodes and unused candidate context.
    # Output: Weighted Boundary bonus.
    def bonus(_tested: list[str], reached: set[str], _new: set[str]) -> float:
        return float(config["mu_boundary"]) * boundary_bonus(graph, beliefs, reached)
    return choose(data, beliefs, graph, asked, covered, config, bonus, rng)
