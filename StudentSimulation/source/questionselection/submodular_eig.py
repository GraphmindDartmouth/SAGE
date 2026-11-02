"""Select questions by posterior uncertainty and uncovered graph reach."""

from __future__ import annotations

import random

import networkx as nx

from .common import choose


# Func: Apply the SAGE base uncertainty-plus-coverage selector.
# Input: Candidate data, beliefs, graph, asked set, coverage, settings, and RNG.
# Output: Selected question and its reached graph nodes.
def select(data: dict[str, dict], beliefs: dict[str, dict], graph: nx.Graph,
           asked: set[tuple[str, int]], covered: set[str], config: dict, rng: random.Random):
    return choose(data, beliefs, graph, asked, covered, config, lambda _tested, _reach, _new: 0.0, rng)
