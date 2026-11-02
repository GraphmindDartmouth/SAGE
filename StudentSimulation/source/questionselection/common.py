"""Score available questions using uncertainty, new coverage, and optional graph bonuses."""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import networkx as nx

from source.scoring.beta import uncertainty


# Func: Enumerate questions whose tested concepts remain valid and unasked.
# Input: Cluster data and the set of already asked cluster-question pairs.
# Output: Candidate tuples of cluster ID, question index, and tested concepts.
def candidates(
    data: dict[str, dict], asked: set[tuple[str, int]]
) -> list[tuple[str, int, list[str]]]:
    return [
        (cluster_id, index, list(dict.fromkeys(question["used_terms"])))
        for cluster_id, cluster in data.items()
        for index, question in enumerate(cluster["questions"])
        if (cluster_id, index) not in asked and question["used_terms"]
    ]


# Func: Reach graph nodes up to the configured number of hops from a question.
# Input: Graph, tested concepts, and nonnegative hop count.
# Output: Set of reached concept nodes.
def reach(graph: nx.Graph, tested: list[str], hops: int) -> set[str]:
    if hops < 0:
        raise ValueError("neighbor_hops must be nonnegative")
    reached = set(tested)
    frontier = set(tested)
    for _ in range(hops):
        following = {neighbor for node in frontier for neighbor in graph.neighbors(node)} - reached
        reached.update(following)
        frontier = following
        if not frontier:
            break
    return reached


# Func: Read current posterior uncertainty for one concept node.
# Input: Belief table and node identifier.
# Output: Normalized Beta uncertainty.
def node_uncertainty(beliefs: dict[str, dict], node: str) -> float:
    return uncertainty(beliefs[node]["alpha"], beliefs[node]["beta"])


# Func: Compute SAGE's base uncertainty and uncovered-neighborhood score.
# Input: Graph, beliefs, tested concepts, covered nodes, hops, and coverage weight.
# Output: Base score, reached nodes, and newly reached nodes.
def base_score(
    graph: nx.Graph, beliefs: dict[str, dict], tested: list[str], covered: set[str],
    hops: int, lambda_coverage: float, cover_queried: bool,
) -> tuple[float, set[str], set[str]]:
    if lambda_coverage < 0:
        raise ValueError("lambda_coverage must be nonnegative")
    reached = reach(graph, tested, hops)
    if not cover_queried:
        reached.difference_update(tested)
    new_nodes = reached - covered
    direct = sum(node_uncertainty(beliefs, node) for node in tested)
    coverage = sum(node_uncertainty(beliefs, node) for node in new_nodes)
    return direct + lambda_coverage * coverage, reached, new_nodes


# Func: Pick a question by the configured greedy or softmax policy.
# Input: Candidate data, current beliefs, graph, coverage, settings, bonus, and RNG.
# Output: Selected cluster ID, question index, tested concepts, and reached concepts.
def choose(
    data: dict[str, dict], beliefs: dict[str, dict], graph: nx.Graph,
    asked: set[tuple[str, int]], covered: set[str], config: dict,
    bonus: Callable[[list[str], set[str], set[str]], float], rng: random.Random,
) -> tuple[str, int, list[str], set[str]] | None:
    options = candidates(data, asked)
    if not options:
        return None
    scores: list[float] = []
    reaches: list[set[str]] = []
    for _, _, tested in options:
        base, reached, new_nodes = base_score(
            graph, beliefs, tested, covered, int(config["neighbor_hops"]),
            float(config["lambda_coverage"]), bool(config["cover_queried"]),
        )
        scores.append(base + bonus(tested, reached, new_nodes))
        reaches.append(reached)
    if bool(config["greedy"]):
        selected = max(range(len(options)), key=lambda index: scores[index])
    else:
        temperature = float(config["temperature"])
        if temperature <= 0:
            raise ValueError("Question-selection temperature must be positive")
        maximum = max(scores)
        weights = [math.exp((score - maximum) / temperature) for score in scores]
        selected = rng.choices(range(len(options)), weights=weights, k=1)[0]
    cluster_id, index, tested = options[selected]
    return cluster_id, index, tested, reaches[selected]
