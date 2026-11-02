"""Combine answer correctness and judged reasoning into direct node evidence."""

from __future__ import annotations

from .beta import add_evidence


# Func: Build the paper's clipped answer-plus-reasoning evidence per cluster node.
# Input: Cluster nodes, directly tested nodes, correctness, judge signs, and weights.
# Output: Signed evidence for every cluster node.
def response_evidence(
    cluster_nodes: list[str],
    used_terms: list[str],
    correct: bool,
    judge_signs: dict[str, int],
    answer_weight: float,
    reasoning_weight: float,
) -> dict[str, float]:
    if answer_weight < 0 or reasoning_weight < 0:
        raise ValueError("Scoring weights must be nonnegative")
    tested = set(used_terms)
    answer_sign = 1.0 if correct else -1.0
    evidence: dict[str, float] = {}
    for node in cluster_nodes:
        reasoning = float(judge_signs.get(node, 0))
        if reasoning not in {-1.0, 0.0, 1.0}:
            raise ValueError("Judge signs must be -1, 0, or 1")
        value = reasoning_weight * reasoning
        if node in tested:
            value += answer_weight * answer_sign
        evidence[node] = max(-1.0, min(1.0, value))
    return evidence


# Func: Apply direct response evidence to the corresponding Beta beliefs.
# Input: Belief table, node evidence, and local evidence multiplier.
# Output: Updated beliefs and observation counts in place.
def apply_direct_evidence(
    beliefs: dict[str, dict[str, float]], evidence: dict[str, float], alpha_local: float
) -> None:
    if alpha_local < 0:
        raise ValueError("scoring.alpha_local must be nonnegative")
    for node, value in evidence.items():
        if node not in beliefs:
            raise ValueError(f"Evidence refers to an unknown graph node: {node}")
        add_evidence(beliefs[node], alpha_local * value)
        if value != 0 and alpha_local > 0:
            beliefs[node]["n_observations"] += 1
