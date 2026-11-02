"""Maintain exact Beta-Bernoulli evidence and normalized posterior uncertainty."""

from __future__ import annotations

import math


PRIOR_STD = math.sqrt(1.0 / 12.0)


# Func: Convert Beta parameters to signed mastery belief.
# Input: Positive alpha and beta parameters.
# Output: Knowledge in the interval from minus one to one.
def knowledge(alpha: float, beta: float) -> float:
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive")
    return (alpha - beta) / (alpha + beta)


# Func: Compute posterior standard deviation relative to Beta(1,1).
# Input: Positive alpha and beta parameters.
# Output: Normalized uncertainty, with one at the prior.
def uncertainty(alpha: float, beta: float) -> float:
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta parameters must be positive")
    total = alpha + beta
    variance = alpha * beta / (total * total * (total + 1.0))
    return math.sqrt(variance) / PRIOR_STD


# Func: Add signed direct or virtual evidence without overwriting old evidence.
# Input: Node state and a signed evidence amount.
# Output: Updated node state in place.
def add_evidence(state: dict[str, float], evidence: float) -> None:
    if not math.isfinite(evidence):
        raise ValueError("Evidence must be finite")
    state["alpha"] += max(0.0, evidence)
    state["beta"] += max(0.0, -evidence)


# Func: Initialize every graph node at the configured Beta prior.
# Input: Ordered node IDs and positive prior parameters.
# Output: Node states with alpha, beta, and direct-observation count.
def initialize_beliefs(node_ids: list[str], prior_alpha: float, prior_beta: float) -> dict[str, dict[str, float]]:
    if not all(math.isfinite(value) and value >= 1 for value in (prior_alpha, prior_beta)):
        raise ValueError("Beta prior parameters must be finite and at least one")
    return {
        node: {"alpha": float(prior_alpha), "beta": float(prior_beta), "n_observations": 0}
        for node in node_ids
    }
