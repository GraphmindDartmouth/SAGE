"""Summarize hidden mastery recovery directly from current Beta beliefs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .beta import knowledge, uncertainty


# Func: Measure prediction quality and belief uncertainty against hidden labels.
# Input: Current beliefs and node-level ground truth.
# Output: Accuracy, macro F1, mean squared error, and uncertainty diagnostics.
def summarize(beliefs: dict[str, dict[str, float]], truth: dict[str, bool]) -> dict[str, float | int]:
    ids = list(beliefs)
    actual = np.asarray([int(truth[node]) for node in ids], dtype=int)
    signed = np.asarray([knowledge(beliefs[node]["alpha"], beliefs[node]["beta"]) for node in ids])
    predicted = (signed > 0).astype(int)
    probabilities = (signed + 1.0) / 2.0
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", labels=[0, 1], zero_division=0)),
        "mse": float(np.mean((probabilities - actual) ** 2)),
        "mean_uncertainty": float(np.mean([uncertainty(beliefs[node]["alpha"], beliefs[node]["beta"]) for node in ids])),
        "directly_observed_nodes": sum(beliefs[node]["n_observations"] > 0 for node in ids),
    }
