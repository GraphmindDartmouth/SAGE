"""Expand GCC labels using ratios of propagated-feature centroid distances."""

from __future__ import annotations

import numpy as np

from .overlap import duplication_rate, search_parameter


# Func: Compute GCC Soft's nearest-centroid distance ratios.
# Input: Propagated semantic features and hard cluster labels.
# Output: A node-by-cluster matrix of distance ratios.
def distance_ratios(propagated: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_clusters = int(labels.max()) + 1
    centroids = np.zeros((n_clusters, propagated.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_clusters)
    np.add.at(centroids, labels, propagated)
    present = counts > 0
    centroids[present] /= counts[present, None]
    squared = np.maximum(
        np.sum(propagated**2, axis=1, keepdims=True)
        + np.sum(centroids**2, axis=1)
        - 2.0 * propagated @ centroids.T,
        0.0,
    )
    nearest = squared.min(axis=1)
    return nearest[:, None] / np.where(squared > 0, squared, 1e-10)


# Func: Assign extra clusters whose distance ratio meets a threshold.
# Input: Distance ratios, hard labels, and the ratio threshold.
# Output: One or more cluster memberships for every node.
def memberships_at(ratios: np.ndarray, labels: np.ndarray, threshold: float) -> dict[int, list[int]]:
    return {
        index: sorted(set(np.where(ratios[index] >= threshold)[0].tolist()) | {int(labels[index])})
        for index in range(len(labels))
    }


# Func: Tune GCC Soft's ratio threshold to a duplication target.
# Input: Propagated features, hard labels, and search settings.
# Output: Overlapping memberships, selected threshold, and achieved rate.
def cluster(
    propagated: np.ndarray,
    labels: np.ndarray,
    *,
    target_duplication_rate: float,
    search_iterations: int,
    search_tolerance: float,
) -> tuple[dict[int, list[int]], float, float]:
    ratios = distance_ratios(propagated, labels)

    # Func: Evaluate duplication at one GCC Soft distance threshold.
    # Input: A candidate distance-ratio threshold.
    # Output: The duplicated-node fraction.
    def rate_at(threshold: float) -> float:
        return duplication_rate(memberships_at(ratios, labels, threshold), len(labels))

    threshold, rate = search_parameter(
        rate_at, target_duplication_rate, 0.0, 1.0, False, search_iterations, search_tolerance
    )
    return memberships_at(ratios, labels, threshold), threshold, rate
