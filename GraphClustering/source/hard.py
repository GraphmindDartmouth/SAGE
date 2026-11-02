"""Run SAGE's graph convolutional hard clustering on concept node embeddings."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


# Func: Apply SAGE's adjacency and feature normalization before diffusion.
# Input: Symmetric adjacency and aligned semantic node features.
# Output: Normalized sparse adjacency and row-normalized features.
def preprocess(adjacency: sp.csr_matrix, features: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray]:
    n_nodes = adjacency.shape[0]
    augmented = adjacency + sp.eye(n_nodes, format="csr")
    degree = np.asarray(augmented.sum(axis=1)).ravel()
    inverse_sqrt = sp.diags(np.power(degree, -0.5))
    symmetric = (inverse_sqrt @ augmented @ inverse_sqrt).tocsr()
    row_augmented = symmetric + sp.eye(n_nodes, format="csr")
    row_degree = np.asarray(row_augmented.sum(axis=1)).ravel()
    diffused_adjacency = (sp.diags(1.0 / row_degree) @ row_augmented).tocsr()
    return diffused_adjacency, normalize(features, norm="l2", axis=1)


# Func: Diffuse semantic features through the normalized line graph.
# Input: Normalized adjacency, features, and diffusion power.
# Output: A dense propagated feature matrix.
def propagate(adjacency: sp.csr_matrix, features: np.ndarray, power: int) -> np.ndarray:
    if power < 0:
        raise ValueError("method.power must be non-negative")
    propagated = features.copy()
    for _ in range(power):
        propagated = adjacency @ propagated
    return np.asarray(propagated, dtype=np.float64)


# Func: Convert adjacency to the row-weighted sparse tensor used by GCC.
# Input: A nonnegative sparse adjacency matrix.
# Output: A TensorFlow sparse tensor with row-normalized edge weights.
def adjacency_tensor(adjacency: sp.csr_matrix) -> tf.SparseTensor:
    edges = adjacency.tocoo()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    tensor = tf.SparseTensor(
        indices=np.column_stack((edges.row, edges.col)).astype(np.int64),
        values=(edges.data / degree[edges.row]).astype(np.float64),
        dense_shape=adjacency.shape,
    )
    return tf.sparse.reorder(tensor)


# Func: Optimize SAGE's alternating assignment, centroid, and projection steps.
# Input: Propagated features, graph adjacency, cluster count, and run settings.
# Output: Final integer labels and reconstruction loss for one run.
def optimize(
    features: np.ndarray,
    adjacency: sp.csr_matrix,
    n_clusters: int,
    lam: float,
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    if max_iter < 1 or tol < 0 or lam < 0:
        raise ValueError("method.max_iter must be positive; method.tol and method.lam must be nonnegative")
    n_components = min(n_clusters, features.shape[1], features.shape[0])
    projection = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(features).components_.T
    initial = features @ projection
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(initial)
    labels = tf.convert_to_tensor(kmeans.labels_, dtype=tf.int32)
    centroids = tf.convert_to_tensor(kmeans.cluster_centers_, dtype=tf.float64)
    feature_tensor = tf.convert_to_tensor(features, dtype=tf.float64)
    graph_tensor = adjacency_tensor(adjacency)
    previous_loss = float("inf")
    for _ in range(max_iter):
        cross = tf.transpose(feature_tensor) @ tf.gather(centroids, labels)
        _, left, right = tf.linalg.svd(cross, full_matrices=False)
        projection = left @ tf.transpose(right)
        projected = feature_tensor @ projection
        distances = (
            tf.reduce_sum(projected**2, axis=1, keepdims=True)
            + tf.reduce_sum(centroids**2, axis=1)
            - 2.0 * projected @ tf.transpose(centroids)
        )
        affinity = tf.sparse.sparse_dense_matmul(graph_tensor, tf.one_hot(labels, n_clusters, dtype=tf.float64))
        labels = tf.argmin(distances - lam * affinity, axis=1, output_type=tf.int32)
        centroids = tf.math.unsorted_segment_mean(projected, labels, n_clusters)
        reconstruction = tf.gather(centroids @ tf.transpose(projection), labels)
        loss = float(tf.linalg.norm(feature_tensor - reconstruction).numpy())
        if not np.isfinite(loss):
            raise RuntimeError("GCC optimization produced a non-finite loss")
        if abs(previous_loss - loss) < tol:
            break
        previous_loss = loss
    return labels.numpy(), loss


# Func: Repeat the GCC optimizer with distinct seeds and keep the best run.
# Input: Adjacency, semantic features, cluster count, seed, and method settings.
# Output: Best labels, loss, all run losses, and propagated features.
def cluster(
    adjacency: sp.csr_matrix,
    features: np.ndarray,
    n_clusters: int,
    seed: int,
    *,
    power: int,
    lam: float,
    runs: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, float, list[float], np.ndarray]:
    n_nodes = features.shape[0]
    if not 2 <= n_clusters <= n_nodes:
        raise ValueError(f"clustering.n_clusters must be between 2 and the number of nodes ({n_nodes})")
    if runs < 1:
        raise ValueError("method.runs must be positive")
    normalized_adjacency, normalized_features = preprocess(adjacency, features)
    propagated = propagate(normalized_adjacency, normalized_features, power)
    losses: list[float] = []
    best_labels: np.ndarray | None = None
    best_loss = float("inf")
    for run in range(runs):
        labels, loss = optimize(propagated, adjacency, n_clusters, lam, max_iter, tol, seed + run)
        losses.append(loss)
        if loss < best_loss:
            best_labels, best_loss = labels.copy(), loss
    if best_labels is None:
        raise RuntimeError("GCC produced no cluster assignments")
    return best_labels, best_loss, losses, propagated
