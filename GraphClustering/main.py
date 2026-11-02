"""Cluster an embedded SAGE line graph with hard GCC or an overlapping method."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from source import ego, ego_mod, gcc_soft, hard
from source.data import load_graph_data
from source.overlap import graph_metrics


LOGGER = logging.getLogger(__name__)


# Func: Resolve a configured path from the GraphClustering directory.
# Input: Project directory and an absolute or relative path string.
# Output: An absolute normalized Path.
def resolve_path(project_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


# Func: Convert complete node memberships to downstream cluster JSON.
# Input: Ordered node IDs and memberships indexed by matrix row.
# Output: A mapping of cluster IDs to lists of concept node IDs.
def cluster_payload(node_ids: list[str], memberships: dict[int, list[int]]) -> dict[str, dict[str, list[str]]]:
    if set(memberships) != set(range(len(node_ids))) or any(not labels for labels in memberships.values()):
        raise ValueError("Every concept node must have at least one cluster membership")
    clusters: dict[str, list[str]] = {}
    for row, node_id in enumerate(node_ids):
        for label in memberships[row]:
            clusters.setdefault(str(label), []).append(node_id)
    return {"clusters": dict(sorted(clusters.items(), key=lambda item: int(item[0])))}


# Func: Save one JSON artifact and create its parent directory if necessary.
# Input: Destination path and JSON-serializable content.
# Output: The written artifact path.
def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# Func: Run configured GCC and optionally expand its overlapping memberships.
# Input: Hydra configuration with paths, method, cluster count, and seed.
# Output: Cluster assignments and metrics under the configured output directory.
@hydra.main(version_base=None, config_path="configurations", config_name="main")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    project_dir = Path(__file__).resolve().parent
    matrix_path = resolve_path(project_dir, str(cfg.paths.matrix))
    graph_path = resolve_path(project_dir, str(cfg.paths.graph))
    node_ids_path = resolve_path(project_dir, str(cfg.paths.node_ids))
    output_root = resolve_path(project_dir, str(cfg.paths.output))
    data = load_graph_data(matrix_path, graph_path, node_ids_path)
    method = str(cfg.method.name)
    if method not in {"hard", "ego", "ego_mod", "gcc_soft"}:
        raise ValueError(f"Unsupported clustering method: {method}")
    n_clusters = int(cfg.clustering.n_clusters)
    seed = int(cfg.clustering.seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    labels, best_loss, run_losses, propagated = hard.cluster(
        data.adjacency,
        data.features,
        n_clusters,
        seed,
        power=int(cfg.method.power),
        lam=float(cfg.method.lam),
        runs=int(cfg.method.runs),
        max_iter=int(cfg.method.max_iter),
        tol=float(cfg.method.tol),
    )
    memberships = {index: [int(label)] for index, label in enumerate(labels)}
    selected_parameter: dict[str, float] | None = None
    if method == "ego":
        memberships, value, _ = ego.cluster(
            data.graph, data.node_ids, data.features, labels,
            target_duplication_rate=float(cfg.method.target_duplication_rate),
            search_iterations=int(cfg.method.search_iterations),
            search_tolerance=float(cfg.method.search_tolerance),
        )
        selected_parameter = {"percentile": value}
    elif method == "ego_mod":
        memberships, value, _ = ego_mod.cluster(
            data.graph, data.node_ids, data.features, labels,
            fixed_percentile=float(cfg.method.fixed_percentile),
            target_duplication_rate=float(cfg.method.target_duplication_rate),
            search_iterations=int(cfg.method.search_iterations),
            search_tolerance=float(cfg.method.search_tolerance),
        )
        selected_parameter = {"epsilon": value}
    elif method == "gcc_soft":
        memberships, value, _ = gcc_soft.cluster(
            propagated, labels,
            target_duplication_rate=float(cfg.method.target_duplication_rate),
            search_iterations=int(cfg.method.search_iterations),
            search_tolerance=float(cfg.method.search_tolerance),
        )
        selected_parameter = {"ratio_threshold": value}
    output_dir = output_root / method
    clusters = cluster_payload(data.node_ids, memberships)
    metrics = {
        "method": method,
        "matrix_path": str(matrix_path),
        "graph_path": str(graph_path),
        "node_ids_path": str(node_ids_path),
        "n_nodes": len(data.node_ids),
        "n_clusters_requested": n_clusters,
        "n_clusters_written": len(clusters["clusters"]),
        "parameters": {"seed": seed, **OmegaConf.to_container(cfg.method, resolve=True)},
        "selected_parameter": selected_parameter,
        "best_loss": best_loss,
        "run_losses": run_losses,
        "metrics": graph_metrics(data.adjacency, memberships),
    }
    clusters_path = save_json(output_dir / "clusters.json", clusters)
    metrics_path = save_json(output_dir / "metrics.json", metrics)
    LOGGER.info("Wrote %s and %s", clusters_path, metrics_path)
    print(f"Saved {method} clusters for {len(data.node_ids)} concept nodes to {clusters_path}")


if __name__ == "__main__":
    main()
