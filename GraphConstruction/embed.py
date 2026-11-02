"""Embed concept graph nodes and save features in SAGE clustering formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from omegaconf import DictConfig


# Func: Encode complete concept statements with the configured embedding model.
# Input: Ordered node texts and embedding configuration.
# Output: A float32 feature matrix with one row per graph node.
def embed_nodes(node_texts: list[str], cfg: DictConfig) -> np.ndarray:
    if not node_texts:
        raise ValueError("Cannot embed an empty line graph")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed; install this project first") from exc
    device = None if str(cfg.device) == "auto" else str(cfg.device)
    encoder = SentenceTransformer(
        str(cfg.model),
        device=device,
        trust_remote_code=bool(cfg.trust_remote_code),
    )
    encoder.max_seq_length = int(cfg.max_length)
    texts = [text.casefold() for text in node_texts] if bool(cfg.lowercase) else node_texts
    features = encoder.encode(
        texts,
        batch_size=int(cfg.batch_size),
        show_progress_bar=bool(cfg.show_progress),
        convert_to_numpy=True,
        normalize_embeddings=bool(cfg.normalize),
    )
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(node_texts) or not np.isfinite(features).all():
        raise ValueError("Embedding model returned invalid or misaligned features")
    return features


# Func: Construct sparse adjacency aligned with the graph node order.
# Input: Line graph, ordered node identifiers, and adjacency settings.
# Output: A float32 CSR adjacency matrix.
def build_adjacency(graph: dict[str, Any], node_ids: list[str], cfg: DictConfig) -> sp.csr_matrix:
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    rows: list[int] = []
    columns: list[int] = []
    for link in graph["links"]:
        source = index[str(link["source"])]
        target = index[str(link["target"])]
        if bool(cfg.remove_self_loops) and source == target:
            continue
        rows.append(source)
        columns.append(target)
        if bool(cfg.undirected_adjacency) and source != target:
            rows.append(target)
            columns.append(source)
    values = np.ones(len(rows), dtype=np.float32)
    adjacency = sp.csr_matrix((values, (rows, columns)), shape=(len(node_ids), len(node_ids)), dtype=np.float32)
    adjacency.data[:] = 1.0
    return adjacency


# Func: Save the ordered feature matrix as an inspectable NumPy file.
# Input: Destination path and float32 feature matrix.
# Output: The written .npy path.
def save_embeddings(path: Path, features: np.ndarray) -> Path:
    np.save(path, features, allow_pickle=False)
    return path


# Func: Save features and adjacency in SAGE's MATLAB matrix format.
# Input: Destination, features, adjacency, node IDs, and matrix settings.
# Output: The written .mat path containing fea, W, gnd, and optional node IDs.
def save_matrix(
    path: Path,
    features: np.ndarray,
    adjacency: sp.csr_matrix,
    node_ids: list[str],
    cfg: DictConfig,
) -> Path:
    if features.shape[0] != len(node_ids) or adjacency.shape != (len(node_ids), len(node_ids)):
        raise ValueError("Features and adjacency must follow the node_ids.json order")
    ground_truth = np.full((len(node_ids), 1), int(cfg.default_ground_truth_label), dtype=np.int32)
    payload: dict[str, Any] = {"fea": features, "W": adjacency, "gnd": ground_truth}
    if bool(cfg.include_node_ids):
        payload["node_ids"] = np.asarray(node_ids, dtype=object).reshape(-1, 1)
    sio.savemat(path, payload, do_compression=True)
    return path


# Func: Build all feature arrays in the exact line-graph node order.
# Input: A concept line graph and the embedding and matrix configuration.
# Output: Ordered node IDs, embeddings, and aligned adjacency.
def prepare_features(graph: dict[str, Any], cfg: DictConfig) -> tuple[list[str], np.ndarray, sp.csr_matrix]:
    node_ids = [str(node["id"]) for node in graph["nodes"]]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Line-graph node identifiers must be unique")
    features = embed_nodes(node_ids, cfg.embedding)
    adjacency = build_adjacency(graph, node_ids, cfg.matrix)
    return node_ids, features, adjacency


# Func: Persist aligned node IDs, embeddings, and SAGE matrix data.
# Input: Output directory, configuration, ordered IDs, features, and adjacency.
# Output: Paths to the three written feature artifacts.
def save_feature_artifacts(
    output_dir: Path,
    cfg: DictConfig,
    node_ids: list[str],
    features: np.ndarray,
    adjacency: sp.csr_matrix,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    node_ids_path = output_dir / str(cfg.output.node_ids_file)
    node_ids_path.write_text(json.dumps(node_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    embeddings_path = save_embeddings(output_dir / str(cfg.output.embeddings_file), features)
    matrix_path = save_matrix(output_dir / str(cfg.output.matrix_file), features, adjacency, node_ids, cfg.matrix)
    return node_ids_path, embeddings_path, matrix_path


# Func: Regenerate embeddings and matrix files from a saved line graph.
# Input: Hydra configuration and the line graph in the configured destination.
# Output: Updated node IDs, embeddings, and SAGE matrix without KGGen calls.
@hydra.main(version_base=None, config_path="configurations", config_name="main")
def main(cfg: DictConfig) -> None:
    project_dir = Path(__file__).resolve().parent
    source_path = Path(str(cfg.paths.text)).expanduser()
    destination_path = Path(str(cfg.paths.destination)).expanduser()
    if not source_path.is_absolute():
        source_path = project_dir / source_path
    if not destination_path.is_absolute():
        destination_path = project_dir / destination_path
    output_dir = destination_path.resolve() / source_path.stem
    graph_path = output_dir / str(cfg.output.line_graph_file)
    if not graph_path.is_file():
        raise FileNotFoundError(f"Line graph does not exist: {graph_path}; run main.py first")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    node_ids, features, adjacency = prepare_features(graph, cfg)
    save_feature_artifacts(output_dir, cfg, node_ids, features, adjacency)
    print(f"Saved {len(node_ids)} aligned node embeddings to {output_dir}")


if __name__ == "__main__":
    main()
