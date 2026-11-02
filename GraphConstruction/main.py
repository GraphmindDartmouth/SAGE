"""Build a cleaned SAGE line graph and its embeddings from educational text using KGGen."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig

from clean import build_line_graph, clean_graph, graph_statistics
from embed import prepare_features, save_feature_artifacts


LOGGER = logging.getLogger(__name__)


# Func: Resolve a configured path relative to the GraphConstruction directory.
# Input: The project directory and an absolute or relative path.
# Output: An absolute normalized Path.
def resolve_path(project_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


# Func: Split source text into overlapping character-bounded chunks.
# Input: Full text, maximum chunk size, and overlap size.
# Output: An ordered list of non-empty text chunks.
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("kggen.chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("kggen.chunk_overlap must be non-negative and smaller than chunk_size")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


# Func: Load and validate the source educational text.
# Input: Path to a UTF-8 text file.
# Output: Non-empty source text.
def load_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Input text file does not exist: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Input text file is empty: {path}")
    return text


# Func: Create one configured KGGen client for the complete extraction.
# Input: KGGen model, temperature, and API-key configuration.
# Output: An initialized KGGen client.
def create_kggen_client(cfg: DictConfig) -> Any:
    try:
        from kg_gen import KGGen
    except ImportError as exc:
        raise RuntimeError("KGGen is not installed; install this project before running the pipeline") from exc
    api_key = os.getenv(str(cfg.api_key_env)) if cfg.api_key_env else None
    if bool(cfg.require_api_key) and not api_key:
        raise RuntimeError(f"Required API key environment variable is not set: {cfg.api_key_env}")
    kwargs: dict[str, Any] = {"model": str(cfg.model), "temperature": float(cfg.temperature)}
    if api_key:
        kwargs["api_key"] = api_key
    return KGGen(**kwargs)


# Func: Convert one KGGen relation into a head-relation-tail triplet.
# Input: A relation represented as a sequence, mapping, or object.
# Output: A raw triplet or None when the relation is invalid.
def unpack_relation(relation: Any) -> tuple[str, str, str] | None:
    if isinstance(relation, dict):
        head = relation.get("head", relation.get("subject", relation.get("source")))
        predicate = relation.get("relation", relation.get("predicate", relation.get("label")))
        tail = relation.get("tail", relation.get("object", relation.get("target")))
    elif isinstance(relation, (list, tuple)) and len(relation) == 3:
        head, predicate, tail = relation
    else:
        head = getattr(relation, "head", getattr(relation, "subject", None))
        predicate = getattr(relation, "relation", getattr(relation, "predicate", None))
        tail = getattr(relation, "tail", getattr(relation, "object", None))
    if head is None or predicate is None or tail is None:
        return None
    return str(head), str(predicate), str(tail)


# Func: Extract entities and valid triplets from a KGGen response.
# Input: A KGGen graph response.
# Output: Raw entity strings and head-relation-tail triplets.
def extract_kggen_payload(graph: Any) -> tuple[list[str], list[tuple[str, str, str]]]:
    entities = [str(value) for value in getattr(graph, "entities", [])]
    triples = [triple for item in getattr(graph, "relations", []) if (triple := unpack_relation(item)) is not None]
    return entities, triples


# Func: Generate a directed labeled knowledge graph from text with KGGen.
# Input: Source text and KGGen extraction configuration.
# Output: Entity nodes and directed, labeled relation links.
def generate_raw_graph(text: str, cfg: DictConfig) -> dict[str, list[dict[str, str]]]:
    chunks = chunk_text(text, int(cfg.chunk_size), int(cfg.chunk_overlap))
    client = create_kggen_client(cfg)
    entities: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    LOGGER.info("Extracting knowledge from %d text chunks", len(chunks))
    for chunk_index, chunk in enumerate(chunks, start=1):
        for repeat_index in range(1, int(cfg.repeats_per_chunk) + 1):
            for attempt in range(int(cfg.max_retries) + 1):
                try:
                    response = client.generate(input_data=chunk, context=str(cfg.context), cluster=False)
                    break
                except Exception:
                    if attempt >= int(cfg.max_retries):
                        raise
                    delay = float(cfg.retry_delay_seconds) * (2**attempt)
                    LOGGER.warning(
                        "KGGen failed for chunk %d/%d repeat %d/%d; retrying in %.1f seconds",
                        chunk_index,
                        len(chunks),
                        repeat_index,
                        int(cfg.repeats_per_chunk),
                        delay,
                    )
                    time.sleep(delay)
            chunk_entities, chunk_triples = extract_kggen_payload(response)
            entities.update(chunk_entities)
            triples.update(chunk_triples)
        LOGGER.info("Processed text chunk %d/%d", chunk_index, len(chunks))
    for head, _, tail in triples:
        entities.update((head, tail))
    return {
        "nodes": [{"id": entity} for entity in sorted(entities)],
        "links": [
            {"source": head, "label": relation, "target": tail}
            for head, relation, tail in sorted(triples)
        ],
    }


# Func: Save one graph or ordered node list as formatted JSON.
# Input: Destination path and a JSON-serializable payload.
# Output: The written destination path.
def save_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


# Func: Run extraction, cleaning, line-graph construction, and feature export.
# Input: The Hydra configuration from configurations/main.yaml.
# Output: Graph JSON, ordered node IDs, embeddings, and a SAGE matrix in the destination directory.
@hydra.main(version_base=None, config_path="configurations", config_name="main")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    random.seed(int(cfg.runtime.seed))
    np.random.seed(int(cfg.runtime.seed))
    project_dir = Path(__file__).resolve().parent
    source_path = resolve_path(project_dir, str(cfg.paths.text))
    destination_root = resolve_path(project_dir, str(cfg.paths.destination))
    text = load_text(source_path)
    if not str(cfg.kggen.context).strip():
        raise ValueError("kggen.context must describe the source material")

    raw_graph = generate_raw_graph(text, cfg.kggen)
    cleaned_graph, cleaning_report = clean_graph(raw_graph, cfg.cleaning)
    if not cleaned_graph["links"]:
        raise ValueError("Cleaning removed every extracted relation")
    line_graph = build_line_graph(cleaned_graph, cfg.line_graph)
    node_ids, features, adjacency = prepare_features(line_graph, cfg)

    output_dir = destination_root / source_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / str(cfg.output.cleaned_graph_file), cleaned_graph)
    save_json(output_dir / str(cfg.output.line_graph_file), line_graph)
    save_feature_artifacts(output_dir, cfg, node_ids, features, adjacency)

    LOGGER.info("Cleaning: %s", cleaning_report)
    LOGGER.info("Line graph: %s", graph_statistics(line_graph))
    LOGGER.info("Artifacts written to %s", output_dir)


if __name__ == "__main__":
    main()
