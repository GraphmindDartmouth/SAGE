"""Clean KGGen relations and convert them into a SAGE concept line graph."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

import networkx as nx
import numpy as np
from omegaconf import DictConfig


WHITESPACE_PATTERN = re.compile(r"\s+")
REPEATED_WORD_PATTERN = re.compile(r"\b([^\W\d_]+)\b(?:\s+\1\b)+", re.IGNORECASE)
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")


# Func: Normalize an entity or relation without changing its meaning.
# Input: Raw text and configured Unicode, casing, and repetition settings.
# Output: A normalized single-line string.
def normalize_text(value: str, unicode_form: str, lowercase: bool, collapse_repeated_words: bool) -> str:
    text = unicodedata.normalize(unicode_form, str(value))
    text = ZERO_WIDTH_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    if collapse_repeated_words:
        text = REPEATED_WORD_PATTERN.sub(r"\1", text)
    return text.casefold() if lowercase else text


# Func: Build a comparison key for entity deduplication.
# Input: A normalized entity string.
# Output: A case-insensitive, punctuation-stable key.
def entity_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"[\u2018\u2019\u201c\u201d\"`]", "", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


# Func: Load a person-name detector when the optional filter is enabled.
# Input: Cleaning configuration with the filter flag and spaCy model name.
# Output: A spaCy pipeline when enabled, otherwise None.
def load_person_detector(cfg: DictConfig) -> Any | None:
    if not bool(cfg.remove_person_entities):
        return None
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("Install the person-filtering extra to enable person-name removal") from exc
    model_name = str(cfg.spacy_model)
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(f"Install the spaCy model before enabling person-name removal: {model_name}") from exc


# Func: Check whether an entity is entirely a detected person name.
# Input: An entity string and an optional spaCy pipeline.
# Output: True only when a PERSON span matches the complete entity.
def is_person_entity(value: str, detector: Any | None) -> bool:
    if detector is None:
        return False
    key = entity_key(value)
    return any(entity.label_ == "PERSON" and entity_key(entity.text) == key for entity in detector(value).ents)


# Func: Validate an entity against configured cleaning rules.
# Input: Normalized entity text and cleaning configuration.
# Output: A removal reason or None when the entity is valid.
def entity_removal_reason(value: str, cfg: DictConfig) -> str | None:
    if not value:
        return "empty_entity"
    lowered = value.casefold()
    if lowered in {str(item).casefold() for item in cfg.blocked_exact_entities}:
        return "blocked_exact_entity"
    if any(str(item).casefold() in lowered for item in cfg.blocked_substrings):
        return "blocked_substring"
    if len(value) < int(cfg.min_entity_characters):
        return "entity_too_short"
    if len(value.split()) > int(cfg.max_entity_tokens):
        return "entity_too_long"
    return None


# Func: Clean and deduplicate a directed labeled knowledge graph.
# Input: Raw KGGen nodes and links plus cleaning configuration.
# Output: A cleaned graph and counts of retained and removed items.
def clean_graph(
    raw_graph: dict[str, list[dict[str, str]]], cfg: DictConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    detector = load_person_detector(cfg)
    aliases: dict[str, str] = {}
    rejected: set[str] = set()
    canonical_by_key: dict[str, str] = {}
    removal_counts: Counter[str] = Counter()
    candidates = [str(node["id"]) for node in raw_graph["nodes"]]
    candidates.extend(str(link[endpoint]) for link in raw_graph["links"] for endpoint in ("source", "target"))
    for raw_entity in candidates:
        if raw_entity in aliases or raw_entity in rejected:
            continue
        cleaned = normalize_text(
            raw_entity,
            str(cfg.unicode_form),
            bool(cfg.lowercase_entities),
            bool(cfg.collapse_repeated_words),
        )
        reason = entity_removal_reason(cleaned, cfg)
        if reason is None and is_person_entity(cleaned, detector):
            reason = "person_entity"
        if reason is not None:
            removal_counts[reason] += 1
            rejected.add(raw_entity)
            continue
        key = entity_key(cleaned)
        canonical = canonical_by_key.get(key)
        if canonical is None:
            canonical_by_key[key] = cleaned
            canonical = cleaned
        elif canonical != cleaned:
            removal_counts["duplicate_entity"] += 1
        aliases[raw_entity] = canonical

    blocked_relations = {str(item).casefold() for item in cfg.blocked_relations}
    triples: set[tuple[str, str, str]] = set()
    for link in raw_graph["links"]:
        head = aliases.get(str(link["source"]))
        tail = aliases.get(str(link["target"]))
        if head is None or tail is None:
            removal_counts["relation_with_removed_entity"] += 1
            continue
        relation = normalize_text(
            str(link.get("label", "")),
            str(cfg.unicode_form),
            bool(cfg.lowercase_relations),
            bool(cfg.collapse_repeated_words),
        )
        if bool(cfg.remove_empty_relations) and not relation:
            removal_counts["empty_relation"] += 1
            continue
        if len(relation.split()) > int(cfg.max_relation_tokens):
            removal_counts["relation_too_long"] += 1
            continue
        if relation.casefold() in blocked_relations:
            removal_counts["blocked_relation"] += 1
            continue
        if bool(cfg.remove_self_relations) and entity_key(head) == entity_key(tail):
            removal_counts["self_relation"] += 1
            continue
        triple = (head, relation, tail)
        if triple in triples:
            removal_counts["duplicate_relation"] += 1
            continue
        triples.add(triple)

    used_entities = sorted({entity for head, _, tail in triples for entity in (head, tail)})
    cleaned_graph = {
        "directed": True,
        "multigraph": True,
        "nodes": [{"id": entity} for entity in used_entities],
        "links": [
            {"source": head, "label": relation, "target": tail}
            for head, relation, tail in sorted(triples)
        ],
    }
    report = {
        "raw_entities": len(raw_graph["nodes"]),
        "raw_relations": len(raw_graph["links"]),
        "cleaned_entities": len(used_entities),
        "cleaned_relations": len(triples),
        "removals": dict(sorted(removal_counts.items())),
    }
    return cleaned_graph, report


# Func: Render one concept triplet as a readable line-graph node identifier.
# Input: A head-relation-tail triplet and a configurable text template.
# Output: A normalized concept statement string.
def render_triple(triple: tuple[str, str, str], template: str) -> str:
    head, relation, tail = triple
    return WHITESPACE_PATTERN.sub(" ", template.format(head=head, relation=relation, tail=tail)).strip()


# Func: Build a concept line graph from the cleaned entity graph.
# Input: Cleaned directed graph and line-graph configuration.
# Output: Concept nodes and links between consecutive or endpoint-sharing concepts.
def build_line_graph(cleaned_graph: dict[str, Any], cfg: DictConfig) -> dict[str, Any]:
    triples = [
        (str(link["source"]), str(link["label"]), str(link["target"]))
        for link in cleaned_graph["links"]
    ]
    triple_ids: dict[tuple[str, str, str], str] = {}
    id_owners: dict[str, tuple[str, str, str]] = {}
    for triple in triples:
        node_id = render_triple(triple, str(cfg.node_template))
        owner = id_owners.get(node_id)
        if owner is not None and owner != triple:
            raise ValueError(f"Line-graph node template produced a duplicate identifier: {node_id}")
        triple_ids[triple] = node_id
        id_owners[node_id] = triple

    edge_pairs: set[tuple[str, str]] = set()
    mode = str(cfg.mode)
    if mode == "consecutive":
        incoming: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        outgoing: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for triple in triples:
            outgoing[triple[0]].append(triple)
            incoming[triple[2]].append(triple)
        for middle in sorted(set(incoming) & set(outgoing)):
            for left in incoming[middle]:
                for right in outgoing[middle]:
                    if left != right:
                        edge_pairs.add(tuple(sorted((triple_ids[left], triple_ids[right]))))
    elif mode == "shared_endpoint":
        by_entity: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for triple in triples:
            by_entity[triple[0]].append(triple)
            if triple[2] != triple[0]:
                by_entity[triple[2]].append(triple)
        for related in by_entity.values():
            for left_index, left in enumerate(related):
                for right in related[left_index + 1 :]:
                    if left != right:
                        edge_pairs.add(tuple(sorted((triple_ids[left], triple_ids[right]))))
    else:
        raise ValueError("line_graph.mode must be 'consecutive' or 'shared_endpoint'")

    nodes = [
        {
            "id": triple_ids[triple],
            "head": triple[0],
            "relation": triple[1],
            "tail": triple[2],
        }
        for triple in sorted(triples)
    ]
    return {
        "directed": False,
        "multigraph": False,
        "nodes": nodes,
        "links": [{"source": source, "target": target} for source, target in sorted(edge_pairs)],
    }


# Func: Compute topology diagnostics for a concept line graph.
# Input: A line-graph payload.
# Output: Node, edge, degree, density, isolate, and component statistics.
def graph_statistics(graph: dict[str, Any]) -> dict[str, int | float]:
    network = nx.Graph()
    network.add_nodes_from(str(node["id"]) for node in graph["nodes"])
    network.add_edges_from((str(link["source"]), str(link["target"])) for link in graph["links"])
    components = list(nx.connected_components(network))
    degrees = [degree for _, degree in network.degree()]
    return {
        "nodes": network.number_of_nodes(),
        "edges": network.number_of_edges(),
        "components": len(components),
        "largest_component": max((len(component) for component in components), default=0),
        "isolates": sum(degree == 0 for degree in degrees),
        "average_degree": float(np.mean(degrees)) if degrees else 0.0,
        "median_degree": float(np.median(degrees)) if degrees else 0.0,
        "maximum_degree": max(degrees, default=0),
        "density": float(nx.density(network)) if network.number_of_nodes() > 1 else 0.0,
    }
