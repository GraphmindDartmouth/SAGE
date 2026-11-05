"""Load aligned SAGE assets and assign hidden mastery without using assessment evidence."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass
class SimulationData:
    graph: nx.Graph
    node_ids: list[str]
    clusters: dict[str, dict]
    truth: dict[str, bool]
    mastered_clusters: set[str]
    dropped_overlap_questions: int


# Func: Read a required UTF-8 JSON input.
# Input: Path to an existing JSON file.
# Output: Parsed JSON content.
def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# Func: Build the concept line graph with its saved node order.
# Input: GraphConstruction line_graph.json path.
# Output: Undirected graph and ordered unique node IDs.
def load_graph(path: Path) -> tuple[nx.Graph, list[str]]:
    payload = read_json(path)
    node_ids = [str(item["id"]) for item in payload["nodes"]]
    if not node_ids or len(node_ids) != len(set(node_ids)):
        raise ValueError("Graph nodes must be nonempty and unique")
    graph = nx.Graph()
    graph.add_nodes_from(node_ids)
    for link in payload.get("links", []):
        source, target = str(link["source"]), str(link["target"])
        if source not in graph or target not in graph:
            raise ValueError("Graph link refers to an unknown node")
        if source != target:
            graph.add_edge(source, target)
    return graph, node_ids


# Func: Validate cluster terms and generated questions against graph node IDs.
# Input: Cluster JSON, grouped question JSON, and graph node set.
# Output: Cluster records with concept lists and valid question records.
def load_clusters_and_questions(cluster_path: Path, question_path: Path, nodes: set[str]) -> dict[str, dict]:
    raw_clusters = read_json(cluster_path)
    raw_questions = read_json(question_path)
    if not isinstance(raw_clusters, dict) or not isinstance(raw_clusters.get("clusters"), dict):
        raise ValueError("clusters.json must contain a clusters mapping")
    if not isinstance(raw_questions, list):
        raise ValueError("questions.json must contain grouped cluster records")
    questions_by_cluster: dict[str, list[dict]] = {}
    for entry in raw_questions:
        cluster_id = str(entry["cluster_id"])
        if cluster_id in questions_by_cluster or not isinstance(entry.get("questions"), list):
            raise ValueError(f"Duplicate or invalid question cluster: {cluster_id}")
        questions_by_cluster[cluster_id] = entry["questions"]
    clusters: dict[str, dict] = {}
    all_terms: set[str] = set()
    for raw_id, raw_terms in raw_clusters["clusters"].items():
        cluster_id = str(raw_id)
        if not isinstance(raw_terms, list) or not raw_terms:
            raise ValueError(f"Cluster {cluster_id} has no concept nodes")
        terms = [str(term) for term in raw_terms]
        if len(terms) != len(set(terms)) or not set(terms).issubset(nodes):
            raise ValueError(f"Cluster {cluster_id} has duplicate or unknown nodes")
        all_terms.update(terms)
        questions: list[dict] = []
        for question in questions_by_cluster.get(cluster_id, []):
            used = [str(term) for term in question.get("used_terms", [])]
            if (
                not isinstance(question.get("question"), str) or not question["question"].strip()
                or not isinstance(question.get("options"), list) or len(question["options"]) != 4
                or any(not isinstance(option, str) or not option.strip() for option in question["options"])
                or str(question.get("answer_letter", "")).upper() not in {"A", "B", "C", "D"}
                or not used or not set(used).issubset(terms)
            ):
                raise ValueError(f"Invalid question or used_terms in cluster {cluster_id}")
            questions.append({
                "question": question["question"],
                "options": list(question["options"]),
                "answer_letter": str(question["answer_letter"]).upper(),
                "used_terms": used,
            })
        clusters[cluster_id] = {"nodes": terms, "questions": questions}
    if all_terms != nodes:
        raise ValueError("Cluster assignments must cover every line-graph node")
    if set(questions_by_cluster) - set(clusters):
        raise ValueError("Question file contains an unknown cluster")
    return clusters


# Func: Connect clusters that meet across a concept graph edge.
# Input: Cluster membership lists and line graph.
# Output: Weighted cluster graph and node-to-cluster memberships.
def cluster_topology(clusters: dict[str, dict], graph: nx.Graph) -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
    memberships: dict[str, set[str]] = defaultdict(set)
    for cluster_id, record in clusters.items():
        for node in record["nodes"]:
            memberships[node].add(cluster_id)
    topology = {cluster_id: defaultdict(int) for cluster_id in clusters}
    for source, target in graph.edges():
        pairs = {
            tuple(sorted((left, right)))
            for left in memberships[source]
            for right in memberships[target]
            if left != right
        }
        for left, right in pairs:
            topology[left][right] += 1
            topology[right][left] += 1
    return {key: dict(value) for key, value in topology.items()}, dict(memberships)


# Func: Choose connected mastered clusters from topology without hidden labels.
# Input: Weighted cluster graph, cluster count, and random seed.
# Output: Set of selected mastered cluster IDs.
def mastered_set(topology: dict[str, dict[str, int]], count: int, seed: int) -> set[str]:
    rng = random.Random(seed)
    ids = sorted(topology)
    if count <= 0:
        return set()
    count = min(count, len(ids))
    weighted_degree = {cluster_id: sum(topology[cluster_id].values()) for cluster_id in ids}
    best_degree = max(weighted_degree.values())
    chosen = {rng.choice([cluster_id for cluster_id in ids if weighted_degree[cluster_id] == best_degree])}
    while len(chosen) < count:
        gains = {cluster_id: sum(topology[cluster_id].get(mastered, 0) for mastered in chosen) for cluster_id in ids if cluster_id not in chosen}
        best_gain = max(gains.values())
        chosen.add(rng.choice([cluster_id for cluster_id in ids if gains.get(cluster_id) == best_gain]))
    return chosen


# Func: Map mastered clusters to binary node truth by majority membership.
# Input: Node-to-cluster memberships and mastered cluster IDs.
# Output: Hidden mastered/unmastered label for every concept node.
def node_truth(memberships: dict[str, set[str]], mastered: set[str]) -> dict[str, bool]:
    return {
        node: sum(cluster_id in mastered for cluster_id in cluster_ids) >= len(cluster_ids) / 2.0
        for node, cluster_ids in memberships.items()
    }


# Func: Choose the smallest connected mastered set reaching the target fraction.
# Input: Cluster topology, node memberships, target mastery fraction, and seed.
# Output: Mastered cluster IDs and resulting node-level truth.
def assign_mastery(
    topology: dict[str, dict[str, int]], memberships: dict[str, set[str]],
    fraction: float, seed: int,
) -> tuple[set[str], dict[str, bool]]:
    if not 0 <= fraction <= 1:
        raise ValueError("simulation.mastery_fraction must be between zero and one")
    target = fraction * len(memberships)
    lower, upper = 0, len(topology)
    while lower < upper:
        midpoint = (lower + upper) // 2
        selected = mastered_set(topology, midpoint, seed)
        if sum(node_truth(memberships, selected).values()) < target:
            lower = midpoint + 1
        else:
            upper = midpoint
    selected = mastered_set(topology, lower, seed)
    return selected, node_truth(memberships, selected)


# Func: Keep each cluster's concepts consistent with its assigned mastery state.
# Input: Overlapping clusters, node truth, and mastered cluster IDs.
# Output: Profile-aligned clusters and count of questions with no tested concepts.
def align_profile(
    clusters: dict[str, dict], truth: dict[str, bool], mastered: set[str]
) -> tuple[dict[str, dict], int]:
    aligned: dict[str, dict] = {}
    dropped = 0
    for cluster_id, record in clusters.items():
        nodes = [node for node in record["nodes"] if truth[node] == (cluster_id in mastered)]
        allowed = set(nodes)
        questions = []
        for question in record["questions"]:
            used = [node for node in question["used_terms"] if node in allowed]
            if not used:
                dropped += 1
                continue
            questions.append({**question, "used_terms": used})
        aligned[cluster_id] = {"nodes": nodes, "questions": questions}
    return aligned, dropped


# Func: Initialize one hidden student profile from saved graph and questions.
# Input: Asset paths, mastery fraction, and deterministic seed.
# Output: Aligned graph, profile-consistent questions, and hidden node truth.
def initialize(
    graph_path: Path, cluster_path: Path, question_path: Path,
    mastery_fraction: float, seed: int,
) -> SimulationData:
    graph, node_ids = load_graph(graph_path)
    clusters = load_clusters_and_questions(cluster_path, question_path, set(node_ids))
    topology, memberships = cluster_topology(clusters, graph)
    mastered, truth = assign_mastery(topology, memberships, mastery_fraction, seed)
    aligned, dropped = align_profile(clusters, truth, mastered)
    if not any(record["questions"] for record in aligned.values()):
        raise ValueError("Profile alignment left no valid assessment questions")
    return SimulationData(graph, node_ids, aligned, truth, mastered, dropped)
