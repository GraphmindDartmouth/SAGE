"""Parse simulated answers and concept-level reasoning judgments safely."""

from __future__ import annotations

import re


ANSWER = re.compile(r"(?im)^\s*Answer\s*:\s*([A-D])\b")
REASONING = re.compile(r"(?is)^\s*Reasoning\s*:\s*(.+)$", re.MULTILINE)
HEADERS = {"well_mastered_nodes": 1, "confused_nodes": -1, "unknown_nodes": 0}


# Func: Extract one answer choice and a nonempty reasoning trace.
# Input: Raw simulated student text.
# Output: Answer letter and reasoning text.
def parse_response(text: str) -> tuple[str, str]:
    answer = ANSWER.search(text)
    reasoning = REASONING.search(text)
    if not answer or not reasoning or not reasoning.group(1).strip():
        raise ValueError("Student response must contain Answer and Reasoning fields")
    return answer.group(1).upper(), reasoning.group(1).strip()


# Func: Resolve a judge concept label only when it matches one exact concept.
# Input: Raw judge node text and cluster node IDs.
# Output: Exact node ID or None for unmatched labels.
def resolve_node(raw: str, nodes: list[str]) -> str | None:
    normalized = " ".join(raw.casefold().split())
    matches = [node for node in nodes if " ".join(node.casefold().split()) == normalized]
    return matches[0] if len(matches) == 1 else None


# Func: Parse three judge categories into signed reasoning evidence.
# Input: Raw judge text and exact cluster node IDs.
# Output: Node-to-sign mapping with unmentioned nodes marked unknown.
def parse_judgment(text: str, nodes: list[str]) -> dict[str, int]:
    signs = {node: 0 for node in nodes}
    assigned: set[str] = set()
    current: int | None = None
    seen_headers: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("`*").strip()
        header = line.rstrip(":").casefold()
        if header in HEADERS and line.endswith(":"):
            current = HEADERS[header]
            seen_headers.add(header)
            continue
        if current is None or not line:
            continue
        item = re.sub(r"^[-*]\s*", "", line).strip().strip("`*").strip()
        if item.casefold() in {"none", "<node or none>", ""}:
            continue
        node = resolve_node(item, nodes)
        if node is None:
            raise ValueError(f"Judge returned a concept outside this cluster: {item}")
        if node in assigned:
            raise ValueError(f"Judge classified a concept more than once: {node}")
        signs[node] = current
        assigned.add(node)
    if seen_headers != set(HEADERS):
        raise ValueError("Judge must provide mastered, confused, and unknown sections")
    if assigned != set(nodes):
        raise ValueError("Judge must classify every cluster concept exactly once")
    return signs
