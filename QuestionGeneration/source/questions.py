"""Normalize generated cluster questions and teacher examples for shared judging."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import read_json


OPTION_RE = re.compile(r"(?m)^\s*([A-H])\s*[.)]\s+(.+?)\s*$", re.IGNORECASE)


# Func: Split a teacher question blob into its stem and labeled choices.
# Input: A qn string with one choice per line.
# Output: Question stem and ordered choice lines.
def split_question_blob(text: str) -> tuple[str, list[str]]:
    matches = list(OPTION_RE.finditer(text))
    if len(matches) < 2:
        raise ValueError("Teacher question needs labeled choices")
    stem = text[:matches[0].start()].strip()
    options = [f"{match.group(1).upper()}. {match.group(2).strip()}" for match in matches]
    if not stem or len({option[0] for option in options}) != len(options):
        raise ValueError("Teacher question has an empty stem or repeated choice labels")
    return stem, options


# Func: Normalize one generated or teacher question into a common record.
# Input: Raw question item and its cluster identifier.
# Output: Stem, choices, answer key if available, and cluster ID.
def normalize_question(item: dict[str, Any], cluster_id: str) -> dict[str, Any]:
    if "qn" in item:
        stem, options = split_question_blob(str(item["qn"]))
    else:
        stem = str(item.get("question", item.get("qstn", ""))).strip()
        options = [str(option).strip() for option in item.get("options", [])]
        if not stem or len(options) < 2 or any(not option for option in options):
            raise ValueError("Question needs a stem and at least two choices")
    return {
        "cluster_id": cluster_id,
        "question": stem,
        "options": options,
        "answer_letter": item.get("answer_letter"),
    }


# Func: Flatten supported question JSON into a single judge-ready list.
# Input: Path to grouped generated questions or a teacher example list.
# Output: Ordered normalized question records.
def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Question file must contain a JSON list: {path}")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Question item {index} must be a JSON object")
        cluster_id = str(item.get("cluster_id", "__teacher__"))
        if "questions" in item:
            if not isinstance(item["questions"], list):
                raise ValueError(f"Cluster {cluster_id} questions must be a list")
            records.extend(normalize_question(question, cluster_id) for question in item["questions"])
        else:
            records.append(normalize_question(item, cluster_id))
    if not records:
        raise ValueError(f"Question file contains no questions: {path}")
    return records
