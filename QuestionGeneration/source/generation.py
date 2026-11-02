"""Generate and validate multiple-choice questions from SAGE concept clusters."""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from . import read_json, save_json


LETTERS = ("A", "B", "C", "D")
OPTION_RE = re.compile(r"(?m)^\s*([A-D])\s*[.)]\s+(.+?)\s*$", re.IGNORECASE)
ANSWER_RE = re.compile(r"(?im)^\s*Answer\s*:\s*([A-D])\s*$")
TERMS_RE = re.compile(r"(?im)^\s*Concept terms used\s*:\s*$")


# Func: Load unique concept node strings for each nonempty cluster.
# Input: Path to GraphClustering's clusters.json.
# Output: Ordered cluster IDs mapped to their concept statements.
def load_clusters(path: Path) -> dict[str, list[str]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("clusters"), dict):
        raise ValueError("Cluster JSON must contain a clusters object")
    clusters: dict[str, list[str]] = {}
    for cluster_id, raw_terms in payload["clusters"].items():
        if not isinstance(raw_terms, list) or any(not isinstance(term, str) for term in raw_terms):
            raise ValueError(f"Cluster {cluster_id} must contain a list of concept strings")
        terms = list(dict.fromkeys(term.strip() for term in raw_terms if term.strip()))
        if terms:
            clusters[str(cluster_id)] = terms
    if not clusters:
        raise ValueError("No nonempty clusters were found")
    return dict(sorted(clusters.items(), key=lambda item: (not item[0].isdigit(), int(item[0]) if item[0].isdigit() else item[0])))


# Func: Load teacher examples used only as style references.
# Input: Path to a JSON list of qn or question records.
# Output: Nonempty example question strings.
def load_few_shots(path: Path) -> list[str]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("Few-shot JSON must be a list")
    examples = []
    for item in payload:
        if isinstance(item, dict):
            question = item.get("qn") or item.get("question")
            if isinstance(question, str) and question.strip():
                examples.append(question.strip())
    if not examples:
        raise ValueError("Few-shot JSON contains no example questions")
    return examples


# Func: Format a cluster and sampled teacher examples as chat messages.
# Input: Prompt templates, topic, concept terms, focus terms, examples, count, and RNG.
# Output: System and user messages for one model request.
def build_messages(
    prompt: dict[str, str], topic: str, terms: list[str], focus_terms: list[str],
    examples: list[str], count: int, rng: random.Random,
) -> list[dict[str, str]]:
    if count < 0:
        raise ValueError("generation.few_shot_examples must be non-negative")
    sampled = rng.sample(examples, min(count, len(examples)))
    shuffled = terms.copy()
    rng.shuffle(shuffled)
    return [
        {"role": "system", "content": prompt["system"].format(topic=topic)},
        {"role": "user", "content": prompt["user_template"].format(
            examples="\n\n".join(sampled) if sampled else "None",
            terms="\n".join(f"- {term}" for term in shuffled),
            focus_terms="\n".join(f"- {term}" for term in focus_terms),
        )},
    ]


# Func: Parse one model response and retain only exact supplied concept labels.
# Input: Raw response text and the cluster's concept statements.
# Output: Validated question, four choices, answer, and used terms.
def parse_question(text: str, terms: list[str]) -> dict[str, Any]:
    cleaned = re.sub(r"\*\*(.*?)\*\*|__(.*?)__", lambda match: match.group(1) or match.group(2), text).strip()
    question_match = re.search(r"(?im)^\s*Question\s*:\s*(.+?)\s*$", cleaned)
    options = list(OPTION_RE.finditer(cleaned))
    answer_match = ANSWER_RE.search(cleaned)
    terms_match = TERMS_RE.search(cleaned)
    if not question_match or len(options) != 4 or not answer_match or not terms_match:
        raise ValueError("Response must contain a question, four choices, an answer, and used terms")
    if [match.group(1).upper() for match in options] != list(LETTERS):
        raise ValueError("Choices must appear once each in A, B, C, D order")
    if not (question_match.end() <= options[0].start() < options[-1].end() <= answer_match.start() < terms_match.start()):
        raise ValueError("Question fields are not in the requested order")
    stem = question_match.group(1).strip()
    choice_texts = [match.group(2).strip() for match in options]
    if len(stem.split()) < 5 or any(not choice for choice in choice_texts):
        raise ValueError("Question stem or answer choice is too short")
    if len({re.sub(r"\W+", "", choice).casefold() for choice in choice_texts}) != 4:
        raise ValueError("Answer choices must be distinct")
    canonical = {re.sub(r"\s+", " ", term).casefold(): term for term in terms}
    used: list[str] = []
    for line in cleaned[terms_match.end():].splitlines():
        candidate = re.sub(r"^\s*[-*]\s*", "", line).strip()
        if not candidate:
            continue
        key = re.sub(r"\s+", " ", candidate).casefold()
        if key not in canonical:
            raise ValueError(f"Model reported a concept outside this cluster: {candidate}")
        term = canonical[key]
        if term not in used:
            used.append(term)
    if not used:
        raise ValueError("Model did not report any used cluster concepts")
    return {
        "question": stem,
        "options": [f"{letter}. {choice}" for letter, choice in zip(LETTERS, choice_texts)],
        "answer_letter": answer_match.group(1).upper(),
        "used_terms": used,
    }


# Func: Fingerprint the generation inputs that determine question wording.
# Input: Generation settings, prompt, model settings, and teacher examples.
# Output: Stable SHA-256 fingerprint for resume validation.
def generation_fingerprint(
    generation: dict[str, Any], prompt: dict[str, str], model: dict[str, Any], examples: list[str]
) -> str:
    settings = {key: value for key, value in generation.items() if key != "questions_per_cluster"}
    content = json.dumps({"generation": settings, "prompt": prompt, "model": model, "examples": examples}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# Func: Validate saved questions before resuming an interrupted run.
# Input: Existing output path, clusters, desired question count, and generation fingerprint.
# Output: Current per-cluster records with preserved valid questions.
def load_progress(path: Path, clusters: dict[str, list[str]], target: int, fingerprint: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {
            cluster_id: {"cluster_id": cluster_id, "cluster_terms": terms, "generation_fingerprint": fingerprint, "questions": []}
            for cluster_id, terms in clusters.items()
        }
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("Existing questions output must be a list")
    records = {str(item["cluster_id"]): item for item in payload}
    if len(records) != len(payload) or set(records) != set(clusters):
        raise ValueError("Existing questions output has different cluster IDs; choose another output path")
    for cluster_id, terms in clusters.items():
        record = records[cluster_id]
        if record.get("cluster_terms") != terms or not isinstance(record.get("questions"), list):
            raise ValueError(f"Saved cluster {cluster_id} no longer matches its input concepts")
        if record.get("generation_fingerprint") != fingerprint:
            raise ValueError(f"Saved cluster {cluster_id} used different generation settings; choose another output path")
        if len(record["questions"]) > target:
            raise ValueError(f"Saved cluster {cluster_id} has more than generation.questions_per_cluster questions")
        signatures: set[str] = set()
        for question in record["questions"]:
            if (
                not isinstance(question, dict)
                or not isinstance(question.get("question"), str)
                or not question["question"].strip()
                or not isinstance(question.get("options"), list)
                or len(question["options"]) != 4
                or any(not isinstance(option, str) or not option.strip() for option in question["options"])
                or question.get("answer_letter") not in LETTERS
                or not isinstance(question.get("used_terms"), list)
                or not question["used_terms"]
                or any(not isinstance(term, str) for term in question["used_terms"])
                or not set(question["used_terms"]).issubset(terms)
            ):
                raise ValueError(f"Saved cluster {cluster_id} contains an invalid question")
            signature = question["question"].casefold()
            if signature in signatures:
                raise ValueError(f"Saved cluster {cluster_id} contains duplicate questions")
            signatures.add(signature)
    return records


# Func: Generate the configured number of questions for every cluster.
# Input: Paths, generation settings, prompt, model settings, and an OpenAI-compatible client.
# Output: Resumable grouped questions saved as JSON.
def generate_questions(
    cluster_path: Path,
    few_shot_path: Path,
    output_path: Path,
    generation: dict[str, Any],
    prompt: dict[str, str],
    model: dict[str, Any],
    client: Any,
) -> Path:
    clusters = load_clusters(cluster_path)
    examples = load_few_shots(few_shot_path)
    target = int(generation["questions_per_cluster"])
    attempts = int(generation["max_attempts_per_question"])
    focus_count = int(generation["focus_terms_per_question"])
    max_terms = int(generation["max_terms_per_prompt"])
    if target < 1 or attempts < 1 or focus_count < 1 or max_terms < 1 or focus_count > max_terms:
        raise ValueError("Question count, attempts, and prompt term limits must be positive; focus count cannot exceed the prompt limit")
    fingerprint = generation_fingerprint(generation, prompt, model, examples)
    records = load_progress(output_path, clusters, target, fingerprint)
    rng = random.Random(int(generation["seed"]))
    for cluster_id, terms in clusters.items():
        questions = records[cluster_id]["questions"]
        signatures = {question["question"].casefold() for question in questions}
        usage = {term: 0 for term in terms}
        for question in questions:
            for term in question.get("used_terms", []):
                if term in usage:
                    usage[term] += 1
        while len(questions) < target:
            error: Exception | None = None
            for _ in range(attempts):
                focus_terms = terms.copy()
                rng.shuffle(focus_terms)
                focus_terms.sort(key=lambda term: usage[term])
                presented_terms = focus_terms[:max_terms]
                messages = build_messages(
                    prompt, str(generation["topic"]), presented_terms, presented_terms[:focus_count],
                    examples, int(generation["few_shot_examples"]), rng,
                )
                try:
                    response = client.chat.completions.create(
                        model=str(model["id"]),
                        messages=messages,
                        temperature=float(model["temperature"]),
                        max_tokens=int(model["max_tokens"]),
                    )
                except Exception as exc:
                    error = exc
                    continue
                try:
                    record = parse_question(response.choices[0].message.content or "", presented_terms)
                    signature = record["question"].casefold()
                    if signature in signatures:
                        raise ValueError("Duplicate question for this cluster")
                except ValueError as exc:
                    error = exc
                    continue
                questions.append(record)
                signatures.add(signature)
                for term in record["used_terms"]:
                    usage[term] += 1
                save_json(output_path, list(records.values()))
                break
            else:
                raise RuntimeError(f"Could not generate question {len(questions) + 1}/{target} for cluster {cluster_id}: {error}")
    return save_json(output_path, list(records.values()))
