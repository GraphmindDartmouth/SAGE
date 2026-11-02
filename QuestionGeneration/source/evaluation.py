"""Run optional three-model question judging and save resumable score files."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

from .earth_mover import compare_score_files
from . import read_json, save_json
from .judge import load_judge, score_question
from .questions import load_questions


# Func: Fingerprint the judge models and rubric that produced saved scores.
# Input: Model registry and evaluation settings.
# Output: Stable SHA-256 configuration fingerprint.
def judge_fingerprint(models: dict[str, Any], evaluation: dict[str, Any]) -> str:
    content = json.dumps({"models": models, "evaluation": evaluation}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# Func: Resume saved scores only when questions and rubric match exactly.
# Input: Score path, normalized source questions, and judge fingerprint.
# Output: Score records with existing validated results preserved.
def prepare_scores(path: Path, questions: list[dict[str, Any]], fingerprint: str) -> list[dict[str, Any]]:
    if not path.exists():
        return [{**question, "judge_fingerprint": fingerprint, "scores": {}} for question in questions]
    records = read_json(path)
    if not isinstance(records, list) or len(records) != len(questions):
        raise ValueError(f"Saved scores do not match the current question count: {path}")
    for index, (record, question) in enumerate(zip(records, questions)):
        if record.get("judge_fingerprint") != fingerprint:
            raise ValueError(f"Saved scores used a different judge configuration: {path}")
        if any(record.get(key) != question[key] for key in ("cluster_id", "question", "options", "answer_letter")):
            raise ValueError(f"Saved question {index} differs from the current input: {path}")
        if not isinstance(record.get("scores"), dict):
            raise ValueError(f"Saved question {index} has invalid scores: {path}")
    return records


# Func: Determine whether a score record has every rubric dimension for a model.
# Input: Score record, model name, and required dimensions.
# Output: True when the record is complete for that model.
def score_complete(record: dict[str, Any], model_name: str, dimensions: list[str]) -> bool:
    result = record["scores"].get(model_name)
    return isinstance(result, dict) and all(dimension in result for dimension in dimensions)


# Func: Score all pending questions with one loaded judge model.
# Input: Judge objects, model settings, rubric, both score sets, and output paths.
# Output: Updated score files saved after each completed question.
def score_pending(
    model_name: str,
    model_objects: tuple[Any, Any, Any],
    model_cfg: dict[str, Any],
    evaluation: dict[str, Any],
    teacher_records: list[dict[str, Any]],
    generated_records: list[dict[str, Any]],
    teacher_path: Path,
    generated_path: Path,
) -> None:
    dimensions = list(evaluation["dimensions"])
    model, tokenizer, device = model_objects
    for records, path in ((teacher_records, teacher_path), (generated_records, generated_path)):
        for index, record in enumerate(records):
            if score_complete(record, model_name, dimensions):
                continue
            record["scores"][model_name] = score_question(model, tokenizer, device, model_cfg, evaluation, record)
            save_json(path, records)
            print(f"{model_name}: scored {index + 1}/{len(records)} in {path.name}", flush=True)


# Func: Score teacher and generated questions, then compare their distributions.
# Input: All configured question paths, model settings, rubric, and output paths.
# Output: Teacher scores, generated scores, and Earth Mover comparison paths.
def run_evaluation(paths: dict[str, Path], models: dict[str, Any], evaluation: dict[str, Any]) -> tuple[Path, Path, Path]:
    teacher_questions = load_questions(paths["teacher_questions"])
    generated_questions = load_questions(paths["generated_questions"])
    fingerprint = judge_fingerprint(models, evaluation)
    teacher = prepare_scores(paths["teacher_scores"], teacher_questions, fingerprint)
    generated = prepare_scores(paths["generated_scores"], generated_questions, fingerprint)
    dimensions = list(evaluation["dimensions"])
    for model_name, model_cfg in models.items():
        if all(score_complete(record, model_name, dimensions) for record in teacher + generated):
            continue
        loaded = load_judge(str(model_cfg["path"]), str(evaluation["dtype"]), str(evaluation["device_map"]))
        try:
            score_pending(
                model_name, loaded, model_cfg, evaluation, teacher, generated,
                paths["teacher_scores"], paths["generated_scores"],
            )
        finally:
            del loaded
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    comparison = compare_score_files(
        paths["teacher_scores"], paths["generated_scores"], paths["comparison"],
        list(models), dimensions, float(evaluation["score_min"]), float(evaluation["score_max"]),
    )
    return paths["teacher_scores"], paths["generated_scores"], comparison
