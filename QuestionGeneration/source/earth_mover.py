"""Compare teacher and generated score distributions with exact one-dimensional Earth Mover distance."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wasserstein_distance

from . import read_json, save_json


# Func: Extract complete scores for one judge model and evaluation dimension.
# Input: Scored question records, model name, and dimension name.
# Output: A finite one-dimensional score array.
def score_values(records: list[dict[str, Any]], model: str, dimension: str) -> np.ndarray:
    values: list[float] = []
    for index, record in enumerate(records):
        try:
            value = float(record["scores"][model][dimension])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing {model}/{dimension} score at question {index}") from exc
        if not np.isfinite(value):
            raise ValueError(f"Non-finite {model}/{dimension} score at question {index}")
        values.append(value)
    if not values:
        raise ValueError("Both question sets must contain at least one scored question")
    return np.asarray(values, dtype=np.float64)


# Func: Compute exact Earth Mover distance and normalized similarity on a score axis.
# Input: Teacher scores, generated scores, and the score range.
# Output: Distance in score points and similarity between zero and one.
def compare_axis(teacher: np.ndarray, generated: np.ndarray, score_min: float, score_max: float) -> dict[str, float]:
    if score_max <= score_min:
        raise ValueError("evaluation.score_max must exceed score_min")
    if np.any((teacher < score_min) | (teacher > score_max)) or np.any((generated < score_min) | (generated > score_max)):
        raise ValueError("Judge scores must lie inside the configured score range")
    distance = float(wasserstein_distance(teacher, generated))
    return {
        "earth_mover_distance": distance,
        "similarity": float(np.clip(1.0 - distance / (score_max - score_min), 0.0, 1.0)),
    }


# Func: Compare every configured model and rubric dimension without subsampling.
# Input: Teacher and generated score records, models, dimensions, and score range.
# Output: A complete comparison including counts and overall mean similarity.
def compare_distributions(
    teacher_records: list[dict[str, Any]],
    generated_records: list[dict[str, Any]],
    models: list[str],
    dimensions: list[str],
    score_min: float,
    score_max: float,
) -> dict[str, Any]:
    if not models or not dimensions:
        raise ValueError("At least one judge model and one rubric dimension are required")
    results: dict[str, dict[str, dict[str, float]]] = {}
    similarities: list[float] = []
    for model in models:
        results[model] = {}
        for dimension in dimensions:
            teacher = score_values(teacher_records, model, dimension)
            generated = score_values(generated_records, model, dimension)
            result = compare_axis(teacher, generated, score_min, score_max)
            results[model][dimension] = result
            similarities.append(result["similarity"])
    return {
        "teacher_count": len(teacher_records),
        "generated_count": len(generated_records),
        "teacher_choice_counts": dict(sorted(Counter(len(record.get("options", [])) for record in teacher_records).items())),
        "generated_choice_counts": dict(sorted(Counter(len(record.get("options", [])) for record in generated_records).items())),
        "score_range": [score_min, score_max],
        "models": results,
        "mean_similarity": float(np.mean(similarities)),
    }


# Func: Load both score files and persist their Earth Mover comparison.
# Input: Teacher scores path, generated scores path, output path, and comparison settings.
# Output: Path to the saved comparison JSON.
def compare_score_files(
    teacher_path: Path,
    generated_path: Path,
    output_path: Path,
    models: list[str],
    dimensions: list[str],
    score_min: float,
    score_max: float,
) -> Path:
    teacher_records = read_json(teacher_path)
    generated_records = read_json(generated_path)
    if not isinstance(teacher_records, list) or not isinstance(generated_records, list):
        raise ValueError("Both score files must contain JSON lists")
    report = compare_distributions(teacher_records, generated_records, models, dimensions, score_min, score_max)
    return save_json(output_path, report)
