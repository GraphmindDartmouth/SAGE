"""Simulate a SAGE learner and assess Beta-Bernoulli beliefs over a concept line graph."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from initialization import SimulationData, initialize
from source.propagation import chebyshev, distance_weighted, heat_kernel, ppnrp
from source.propagation.common import apply_virtual_evidence, knowledge_vector
from source.questionselection import boundary_submodular_eig, cutmass_submodular_eig
from source.questionselection import frontier_submodular_eig, random as random_selector, submodular_eig
from source.scoring.beta import initialize_beliefs
from source.scoring.evidence import apply_direct_evidence, response_evidence
from source.scoring.judge import parse_judgment, parse_response
from source.scoring.metrics import summarize
from student import LLMResponder, load_examples


# Func: Resolve a configured path relative to the StudentSimulation directory.
# Input: Project directory and absolute or relative path string.
# Output: Absolute normalized path.
def resolve_path(project_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


# Func: Atomically save the current trajectory without partial JSON output.
# Input: Output path and JSON-serializable simulation record.
# Output: Written output path.
def save_trajectory(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


# Func: Construct one configured graph diffusion operator before assessment.
# Input: Graph data and propagation method settings.
# Output: Reusable operator or None for Distance-Weighted.
def build_operator(data: SimulationData, config: dict) -> Any:
    name = str(config["name"])
    if name == "ppnrp":
        return ppnrp.build_operator(data.graph, data.node_ids, float(config["restart"]))
    if name == "heat_kernel":
        return heat_kernel.build_operator(data.graph, data.node_ids, float(config["time"]))
    if name == "chebyshev":
        return chebyshev.build_operator(
            data.graph, data.node_ids, int(config["terms"]),
            config.get("coefficients"), bool(config["jackson_damping"]),
        )
    if name == "distance_weighted":
        return None
    raise ValueError(f"Unknown propagation method: {name}")


# Func: Apply one of the paper's four signed propagation operators.
# Input: Method name, prepared operator, and current knowledge vector.
# Output: Signed propagated signal in node order.
def propagate_signal(name: str, operator: Any, vector: Any):
    if name == "ppnrp":
        return ppnrp.propagate(operator, vector)
    if name == "heat_kernel":
        return heat_kernel.propagate(operator, vector)
    if name == "chebyshev":
        return chebyshev.propagate(operator, vector)
    if name == "distance_weighted":
        return distance_weighted.propagate(vector)
    raise ValueError(f"Unknown propagation method: {name}")


# Func: Select one unasked question with the configured policy.
# Input: Simulation data, beliefs, asked and covered sets, policy settings, and RNG.
# Output: Selected question and graph reach, or None when exhausted.
def select_question(data: SimulationData, beliefs: dict, asked: set, covered: set,
                    config: dict, rng: random.Random):
    name = str(config["name"])
    if name == "random":
        return random_selector.select(data.clusters, asked, data.graph, int(config["neighbor_hops"]), rng)
    policies = {
        "submodular_eig": submodular_eig,
        "boundary_submodular_eig": boundary_submodular_eig,
        "cutmass_submodular_eig": cutmass_submodular_eig,
        "frontier_submodular_eig": frontier_submodular_eig,
    }
    if name not in policies:
        raise ValueError(f"Unknown question-selection policy: {name}")
    return policies[name].select(data.clusters, beliefs, data.graph, asked, covered, config, rng)


# Func: Run a fully initialized learner simulation with an injectable responder.
# Input: Data, grouped settings, response examples, responder, and optional output path.
# Output: Complete step trajectory, final beliefs, hidden truth, and summary.
def run_simulation(
    data: SimulationData, config: dict, examples: dict[str, list],
    responder: Any, output_path: Path | None = None,
) -> dict:
    scoring = config["scoring"]
    propagation = config["propagation"]
    selection = config["QuestionSelection"]
    simulation = config["simulation"]
    profile_prompts = config["StudentProfiles"]
    model = config["model"]
    seed = int(simulation["seed"])
    selection_rng = random.Random(seed)
    example_rng = random.Random(f"examples:{seed}")
    beliefs = initialize_beliefs(data.node_ids, float(scoring["prior_alpha"]), float(scoring["prior_beta"]))
    operator = build_operator(data, propagation)
    asked: set[tuple[str, int]] = set()
    covered: set[str] = set()
    steps: list[dict] = []
    skipped_questions: list[dict] = []
    max_questions = int(simulation["max_questions"])
    attempts = int(model["max_attempts_per_question"])
    if max_questions < 1 or attempts < 1:
        raise ValueError("simulation.max_questions and model.max_attempts_per_question must be positive")
    if int(simulation["examples_per_response"]) < 0:
        raise ValueError("simulation.examples_per_response must be nonnegative")
    trajectory = {
        "propagation": propagation["name"],
        "question_selection": selection["name"],
        "mastered_clusters": sorted(data.mastered_clusters),
        "ground_truth": data.truth,
        "dropped_overlap_questions": data.dropped_overlap_questions,
        "steps": steps,
        "skipped_questions": skipped_questions,
    }
    while len(steps) < max_questions:
        selected = select_question(data, beliefs, asked, covered, selection, selection_rng)
        if selected is None:
            break
        cluster_id, index, tested, reached = selected
        question = data.clusters[cluster_id]["questions"][index]
        cluster_nodes = data.clusters[cluster_id]["nodes"]
        profile = "mastered" if cluster_id in data.mastered_clusters else "unmastered"
        failure: Exception | None = None
        for _attempt in range(attempts):
            try:
                response = responder.respond(
                    profile, question, tested, examples.get(profile, []),
                    profile_prompts, str(simulation["topic"]),
                    int(simulation["examples_per_response"]), example_rng,
                )
                answer, reasoning = parse_response(response)
                chunk_size = int(scoring["judge_chunk_size"])
                if chunk_size < 1:
                    raise ValueError("scoring.judge_chunk_size must be positive")
                chunks = [
                    cluster_nodes[start:start + chunk_size]
                    for start in range(0, len(cluster_nodes), chunk_size)
                ]
                signs = {}
                for chunk in chunks:
                    judgment = responder.judge(
                        question, answer, reasoning, chunk, scoring, str(simulation["topic"]),
                    )
                    signs.update(parse_judgment(judgment, chunk))
                break
            except (ValueError, KeyError) as exc:
                failure = exc
        else:
            skipped_questions.append({
                "cluster_id": cluster_id, "question_index": index, "reason": str(failure),
            })
            asked.add((cluster_id, index))
            continue
        correct = answer == question["answer_letter"]
        evidence = response_evidence(
            cluster_nodes, tested, correct, signs,
            float(scoring["answer_weight"]), float(scoring["reasoning_weight"]),
        )
        apply_direct_evidence(beliefs, evidence, float(scoring["alpha_local"]))
        vector = knowledge_vector(beliefs, data.node_ids)
        signal = propagate_signal(str(propagation["name"]), operator, vector)
        changed = apply_virtual_evidence(
            beliefs, data.graph, set(tested), data.node_ids, signal, float(propagation["strength"]),
        )
        asked.add((cluster_id, index))
        covered.update(reached)
        steps.append({
            "step": len(steps) + 1,
            "cluster_id": cluster_id,
            "question_index": index,
            "question": question["question"],
            "tested_nodes": tested,
            "student_profile": profile,
            "answer": answer,
            "correct": correct,
            "reasoning": reasoning,
            "judge_signs": signs,
            "direct_evidence": evidence,
            "propagated_nodes_changed": changed,
            "metrics": summarize(beliefs, data.truth),
        })
        trajectory["final_beliefs"] = beliefs
        trajectory["summary"] = steps[-1]["metrics"]
        if output_path is not None:
            save_trajectory(output_path, trajectory)
    if not steps:
        trajectory["final_beliefs"] = beliefs
        trajectory["summary"] = summarize(beliefs, data.truth)
    trajectory["questions_asked"] = len(steps)
    trajectory["stop_reason"] = "budget" if len(steps) == max_questions else "questions_exhausted"
    if output_path is not None:
        save_trajectory(output_path, trajectory)
    return trajectory


# Func: Load Hydra settings, initialize a hidden student, and run assessment.
# Input: Paths, selected methods, scoring settings, model, and question budget.
# Output: Saved trajectory JSON under the configured destination.
def main(cfg: Any) -> None:
    from omegaconf import OmegaConf

    project_dir = Path(__file__).resolve().parent
    paths = {
        key: resolve_path(project_dir, str(value)) if value is not None else None
        for key, value in cfg.paths.items()
    }
    data = initialize(
        paths["graph"], paths["clusters"], paths["questions"],
        float(cfg.simulation.mastery_fraction), int(cfg.simulation.seed),
    )
    key = os.getenv(str(cfg.model.api_key_env))
    if not key:
        raise RuntimeError(f"Required API key environment variable is not set: {cfg.model.api_key_env}")
    settings = OmegaConf.to_container(cfg, resolve=True)
    responder = LLMResponder(key, settings["model"])
    examples = {
        "mastered": load_examples(paths["mastered_examples"]),
        "unmastered": load_examples(paths["unmastered_examples"]),
    }
    result = run_simulation(data, settings, examples, responder, paths["output"])
    print(f"Saved {result['questions_asked']} simulated interactions to {paths['output']}")


# Func: Start the configured simulation through Hydra's command-line interface.
# Input: Optional Hydra overrides from the command line.
# Output: Saved trajectory from main.
def cli() -> None:
    import hydra

    hydra.main(version_base=None, config_path="configurations", config_name="main")(main)()


if __name__ == "__main__":
    cli()
