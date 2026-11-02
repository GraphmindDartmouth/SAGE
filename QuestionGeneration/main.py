"""Generate validated multiple-choice questions for each SAGE concept cluster."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from source.generation import generate_questions


# Func: Resolve a configured path from the QuestionGeneration directory.
# Input: Project directory and absolute or relative path string.
# Output: Absolute normalized Path.
def resolve_path(project_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


# Func: Generate cluster questions using the configured OpenAI-compatible model.
# Input: Hydra configuration with cluster, examples, output, model, and prompt settings.
# Output: A resumable questions.json file with the requested questions per cluster.
@hydra.main(version_base=None, config_path="configurations", config_name="main")
def main(cfg: DictConfig) -> None:
    project_dir = Path(__file__).resolve().parent
    cluster_path = resolve_path(project_dir, str(cfg.paths.clusters))
    few_shot_path = resolve_path(project_dir, str(cfg.paths.few_shots))
    output_path = resolve_path(project_dir, str(cfg.paths.output))
    api_key = os.getenv(str(cfg.model.api_key_env))
    if not api_key:
        raise RuntimeError(f"Required API key environment variable is not set: {cfg.model.api_key_env}")
    from openai import OpenAI

    client_args = {"api_key": api_key}
    if cfg.model.base_url:
        client_args["base_url"] = str(cfg.model.base_url)
    client = OpenAI(**client_args)
    output = generate_questions(
        cluster_path, few_shot_path, output_path,
        OmegaConf.to_container(cfg.generation, resolve=True),
        OmegaConf.to_container(cfg.prompt, resolve=True),
        OmegaConf.to_container(cfg.model, resolve=True),
        client,
    )
    print(f"Saved cluster questions to {output}")


if __name__ == "__main__":
    main()
