"""Optionally judge generated and teacher questions, then compare their score distributions."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from source.evaluation import run_evaluation


# Func: Resolve a configured path from the QuestionGeneration directory.
# Input: Project directory and absolute or relative path string.
# Output: Absolute normalized Path.
def resolve_path(project_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return path.resolve() if path.is_absolute() else (project_dir / path).resolve()


# Func: Score both question sets with the configured judges and compute Earth Mover distances.
# Input: Hydra evaluation configuration with paths, models, and rubric prompts.
# Output: Two resumable score JSON files and one comparison JSON file.
@hydra.main(version_base=None, config_path="configurations", config_name="evaluations")
def main(cfg: DictConfig) -> None:
    project_dir = Path(__file__).resolve().parent
    paths = {name: resolve_path(project_dir, str(value)) for name, value in cfg.paths.items()}
    teacher, generated, comparison = run_evaluation(
        paths,
        OmegaConf.to_container(cfg.models, resolve=True),
        OmegaConf.to_container(cfg.evaluation, resolve=True),
    )
    print(f"Saved teacher scores to {teacher}")
    print(f"Saved generated scores to {generated}")
    print(f"Saved Earth Mover comparison to {comparison}")


if __name__ == "__main__":
    main()
