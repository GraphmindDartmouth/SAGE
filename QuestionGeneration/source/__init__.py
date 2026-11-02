"""Shared JSON operations for question generation and evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


# Func: Read a required JSON input or completed output.
# Input: Path to an existing UTF-8 JSON file.
# Output: Parsed JSON content.
def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# Func: Save JSON through a temporary file to avoid partial writes.
# Input: Destination path and JSON-serializable content.
# Output: The written destination path.
def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
