"""Lineage emission utilities."""

from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from .utils import current_git_sha, ensure_dir, is_git_dirty


def write_lineage(
    output_dir: Path,
    *,
    run_id: str,
    command: str,
    params: dict[str, Any] | None = None,
    inputs: dict[str, str] | None = None,
    outputs: dict[str, str] | None = None,
) -> Path:
    """Persist a lineage JSON document summarising the run."""

    payload: dict[str, Any] = {
        "run_id": run_id,
        "command": command,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "git": {
            "sha": current_git_sha(),
            "dirty": is_git_dirty(),
        },
        "params": params or {},
        "inputs": inputs or {},
        "outputs": outputs or {},
    }

    lineage_dir = ensure_dir(Path(output_dir) / "lineage")
    lineage_path = lineage_dir / f"{run_id}.json"
    lineage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return lineage_path
