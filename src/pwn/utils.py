"""Utility helpers for the potato-weight-nutrition pipeline."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

LOGGER_NAME = "pwn"


def ensure_dir(path: Path) -> Path:
    """Create a directory if it is missing and return the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and hash randomisation for determinism."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = "0"


@dataclass(frozen=True)
class RunMetadata:
    """Basic metadata captured for each CLI invocation."""

    run_id: str
    command: str
    started_at: datetime

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "command": self.command,
            "started_at": self.started_at.isoformat(timespec="seconds"),
        }


def json_dumps(payload: dict[str, Any]) -> str:
    """Serialise payload to pretty JSON for CLI output."""

    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def current_git_sha() -> str | None:
    """Return the current git commit SHA if available."""

    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - git may be unavailable in CI artefacts
        return None


def is_git_dirty() -> bool:
    """Return True when the working tree has uncommitted changes."""

    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode != 0
    except Exception:  # pragma: no cover - git may be unavailable
        return False


def human_join(items: Iterable[str], final: str = "and") -> str:
    """Return a human-readable joined string."""

    items_list = list(items)
    if not items_list:
        return ""
    if len(items_list) == 1:
        return items_list[0]
    return f"{', '.join(items_list[:-1])} {final} {items_list[-1]}"


def configure_logging(verbose: bool = False) -> logging.Logger:
    """Configure JSON console logging for the pipeline."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    for handler in list(logger.handlers):  # reset when invoked multiple times
        logger.removeHandler(handler)

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting only
            payload: dict[str, Any] = {
                "timestamp": datetime.utcfromtimestamp(record.created).isoformat(timespec="milliseconds") + "Z",
                "level": record.levelname,
                "message": record.getMessage(),
            }
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            for key in ("run_id", "step"):
                value = getattr(record, key, None)
                if value is not None:
                    payload[key] = value
            return json.dumps(payload)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger
