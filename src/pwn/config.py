"""Configuration loading for the potato-weight-nutrition pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PipelineSettings(BaseModel):
    """Typed configuration for the pipeline."""

    input_dir: Path = Field(default=Path("data"))
    output_dir: Path = Field(default=Path("reports"))
    analysis_dir: Path = Field(default=Path("reports/analysis"))
    figures_dir: Path = Field(default=Path("reports/figs"))
    reports_dir: Path = Field(default=Path("reports"))
    seed: int = 42
    metrics_filename: str = "metrics.json"
    teaser_filename: str = "teaser.png"
    lineage_enabled: bool = False
    observability_port: int = 9157
    model: str = "ols"

    class Config:
        arbitrary_types_allowed = True


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_toml(path: Path) -> dict[str, Any]:
    import tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_settings(config_path: Path | None) -> PipelineSettings:
    """Load settings from optional config file."""

    data: dict[str, Any] = {}
    if config_path is not None:
        suffix = config_path.suffix.lower()
        if suffix in {".yml", ".yaml"}:
            data = _load_yaml(config_path)
        elif suffix == ".toml":
            data = _load_toml(config_path)
        elif suffix == ".json":
            data = _load_json(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    return PipelineSettings(**data)
