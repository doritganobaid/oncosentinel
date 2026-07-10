from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from oncosentinel.knowledge.settings_schema import CohortConfig, RunConfig

BASE_FILES = (
    ("model", "kernel.toml"),
    ("model", "backbone.toml"),
    ("model", "compute.toml"),
    ("train", "calibration.toml"),
)


def _read(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_cohorts(anchors: list[dict[str, Any]], selected: list[str]) -> list[CohortConfig]:
    catalogue = {entry["name"]: entry for entry in anchors}
    missing = [name for name in selected if name not in catalogue]
    if missing:
        raise KeyError(f"unknown cohorts requested: {missing}")
    return [CohortConfig(**catalogue[name]) for name in selected]


def load_config(experiment: str, configs_dir: Path) -> RunConfig:
    merged: dict[str, Any] = {}
    for sub, name in BASE_FILES:
        _deep_merge(merged, _read(configs_dir / sub / name))
    anchors = _read(configs_dir / "data" / "anchors.toml")["cohort"]
    experiment_data = _read(configs_dir / "experiment" / f"{experiment}.toml")
    _deep_merge(merged, experiment_data)
    selected = list(merged.get("run", {}).get("cohorts", []))
    cohorts = _resolve_cohorts(anchors, selected)
    return RunConfig.from_merged(merged, cohorts)


def default_configs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"
