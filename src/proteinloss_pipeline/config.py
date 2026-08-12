from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

ENV_OVERRIDES = {
    "PROTEINLOSS_DATA_ROOT": ("paths", "data_root"),
    "PROTEINLOSS_RUN_ROOT": ("paths", "run_root"),
    "PROTEINLOSS_SCRATCH_ROOT": ("paths", "scratch_root"),
    "PROTEINLOSS_SOURCES_CONFIG": ("sources_config",),
    "PROTEINLOSS_DEVICE": ("training", "device"),
    "PROTEINLOSS_NUM_WORKERS": ("training", "num_workers"),
    "PROTEINLOSS_SEED": ("seed",),
}


def read_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {source}")
    return payload


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce_env(value: str, existing: Any) -> Any:
    if isinstance(existing, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value


def apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(config)
    for env_name, keys in ENV_OVERRIDES.items():
        if env_name not in os.environ:
            continue
        cursor = out
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        leaf = keys[-1]
        cursor[leaf] = _coerce_env(os.environ[env_name], cursor.get(leaf))
    return out


def resolve_config(paths: list[str | Path], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one configuration file is required")
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, read_yaml(path))
    if overrides:
        merged = deep_merge(merged, overrides)
    merged = apply_env_overrides(merged)
    merged.setdefault("seed", 7)
    merged.setdefault("paths", {})
    merged["paths"].setdefault("data_root", "data")
    merged["paths"].setdefault("run_root", "runs")
    merged["paths"].setdefault("scratch_root", "scratch")
    return merged


def write_snapshot(config: dict[str, Any], directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "resolved_config.yaml"
    temporary = path.with_suffix(".yaml.part")
    temporary.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    (out / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def resolve_path(value: str | Path, *, base: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path if path.is_absolute() else Path(base) / path
