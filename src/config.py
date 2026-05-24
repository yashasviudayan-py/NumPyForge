"""JSON configuration helpers for pipeline and serving entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar, cast

from src.types import PathLikeString

Config = dict[str, Any]
T = TypeVar("T")


def load_json_config(path: PathLikeString) -> Config:
    """Load a JSON object from disk."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Config must contain a JSON object: {config_path}")
    return cast(Config, value)


def deep_merge(base: Config, override: Config) -> Config:
    """Return a recursive merge where override values win."""
    merged: Config = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(cast(Config, current), cast(Config, value))
        else:
            merged[key] = value
    return merged


def get_config_value(config: Config, dotted_path: str, expected_type: type[T]) -> T:
    """Read and type-check a nested config value."""
    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing config value: {dotted_path}")
        value = value[key]
    if not isinstance(value, expected_type):
        raise TypeError(f"Config value {dotted_path!r} must be {expected_type.__name__}.")
    return value
