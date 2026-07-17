from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def get_by_dotted_path(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split('.'):
        current = current[part]
    return current


def set_by_dotted_path(payload: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    current: Any = payload
    parts = path.split('.')
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value
    return payload
