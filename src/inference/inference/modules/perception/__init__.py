"""Perception package: blob corridor (default) + legacy polyfit (reference)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_perception_backend(config_path: Path | None = None) -> str:
    """Return ``blob`` or ``legacy`` from lane_vision.yaml ``perception.backend``."""

    import yaml

    if config_path is None:
        # Walk up to repo root config/lane_vision.yaml
        for parent in Path(__file__).resolve().parents:
            candidate = parent / 'config' / 'lane_vision.yaml'
            if candidate.is_file():
                config_path = candidate
                break
    if config_path is None or not Path(config_path).is_file():
        return 'blob'
    try:
        with open(config_path, encoding='utf-8') as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    except OSError:
        return 'blob'
    block = data.get('perception') or {}
    raw = str(block.get('backend', 'blob')).strip().lower()
    if raw in ('legacy', 'polyfit', 'rail'):
        return 'legacy'
    return 'blob'
