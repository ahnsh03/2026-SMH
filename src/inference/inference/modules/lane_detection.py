"""Lane perception façade.

Default backend: **blob** (road mask denoise → centerline).
Legacy polyfit loads only when ``backend=legacy`` or ``enable_fork`` needs it.
"""

from __future__ import annotations

import importlib
from typing import Any

from inference.modules.perception import load_perception_backend
from inference.modules.perception.types import (  # noqa: F401
    ForkLanePair,
    LaneBoundary,
    LaneDebugFrame,
    LaneDetections,
    LaneMarking,
    RoadBranch,
)

_BACKEND = None
_legacy_mod = None


def _legacy():
    global _legacy_mod
    if _legacy_mod is None:
        _legacy_mod = importlib.import_module(
            'inference.modules.perception.legacy.lane_detection'
        )
    return _legacy_mod


def _blob_detect_mod():
    return importlib.import_module('inference.modules.perception.blob.detect')


def get_perception_backend() -> str:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = load_perception_backend()
    return _BACKEND


def set_perception_backend(name: str | None) -> None:
    global _BACKEND
    if name is None:
        _BACKEND = None
        return
    key = str(name).strip().lower()
    _BACKEND = 'legacy' if key in ('legacy', 'polyfit', 'rail') else 'blob'


def reset_tracking_state() -> None:
    if get_perception_backend() == 'blob':
        _blob_detect_mod().reset_tracking_state()
        return
    _legacy().reset_tracking_state()


def detect(
    frame: Any,
    *,
    active_branch_rank: int | None = None,
    prefer_yellow: bool | None = None,
    enable_fork: bool = True,
) -> Any:
    if get_perception_backend() == 'blob':
        return _blob_detect_mod().detect(
            frame,
            active_branch_rank=active_branch_rank,
            prefer_yellow=prefer_yellow,
            enable_fork=enable_fork,
        )
    return _legacy().detect(
        frame,
        active_branch_rank=active_branch_rank,
        prefer_yellow=prefer_yellow,
        enable_fork=enable_fork,
    )


def detect_with_debug(
    frame: Any,
    *,
    active_branch_rank: int | None = None,
    prefer_yellow: bool | None = None,
    enable_fork: bool = True,
) -> tuple[Any, Any]:
    if get_perception_backend() == 'blob':
        return _blob_detect_mod().detect_with_debug(
            frame,
            active_branch_rank=active_branch_rank,
            prefer_yellow=prefer_yellow,
            enable_fork=enable_fork,
        )
    return _legacy().detect_with_debug(
        frame,
        active_branch_rank=active_branch_rank,
        prefer_yellow=prefer_yellow,
        enable_fork=enable_fork,
    )


def __getattr__(name: str) -> Any:
    """Lazy proxy for legacy constants/helpers (tuners / tests)."""

    return getattr(_legacy(), name)
