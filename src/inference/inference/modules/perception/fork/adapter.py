"""Sign-gated fork: call legacy fork discrimination, merge into blob result."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


def merge_fork_from_legacy(
    frame: np.ndarray,
    detections: Any,
    debug: Any,
    *,
    prefer_yellow: bool | None,
    active_branch_rank: int | None,
) -> tuple[Any, Any]:
    """Run legacy ``detect_with_debug(..., enable_fork=True)`` for fork fields only.

    Keeps blob ``drivable_area`` / course centerline; overlays fork_active,
    branches, fork_lane_pairs, and related debug. Intended only when the
    planner already passed ``enable_fork`` (traffic-sign gate).
    """

    from inference.modules.perception.legacy import lane_detection as legacy

    _legacy_det, legacy_debug = legacy.detect_with_debug(
        frame,
        active_branch_rank=active_branch_rank,
        prefer_yellow=prefer_yellow,
        enable_fork=True,
    )

    fork_active = bool(getattr(legacy_debug, 'fork_active', False))
    pairs = tuple(getattr(legacy_debug, 'fork_lane_pairs', ()) or ())
    branches = tuple(getattr(_legacy_det, 'branches', ()) or ())
    split_src = str(getattr(legacy_debug, 'fork_split_source', '') or '')

    # Prefer legacy branches when fork is active; else keep blob centerline path.
    new_det = replace(
        detections,
        fork_active=fork_active,
        branches=branches if fork_active and branches else getattr(detections, 'branches', ()),
    )

    # Build a new debug frame: blob masks + legacy fork annotations.
    # LaneDebugFrame is not frozen — mutate when possible, else replace fields.
    debug.fork_active = fork_active
    debug.fork_lane_pairs = pairs
    debug.fork_split_source = split_src
    debug.fork_mark_tracks = tuple(getattr(legacy_debug, 'fork_mark_tracks', ()) or ())
    debug.road_branches = tuple(getattr(legacy_debug, 'road_branches', ()) or ())
    debug.road_cells = getattr(legacy_debug, 'road_cells', debug.road_cells)
    debug.ego_road_color = getattr(legacy_debug, 'ego_road_color', None)

    # When fork is active, expose legacy rails for fork pair preview only.
    if fork_active:
        for attr in ('white_left', 'white_right', 'yellow_left', 'yellow_right'):
            src = getattr(legacy_debug, attr, None)
            if src is not None and getattr(src, 'size', 0):
                setattr(debug, attr, np.asarray(src, dtype=np.float32))

    return new_det, debug
