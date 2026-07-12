"""Adapters between Won Tae module dataclasses, lane_msgs, and inference.types.

SSOT for planners is ``inference.types.LaneDetections``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from inference import types as T


def _as_xy(points: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    return pts[:, :2].copy()


def _as_xyz(points: Any) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if pts.ndim != 2:
        return np.empty((0, 3), dtype=np.float32)
    if pts.shape[1] >= 3:
        return pts[:, :3].copy()
    if pts.shape[1] == 2:
        z = np.zeros((pts.shape[0], 1), dtype=np.float32)
        return np.hstack([pts, z])
    return np.empty((0, 3), dtype=np.float32)


def _marking_from_module(marking: Any) -> T.LaneMarking:
    return T.LaneMarking(
        id=int(getattr(marking, 'id', 0)),
        color=int(getattr(marking, 'color', 0)),
        side_hint=int(getattr(marking, 'side_hint', 0)),
        confidence=float(getattr(marking, 'confidence', 0.0)),
        length=float(getattr(marking, 'length', 0.0)),
        heading=float(getattr(marking, 'heading', 0.0)),
        curvature=float(getattr(marking, 'curvature', 0.0)),
        points=_as_xy(getattr(marking, 'points', ())),
    )


def _branch_from_module(branch: Any) -> T.RoadBranch:
    return T.RoadBranch(
        lateral_rank=int(getattr(branch, 'lateral_rank', 0)),
        confidence=float(getattr(branch, 'confidence', 0.0)),
        width=float(getattr(branch, 'width', 0.0)),
        points=_as_xyz(getattr(branch, 'points', ())),
    )


def detections_from_module(
    dc: Any,
    *,
    meters_per_pixel: float = 0.0,
    x_forward_max: float = 0.0,
) -> T.LaneDetections:
    """Convert ``lane_detection.LaneDetections`` dataclass → ``types.LaneDetections``."""

    lanes = tuple(_marking_from_module(m) for m in getattr(dc, 'lanes', ()) or ())
    branches = tuple(
        _branch_from_module(b) for b in getattr(dc, 'branches', ()) or ()
    )
    drivable = np.asarray(
        getattr(dc, 'drivable_area', np.empty((0, 0), dtype=np.uint8)),
        dtype=np.uint8,
    )
    return T.LaneDetections(
        lanes=lanes,
        white_visible=bool(getattr(dc, 'white_visible', False)),
        yellow_visible=bool(getattr(dc, 'yellow_visible', False)),
        left_visible=bool(getattr(dc, 'left_visible', False)),
        right_visible=bool(getattr(dc, 'right_visible', False)),
        white_confidence=float(getattr(dc, 'white_confidence', 0.0)),
        yellow_confidence=float(getattr(dc, 'yellow_confidence', 0.0)),
        left_confidence=float(getattr(dc, 'left_confidence', 0.0)),
        right_confidence=float(getattr(dc, 'right_confidence', 0.0)),
        white_centerline=_as_xy(getattr(dc, 'white_centerline', ())),
        yellow_centerline=_as_xy(getattr(dc, 'yellow_centerline', ())),
        yellow_crossing_line=bool(getattr(dc, 'yellow_crossing_line', False)),
        fork_active=bool(getattr(dc, 'fork_active', False)),
        branches=branches,
        drivable_area=drivable,
        meters_per_pixel=float(meters_per_pixel),
        x_forward_max=float(x_forward_max),
    )


def _points_from_msg(points: Any) -> np.ndarray:
    if not points:
        return np.empty((0, 3), dtype=np.float32)
    rows = []
    for p in points:
        rows.append(
            (
                float(getattr(p, 'x', 0.0)),
                float(getattr(p, 'y', 0.0)),
                float(getattr(p, 'z', 0.0)),
            )
        )
    return np.asarray(rows, dtype=np.float32)


def detections_from_msg(msg: Any) -> T.LaneDetections:
    """Convert ``lane_msgs/LaneDetections`` → ``types.LaneDetections``."""

    lanes = []
    for marking in getattr(msg, 'lanes', []) or []:
        xyz = _points_from_msg(getattr(marking, 'points', []))
        lanes.append(
            T.LaneMarking(
                id=int(getattr(marking, 'id', 0)),
                color=int(getattr(marking, 'color', 0)),
                side_hint=int(getattr(marking, 'side_hint', 0)),
                confidence=float(getattr(marking, 'confidence', 0.0)),
                length=float(getattr(marking, 'length', 0.0)),
                heading=float(getattr(marking, 'heading', 0.0)),
                curvature=float(getattr(marking, 'curvature', 0.0)),
                points=xyz[:, :2] if xyz.size else np.empty((0, 2), dtype=np.float32),
            )
        )

    branches = []
    for branch in getattr(msg, 'branches', []) or []:
        branches.append(
            T.RoadBranch(
                lateral_rank=int(getattr(branch, 'branch_id', 0)),
                confidence=float(getattr(branch, 'confidence', 0.0)),
                width=float(getattr(branch, 'width', 0.0)),
                points=_points_from_msg(getattr(branch, 'centerline', [])),
            )
        )

    drivable = np.empty((0, 0), dtype=np.uint8)
    image = getattr(msg, 'drivable_area', None)
    if image is not None and int(getattr(image, 'height', 0)) > 0:
        h = int(image.height)
        w = int(image.width)
        drivable = np.frombuffer(bytes(image.data), dtype=np.uint8).reshape((h, w))

    return T.LaneDetections(
        lanes=tuple(lanes),
        white_visible=bool(getattr(msg, 'white_visible', False)),
        yellow_visible=bool(getattr(msg, 'yellow_visible', False)),
        left_visible=bool(getattr(msg, 'left_visible', False)),
        right_visible=bool(getattr(msg, 'right_visible', False)),
        white_confidence=float(getattr(msg, 'white_confidence', 0.0)),
        yellow_confidence=float(getattr(msg, 'yellow_confidence', 0.0)),
        left_confidence=float(getattr(msg, 'left_confidence', 0.0)),
        right_confidence=float(getattr(msg, 'right_confidence', 0.0)),
        white_centerline=_as_xy(_points_from_msg(getattr(msg, 'white_centerline', []))),
        yellow_centerline=_as_xy(_points_from_msg(getattr(msg, 'yellow_centerline', []))),
        yellow_crossing_line=bool(getattr(msg, 'yellow_crossing_line', False)),
        fork_active=bool(getattr(msg, 'fork_active', False)),
        branches=tuple(branches),
        drivable_area=drivable,
        meters_per_pixel=float(getattr(msg, 'meters_per_pixel', 0.0)),
        x_forward_max=float(getattr(msg, 'x_forward_max', 0.0)),
    )
