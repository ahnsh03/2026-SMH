"""Lane detection module — 담당: 장원태 (인지).

Phase 2 interim: temporary **white-only** stub (Metric IPM + HSV from
``config/lane_vision.yaml``) that feeds ``lane_planner``. Replace
``detect_markings`` with Won Tae perception when PR merges; keep returning
``LaneResult`` via planner for pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from inference.modules.lane_planner import get_shared_planner, plan
from inference.types import LaneDetections, LaneMarking, LaneResult


def _repo_root() -> Path:
    for base in Path(__file__).resolve().parents:
        if (base / 'config' / 'lane_vision.yaml').exists():
            return base
    return Path(__file__).resolve().parents[4]


def _ensure_vision_tune():
    tune_dir = _repo_root() / 'scripts' / 'vision_tune'
    if str(tune_dir) not in sys.path:
        sys.path.insert(0, str(tune_dir))
    from hsv import load_hsv_ranges, make_mask  # noqa: WPS433
    from metric_ipm import (  # noqa: WPS433
        bev_uv_to_xy,
        load_metric_ipm,
        warp_metric_ipm,
    )

    return load_metric_ipm, warp_metric_ipm, bev_uv_to_xy, load_hsv_ranges, make_mask


def _extract_lr_polylines(
    mask: np.ndarray,
    *,
    row_step: int = 4,
    min_pix: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (left_uv Nx2, right_uv Nx2, confidence) in BEV pixel coords."""
    h, w = mask.shape[:2]
    cx = w // 2
    left_uv: list[tuple[float, float]] = []
    right_uv: list[tuple[float, float]] = []

    for v in range(0, h, max(1, row_step)):
        xs = np.flatnonzero(mask[v] > 0)
        if xs.size < min_pix:
            continue
        left_xs = xs[xs < cx]
        right_xs = xs[xs >= cx]
        if left_xs.size >= min_pix:
            left_uv.append((float(left_xs[-1]), float(v)))
        if right_xs.size >= min_pix:
            right_uv.append((float(right_xs[0]), float(v)))

    left = (
        np.asarray(left_uv, dtype=np.float32)
        if left_uv
        else np.empty((0, 2), dtype=np.float32)
    )
    right = (
        np.asarray(right_uv, dtype=np.float32)
        if right_uv
        else np.empty((0, 2), dtype=np.float32)
    )
    conf = 0.0
    if left.shape[0] >= 3 and right.shape[0] >= 3:
        conf = 0.85
    elif left.shape[0] >= 3 or right.shape[0] >= 3:
        conf = 0.45
    return left, right, conf


def _uv_to_marking(
    uv: np.ndarray,
    *,
    marking_id: int,
    side: int,
    confidence: float,
    bev_uv_to_xy,
    params,
) -> LaneMarking:
    if uv.shape[0] == 0:
        return LaneMarking(
            id=marking_id,
            color=LaneMarking.COLOR_WHITE,
            side_hint=side,
            confidence=0.0,
        )
    x, y = bev_uv_to_xy(uv[:, 0], uv[:, 1], params)
    pts = np.stack([x, y], axis=1).astype(np.float32)
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    length = (
        float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        if pts.shape[0] > 1
        else 0.0
    )
    return LaneMarking(
        id=marking_id,
        color=LaneMarking.COLOR_WHITE,
        side_hint=side,
        confidence=confidence,
        length=length,
        points=pts,
    )


def detect_markings(frame: np.ndarray) -> LaneDetections:
    """Temporary white-only perception stub (Metric IPM + yaml HSV)."""
    if frame is None or frame.size == 0:
        return LaneDetections()

    (
        load_metric_ipm,
        warp_metric_ipm,
        bev_uv_to_xy,
        load_hsv_ranges,
        make_mask,
    ) = _ensure_vision_tune()
    vision_yaml = _repo_root() / 'config' / 'lane_vision.yaml'
    params = load_metric_ipm(vision_yaml)
    bev = warp_metric_ipm(frame, params)
    white_rng = load_hsv_ranges(vision_yaml)['white']
    mask = make_mask(bev, white_rng)
    left_uv, right_uv, conf = _extract_lr_polylines(mask)

    left = _uv_to_marking(
        left_uv,
        marking_id=1,
        side=LaneMarking.SIDE_LEFT,
        confidence=conf,
        bev_uv_to_xy=bev_uv_to_xy,
        params=params,
    )
    right = _uv_to_marking(
        right_uv,
        marking_id=2,
        side=LaneMarking.SIDE_RIGHT,
        confidence=conf,
        bev_uv_to_xy=bev_uv_to_xy,
        params=params,
    )
    white_visible = left.xy().shape[0] > 0 or right.xy().shape[0] > 0
    return LaneDetections(
        lanes=(left, right),
        white_visible=white_visible,
        yellow_visible=False,
    )


def detect(frame: np.ndarray) -> LaneResult:
    """
    Detect white lanes and plan steering.

    Returns LaneResult for ``pipeline.fuse_control``. Steering comes from
    ``lane_planner`` only — not from perception.
    """
    detections = detect_markings(frame)
    return plan(detections, get_shared_planner())
