"""Blob backend: masks → morph → road_in drivable → centerline (+ optional fork)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from inference.modules.perception.blob.centerline import centerline_from_blob
from inference.modules.perception.blob.corridor import extract_drivable_blob
from inference.modules.perception.blob.masks import extract_bev_masks, get_ipm_params
from inference.modules.perception.blob.rail_corridor import yellow_lane_present
from inference.modules.perception.fork.adapter import merge_fork_from_legacy
from inference.modules.perception.types import LaneDebugFrame, LaneDetections


def reset_tracking_state() -> None:
    """Blob path is frame-independent; no temporal rails to clear."""

    return


def detect(
    frame: np.ndarray,
    *,
    active_branch_rank: int | None = None,
    prefer_yellow: bool | None = None,
    enable_fork: bool = False,
) -> LaneDetections:
    detections, _debug = detect_with_debug(
        frame,
        active_branch_rank=active_branch_rank,
        prefer_yellow=prefer_yellow,
        enable_fork=enable_fork,
    )
    return detections


def _assign_centerlines(
    centerline: np.ndarray,
    *,
    prefer: bool,
    used_yellow: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if centerline.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    if prefer and used_yellow:
        return (
            np.empty((0, 2), dtype=np.float32),
            centerline.astype(np.float32, copy=False),
        )
    return centerline.astype(np.float32, copy=False), np.empty((0, 2), dtype=np.float32)


def detect_with_debug(
    frame: np.ndarray,
    *,
    active_branch_rank: int | None = None,
    prefer_yellow: bool | None = None,
    enable_fork: bool = False,
) -> tuple[LaneDetections, LaneDebugFrame]:
    """Runtime detections + debug masks."""

    if frame is None or getattr(frame, 'size', 0) == 0:
        return LaneDetections(), LaneDebugFrame()

    prefer = bool(prefer_yellow) if prefer_yellow is not None else False
    masks = extract_bev_masks(frame)
    ipm = get_ipm_params()
    mpp = float(ipm.meters_per_pixel)
    track_w = float(ipm.track_width_m)
    bev = masks['bev']
    h_bev = int(masks['road_raw'].shape[0]) if masks['road_raw'].size else 0

    blob, lane_mask, corr_stats, left_rails, right_rails, used_yellow = (
        extract_drivable_blob(
            masks['road_raw'],
            masks['white'],
            masks['yellow'],
            prefer_yellow=prefer,
            track_width_m=track_w,
            meters_per_pixel=mpp,
            x_max_m=float(ipm.x_max_m),
        )
    )

    rail_conf = float(corr_stats.rail_valid_ratio)
    # Prefer DT-ridge + poly2 on the strip (A/B: low jerk). Rails only as edge viz.
    centerline, mids = centerline_from_blob(
        blob,
        x_max_m=float(ipm.x_max_m),
        meters_per_pixel=mpp,
        bev_width=int(ipm.bev_width),
        mode='dt_ridge',
    )
    use_rail_center = False
    _ = (use_rail_center,)

    conf = min(1.0, float(centerline.shape[0]) / 40.0) if centerline.shape[0] else 0.0
    if rail_conf >= 0.22:
        conf = min(1.0, max(conf, rail_conf))

    white_cl, yellow_cl = _assign_centerlines(
        centerline, prefer=prefer, used_yellow=used_yellow
    )
    yellow_vis = yellow_lane_present(masks['yellow']) if prefer else False
    white_vis = bool(np.any(masks['white'])) and (not prefer or not yellow_vis)

    detections = LaneDetections(
        drivable_area=blob.copy(),
        white_centerline=white_cl,
        yellow_centerline=yellow_cl,
        white_visible=white_vis,
        yellow_visible=yellow_vis,
        left_visible=centerline.shape[0] >= 2,
        right_visible=centerline.shape[0] >= 2,
        white_confidence=conf if white_cl.shape[0] else 0.0,
        yellow_confidence=conf if yellow_cl.shape[0] else 0.0,
        left_confidence=conf,
        right_confidence=conf,
        confidence=conf,
        fork_active=False,
        branches=(),
        active_branch_rank=active_branch_rank,
        lane_policy='explore',
        meters_per_pixel=mpp,
        x_forward_max=float(ipm.x_max_m),
    )

    if left_rails.size == h_bev and h_bev > 0:
        white_left = left_rails.copy() if not used_yellow else np.full(h_bev, np.nan, np.float32)
        white_right = right_rails.copy() if not used_yellow else np.full(h_bev, np.nan, np.float32)
        yellow_left = left_rails.copy() if used_yellow else np.full(h_bev, np.nan, np.float32)
        yellow_right = right_rails.copy() if used_yellow else np.full(h_bev, np.nan, np.float32)
    else:
        nan_rails = np.full(h_bev, np.nan, dtype=np.float32) if h_bev > 0 else np.empty(0, np.float32)
        white_left = nan_rails.copy()
        white_right = nan_rails.copy()
        yellow_left = nan_rails.copy()
        yellow_right = nan_rails.copy()

    debug = LaneDebugFrame(
        bev=bev,
        white_bev=masks['white'],
        yellow_bev=masks['yellow'],
        black_bev=masks['black'],
        red_bev=masks['red'],
        road_raw=masks['road_raw'],
        road_clean=blob.copy(),
        white_left=white_left,
        white_right=white_right,
        yellow_left=yellow_left,
        yellow_right=yellow_right,
        fork_active=False,
        prefer_yellow=prefer_yellow,
        active_branch_rank=active_branch_rank,
        lane_policy='explore',
        red_coverage=float(np.mean(masks['red'] > 0)) if masks['red'].size else 0.0,
        red_pixel_count=int(np.count_nonzero(masks['red'])),
    )
    _ = (lane_mask, mids, corr_stats)

    if enable_fork:
        detections, debug = merge_fork_from_legacy(
            frame,
            detections,
            debug,
            prefer_yellow=prefer_yellow,
            active_branch_rank=active_branch_rank,
        )
        # Fork rails from legacy override paint rails when present.
        if used_yellow:
            if np.any(np.isfinite(debug.yellow_left)):
                yellow_left = np.asarray(debug.yellow_left, dtype=np.float32)
                yellow_right = np.asarray(debug.yellow_right, dtype=np.float32)
        else:
            if np.any(np.isfinite(debug.white_left)):
                white_left = np.asarray(debug.white_left, dtype=np.float32)
                white_right = np.asarray(debug.white_right, dtype=np.float32)

    if enable_fork or active_branch_rank is not None or detections.fork_active:
        try:
            from inference.modules import active_lane as al

            detections, debug = al.apply_active_lane_policy(
                detections,
                debug,
                active_branch_rank,
            )
        except Exception:
            if active_branch_rank is not None:
                detections = replace(
                    detections,
                    active_branch_rank=active_branch_rank,
                    lane_policy='locked',
                )

    return detections, debug
