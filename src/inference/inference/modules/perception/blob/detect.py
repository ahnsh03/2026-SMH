"""Blob backend: bag-SSOT ego blob → mask COM control (+ optional fork)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from inference.modules.perception.blob.centerline import centerline_from_blob
from inference.modules.perception.blob.corridor import (
    CorridorStats,
    extract_drivable_blob,
)
from inference.modules.perception.blob.masks import (
    course_ego_blob,
    extract_bev_masks,
    get_ipm_params,
)
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


def _rails_from_blob(blob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row leftmost/rightmost ego pixels (BEV u)."""

    if blob is None or getattr(blob, 'size', 0) == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    h, _w = blob.shape[:2]
    left = np.full(h, np.nan, np.float32)
    right = np.full(h, np.nan, np.float32)
    for v in range(h):
        cols = np.flatnonzero(blob[v] > 0)
        if cols.size >= 2:
            left[v] = float(cols[0])
            right[v] = float(cols[-1])
    return left, right


def detect_with_debug(
    frame: np.ndarray,
    *,
    active_branch_rank: int | None = None,
    prefer_yellow: bool | None = None,
    enable_fork: bool = False,
) -> tuple[LaneDetections, LaneDebugFrame]:
    """Runtime detections + debug masks.

    Drivable SSOT matches monitor / ``extract_five``:
    paint|road → morph open3/close13 → ``keep_bottom_ego_blob``.
    Mask-COM (``mask_p``) tracks that ego region.
    """

    if frame is None or getattr(frame, 'size', 0) == 0:
        return LaneDetections(), LaneDebugFrame()

    prefer = bool(prefer_yellow) if prefer_yellow is not None else False
    masks = extract_bev_masks(frame, prefer_yellow=prefer)
    ipm = get_ipm_params()
    mpp = float(ipm.meters_per_pixel)
    track_w = float(ipm.track_width_m)
    bev = masks['bev']
    h_bev = int(masks['road_raw'].shape[0]) if masks['road_raw'].size else 0

    _white_ssot, ego, _bev_ssot = course_ego_blob(frame, prefer_yellow=prefer)
    used_yellow = bool(prefer and yellow_lane_present(masks['yellow']))
    method = 'ego_blob'
    left_rails = np.empty(0, dtype=np.float32)
    right_rails = np.empty(0, dtype=np.float32)
    corr_stats = CorridorStats(method=method, used_yellow=used_yellow)
    lane_mask = np.zeros_like(masks['road_raw']) if masks['road_raw'].size else np.zeros(
        (0, 0), dtype=np.uint8
    )

    if ego.size and int(np.count_nonzero(ego)) >= 80:
        blob = np.asarray(ego, dtype=np.uint8)
        left_rails, right_rails = _rails_from_blob(blob)
        valid_rows = int(np.sum(np.isfinite(left_rails) & np.isfinite(right_rails)))
        corr_stats = CorridorStats(
            method=method,
            used_yellow=used_yellow,
            union_coverage=float(np.mean(blob > 0)),
            between_rows=valid_rows,
            rail_valid_ratio=float(valid_rows) / float(max(h_bev, 1)),
        )
    else:
        # Short flicker / empty BEV-HSV → DT strip fallback (same paint walls path).
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
        method = str(corr_stats.method or 'dt_strip')

    rail_conf = float(corr_stats.rail_valid_ratio)
    # Ridge mid on the ego region — stable lateral target for PP / paint blend.
    centerline, mids = centerline_from_blob(
        blob,
        x_max_m=float(ipm.x_max_m),
        meters_per_pixel=mpp,
        bev_width=int(ipm.bev_width),
        mode='dt_ridge',
    )

    conf = min(1.0, float(centerline.shape[0]) / 40.0) if centerline.shape[0] else 0.0
    if rail_conf >= 0.22:
        conf = min(1.0, max(conf, rail_conf))

    white_cl, yellow_cl = _assign_centerlines(
        centerline, prefer=prefer, used_yellow=used_yellow
    )
    yellow_vis = yellow_lane_present(masks['yellow']) if prefer else False
    white_vis = bool(np.any(masks['white'])) and (not prefer or not yellow_vis)

    out_capture = False
    in_moment = False
    try:
        from inference.modules.perception.fork.capture import score_out_fork_capture
        from inference.modules.perception.fork.moment import (
            score_in_circle_fork_moment,
        )

        if prefer:
            in_moment = bool(
                score_in_circle_fork_moment(
                    masks['yellow'], masks['road_raw']
                ).hard
            )
        else:
            out_capture = bool(
                score_out_fork_capture(
                    masks['white'], masks['road_raw'], blob
                ).capture
            )
    except Exception:
        out_capture = False
        in_moment = False

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
        out_fork_capture=out_capture,
        in_circle_fork_moment=in_moment,
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
    _ = (lane_mask, mids, corr_stats, method)

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
