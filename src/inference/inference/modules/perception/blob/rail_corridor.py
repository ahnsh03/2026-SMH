"""Lane-rail corridor helpers for blob perception (road_in compose).

Ports the preview ``out_drivable`` fill/ego-band logic into the inference
package so runtime can clip ``road_raw`` with course-specific lane paint.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from inference.modules.perception.blob.centerline import (
    mids_to_vehicle_points,
    smooth_mids_1d,
)

_WIDTH_CLIP_SCALE = 1.35
_WIDTH_SKIP_SCALE = 2.50
_MIN_YELLOW_PX = 120
_MIN_RAIL_VALID_RATIO = 0.22


@dataclass(frozen=True)
class VehiclePlanarGeom:
    wheelbase_m: float = 0.175
    tread_m: float = 0.120
    ahead_front_axle_m: float = 0.025
    r_min_m: float = 0.385
    track_width_m: float = 0.35
    band_margin_m: float = 0.08

    @property
    def front_axle_x_cam_m(self) -> float:
        return float(-self.ahead_front_axle_m)


@dataclass(frozen=True)
class FillStats:
    valid_rows: int = 0
    clipped_rows: int = 0
    one_sided_rows: int = 0
    skipped_rows: int = 0
    total_rows: int = 0
    guard_clipped_rows: int = 0

    @property
    def valid_ratio(self) -> float:
        if self.total_rows <= 0:
            return 0.0
        return float(self.valid_rows) / float(self.total_rows)


@dataclass(frozen=True)
class RailExtractStats:
    valid_rows: int = 0
    one_sided_rows: int = 0
    total_rows: int = 0
    used_yellow: bool = False

    @property
    def valid_ratio(self) -> float:
        if self.total_rows <= 0:
            return 0.0
        return float(self.valid_rows) / float(self.total_rows)


def vehicle_geom_for_platform(platform: str = 'sim') -> VehiclePlanarGeom:
    key = platform.strip().lower()
    if key in ('sim', 'gazebo', 'limo'):
        return VehiclePlanarGeom(wheelbase_m=0.24, tread_m=0.168)
    return VehiclePlanarGeom()


def yellow_lane_present(yellow: np.ndarray, *, min_px: int = _MIN_YELLOW_PX) -> bool:
    if yellow is None or not getattr(yellow, 'size', 0):
        return False
    return int(np.count_nonzero(yellow > 0)) >= int(min_px)


def resolve_course_lane_mask(
    white: np.ndarray,
    yellow: np.ndarray,
    *,
    prefer_yellow: bool,
    min_yellow_px: int = _MIN_YELLOW_PX,
) -> tuple[np.ndarray, bool]:
    """Pick a single course color mask — never OR white and yellow."""

    if prefer_yellow and yellow_lane_present(yellow, min_px=min_yellow_px):
        return yellow, True
    if white is not None and getattr(white, 'size', 0):
        return white, False
    if yellow is not None and getattr(yellow, 'size', 0):
        return yellow, True
    return np.zeros((0, 0), dtype=np.uint8), False


def kinematic_lateral_reach_m(x_cam_m: float, geom: VehiclePlanarGeom) -> float:
    s = float(x_cam_m) - float(geom.front_axle_x_cam_m)
    if s <= 0.0:
        return 0.5 * float(geom.tread_m)
    r = max(0.05, float(geom.r_min_m))
    s_cap = min(s, float(np.pi) * r * 0.45)
    return float(r * (1.0 - np.cos(s_cap / r)))


def ego_fill_band_half_width_m(x_cam_m: float, geom: VehiclePlanarGeom) -> float:
    lane_half = 0.5 * float(geom.track_width_m)
    reach = kinematic_lateral_reach_m(x_cam_m, geom)
    return float(lane_half + reach + float(geom.band_margin_m))


def ego_mid_assoc_half_width_m(x_cam_m: float, geom: VehiclePlanarGeom) -> float:
    reach = kinematic_lateral_reach_m(x_cam_m, geom)
    soft = float(reach + float(geom.band_margin_m))
    hard = 0.75 * float(geom.track_width_m)
    return float(min(soft, hard))


def build_kinematic_ego_band(
    bev_hw: tuple[int, int],
    *,
    meters_per_pixel: float,
    x_max_m: float,
    geom: VehiclePlanarGeom,
    center_u: float | None = None,
    for_fill: bool = True,
) -> np.ndarray:
    h, w = int(bev_hw[0]), int(bev_hw[1])
    mpp = float(meters_per_pixel)
    cu = float((w - 1) * 0.5 if center_u is None else center_u)
    half_fn = ego_fill_band_half_width_m if for_fill else ego_mid_assoc_half_width_m
    band = np.zeros((h, w), dtype=np.uint8)
    for v in range(h):
        x_cam = float(x_max_m - v * mpp)
        half_m = half_fn(x_cam, geom)
        half_u = half_m / mpp
        u0 = int(np.clip(np.floor(cu - half_u), 0, w - 1))
        u1 = int(np.clip(np.ceil(cu + half_u), 0, w - 1))
        if u1 >= u0:
            band[v, u0 : u1 + 1] = 255
    return band


def extract_rails_from_lane_mask(
    lane_mask: np.ndarray,
    *,
    meters_per_pixel: float,
    x_max_m: float,
    track_width_m: float,
    geom: VehiclePlanarGeom | None = None,
) -> tuple[np.ndarray, np.ndarray, RailExtractStats]:
    """Per-row outer bounds of lane paint within the kinematic ego band."""

    if lane_mask is None or not getattr(lane_mask, 'size', 0):
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            RailExtractStats(),
        )

    h, w = lane_mask.shape[:2]
    g = geom or VehiclePlanarGeom(track_width_m=float(track_width_m))
    mpp = float(meters_per_pixel)
    left = np.full(h, np.nan, dtype=np.float32)
    right = np.full(h, np.nan, dtype=np.float32)
    valid = one_sided = 0

    cleaned = cv2.morphologyEx(
        (lane_mask > 0).astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    for v in range(h):
        cols = np.flatnonzero(cleaned[v] > 0)
        if cols.size == 0:
            continue
        x_cam = float(x_max_m - v * mpp)
        half_u = ego_mid_assoc_half_width_m(x_cam, g) / mpp
        ego_u = (w - 1) * 0.5
        near = cols[np.abs(cols.astype(np.float64) - ego_u) <= half_u + 2.0]
        if near.size == 0:
            near = cols
        lu = float(np.min(near))
        ru = float(np.max(near))
        if ru - lu < 1.0:
            continue
        left[v] = lu
        right[v] = ru
        valid += 1
        width_px = float(track_width_m) / mpp
        if ru - lu < 0.55 * width_px:
            one_sided += 1

    return left, right, RailExtractStats(
        valid_rows=valid,
        one_sided_rows=one_sided,
        total_rows=h,
    )


def guard_rails_with_ego_band(
    left_u: np.ndarray,
    right_u: np.ndarray,
    bev_hw: tuple[int, int],
    *,
    meters_per_pixel: float,
    x_max_m: float,
    track_width_m: float,
    geom: VehiclePlanarGeom | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    h, w = int(bev_hw[0]), int(bev_hw[1])
    g = geom or VehiclePlanarGeom(track_width_m=float(track_width_m))
    mpp = float(meters_per_pixel)
    half_lane_u = (0.5 * g.track_width_m) / mpp
    ego_u = (w - 1) * 0.5
    left = np.asarray(left_u, dtype=np.float64).copy().reshape(-1)
    right = np.asarray(right_u, dtype=np.float64).copy().reshape(-1)
    n = min(h, left.size, right.size)
    guarded = 0
    for v in range(n):
        lu, ru = left[v], right[v]
        if not (np.isfinite(lu) and np.isfinite(ru)):
            continue
        if ru < lu:
            lu, ru = ru, lu
        mid = 0.5 * (lu + ru)
        x_cam = float(x_max_m - v * mpp)
        half_u = ego_mid_assoc_half_width_m(x_cam, g) / mpp
        if abs(mid - ego_u) <= half_u:
            left[v], right[v] = lu, ru
            continue
        left[v] = ego_u - half_lane_u
        right[v] = ego_u + half_lane_u
        guarded += 1
    return left.astype(np.float32), right.astype(np.float32), guarded


def fill_between_rails(
    left_u: np.ndarray,
    right_u: np.ndarray,
    bev_hw: tuple[int, int],
    *,
    track_width_m: float,
    meters_per_pixel: float,
    allow_one_sided: bool = True,
) -> tuple[np.ndarray, FillStats]:
    h, w = int(bev_hw[0]), int(bev_hw[1])
    filled = np.zeros((h, w), dtype=np.uint8)
    left = np.asarray(left_u, dtype=np.float64).reshape(-1)
    right = np.asarray(right_u, dtype=np.float64).reshape(-1)
    n = min(h, left.size, right.size)
    if n <= 0:
        return filled, FillStats(total_rows=h)

    width_px = float(track_width_m) / float(meters_per_pixel)
    half = 0.5 * width_px
    clip_max = width_px * _WIDTH_CLIP_SCALE
    skip_max = width_px * _WIDTH_SKIP_SCALE

    valid = clipped = one_sided = skipped = 0
    for y in range(n):
        lu = left[y]
        ru = right[y]
        l_ok = np.isfinite(lu)
        r_ok = np.isfinite(ru)

        if l_ok and r_ok:
            if ru < lu:
                lu, ru = ru, lu
            span = float(ru - lu)
            if span < 1.0 or span > skip_max:
                skipped += 1
                continue
            if span > clip_max:
                mid = 0.5 * (lu + ru)
                lu = mid - half
                ru = mid + half
                clipped += 1
            x0 = int(np.clip(np.floor(lu), 0, w - 1))
            x1 = int(np.clip(np.ceil(ru), 0, w - 1))
            if x1 - x0 < 1:
                skipped += 1
                continue
            filled[y, x0 : x1 + 1] = 255
            valid += 1
            continue

        if allow_one_sided and (l_ok or r_ok):
            u = float(lu if l_ok else ru)
            x0 = int(np.clip(np.floor(u - half), 0, w - 1))
            x1 = int(np.clip(np.ceil(u + half), 0, w - 1))
            if x1 - x0 < 1:
                skipped += 1
                continue
            filled[y, x0 : x1 + 1] = 255
            valid += 1
            one_sided += 1
            continue

        skipped += 1

    return filled, FillStats(
        valid_rows=valid,
        clipped_rows=clipped,
        one_sided_rows=one_sided,
        skipped_rows=skipped + (h - n),
        total_rows=h,
    )


def compose_road_in(road: np.ndarray, corridor: np.ndarray) -> np.ndarray:
    if road.size == 0 or corridor.size == 0:
        return road.copy() if road.size else corridor.copy()
    return cv2.bitwise_and(road, corridor)


def build_lane_corridor(
    lane_mask: np.ndarray,
    *,
    track_width_m: float,
    meters_per_pixel: float,
    x_max_m: float,
    geom: VehiclePlanarGeom | None = None,
) -> tuple[np.ndarray, FillStats, np.ndarray, np.ndarray]:
    """Fill between paint rails, clip to kinematic ego band."""

    if lane_mask is None or not getattr(lane_mask, 'size', 0):
        empty = np.zeros((0, 0), dtype=np.uint8)
        return empty, FillStats(), np.empty(0, np.float32), np.empty(0, np.float32)

    h, w = lane_mask.shape[:2]
    g = geom or VehiclePlanarGeom(track_width_m=float(track_width_m))
    left, right, _ = extract_rails_from_lane_mask(
        lane_mask,
        meters_per_pixel=meters_per_pixel,
        x_max_m=x_max_m,
        track_width_m=track_width_m,
        geom=g,
    )
    left, right, guard_n = guard_rails_with_ego_band(
        left,
        right,
        (h, w),
        meters_per_pixel=meters_per_pixel,
        x_max_m=x_max_m,
        track_width_m=track_width_m,
        geom=g,
    )
    between, stats = fill_between_rails(
        left,
        right,
        (h, w),
        track_width_m=track_width_m,
        meters_per_pixel=meters_per_pixel,
    )
    if guard_n:
        stats = FillStats(
            valid_rows=stats.valid_rows,
            clipped_rows=stats.clipped_rows,
            one_sided_rows=stats.one_sided_rows,
            skipped_rows=stats.skipped_rows,
            total_rows=stats.total_rows,
            guard_clipped_rows=guard_n,
        )
    band = build_kinematic_ego_band(
        (h, w),
        meters_per_pixel=meters_per_pixel,
        x_max_m=x_max_m,
        geom=g,
    )
    between = cv2.bitwise_and(between, band)
    return between, stats, left, right


def centerline_from_rails(
    left_u: np.ndarray,
    right_u: np.ndarray,
    *,
    x_max_m: float,
    meters_per_pixel: float,
    bev_width: int,
    track_width_m: float,
    smooth_win: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    """Parallel-rail center: midpoint when both sides exist, else outer ± half_w."""

    left = np.asarray(left_u, dtype=np.float64).reshape(-1)
    right = np.asarray(right_u, dtype=np.float64).reshape(-1)
    n = min(left.size, right.size)
    if n <= 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, np.float32)

    half_u = (0.5 * float(track_width_m)) / float(meters_per_pixel)
    mids = np.full(n, np.nan, dtype=np.float64)
    for v in range(n):
        lu, ru = left[v], right[v]
        l_ok = np.isfinite(lu)
        r_ok = np.isfinite(ru)
        if l_ok and r_ok:
            if ru < lu:
                lu, ru = ru, lu
            mids[v] = 0.5 * (lu + ru)
        elif l_ok:
            mids[v] = float(lu) + half_u
        elif r_ok:
            mids[v] = float(ru) - half_u

    smooth = smooth_mids_1d(mids.astype(np.float32), win=smooth_win)
    pts = mids_to_vehicle_points(
        smooth,
        x_max_m=x_max_m,
        meters_per_pixel=meters_per_pixel,
        bev_width=bev_width,
    )
    return pts, smooth
