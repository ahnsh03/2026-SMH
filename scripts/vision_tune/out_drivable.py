"""OUT/IN drivable helpers: road=black|red, fill between fitted rails.

Used by ``preview_out_drivable.py``. White and yellow rails stay course-separate
(OUT → white, IN → yellow); never OR'd for the corridor fill.

out_fork parallel-rail fill is opt-in (``use_fork``) — runtime keeps it behind
the traffic-sign gate; preview defaults fork OFF for normal lanes.

Normal-lane fill applies a kinematic ego band (front axle, tread, R_min,
track width) so fitted corridors do not latch onto a neighbour lane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from hsv import HsvRange, make_mask
from metric_ipm import MetricIpmParams, load_metric_ipm

# Max corridor width vs track prior before center+clip (leak guard).
_WIDTH_CLIP_SCALE = 1.35  # allow ~±35% vs 0.35 m before clipping
_WIDTH_SKIP_SCALE = 2.50  # absurd pairs (broken dashes spanning FOV) → drop row


@dataclass(frozen=True)
class VehiclePlanarGeom:
    """Planar mount / bicycle priors for ego-lane association guards.

    SSOT lengths: docs/vehicle-geometry.md §4.1 / §4.1.1.
    """

    wheelbase_m: float = 0.175
    tread_m: float = 0.120
    ahead_front_axle_m: float = 0.025  # camera ahead of front axle
    r_min_m: float = 0.385
    track_width_m: float = 0.35
    # Extra lateral slack on the kinematic band (m).
    band_margin_m: float = 0.08

    @property
    def rear_axle_to_camera_m(self) -> float:
        return float(self.wheelbase_m + self.ahead_front_axle_m)

    @property
    def front_axle_x_cam_m(self) -> float:
        """Front axle x in camera/IPM frame (negative = behind camera)."""
        return float(-self.ahead_front_axle_m)


def vehicle_geom_for_platform(platform: str = 'real') -> VehiclePlanarGeom:
    key = platform.strip().lower()
    if key in ('sim', 'gazebo', 'limo'):
        return VehiclePlanarGeom(wheelbase_m=0.24, tread_m=0.168)
    return VehiclePlanarGeom()


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


def road_mask_from_hsv(
    bev: np.ndarray,
    ranges: dict[str, HsvRange],
    *,
    prefer_yellow: bool = False,
    yellow: np.ndarray | None = None,
) -> np.ndarray:
    """road via compose_road_raw (red-only / course cyan gates)."""

    from viz_raw_hsv_masks import _bin, _keep_near_floor_blob

    try:
        from inference.modules.perception.blob.masks import compose_road_raw
    except ModuleNotFoundError:
        from inference.inference.modules.perception.blob.masks import compose_road_raw

    black = _keep_near_floor_blob(
        _bin(make_mask(bev, ranges['black_road'], morph=False))
    )
    red = _bin(make_mask(bev, ranges['red_road']))
    if yellow is None:
        yellow = (
            _bin(make_mask(bev, ranges['yellow']))
            if 'yellow' in ranges
            else np.zeros_like(black)
        )
    else:
        yellow = _bin(yellow)
    cyan1 = (
        _bin(make_mask(bev, ranges['black_cyan'], morph=False))
        if 'black_cyan' in ranges
        else np.zeros_like(black)
    )
    cyan2 = (
        _bin(make_mask(bev, ranges['black_cyan_2'], morph=False))
        if 'black_cyan_2' in ranges
        else np.zeros_like(black)
    )
    road, _, _ = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow,
        prefer_yellow=prefer_yellow,
        keep_near_fn=_keep_near_floor_blob,
    )
    return road



def course_rails_from_debug(
    debug: Any,
    course: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (left_u, right_u, rail_label) for the active course color only."""
    key = course.strip().lower()
    if key in ('in', 'yellow'):
        left = np.asarray(debug.yellow_left, dtype=np.float32)
        right = np.asarray(debug.yellow_right, dtype=np.float32)
        return left, right, 'yellow'
    left = np.asarray(debug.white_left, dtype=np.float32)
    right = np.asarray(debug.white_right, dtype=np.float32)
    return left, right, 'white'


def prefer_yellow_for_course(course: str) -> bool:
    key = course.strip().lower()
    return key in ('in', 'yellow', 'in_cam', 'in_course', 'in_yellow')


def _road_width_px(ipm: MetricIpmParams | None = None) -> float:
    p = (ipm or load_metric_ipm()).clamp()
    return float(p.track_width_m) / float(p.meters_per_pixel)


def _geom_with_ipm_track(
    geom: VehiclePlanarGeom | None,
    ipm: MetricIpmParams,
) -> VehiclePlanarGeom:
    g = geom or vehicle_geom_for_platform('real')
    return VehiclePlanarGeom(
        wheelbase_m=g.wheelbase_m,
        tread_m=g.tread_m,
        ahead_front_axle_m=g.ahead_front_axle_m,
        r_min_m=g.r_min_m,
        track_width_m=float(ipm.track_width_m),
        band_margin_m=g.band_margin_m,
    )


def kinematic_lateral_reach_m(
    x_cam_m: float,
    geom: VehiclePlanarGeom,
) -> float:
    """Max |y| from straight ego path at camera-frame forward x (bicycle, |δ|=δ_max).

    Uses R_min = L / tan(δ_max). Arc from front axle: s = x_cam - x_front_axle.
    Lateral offset ≈ R (1 − cos(s/R)) for 0 ≤ s ≤ π R (capped).
    """
    s = float(x_cam_m) - float(geom.front_axle_x_cam_m)
    if s <= 0.0:
        return 0.5 * float(geom.tread_m)
    r = max(0.05, float(geom.r_min_m))
    s_cap = min(s, float(np.pi) * r * 0.45)
    return float(r * (1.0 - np.cos(s_cap / r)))


def ego_mid_assoc_half_width_m(
    x_cam_m: float,
    geom: VehiclePlanarGeom,
) -> float:
    """Tight mid-association half-width: reject neighbour-lane rail pairs.

    Neighbour lane mid ≈ ±track_width from ego mid. Cap reach so far-field
    continuous max-steer cannot still admit a full lateral lane jump; near
    field stays at reach + margin under image-center (straight) prior.
    """
    reach = kinematic_lateral_reach_m(x_cam_m, geom)
    soft = float(reach + float(geom.band_margin_m))
    hard = 0.75 * float(geom.track_width_m)
    return float(min(soft, hard))


def ego_fill_band_half_width_m(
    x_cam_m: float,
    geom: VehiclePlanarGeom,
) -> float:
    """Wider fill AND band: ego mid ± (track/2 + reach + margin)."""
    lane_half = 0.5 * float(geom.track_width_m)
    reach = kinematic_lateral_reach_m(x_cam_m, geom)
    return float(lane_half + reach + float(geom.band_margin_m))


# Back-compat alias used by older call sites / docs.
def ego_band_half_width_m(
    x_cam_m: float,
    geom: VehiclePlanarGeom,
) -> float:
    return ego_fill_band_half_width_m(x_cam_m, geom)


def build_kinematic_ego_band(
    bev_hw: tuple[int, int],
    *,
    ipm: MetricIpmParams | None = None,
    geom: VehiclePlanarGeom | None = None,
    center_u: float | None = None,
    for_fill: bool = True,
) -> np.ndarray:
    """Binary mask: u within ego kinematic band per BEV row.

    ``for_fill=True`` → corridor clipping width (track/2 + reach).
    ``for_fill=False`` → mid-association width (reach + margin only).
    """
    h, w = int(bev_hw[0]), int(bev_hw[1])
    p = (ipm or load_metric_ipm()).clamp()
    g = _geom_with_ipm_track(geom, p)
    mpp = float(p.meters_per_pixel)
    cu = float((w - 1) * 0.5 if center_u is None else center_u)
    half_fn = ego_fill_band_half_width_m if for_fill else ego_mid_assoc_half_width_m
    band = np.zeros((h, w), dtype=np.uint8)
    for v in range(h):
        x_cam = float(p.x_max_m - v * mpp)
        half_m = half_fn(x_cam, g)
        half_u = half_m / mpp
        u0 = int(np.clip(np.floor(cu - half_u), 0, w - 1))
        u1 = int(np.clip(np.ceil(cu + half_u), 0, w - 1))
        if u1 >= u0:
            band[v, u0 : u1 + 1] = 255
    return band


def guard_fitted_rails_with_ego_band(
    left_u: np.ndarray,
    right_u: np.ndarray,
    bev_hw: tuple[int, int],
    *,
    ipm: MetricIpmParams | None = None,
    geom: VehiclePlanarGeom | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Soft-snap rail midpoints that escape the kinematic ego mid band.

    Rows whose mid is outside the *association* band are recentered to ego_u
    with track_width prior (keeps a coherent corridor instead of jumping).
    Returns (left, right, n_guard_rows).
    """
    h, w = int(bev_hw[0]), int(bev_hw[1])
    p = (ipm or load_metric_ipm()).clamp()
    g = _geom_with_ipm_track(geom, p)
    mpp = float(p.meters_per_pixel)
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
        x_cam = float(p.x_max_m - v * mpp)
        half_m = ego_mid_assoc_half_width_m(x_cam, g)
        half_u = half_m / mpp
        if abs(mid - ego_u) <= half_u:
            left[v], right[v] = lu, ru
            continue
        left[v] = ego_u - half_lane_u
        right[v] = ego_u + half_lane_u
        guarded += 1
    return left.astype(np.float32), right.astype(np.float32), guarded


def fill_between_fitted_rails(
    left_u: np.ndarray,
    right_u: np.ndarray,
    bev_hw: tuple[int, int],
    *,
    ipm: MetricIpmParams | None = None,
    allow_one_sided: bool = True,
) -> tuple[np.ndarray, FillStats]:
    """Fill BEV corridor between fitted left/right boundary U per row.

    Width prior = ``track_width_m`` (0.35 m). Oversized pairs are center-clipped;
    extreme outliers are skipped to avoid dashed-lane FOV leaks.
    """
    h, w = int(bev_hw[0]), int(bev_hw[1])
    filled = np.zeros((h, w), dtype=np.uint8)
    left = np.asarray(left_u, dtype=np.float64).reshape(-1)
    right = np.asarray(right_u, dtype=np.float64).reshape(-1)
    n = min(h, left.size, right.size)
    if n <= 0:
        return filled, FillStats(total_rows=h)

    width_px = _road_width_px(ipm)
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
            if span < 1.0:
                skipped += 1
                continue
            if span > skip_max:
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


def compose_drivable(
    road: np.ndarray,
    between: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Compose road / fitted-between masks.

    Modes:
      road     — black|red only
      between  — fitted rail corridor
      road_in  — road ∩ between
      union    — between ∪ road
    """
    if mode == 'road':
        return road.copy()
    if mode == 'between':
        return between.copy()
    if mode == 'road_in':
        return cv2.bitwise_and(road, between)
    return cv2.bitwise_or(between, road)


COMPOSE_MODES = ('road', 'between', 'road_in', 'union')

COMPOSE_HELP = {
    'road': 'black|red road only',
    'between': 'fill between fitted rails',
    'road_in': 'road ∩ fitted between',
    'union': 'fitted between ∪ road',
}


def fill_between_fork_lane_pairs(
    fork_lane_pairs: tuple | list,
    bev_hw: tuple[int, int],
    *,
    ipm: MetricIpmParams | None = None,
    lateral_rank: int | None = None,
) -> tuple[np.ndarray, FillStats]:
    """Fill corridors from out_fork ``ForkLanePair`` outer/inner rails.

    If ``lateral_rank`` is set, only that branch (0=L, 1=R). Otherwise OR both.
    """
    h, w = int(bev_hw[0]), int(bev_hw[1])
    combined = np.zeros((h, w), dtype=np.uint8)
    total = FillStats(total_rows=h)
    used = 0
    for pair in fork_lane_pairs or ():
        rank = int(getattr(pair, 'lateral_rank', -1))
        if lateral_rank is not None and rank != int(lateral_rank):
            continue
        outer = np.asarray(getattr(pair, 'outer_u'), dtype=np.float32)
        inner = np.asarray(getattr(pair, 'inner_u'), dtype=np.float32)
        mask, st = fill_between_fitted_rails(
            outer, inner, (h, w), ipm=ipm, allow_one_sided=True
        )
        combined = cv2.bitwise_or(combined, mask)
        used += 1
        total = FillStats(
            valid_rows=max(total.valid_rows, st.valid_rows),
            clipped_rows=total.clipped_rows + st.clipped_rows,
            one_sided_rows=total.one_sided_rows + st.one_sided_rows,
            skipped_rows=st.skipped_rows,
            total_rows=h,
        )
    if used == 0:
        return combined, FillStats(total_rows=h)
    return combined, total


def corridor_from_debug(
    debug: Any,
    course: str,
    *,
    bev_hw: tuple[int, int],
    ipm: MetricIpmParams | None = None,
    use_fork: bool = False,
    fork_rank: int | None = None,
    kinematic_guard: bool = True,
    geom: VehiclePlanarGeom | None = None,
) -> tuple[np.ndarray, FillStats, str, bool, np.ndarray, np.ndarray]:
    """Pick between-rails mask: primary L/R by default; out_fork only if opted in.

    Returns (between, stats, rail_label, used_fork, left_u, right_u) — rails are
    the ones used for fill (guarded when kinematic_guard and not fork).
    """
    left_u, right_u, rail_label = course_rails_from_debug(debug, course)
    fork_active = bool(getattr(debug, 'fork_active', False))
    pairs = tuple(getattr(debug, 'fork_lane_pairs', ()) or ())
    if use_fork and fork_active and pairs and course.strip().lower() in ('out', 'white'):
        between, stats = fill_between_fork_lane_pairs(
            pairs, bev_hw, ipm=ipm, lateral_rank=fork_rank
        )
        src = getattr(debug, 'fork_split_source', '') or 'fork'
        return between, stats, f'{rail_label}+fork({src})', True, left_u, right_u

    guard_n = 0
    if kinematic_guard:
        left_u, right_u, guard_n = guard_fitted_rails_with_ego_band(
            left_u, right_u, bev_hw, ipm=ipm, geom=geom
        )
    between, stats = fill_between_fitted_rails(
        left_u, right_u, bev_hw, ipm=ipm
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
    if kinematic_guard:
        band = build_kinematic_ego_band(bev_hw, ipm=ipm, geom=geom)
        between = cv2.bitwise_and(between, band)
    return between, stats, rail_label, False, left_u, right_u
