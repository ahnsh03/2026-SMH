"""Drivable mask: near-ego blob + DT-strip (track-width prior).

A/B on bag frames (``viz_mask_postprocess_ab.py`` / ``viz_cyan_ab.py``) showed:
  - rail fill / road_in (old) → zigzag mid, worst score
  - F_dtstrip (near blob + distance-transform ridge ± half_w) wins OUT
  - C_near / E_walls compete on IN when paint is strong
  - ``black_cyan`` / ``black_cyan_2``: near-robot CC only *before* morph
    (billboard wash appears on- and off-lane; keep on-lane wash)

Pipeline
--------
1. morph denoise road (black_near|red|cyan_near) — open 3 / close 13 (bag SSOT)
2. keep largest CC by **near-band mass** touching ego near-band  (= ego blob SSOT)
3. DT ridge mid → clip each row to mid ± track_width/2 ∩ blob
4. optional: if course paint dense, AND with paint-wall flood (lights walls)
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from inference.modules.perception.blob.morph_blob import (
    BlobSelectStats,
    morph_clean_road,
    select_best_blob,
)
from inference.modules.perception.blob.rail_corridor import (
    FillStats,
    build_kinematic_ego_band,
    resolve_course_lane_mask,
    vehicle_geom_for_platform,
    yellow_lane_present,
)


@dataclass(frozen=True)
class CorridorStats(BlobSelectStats):
    union_coverage: float = 0.0
    clipped_rows: int = 0
    between_rows: int = 0
    road_in_mode: bool = False
    used_yellow: bool = False
    rail_valid_ratio: float = 0.0
    method: str = 'dt_strip'


def course_lane_mask(
    white: np.ndarray,
    yellow: np.ndarray,
    *,
    prefer_yellow: bool,
) -> np.ndarray:
    lane, _ = resolve_course_lane_mask(white, yellow, prefer_yellow=prefer_yellow)
    return lane


def denoise_road_mask(road: np.ndarray) -> np.ndarray:
    """Morph denoise road before near-ego CC (bag SSOT open 3 / close 13)."""

    cleaned = morph_clean_road(road, open_k=3, close_k=13, max_hole_px=400)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return cleaned
    out = np.zeros_like(cleaned)
    min_area = 80
    for lab in range(1, n):
        if int(stats[lab, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == lab] = 255
    return out


def _dt_ridge_mids(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    h = binary.shape[0]
    mids = np.full(h, np.nan, dtype=np.float32)
    if not np.any(binary):
        return mids
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    for v in range(h):
        row = dist[v]
        if float(row.max()) <= 0.0:
            continue
        mids[v] = float(np.argmax(row))
    return mids


def _smooth_mids(mids: np.ndarray, win: int = 15) -> np.ndarray:
    from inference.modules.perception.blob.centerline import smooth_mids_1d

    return smooth_mids_1d(mids, win=win)


def dt_strip_from_blob(
    blob: np.ndarray,
    *,
    track_width_m: float,
    meters_per_pixel: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip blob to DT-ridge ± half track width. Returns (strip, left_u, right_u)."""

    if blob is None or blob.size == 0 or not np.any(blob):
        empty = np.zeros_like(blob) if blob is not None else np.zeros((0, 0), np.uint8)
        return empty, np.empty(0, np.float32), np.empty(0, np.float32)

    h, w = blob.shape[:2]
    half = 0.5 * float(track_width_m) / float(meters_per_pixel)
    mids = _smooth_mids(_dt_ridge_mids(blob), win=15)
    out = np.zeros_like(blob)
    left = np.full(h, np.nan, dtype=np.float32)
    right = np.full(h, np.nan, dtype=np.float32)
    ego = (w - 1) * 0.5
    for v in range(h):
        mid = float(mids[v]) if np.isfinite(mids[v]) else ego
        u0 = int(np.clip(np.floor(mid - half), 0, w - 1))
        u1 = int(np.clip(np.ceil(mid + half), 0, w - 1))
        row = blob[v, u0 : u1 + 1]
        out[v, u0 : u1 + 1] = row
        cols = np.flatnonzero(out[v] > 0)
        if cols.size >= 2:
            left[v] = float(cols[0])
            right[v] = float(cols[-1])
        elif cols.size == 1:
            left[v] = right[v] = float(cols[0])
    return out, left, right


def _paint_wall_flood(
    road: np.ndarray,
    lane: np.ndarray,
    *,
    meters_per_pixel: float,
    dilate_m: float = 0.04,
) -> np.ndarray:
    h, w = road.shape[:2]
    k = max(3, int(round(dilate_m / meters_per_pixel)))
    if k % 2 == 0:
        k += 1
    walls = np.zeros((h, w), dtype=np.uint8)
    if lane is not None and lane.size and lane.shape[:2] == (h, w):
        walls = cv2.dilate(
            (lane > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
    seedable = ((road > 0) & (walls == 0)).astype(np.uint8) * 255
    seed_x = w // 2
    found = False
    seed_y = h - 2
    for dy in range(0, min(40, h)):
        y = h - 1 - dy
        if seedable[y, seed_x] > 0:
            seed_y, found = y, True
            break
        for dx in range(1, 25):
            for sx in (seed_x - dx, seed_x + dx):
                if 0 <= sx < w and seedable[y, sx] > 0:
                    seed_y, seed_x, found = y, sx, True
                    break
            if found:
                break
        if found:
            break
    if not found:
        return seedable
    flood = seedable.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (int(seed_x), int(seed_y)), 128)
    return np.where(flood == 128, np.uint8(255), np.uint8(0))


def extract_drivable_blob(
    road_raw: np.ndarray,
    white: np.ndarray,
    yellow: np.ndarray,
    *,
    prefer_yellow: bool,
    track_width_m: float,
    meters_per_pixel: float,
    x_max_m: float = 1.5,
    use_road_in: bool = True,
) -> tuple[np.ndarray, np.ndarray, CorridorStats, np.ndarray, np.ndarray, bool]:
    """DT-strip drivable (default). ``use_road_in`` retained for API compat."""

    del use_road_in
    morph = denoise_road_mask(road_raw)
    lane_mask, used_yellow = resolve_course_lane_mask(
        white, yellow, prefer_yellow=prefer_yellow
    )
    left = np.empty(0, dtype=np.float32)
    right = np.empty(0, dtype=np.float32)
    method = 'empty'
    between = np.zeros_like(morph) if morph.size else np.zeros((0, 0), dtype=np.uint8)

    if morph.size == 0 or not np.any(morph):
        stats = CorridorStats(method=method, used_yellow=used_yellow)
        return morph, between, stats, left, right, used_yellow

    near, sel = select_best_blob(
        morph,
        track_width_m=float(track_width_m),
        meters_per_pixel=float(meters_per_pixel),
        lane_bonus=lane_mask if lane_mask.size and lane_mask.shape == morph.shape else None,
    )
    if not np.any(near):
        near = morph

    # Soft ego band to drop far side-course bleed before DT.
    geom = replace(vehicle_geom_for_platform('sim'), track_width_m=float(track_width_m))
    band = build_kinematic_ego_band(
        near.shape[:2],
        meters_per_pixel=float(meters_per_pixel),
        x_max_m=float(x_max_m),
        geom=geom,
    )
    near_band = cv2.bitwise_and(near, band)
    if int(np.count_nonzero(near_band)) < 80:
        near_band = near

    strip, left, right = dt_strip_from_blob(
        near_band,
        track_width_m=float(track_width_m),
        meters_per_pixel=float(meters_per_pixel),
    )
    method = 'dt_strip'
    between = strip.copy()

    # When course paint is dense enough, prefer paint-wall flood ∩ strip.
    lane_px = int(np.count_nonzero(lane_mask)) if lane_mask.size else 0
    paint_ok = lane_px >= (400 if used_yellow else 250)
    if paint_ok and lane_mask.size and lane_mask.shape[:2] == morph.shape[:2]:
        walls = _paint_wall_flood(
            near_band, lane_mask, meters_per_pixel=float(meters_per_pixel)
        )
        wall_px = int(np.count_nonzero(walls))
        strip_px = int(np.count_nonzero(strip))
        if wall_px >= max(80, int(0.4 * strip_px)):
            fused = cv2.bitwise_and(strip, walls)
            if int(np.count_nonzero(fused)) >= max(60, int(0.35 * strip_px)):
                strip = fused
                method = 'dt_strip+walls'
                # Refresh rails from fused
                h, w = strip.shape
                left = np.full(h, np.nan, np.float32)
                right = np.full(h, np.nan, np.float32)
                for v in range(h):
                    cols = np.flatnonzero(strip[v] > 0)
                    if cols.size >= 2:
                        left[v] = float(cols[0])
                        right[v] = float(cols[-1])

    valid_rows = int(np.sum(np.isfinite(left) & np.isfinite(right))) if left.size else 0
    cov = float(np.mean(strip > 0)) if strip.size else 0.0
    stats = CorridorStats(
        n_components=int(sel.n_components),
        chosen_label=int(sel.chosen_label),
        chosen_area=int(np.count_nonzero(strip)),
        score=float(np.count_nonzero(strip)),
        union_coverage=cov,
        clipped_rows=0,
        between_rows=valid_rows,
        road_in_mode=True,
        used_yellow=used_yellow,
        rail_valid_ratio=float(valid_rows) / float(max(strip.shape[0], 1)),
        method=method,
    )
    _ = (FillStats, yellow_lane_present)
    return strip, between, stats, left, right, used_yellow
