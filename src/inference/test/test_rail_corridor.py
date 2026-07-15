"""Tests for blob rail corridor and road_in drivable synthesis."""

from __future__ import annotations

import numpy as np

from inference.modules.perception.blob.corridor import extract_drivable_blob
from inference.modules.perception.blob.rail_corridor import (
    centerline_from_rails,
    compose_road_in,
    fill_between_rails,
    resolve_course_lane_mask,
    yellow_lane_present,
)


def _blank(h: int = 120, w: int = 160) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def test_yellow_lane_present_threshold():
    y = _blank()
    y[80:100, 70:90] = 255
    assert yellow_lane_present(y, min_px=50)
    assert not yellow_lane_present(y, min_px=5000)


def test_resolve_course_lane_never_or():
    h, w = 80, 100
    white = np.zeros((h, w), dtype=np.uint8)
    yellow = np.zeros((h, w), dtype=np.uint8)
    white[40:60, 20:80] = 255
    yellow[40:60, 30:70] = 255
    lane, used_y = resolve_course_lane_mask(white, yellow, prefer_yellow=True)
    assert used_y
    assert np.array_equal(lane, yellow)
    lane_out, used_w = resolve_course_lane_mask(white, yellow, prefer_yellow=False)
    assert not used_w
    assert np.array_equal(lane_out, white)


def test_fill_between_rails_basic():
    h, w = 60, 80
    left = np.full(h, 20.0, dtype=np.float32)
    right = np.full(h, 60.0, dtype=np.float32)
    filled, stats = fill_between_rails(
        left,
        right,
        (h, w),
        track_width_m=0.35,
        meters_per_pixel=0.004,
    )
    assert stats.valid_rows == h
    assert np.count_nonzero(filled) > h * 30


def test_compose_road_in_clips_bleed():
    road = np.zeros((40, 60), dtype=np.uint8)
    road[:, :] = 255
    corridor = np.zeros_like(road)
    corridor[10:30, 20:40] = 255
    clipped = compose_road_in(road, corridor)
    assert np.count_nonzero(clipped) < np.count_nonzero(road)
    assert np.count_nonzero(clipped) == np.count_nonzero(corridor)


def test_extract_drivable_dt_strip():
    h, w = 100, 120
    road = np.zeros((h, w), dtype=np.uint8)
    road[50:95, 35:85] = 255
    white = np.zeros((h, w), dtype=np.uint8)
    white[50:95, 35:45] = 255
    white[50:95, 75:85] = 255
    yellow = _blank(h, w)
    blob, _between, stats, left, right, used_y = extract_drivable_blob(
        road,
        white,
        yellow,
        prefer_yellow=False,
        track_width_m=0.35,
        meters_per_pixel=0.004,
        x_max_m=1.2,
    )
    assert not used_y
    assert stats.method.startswith('dt_strip')
    assert stats.road_in_mode
    assert blob.shape == road.shape
    assert int(np.count_nonzero(blob)) < int(np.count_nonzero(road))
    assert left.size == h and right.size == h


def test_centerline_dt_ridge_smoother_than_row_mid():
    from inference.modules.perception.blob.centerline import (
        blob_row_mids,
        centerline_from_blob,
        dt_ridge_mids,
        smooth_mids_1d,
    )

    h, w = 80, 100
    blob = np.zeros((h, w), dtype=np.uint8)
    # tapered corridor with jagged edges
    for v in range(20, 75):
        mid = 50 + int(3 * np.sin(v / 5.0))
        half = 20
        blob[v, mid - half : mid + half] = 255
        if v % 4 == 0:
            blob[v, mid + half : mid + half + 8] = 255
    raw = smooth_mids_1d(blob_row_mids(blob), win=5)
    ridge_pts, ridge = centerline_from_blob(
        blob, x_max_m=1.0, meters_per_pixel=0.004, bev_width=w, mode='dt_ridge'
    )
    assert ridge_pts.shape[0] >= 10
    raw_j = float(np.nanmean(np.abs(np.diff(raw[np.isfinite(raw)]))))
    ridge_j = float(np.nanmean(np.abs(np.diff(ridge[np.isfinite(ridge)]))))
    assert ridge_j <= raw_j * 1.05 + 0.5

