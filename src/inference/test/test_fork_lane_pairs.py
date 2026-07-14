"""Tests for yellow fork L/R marking pair split."""

from __future__ import annotations

import numpy as np


def _synthetic_yellow_fork_mask(h: int, w: int) -> np.ndarray:
    """Four diverging yellow polylines (L-out, L-in, R-in, R-out)."""

    mask = np.zeros((h, w), dtype=np.uint8)
    mid = (w - 1) / 2.0
    for row in range(h - 1, h // 5, -1):
        t = 1.0 - row / float(h - 1)  # 0 near, 1 far
        # Outers flare; inners form a V gore.
        lo = mid - 40 - 35 * t
        li = mid - 8 - 25 * t
        ri = mid + 8 + 25 * t
        ro = mid + 40 + 35 * t
        for u in (lo, li, ri, ro):
            c = int(round(u))
            if 1 <= c < w - 1:
                mask[row, c - 1 : c + 2] = 255
    return mask


def test_track_and_pair_synthetic_fork():
    from inference.modules import lane_detection as ld

    mask = _synthetic_yellow_fork_mask(ld.BEV_HEIGHT, ld.BEV_WIDTH)
    pairs, tracks = ld.extract_yellow_fork_lane_pairs(mask)
    assert len(tracks) >= 3, f"expected >=3 tracks, got {len(tracks)}"
    assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}"
    assert pairs[0].lateral_rank == 0
    assert pairs[1].lateral_rank == 1

    # Left center should sit left of right center (larger y_left / smaller u).
    c0 = float(np.nanmedian(pairs[0].center_u))
    c1 = float(np.nanmedian(pairs[1].center_u))
    assert c0 < c1

    branches = ld.fork_lane_pairs_to_road_branches(pairs)
    assert len(branches) == 2
    assert branches[0].lateral_rank == 0
    assert len(branches[0].points) >= 2
    # Rank 0 is leftmost → larger median y.
    assert float(np.median(branches[0].points[:, 1])) > float(
        np.median(branches[1].points[:, 1])
    )


def test_dual_courses_to_fork_pairs():
    """WonJung primary+alt courses → left/right ForkLanePair by far mid-u."""
    from inference.modules import lane_detection as ld

    h = ld.BEV_HEIGHT
    mid = (ld.BEV_WIDTH - 1) / 2.0
    primary_l = np.full(h, np.nan, dtype=np.float32)
    primary_r = np.full(h, np.nan, dtype=np.float32)
    alt_l = np.full(h, np.nan, dtype=np.float32)
    alt_r = np.full(h, np.nan, dtype=np.float32)
    for row in range(h - 1, h // 5, -1):
        t = 1.0 - row / float(h - 1)
        # Primary = left lane; alt = right lane (diverging).
        primary_l[row] = mid - 40 - 30 * t
        primary_r[row] = mid - 10 - 20 * t
        alt_l[row] = mid + 10 + 20 * t
        alt_r[row] = mid + 40 + 30 * t

    pairs = ld.fork_lane_pairs_from_dual_courses(
        primary_l, primary_r, alt_l, alt_r
    )
    assert len(pairs) == 2
    assert pairs[0].lateral_rank == 0
    assert pairs[1].lateral_rank == 1
    c0 = float(np.nanmedian(pairs[0].center_u))
    c1 = float(np.nanmedian(pairs[1].center_u))
    assert c0 < c1

    # Swapped mid order should still rank left=0.
    swapped = ld.fork_lane_pairs_from_dual_courses(
        alt_l, alt_r, primary_l, primary_r
    )
    assert len(swapped) == 2
    assert float(np.nanmedian(swapped[0].center_u)) < float(
        np.nanmedian(swapped[1].center_u)
    )


def test_fork_preview_with_pairs():
    from inference.modules import lane_detection as ld

    mask = _synthetic_yellow_fork_mask(ld.BEV_HEIGHT, ld.BEV_WIDTH)
    pairs, tracks = ld.extract_yellow_fork_lane_pairs(mask)
    debug = ld.LaneDebugFrame(
        bev=np.zeros((ld.BEV_HEIGHT, ld.BEV_WIDTH, 3), dtype=np.uint8),
        road_clean=np.zeros((ld.BEV_HEIGHT, ld.BEV_WIDTH), dtype=np.uint8),
        yellow_connected_bev=mask,
        fork_lane_pairs=tuple(pairs),
        fork_mark_tracks=tuple(tracks),
        fork_split_source='yellow_marks',
        road_branches=tuple(ld.fork_lane_pairs_to_road_branches(pairs)),
    )
    for focus in ('all', 'left', 'right'):
        preview = ld.make_fork_lane_pair_preview(debug, focus=focus)
        assert preview.shape == (ld.BEV_HEIGHT, ld.BEV_WIDTH, 3)


def test_two_diverging_outers_synthesize_inners():
    """Wide 2-track fork (white outers) should still yield 2 path pairs.

    Synthetic inner must be a FULL lane width from the outer so the midpoint
    is the lane center (half-width), not the quarter-mark.
    """

    from inference.modules import lane_detection as ld

    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    mask = np.zeros((h, w), dtype=np.uint8)
    mid = (w - 1) / 2.0
    for row in range(h - 1, h // 5, -1):
        t = 1.0 - row / float(h - 1)
        lo = mid - 25 - 55 * t
        ro = mid + 25 + 55 * t
        for u in (lo, ro):
            c = int(round(u))
            if 1 <= c < w - 1:
                mask[row, c - 1 : c + 2] = 255
    pairs, tracks = ld.extract_marking_fork_lane_pairs(mask)
    assert len(tracks) == 2
    assert len(pairs) == 2, f"expected 2 pairs from diverging outers, got {len(pairs)}"
    far_end = max(1, int(round(ld.BEV_HEIGHT * ld.FORK_FAR_ZONE_RATIO)))
    c0 = float(np.nanmedian(pairs[0].center_u[:far_end]))
    c1 = float(np.nanmedian(pairs[1].center_u[:far_end]))
    assert c0 < c1

    half_w = (0.5 * ld.FORK_PAIR_WIDTH_M) / ld.METERS_PER_PIXEL
    # Far-zone: each center should sit ~half lane inside its outer.
    far = ~np.isnan(pairs[0].outer_u[:far_end]) & ~np.isnan(
        pairs[0].center_u[:far_end]
    )
    assert np.any(far)
    ctr_off = float(
        np.nanmedian(
            pairs[0].center_u[:far_end][far] - pairs[0].outer_u[:far_end][far]
        )
    )
    assert abs(ctr_off - half_w) < half_w * 0.55, (
        f"far center offset {ctr_off} not near half-width {half_w}"
    )
def test_centerline_synthesizes_missing_side():
    from inference.modules import lane_detection as ld

    left = np.full(20, 100.0, dtype=np.float32)
    right = np.full(20, np.nan, dtype=np.float32)
    center = ld.centerline_from_boundaries(left, right)
    half = (0.5 * ld.ROAD_WIDTH_M) / ld.METERS_PER_PIXEL
    assert np.allclose(center, 100.0 + half)


def test_road_split_fork_pairs():
    from inference.modules import lane_detection as ld

    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    road = np.zeros((h, w), dtype=np.uint8)
    mid = w // 2
    # Near: one corridor. Far: two corridors.
    for row in range(h - 1, h // 2, -1):
        road[row, mid - 40 : mid + 40] = 255
    for row in range(h // 2, h // 6, -1):
        t = 1.0 - row / float(h - 1)
        gap = int(10 + 40 * t)
        road[row, mid - 70 : mid - gap] = 255
        road[row, mid + gap : mid + 70] = 255
    pairs, tracks = ld.extract_road_split_fork_lane_pairs(road)
    assert len(pairs) == 2, f"road-split expected 2 pairs, got {len(pairs)}"
    assert len(tracks) == 4


if __name__ == '__main__':
    test_track_and_pair_synthetic_fork()
    test_fork_preview_with_pairs()
    test_two_diverging_outers_synthesize_inners()
    test_centerline_synthesizes_missing_side()
    test_road_split_fork_pairs()
    print('ok')
