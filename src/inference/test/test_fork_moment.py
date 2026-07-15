"""Unit tests for IN/OUT fork-moment mask scorers."""

from __future__ import annotations

import numpy as np

from inference.modules.perception.fork.moment import (
    score_in_circle_fork_moment,
    score_out_fork_moment,
)


def _empty(h: int = 120, w: int = 160) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)


def test_in_moment_rejects_empty():
    y, r = _empty()
    s = score_in_circle_fork_moment(y, r)
    assert s.hard is False
    assert s.hard_base is False


def test_out_moment_rejects_empty():
    w, r = _empty()
    s = score_out_fork_moment(w, r)
    assert s.hard is False


def test_in_moment_hits_on_synthetic_dual_yellow_and_free():
    """Far dual yellow + dual free corridor + wide far span → hard."""

    h, w = 200, 300
    yellow = np.zeros((h, w), dtype=np.uint8)
    road = np.zeros((h, w), dtype=np.uint8)
    # Far/mid: two yellow rails with gap; free fill in roads between/around.
    for v in range(0, int(h * 0.70)):
        yellow[v, 40:55] = 1
        yellow[v, 245:260] = 1
        road[v, 20:280] = 1
    # Near: single corridor (narrower) so span_ratio rises.
    for v in range(int(h * 0.70), h):
        yellow[v, 120:140] = 1
        yellow[v, 160:180] = 1
        road[v, 100:200] = 1

    s = score_in_circle_fork_moment(yellow * 255, road * 255)
    assert s.far_dual_yellow >= 70.0
    assert s.hard_base is True
    assert s.far_dual_free >= 70.0
    assert s.span_ratio >= 1.3
    assert s.hard is True


def test_out_moment_hits_on_wide_white_sep():
    h, w = 200, 320
    white = np.zeros((h, w), dtype=np.uint8)
    road = np.zeros((h, w), dtype=np.uint8)
    for v in range(0, int(h * 0.70)):
        white[v, 30:50] = 1
        white[v, 270:290] = 1  # sep mid ≈ 240 > 150
        # Split road (gore) so far_dual_road fires — solid fill is a single run.
        road[v, 20:140] = 1
        road[v, 180:300] = 1
    for v in range(int(h * 0.70), h):
        white[v, 130:150] = 1
        road[v, 110:210] = 1

    s = score_out_fork_moment(white * 255, road * 255)
    assert s.sep_white >= 150.0
    assert s.far_dual_white >= 90.0
    assert s.far_dual_road >= 80.0
    assert s.hard is True


def test_out_moment_rejects_narrow_parallel_rails():
    """Lane-width white L/R (~90px sep) must not hard-fire."""

    h, w = 200, 320
    white = np.zeros((h, w), dtype=np.uint8)
    road = np.zeros((h, w), dtype=np.uint8)
    for v in range(h):
        white[v, 110:125] = 1
        white[v, 195:210] = 1  # mid sep ≈ 85
        road[v, 90:230] = 1

    s = score_out_fork_moment(white * 255, road * 255)
    assert s.sep_white < 150.0
    assert s.hard is False


def test_out_fork_capture_fuses_tip_and_stretch():
    from inference.modules.perception.fork.capture import score_out_fork_capture

    h, w = 200, 320
    white = np.zeros((h, w), dtype=np.uint8)
    road = np.zeros((h, w), dtype=np.uint8)
    ego = np.zeros((h, w), dtype=np.uint8)
    # Tip-like white dual far
    for v in range(0, int(h * 0.70)):
        white[v, 30:50] = 255
        white[v, 270:290] = 255
        road[v, 20:140] = 255
        road[v, 180:300] = 255
        ego[v, 20:140] = 255
        ego[v, 180:300] = 255
    for v in range(int(h * 0.70), h):
        white[v, 130:150] = 255
        road[v, 110:210] = 255
        ego[v, 110:210] = 255

    s = score_out_fork_capture(white, road, ego)
    assert s.tip is True
    assert s.in_stretch is True or s.ego.soft is True
    assert s.capture is True
