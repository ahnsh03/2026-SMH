"""Smoke tests for LaneDebugFrame / mode previews (no ROS / camera)."""

from __future__ import annotations

import numpy as np


def test_detect_with_debug_empty():
    from inference.modules import lane_detection as ld

    dets, debug = ld.detect_with_debug(np.zeros((0, 0, 3), dtype=np.uint8))
    assert dets.fork_active is False
    assert debug.red_coverage == 0.0


def test_detect_with_debug_dummy_frame_and_modes():
    from inference.modules import lane_detection as ld

    ld.VISUALIZE = False
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    # Dark road + faint white lines so pipeline runs without crashing.
    frame[90:150, 40:50] = (220, 220, 220)
    frame[90:150, 270:280] = (220, 220, 220)
    frame[100:140, 50:270] = (10, 10, 10)

    dets, debug = ld.detect_with_debug(frame)
    assert debug.bev.ndim == 3
    assert debug.white_bev.shape[:2] == debug.bev.shape[:2]
    assert debug.red_bev.shape == debug.white_bev.shape
    assert 0.0 <= debug.red_coverage <= 1.0

    for mode in (
        'white',
        'yellow',
        'dash',
        'dash_left',
        'dash_right',
        'fork',
        'fork_left',
        'fork_right',
        'red',
        'crossing',
    ):
        preview = ld.render_mode_preview(mode, debug)
        assert preview.ndim == 3
        assert preview.shape[0] > 0 and preview.shape[1] > 0


def test_apply_detect_tune_roundtrip():
    from inference.modules import lane_detection as ld

    before = ld.get_detect_tune()
    ld.apply_detect_tune(
        crossing_coverage_ratio=0.55,
        crossing_min_rows=5,
        min_branch_separation_m=0.20,
        dash_max_lateral_error_m=0.06,
        red_h_low_wrap=8,
    )
    mid = ld.get_detect_tune()
    assert abs(float(mid['crossing_coverage_ratio']) - 0.55) < 1e-6
    assert int(mid['crossing_min_rows']) == 5
    assert abs(float(mid['min_branch_separation_m']) - 0.20) < 1e-6
    assert int(mid['red_h_low_wrap']) == 8
    ld.apply_detect_tune(
        crossing_coverage_ratio=float(before['crossing_coverage_ratio']),
        crossing_min_rows=int(before['crossing_min_rows']),
        min_branch_separation_m=float(before['min_branch_separation_m']),
        dash_max_lateral_error_m=float(before['dash_max_lateral_error_m']),
        red_h_low_wrap=int(before['red_h_low_wrap']),
    )


if __name__ == '__main__':
    test_detect_with_debug_empty()
    test_detect_with_debug_dummy_frame_and_modes()
    test_apply_detect_tune_roundtrip()
    print('ok')
