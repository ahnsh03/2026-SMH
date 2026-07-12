"""Unit tests for white-lane planner (no ROS / camera required)."""

from __future__ import annotations

import sys
from pathlib import Path

_INFERENCE_SRC = Path(__file__).resolve().parents[1]
if str(_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_SRC))

from inference.modules.lane_planner import (  # noqa: E402
    LaneControlParams,
    LanePlanner,
    centerline_y_at_lookahead,
    mock_white_lane,
)
from inference.pipeline import fuse_control  # noqa: E402
from inference.types import LaneDetections, LaneResult, PipelineContext  # noqa: E402


def test_centerline_straight_zero():
    dets = mock_white_lane(y_left=0.175, y_right=-0.175)
    y_c, conf = centerline_y_at_lookahead(dets, 0.8, 0.175)
    assert y_c is not None
    assert abs(y_c) < 1e-4
    assert conf > 0.4


def test_centerline_offset_right_of_vehicle():
    # Both markings shifted so center is at y=-0.1 (right of vehicle).
    dets = mock_white_lane(y_left=0.075, y_right=-0.275)
    y_c, _ = centerline_y_at_lookahead(dets, 0.8, 0.175)
    assert y_c is not None
    assert y_c < -0.05


def test_planner_steer_sign_d_racer():
    """Centerline left (y>0) → negative steering (left). Center right → +steer."""
    params = LaneControlParams(kp=2.0, ema_alpha=1.0, steer_rate_limit=1.0)
    planner = LanePlanner(params)

    left_of_car = mock_white_lane(y_left=0.30, y_right=-0.05)
    r1 = planner.step(left_of_car)
    assert r1.steering_offset < 0.0
    assert r1.confidence > 0.1

    planner.reset()
    right_of_car = mock_white_lane(y_left=0.05, y_right=-0.30)
    r2 = planner.step(right_of_car)
    assert r2.steering_offset > 0.0


def test_planner_one_side_left_only():
    dets = mock_white_lane(y_left=0.20, y_right=-0.20)
    left = dets.white_left()
    assert left is not None
    one = LaneDetections(lanes=(left,), white_visible=True)
    y_c, conf = centerline_y_at_lookahead(one, 0.8, 0.175)
    assert y_c is not None
    assert conf >= 0.3
    # left at 0.20 → center ≈ 0.025
    assert abs(y_c - 0.025) < 1e-3


def test_fuse_throttle_scale():
    ctx = PipelineContext(
        lane=LaneResult(steering_offset=0.6, confidence=0.9, throttle_scale=0.8)
    )
    cmd = fuse_control(ctx, cruise_throttle=0.5)
    assert abs(cmd.throttle - 0.4) < 1e-6
    assert abs(cmd.steering - 0.6) < 1e-6


def test_hold_decay_when_lost():
    params = LaneControlParams(ema_alpha=1.0, steer_rate_limit=1.0, hold_decay=0.5)
    planner = LanePlanner(params)
    dets = mock_white_lane(y_left=0.25, y_right=-0.10)
    first = planner.step(dets)
    assert abs(first.steering_offset) > 0.05
    lost = planner.step(LaneDetections())
    assert abs(lost.steering_offset) < abs(first.steering_offset)


if __name__ == '__main__':
    test_centerline_straight_zero()
    test_centerline_offset_right_of_vehicle()
    test_planner_steer_sign_d_racer()
    test_planner_one_side_left_only()
    test_fuse_throttle_scale()
    test_hold_decay_when_lost()
    print('ok')
