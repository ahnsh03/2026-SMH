"""Unit tests for Pure Pursuit lane planner (no ROS / camera required)."""

from __future__ import annotations

import math
from pathlib import Path


from inference.modules.lane_planner import (  # noqa: E402
    LaneControlParams,
    LanePlanner,
    centerline_y_at_lookahead,
    mock_white_lane,
    pure_pursuit_steer,
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


def test_pp_straight_zero_steer():
    raw, alpha, delta = pure_pursuit_steer(
        0.8,
        0.0,
        wheelbase_m=0.24,
        lookahead_m=0.8,
        max_steer_angle_rad=0.5236,
    )
    assert abs(raw) < 1e-6
    assert abs(alpha) < 1e-6
    assert abs(delta) < 1e-6


def test_pp_left_target_negative_steer():
    """Target left (y>0) → negative /control steering (D-Racer left)."""
    raw, alpha, _delta = pure_pursuit_steer(
        0.8,
        0.15,
        wheelbase_m=0.24,
        lookahead_m=0.8,
        max_steer_angle_rad=0.5236,
    )
    assert alpha > 0.0
    assert raw < 0.0


def test_pp_shorter_ld_stronger_steer():
    soft, _, _ = pure_pursuit_steer(
        1.2,
        0.12,
        wheelbase_m=0.24,
        lookahead_m=1.2,
        max_steer_angle_rad=0.5236,
    )
    hard, _, _ = pure_pursuit_steer(
        0.5,
        0.12,
        wheelbase_m=0.24,
        lookahead_m=0.5,
        max_steer_angle_rad=0.5236,
    )
    assert abs(hard) > abs(soft)


def test_planner_steer_sign_d_racer():
    """Centerline left (y>0) → negative steering (left). Center right → +steer."""
    params = LaneControlParams(ema_alpha=1.0, steer_rate_limit=1.0)
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


def test_yellow_follow_centerline():
    from inference.modules.lane_planner import mock_lane
    from inference.types import LaneMarking

    dets = mock_lane(0.175, -0.175, color=LaneMarking.COLOR_YELLOW)
    y_c, conf = centerline_y_at_lookahead(dets, 0.8, 0.175, follow_color='yellow')
    assert y_c is not None
    assert abs(y_c) < 1e-4
    assert conf > 0.4
    assert dets.white_left() is None
    assert dets.yellow_left() is not None


def test_hold_decay_when_lost():
    params = LaneControlParams(ema_alpha=1.0, steer_rate_limit=1.0, hold_decay=0.5)
    planner = LanePlanner(params)
    dets = mock_white_lane(y_left=0.25, y_right=-0.10)
    first = planner.step(dets)
    assert abs(first.steering_offset) > 0.05
    lost = planner.step(LaneDetections())
    assert abs(lost.steering_offset) < abs(first.steering_offset)


def test_larger_wheelbase_less_normalized_steer_for_same_curvature_path():
    """Same target geometry: larger L → larger |δ|, but we normalize by δ_max.
    With fixed δ_max, larger L yields larger |raw| until sat.
    """
    small_l, _, d_s = pure_pursuit_steer(
        0.8, 0.1, wheelbase_m=0.174, lookahead_m=0.8, max_steer_angle_rad=0.5236
    )
    large_l, _, d_l = pure_pursuit_steer(
        0.8, 0.1, wheelbase_m=0.24, lookahead_m=0.8, max_steer_angle_rad=0.5236
    )
    assert abs(d_l) > abs(d_s)
    assert abs(large_l) > abs(small_l)
    assert math.isfinite(small_l) and math.isfinite(large_l)


if __name__ == '__main__':
    test_centerline_straight_zero()
    test_centerline_offset_right_of_vehicle()
    test_pp_straight_zero_steer()
    test_pp_left_target_negative_steer()
    test_pp_shorter_ld_stronger_steer()
    test_planner_steer_sign_d_racer()
    test_planner_one_side_left_only()
    test_fuse_throttle_scale()
    test_yellow_follow_centerline()
    test_hold_decay_when_lost()
    test_larger_wheelbase_less_normalized_steer_for_same_curvature_path()
    print('ok')
