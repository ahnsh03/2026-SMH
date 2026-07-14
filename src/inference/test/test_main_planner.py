"""Unit tests for MainPlanner primitives (no ROS required)."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

_INFERENCE_SRC = Path(__file__).resolve().parents[1]
if str(_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_SRC))

from inference.pipeline import (  # noqa: E402
    MainPlanner,
    PlannerConfig,
    RisingEventCounter,
)
from inference.types import (  # noqa: E402
    ArucoResult,
    DrivingState,
    PathSource,
    RouteMode,
    TrafficResult,
    TurnSign,
)


def test_event_counter_debounces_and_rearms():
    counter = RisingEventCounter(on_frames=2, off_frames=2)
    events = [counter.update(value) for value in (True, True, True, False, False, True, True)]
    assert events == [False, True, False, False, False, False, True]
    assert counter.events == 2


def test_pure_pursuit_steering_sign():
    planner = MainPlanner(
        PlannerConfig(
            min_points=2,
            lookahead_m=0.5,
            steering_rate_limit_per_sec=10.0,
        )
    )
    left_path = np.array([[0.2, 0.0], [0.8, 0.2]], dtype=np.float32)
    result = planner._pure_pursuit(left_path)
    assert result.valid
    assert result.steering < 0.0  # D-Racer: negative steering is left.

    planner._steering = 0.0
    right_path = np.array([[0.2, 0.0], [0.8, -0.2]], dtype=np.float32)
    result = planner._pure_pursuit(right_path)
    assert result.valid
    assert result.steering > 0.0


def test_pure_pursuit_rejects_short_path():
    planner = MainPlanner(PlannerConfig(min_points=3))
    result = planner._pure_pursuit(np.array([[0.5, 0.0]], dtype=np.float32))
    assert not result.valid


def test_perception_path_is_translated_once_into_rear_axle_frame():
    planner = MainPlanner(
        PlannerConfig(
            min_points=3,
            perception_to_rear_axle_x_m=0.265,
            lookahead_m=0.65,
            curve_lookahead_m=0.65,
            steering_rate_limit_per_sec=10.0,
        )
    )
    camera_path = np.array(
        [[0.22, 0.00], [0.40, 0.03], [0.80, 0.12]], dtype=np.float32
    )
    original = camera_path.copy()

    rear_path = planner._path_in_rear_axle_frame(camera_path)
    np.testing.assert_allclose(rear_path[:, 0], camera_path[:, 0] + 0.265)
    np.testing.assert_allclose(rear_path[:, 1], camera_path[:, 1])
    np.testing.assert_array_equal(camera_path, original)

    with patch.object(
        planner,
        '_cross_track_error',
        wraps=planner._cross_track_error,
    ) as cte:
        result = planner._pure_pursuit(camera_path)
    assert result.valid
    control_path = cte.call_args.args[0]
    np.testing.assert_allclose(control_path[-rear_path.shape[0] :], rear_path)
    assert abs(float(control_path[0, 0]) - planner.config.wheelbase_m) < 1e-6


def test_near_path_is_curve_extrapolated_to_front_axle():
    planner = MainPlanner(
        PlannerConfig(
            wheelbase_m=0.24,
            near_path_fit_span_m=0.30,
            near_path_extrapolation_max_m=0.35,
        )
    )
    x = np.linspace(0.485, 0.90, 20, dtype=np.float32)
    y = (0.40 * x * x - 0.05 * x + 0.01).astype(np.float32)
    observed = np.column_stack((x, y))

    extended, distance = planner._extend_path_to_front_axle(observed)

    assert abs(distance - 0.245) < 1e-5
    assert abs(float(extended[0, 0]) - planner.config.wheelbase_m) < 1e-6
    expected_y = 0.40 * planner.config.wheelbase_m**2 - 0.05 * planner.config.wheelbase_m + 0.01
    assert abs(float(extended[0, 1]) - expected_y) < 1e-4
    np.testing.assert_allclose(extended[-observed.shape[0] :], observed)


def test_near_path_extrapolation_rejects_unbounded_gap():
    planner = MainPlanner(
        PlannerConfig(
            wheelbase_m=0.24,
            near_path_extrapolation_max_m=0.20,
        )
    )
    observed = np.array([[0.50, 0.1], [0.70, 0.2]], dtype=np.float32)

    unchanged, distance = planner._extend_path_to_front_axle(observed)

    assert distance == 0.0
    np.testing.assert_array_equal(unchanged, observed)


def test_cte_correction_steers_toward_offset_path():
    planner = MainPlanner(
        PlannerConfig(
            min_points=2,
            lookahead_m=0.5,
            curve_lookahead_m=0.5,
            steering_rate_limit_per_sec=10.0,
            cte_gain=0.2,
            cte_deadband_m=0.0,
        )
    )
    left_offset = np.array([[0.1, 0.2], [1.0, 0.2]], dtype=np.float32)
    result = planner._pure_pursuit(left_offset)
    assert result.cross_track_error_m > 0.0
    assert result.cte_steering < 0.0

    planner._steering = 0.0
    right_offset = np.array([[0.1, -0.2], [1.0, -0.2]], dtype=np.float32)
    result = planner._pure_pursuit(right_offset)
    assert result.cross_track_error_m < 0.0
    assert result.cte_steering > 0.0


def test_cte_deadband_ignores_small_centerline_noise():
    planner = MainPlanner(
        PlannerConfig(min_points=2, cte_deadband_m=0.02, cte_gain=0.2)
    )
    path = np.array([[0.1, 0.01], [1.0, 0.01]], dtype=np.float32)
    result = planner._pure_pursuit(path)
    assert result.cte_steering == 0.0


def test_heading_correction_steers_with_path_tangent():
    planner = MainPlanner(
        PlannerConfig(
            min_points=3,
            heading_gain=0.25,
            heading_preview_m=0.30,
            heading_sample_span_m=0.15,
            max_heading_steering=0.20,
            steering_rate_limit_per_sec=10.0,
        )
    )
    left_path = np.array(
        [[0.2, 0.0], [0.35, 0.02], [0.5, 0.12], [0.8, 0.35]],
        dtype=np.float32,
    )
    result = planner._pure_pursuit(left_path)
    assert result.heading_error_rad > 0.0
    assert result.heading_steering < 0.0


def test_roundabout_uses_dedicated_lookahead_and_throttle():
    planner = MainPlanner(
        PlannerConfig(
            min_points=3,
            roundabout_lookahead_m=0.32,
            roundabout_straight_lookahead_m=0.32,
            roundabout_curve_lookahead_m=0.32,
            roundabout_throttle=0.06,
            lookahead_shrink_rate_m_per_sec=10.0,
            lookahead_grow_rate_m_per_sec=10.0,
            steering_rate_limit_per_sec=10.0,
        )
    )
    planner.state = DrivingState.ROUNDABOUT_CIRCLE
    path = np.array(
        [[0.22, 0.0], [0.4, 0.05], [0.7, 0.2]], dtype=np.float32
    )
    result = planner._pure_pursuit(path)
    command = planner._drive(result)
    assert abs(result.lookahead_m - 0.32) < 1e-9
    assert abs(command.throttle - 0.06) < 1e-9


def test_path_loss_returns_stored_steering_toward_neutral():
    planner = MainPlanner(
        PlannerConfig(path_lost_steering_return_rate_per_sec=2.0)
    )
    planner._steering = 0.75
    assert abs(planner._return_steering_to_neutral(0.1) - 0.55) < 1e-9
    planner._steering = -0.15
    assert planner._return_steering_to_neutral(0.1) == 0.0


def test_brief_path_loss_holds_then_returns_steering():
    empty = np.empty((0, 2), dtype=np.float32)
    lane = SimpleNamespace(
        white_centerline=empty,
        yellow_centerline=empty,
        white_confidence=0.0,
        yellow_confidence=0.0,
        white_visible=False,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(),
    )
    planner = MainPlanner(
        PlannerConfig(
            path_lost_hold_frames=2,
            path_lost_hold_max_steering=1.0,
            path_lost_stop_frames=10,
            path_lost_steering_return_rate_per_sec=2.0,
        )
    )
    planner._steering = 0.6
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        first = planner.step(frame, now_sec=0.0)
        second = planner.step(frame, now_sec=0.05)
        third = planner.step(frame, now_sec=0.10)

    assert first.command.steering == 0.6
    assert second.command.steering == 0.6
    assert first.decision.endswith('path_lost_hold')
    assert second.decision.endswith('path_lost_hold')
    assert abs(third.command.steering - 0.5) < 1e-9
    assert third.decision.endswith('path_lost_return')


def test_steering_rate_limit_is_time_based():
    path = np.array([[0.1, 0.0], [1.0, 0.5]], dtype=np.float32)
    config = PlannerConfig(
        min_points=2,
        lookahead_m=0.5,
        curve_lookahead_m=0.5,
        steering_rate_limit_per_sec=1.0,
    )
    short_step = MainPlanner(config)._pure_pursuit(path, dt_sec=0.05)
    long_step = MainPlanner(config)._pure_pursuit(path, dt_sec=0.20)
    assert abs(short_step.steering) <= 0.05 + 1e-9
    assert abs(long_step.steering) <= 0.20 + 1e-9
    assert abs(long_step.steering) > abs(short_step.steering)


def test_lookahead_target_is_interpolated_to_exact_radius():
    planner = MainPlanner(
        PlannerConfig(
            min_points=3,
            lookahead_m=0.65,
            curve_lookahead_m=0.65,
            steering_rate_limit_per_sec=10.0,
        )
    )
    sparse_path = np.array(
        [[0.2, 0.0], [0.4, 0.05], [1.2, 0.2]], dtype=np.float32
    )
    result = planner._pure_pursuit(sparse_path)
    assert result.valid
    assert abs(result.target_distance - 0.65) < 1e-4


def test_curvature_reduces_lookahead_and_throttle():
    planner = MainPlanner(
        PlannerConfig(
            min_points=5,
            lookahead_m=0.8,
            curve_lookahead_m=0.45,
            lookahead_shrink_rate_m_per_sec=10.0,
            curvature_full_scale=0.5,
            cruise_throttle=0.13,
            curve_throttle=0.07,
            throttle_accel_rate_per_sec=10.0,
            throttle_decel_rate_per_sec=10.0,
            steering_rate_limit_per_sec=10.0,
        )
    )
    angles = np.linspace(0.0, 1.2, 30, dtype=np.float32)
    curved_path = np.column_stack(
        (0.2 + 0.6 * np.sin(angles), 0.6 * (1.0 - np.cos(angles)))
    ).astype(np.float32)
    result = planner._pure_pursuit(curved_path)
    command = planner._drive(result)
    assert result.valid
    assert result.curve_ratio > 0.0
    assert result.lookahead_m < planner.config.lookahead_m
    assert command.throttle < planner.config.cruise_throttle


def test_in_course_exits_on_second_debounced_branch():
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    branch = SimpleNamespace(points=path, confidence=0.9)
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path,
        white_confidence=0.9,
        yellow_confidence=0.9,
        white_visible=True,
        yellow_visible=True,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(branch,),
    )
    config = PlannerConfig(
        route_mode=RouteMode.IN,
        prefer_yellow=True,
        yellow_valid_on_frames=1,
        roundabout_entry_confirm_sec=0.0,
        roundabout_entry_require_crossing=False,
        min_points=5,
        min_lap_time_sec=1.0,
        branch_required_events=2,
        branch_select_on_frames=1,
        branch_select_off_frames=1,
        branch_on_frames=1,
        branch_off_frames=1,
        crossing_on_frames=1,
        crossing_off_frames=1,
    )
    planner = MainPlanner(config)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        planner.step(frame, now_sec=0.0)  # Enter roundabout and reset counters.
        lane.fork_active = True
        lane.branches = (branch, branch)
        planner.step(frame, now_sec=0.1)
        lane.fork_active = False
        lane.branches = (branch,)
        planner.step(frame, now_sec=0.2)
        lane.fork_active = True
        lane.branches = (branch, branch)
        output = planner.step(frame, now_sec=1.2)

    assert planner.branch_counter.events == 2
    assert output.state is DrivingState.ROUNDABOUT_EXIT
    assert output.decision == 'roundabout_exit_branch'


def test_out_fork_state_does_not_treat_single_branch_as_turn_path():
    white_path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    misleading_branch = np.array(
        [[0.2, -0.2], [0.4, -0.3], [0.6, -0.4], [0.8, -0.5], [1.0, -0.6]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=white_path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(SimpleNamespace(points=misleading_branch, confidence=0.9),),
    )
    planner = MainPlanner(PlannerConfig(min_points=5))
    planner.state = DrivingState.FORK_TURN
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        output = planner.step(frame, now_sec=0.0)

    assert output.path_source is PathSource.WHITE_CENTERLINE
    assert output.decision == 'out_fork_lane_follow'


def test_out_fork_debug_reports_sign_selected_branch():
    left_path = np.array(
        [[0.2, 0.15], [0.4, 0.2], [0.6, 0.25], [0.8, 0.3], [1.0, 0.35]],
        dtype=np.float32,
    )
    right_path = left_path.copy()
    right_path[:, 1] *= -1.0
    lane = SimpleNamespace(
        white_centerline=left_path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(
            SimpleNamespace(points=left_path, confidence=0.9),
            SimpleNamespace(points=right_path, confidence=0.9),
        ),
    )
    planner = MainPlanner(PlannerConfig(min_points=5))
    planner.desired_turn = TurnSign.RIGHT
    planner.state = DrivingState.FORK_TURN
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        return_value=TrafficResult(turn=TurnSign.RIGHT),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        output = planner.step(frame, now_sec=0.0)

    assert output.path_source is PathSource.RIGHT_BRANCH
    assert output.debug['turn_sign'] == 'right'
    assert output.debug['desired_turn'] == 'right'
    assert output.debug['selected_branch_rank'] == -1
    assert output.debug['branch_selection_reason'] == 'sign_right'


def test_sign_is_confirmed_then_locked_during_fork():
    left_path = np.array(
        [[0.2, 0.1], [0.4, 0.15], [0.6, 0.2], [0.8, 0.25], [1.0, 0.3]],
        dtype=np.float32,
    )
    right_path = left_path.copy()
    right_path[:, 1] *= -1.0
    lane = SimpleNamespace(
        white_centerline=left_path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(
            SimpleNamespace(points=left_path, confidence=0.9),
            SimpleNamespace(points=right_path, confidence=0.9),
        ),
    )
    planner = MainPlanner(
        PlannerConfig(min_points=5, sign_confirm_frames=2, branch_on_frames=1)
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        side_effect=(
            TrafficResult(turn=TurnSign.LEFT),
            TrafficResult(turn=TurnSign.LEFT),
            TrafficResult(turn=TurnSign.RIGHT),
        ),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        first = planner.step(frame, now_sec=0.0)
        lane.fork_active = True
        second = planner.step(frame, now_sec=0.1)
        third = planner.step(frame, now_sec=0.2)

    assert first.debug['desired_turn'] == 'unknown'
    assert second.state is DrivingState.FORK_TURN
    assert second.path_source is PathSource.LEFT_BRANCH
    assert second.debug['fork_locked_turn'] == 'left'
    assert third.debug['turn_sign'] == 'right'
    assert third.debug['desired_turn'] == 'left'
    assert third.debug['fork_locked_turn'] == 'left'
    assert third.path_source is PathSource.LEFT_BRANCH


def test_fork_uses_cached_branch_during_short_detection_flicker():
    left_path = np.array(
        [[0.2, 0.1], [0.4, 0.15], [0.6, 0.2], [0.8, 0.25], [1.0, 0.3]],
        dtype=np.float32,
    )
    right_path = left_path.copy()
    right_path[:, 1] *= -1.0
    branch_left = SimpleNamespace(points=left_path, confidence=0.9)
    branch_right = SimpleNamespace(points=right_path, confidence=0.9)
    lane = SimpleNamespace(
        white_centerline=right_path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(branch_left, branch_right),
    )
    planner = MainPlanner(
        PlannerConfig(min_points=5, fork_path_hold_frames=2)
    )
    planner.desired_turn = TurnSign.LEFT
    planner._lock_fork_selection()
    planner.state = DrivingState.FORK_TURN
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        planner.step(frame, now_sec=0.0)
        lane.fork_active = False
        lane.branches = (branch_right,)
        output = planner.step(frame, now_sec=0.1)

    assert output.path_source is PathSource.LEFT_BRANCH
    assert output.decision == 'out_fork_cached_branch'
    assert output.debug['selected_branch_rank'] == 0


def test_steering_release_and_reversal_use_faster_rate():
    path = np.array([[0.1, 0.0], [1.0, 0.5]], dtype=np.float32)
    planner = MainPlanner(
        PlannerConfig(
            min_points=2,
            lookahead_m=0.5,
            curve_lookahead_m=0.5,
            steering_rate_limit_per_sec=1.0,
            steering_release_rate_limit_per_sec=10.0,
        )
    )
    planner._steering = 0.8

    result = planner._pure_pursuit(path, dt_sec=0.1)

    assert result.raw_steering < 0.0
    assert abs(result.steering - (-0.2)) < 1e-9


def test_saturated_steering_is_not_held_when_path_is_lost():
    empty = np.empty((0, 2), dtype=np.float32)
    lane = SimpleNamespace(
        white_centerline=empty,
        yellow_centerline=empty,
        white_confidence=0.0,
        yellow_confidence=0.0,
        white_visible=False,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(),
    )
    planner = MainPlanner(
        PlannerConfig(
            path_lost_hold_frames=2,
            path_lost_hold_max_steering=0.35,
            path_lost_steering_return_rate_per_sec=8.0,
        )
    )
    planner._steering = 1.0
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        output = planner.step(frame, now_sec=0.0)

    assert abs(output.command.steering - 0.2) < 1e-9
    assert output.decision.endswith('path_lost_return')


def test_curvature_filter_reacts_faster_to_curve_than_to_straight():
    planner = MainPlanner(
        PlannerConfig(
            curvature_full_scale=1.0,
            curvature_rise_tau_sec=0.05,
            curvature_fall_tau_sec=0.50,
        )
    )
    _, rising = planner._filtered_curve_ratio(1.0, True, 0.10)
    _, falling = planner._filtered_curve_ratio(0.0, True, 0.10)

    assert rising > 0.8
    assert falling > 0.6


def test_invalid_curvature_holds_then_selects_conservative_profile():
    planner = MainPlanner(
        PlannerConfig(
            curvature_invalid_hold_sec=0.40,
            curvature_rise_tau_sec=0.05,
        )
    )
    planner._curve_ratio = 0.4

    _, held = planner._filtered_curve_ratio(0.0, False, 0.20)
    _, conservative = planner._filtered_curve_ratio(0.0, False, 0.25)

    assert abs(held - 0.4) < 1e-9
    assert conservative > held


def test_throttle_accelerates_slowly_and_decelerates_quickly():
    planner = MainPlanner(
        PlannerConfig(
            default_throttle=0.10,
            throttle_accel_rate_per_sec=0.20,
            throttle_decel_rate_per_sec=0.80,
        )
    )

    assert abs(planner._update_throttle(0.30, 0.10) - 0.12) < 1e-9
    assert abs(planner._update_throttle(0.05, 0.10) - 0.05) < 1e-9


def test_lookahead_growth_is_time_based_and_speed_gated():
    planner = MainPlanner(
        PlannerConfig(
            lookahead_m=1.0,
            curve_lookahead_m=0.5,
            lookahead_grow_rate_m_per_sec=0.5,
        )
    )
    planner._lookahead_m = 0.5

    desired, ratio = planner._adaptive_lookahead(1.0, 0.10)

    assert desired == 1.0
    assert ratio == 1.0
    assert abs(planner._lookahead_m - 0.55) < 1e-9


def test_roundabout_lookahead_varies_between_curve_and_straight_values():
    planner = MainPlanner(
        PlannerConfig(
            lookahead_m=1.0,
            curve_lookahead_m=0.55,
            curvature_full_scale=0.8,
            roundabout_straight_lookahead_m=0.80,
            roundabout_curve_lookahead_m=0.55,
            lookahead_shrink_rate_m_per_sec=10.0,
            lookahead_grow_rate_m_per_sec=10.0,
        )
    )
    planner._lookahead_m = 0.55

    straight_desired, straight_ratio = planner._adaptive_lookahead(
        1.0,
        0.1,
        straight_lookahead_m=planner.config.roundabout_straight_lookahead_m,
        curve_lookahead_m=planner.config.roundabout_curve_lookahead_m,
    )
    curve_desired, curve_ratio = planner._adaptive_lookahead(
        0.0,
        0.1,
        straight_lookahead_m=planner.config.roundabout_straight_lookahead_m,
        curve_lookahead_m=planner.config.roundabout_curve_lookahead_m,
    )

    assert straight_ratio == 1.0
    assert straight_desired == 0.80
    assert curve_ratio == 0.0
    assert curve_desired == 0.55


def test_previous_recovery_demand_prevents_lookahead_growth():
    planner = MainPlanner(
        PlannerConfig(
            min_points=5,
            lookahead_m=1.0,
            curve_lookahead_m=0.5,
            lookahead_shrink_rate_m_per_sec=10.0,
            lookahead_grow_rate_m_per_sec=10.0,
            cruise_throttle=0.28,
            curve_throttle=0.10,
        )
    )
    planner._lookahead_m = 1.0
    planner._throttle = 0.28
    planner._control_demand_ratio = 1.0
    straight_path = np.array(
        [[0.25, 0.0], [0.45, 0.0], [0.65, 0.0], [0.85, 0.0], [1.05, 0.0]],
        dtype=np.float32,
    )

    result = planner._pure_pursuit(straight_path, 0.10)

    assert result.lookahead_m == 0.5


def test_blended_branch_path_keeps_near_lane_then_joins_branch():
    planner = MainPlanner(PlannerConfig(branch_blend_distance_m=0.35, min_points=3))
    lane_path = np.array(
        [[0.20, 0.00], [0.40, 0.00], [0.60, 0.00], [0.90, 0.00]],
        dtype=np.float32,
    )
    # Branch starts offset laterally and rejoins further out.
    branch_path = np.array(
        [[0.30, 0.20], [0.50, 0.20], [0.70, 0.10], [1.00, 0.00]],
        dtype=np.float32,
    )
    blended = planner._blended_branch_path(lane_path, branch_path)
    assert blended.shape[0] >= 2
    # Near overlap should be pulled toward the lane center (y≈0).
    near = blended[blended[:, 0] <= 0.35]
    assert near.shape[0] >= 1
    assert abs(float(near[0, 1])) < abs(float(branch_path[0, 1]))


def test_roundabout_entry_timer_starts_on_context_and_resets_on_dropout():
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path.copy(),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=True,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(),
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            prefer_yellow=True,
            yellow_valid_on_frames=1,
            roundabout_entry_confirm_sec=0.5,
            roundabout_entry_require_crossing=True,
            min_points=5,
            min_lap_time_sec=100.0,
        )
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        before_context = planner.step(frame, now_sec=100.0)
        lane.yellow_confidence = 0.9
        candidate_started = planner.step(frame, now_sec=101.0)
        still_waiting = planner.step(frame, now_sec=101.4)

        lane.yellow_confidence = 0.0
        planner.step(frame, now_sec=101.45)
        lane.yellow_confidence = 0.9
        restarted = planner.step(frame, now_sec=102.0)
        confirmed_without_crossing = planner.step(frame, now_sec=102.6)
        lane.yellow_crossing_line = True
        entered = planner.step(frame, now_sec=102.7)

    assert before_context.state is DrivingState.NORMAL
    assert before_context.path_source is PathSource.WHITE_CENTERLINE
    assert candidate_started.state is DrivingState.NORMAL
    assert still_waiting.state is DrivingState.NORMAL
    assert restarted.state is DrivingState.NORMAL
    assert restarted.debug['roundabout_entry_candidate_elapsed_sec'] == 0.0
    assert confirmed_without_crossing.state is DrivingState.NORMAL
    assert confirmed_without_crossing.debug['roundabout_entry_context_ready'] is False
    assert entered.state is DrivingState.ROUNDABOUT_CIRCLE
    assert entered.path_source is PathSource.YELLOW_CENTERLINE
    assert entered.debug['roundabout_entry_candidate_elapsed_sec'] == 0.7


def test_roundabout_entry_can_use_stable_yellow_without_crossing_detector():
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path.copy(),
        white_confidence=0.9,
        yellow_confidence=0.9,
        white_visible=True,
        yellow_visible=True,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(),
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            prefer_yellow=True,
            yellow_valid_on_frames=1,
            roundabout_entry_confirm_sec=0.5,
            roundabout_entry_require_crossing=False,
            min_points=5,
            min_lap_time_sec=100.0,
        )
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        waiting = planner.step(frame, now_sec=10.0)
        entered = planner.step(frame, now_sec=10.6)

    assert waiting.state is DrivingState.NORMAL
    assert waiting.path_source is PathSource.WHITE_CENTERLINE
    assert entered.state is DrivingState.ROUNDABOUT_CIRCLE
    assert entered.path_source is PathSource.YELLOW_CENTERLINE
    assert entered.debug['roundabout_entry_context_ready'] is True


def test_legacy_lookahead_rates_convert_via_nominal_dt():
    from inference.pipeline import load_planner_config
    import tempfile
    yaml_text = """
pure_pursuit:
  nominal_control_dt_sec: 0.10
  lookahead_shrink_rate_m: 0.15
  lookahead_grow_rate_m: 0.07
"""
    with tempfile.TemporaryDirectory() as directory:
        cfg = Path(directory) / 'cfg.yaml'
        cfg.write_text(yaml_text)
        loaded = load_planner_config(cfg)
    assert abs(loaded.lookahead_shrink_rate_m_per_sec - 1.50) < 1e-9
    assert abs(loaded.lookahead_grow_rate_m_per_sec - 0.70) < 1e-9


def test_per_sec_lookahead_rates_win_over_legacy_keys():
    from inference.pipeline import load_planner_config
    import tempfile
    yaml_text = """
pure_pursuit:
  nominal_control_dt_sec: 0.10
  lookahead_shrink_rate_m: 0.15
  lookahead_grow_rate_m: 0.07
  lookahead_shrink_rate_m_per_sec: 2.0
  lookahead_grow_rate_m_per_sec: 0.25
"""
    with tempfile.TemporaryDirectory() as directory:
        cfg = Path(directory) / 'cfg.yaml'
        cfg.write_text(yaml_text)
        loaded = load_planner_config(cfg)
    assert abs(loaded.lookahead_shrink_rate_m_per_sec - 2.0) < 1e-9
    assert abs(loaded.lookahead_grow_rate_m_per_sec - 0.25) < 1e-9
