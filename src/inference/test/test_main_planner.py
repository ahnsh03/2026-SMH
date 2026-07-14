"""Unit tests for MainPlanner primitives (no ROS required)."""

from __future__ import annotations

import sys
from pathlib import Path
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
            roundabout_throttle=0.06,
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
            lookahead_shrink_rate_m=1.0,
            curvature_full_scale=0.5,
            cruise_throttle=0.13,
            curve_throttle=0.07,
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
        min_points=5,
        min_lap_time_sec=1.0,
        branch_required_events=2,
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
    assert output.decision == 'roundabout_exit_rank0'


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
    assert output.decision == 'out_fork_color_resume'


def test_forced_turn_right_stays_in_roundabout_circle():
    """IN + forced RIGHT: do not arm exit even when branch events fire."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    branch = SimpleNamespace(
        points=path, confidence=0.9, lateral_rank=0
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path,
        white_confidence=0.9,
        yellow_confidence=0.9,
        white_visible=True,
        yellow_visible=True,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(branch, SimpleNamespace(points=path, confidence=0.9, lateral_rank=1)),
        lane_policy='explore',
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            prefer_yellow=True,
            yellow_valid_on_frames=1,
            min_points=5,
            min_lap_time_sec=0.5,
            branch_required_events=1,
            branch_on_frames=1,
            branch_off_frames=1,
        )
    )
    planner.apply_forced_turn(TurnSign.RIGHT)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        planner.step(frame, now_sec=0.0)  # enter circle
        assert planner.state is DrivingState.ROUNDABOUT_CIRCLE
        out = planner.step(frame, now_sec=1.0)

    assert out.state is DrivingState.ROUNDABOUT_CIRCLE
    assert out.debug['forced_turn'] == 'right'
    assert planner._fork_selection_reason == 'forced_right'


def test_forced_turn_left_arms_roundabout_exit():
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    left = SimpleNamespace(points=path, confidence=0.9, lateral_rank=0)
    right = SimpleNamespace(
        points=path * np.array([1.0, -1.0], dtype=np.float32),
        confidence=0.9,
        lateral_rank=1,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path,
        white_confidence=0.9,
        yellow_confidence=0.9,
        white_visible=True,
        yellow_visible=True,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(left, right),
        lane_policy='explore',
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            prefer_yellow=True,
            yellow_valid_on_frames=1,
            min_points=5,
            min_lap_time_sec=0.5,
            branch_required_events=1,
            branch_on_frames=1,
            branch_off_frames=1,
        )
    )
    planner.apply_forced_turn(TurnSign.LEFT)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect', return_value=lane
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        planner.step(frame, now_sec=0.0)
        out = planner.step(frame, now_sec=1.0)

    assert out.state in (
        DrivingState.ROUNDABOUT_EXIT_READY,
        DrivingState.ROUNDABOUT_EXIT,
    )
    assert out.debug['forced_turn'] == 'left'
    assert planner._fork_selected_rank == 0


def test_force_fork_choice_left_rank0_right_rank1():
    planner = MainPlanner(PlannerConfig(min_points=5))
    planner.force_fork_choice(TurnSign.LEFT, state=DrivingState.FORK_TURN)
    assert planner._fork_selected_rank == 0
    assert planner.state is DrivingState.FORK_TURN
    planner.force_fork_choice(TurnSign.RIGHT, state=DrivingState.ROUNDABOUT_EXIT)
    assert planner._fork_selected_rank == 1
    assert planner.state is DrivingState.ROUNDABOUT_EXIT


def test_ranked_branch_matches_lateral_rank():
    left = SimpleNamespace(points=np.zeros((5, 2), np.float32), confidence=1.0, lateral_rank=0)
    right = SimpleNamespace(points=np.ones((5, 2), np.float32), confidence=1.0, lateral_rank=1)
    # Deliberately reverse list order — lateral_rank must win.
    lane = SimpleNamespace(branches=(right, left))
    assert MainPlanner._ranked_branch(lane, 0) is left
    assert MainPlanner._ranked_branch(lane, 1) is right

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
    assert output.debug['selected_branch_rank'] == 1
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
