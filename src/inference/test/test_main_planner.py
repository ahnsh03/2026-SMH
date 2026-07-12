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
    RouteMode,
    TrafficResult,
)


def test_event_counter_debounces_and_rearms():
    counter = RisingEventCounter(on_frames=2, off_frames=2)
    events = [counter.update(value) for value in (True, True, True, False, False, True, True)]
    assert events == [False, True, False, False, False, False, True]
    assert counter.events == 2


def test_pure_pursuit_steering_sign():
    planner = MainPlanner(
        PlannerConfig(min_points=2, lookahead_m=0.5, steering_rate_limit=1.0)
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
    assert output.decision == 'roundabout_exit_branch'
