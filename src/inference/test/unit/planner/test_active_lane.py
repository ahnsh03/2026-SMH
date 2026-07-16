"""Unit tests for active-lane (selected fork / merge spur) policy."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from inference.modules.active_lane import (
    apply_active_lane_policy,
    collapse_to_selected_lane,
    parse_selected_rank_from_planner_debug,
    suppress_merge_spur_branches,
)
from inference.modules.lane_detection import (
    ForkLanePair,
    LaneDebugFrame,
    LaneDetections,
    RoadBranch,
)
from inference.pipeline import MainPlanner, PlannerConfig
from inference.types import (
    ArucoResult,
    DrivingState,
    PathSource,
    RouteMode,
    TrafficResult,
    TurnSign,
)


def _branch(rank: int, ys: float, *, near: bool = True) -> RoadBranch:
    xs = np.linspace(0.25 if near else 0.90, 1.2, 8, dtype=np.float32)
    ys_arr = np.full_like(xs, ys)
    points = np.column_stack((xs, ys_arr, np.zeros_like(xs))).astype(np.float32)
    return RoadBranch(lateral_rank=rank, confidence=0.9, points=points)


def test_suppress_merge_spur_keeps_near_ego_only():
    ego = _branch(0, 0.0, near=True)
    spur = _branch(1, -0.35, near=False)
    kept = suppress_merge_spur_branches((ego, spur))
    assert len(kept) == 1
    assert int(kept[0].lateral_rank) == 0


def test_collapse_drops_opposite_fork_and_clears_fork_active():
    h = 40
    left_center = np.full(h, 80.0, dtype=np.float32)
    right_center = np.full(h, 140.0, dtype=np.float32)
    nan = np.full(h, np.nan, dtype=np.float32)
    left = ForkLanePair(
        lateral_rank=0,
        outer_u=left_center - 20,
        inner_u=left_center + 20,
        center_u=left_center,
        confidence=0.8,
    )
    right = ForkLanePair(
        lateral_rank=1,
        outer_u=right_center + 20,
        inner_u=right_center - 20,
        center_u=right_center,
        confidence=0.8,
    )
    det = LaneDetections(
        fork_active=True,
        branches=(_branch(0, 0.15), _branch(1, -0.15)),
        white_confidence=0.5,
        white_visible=True,
    )
    dbg = LaneDebugFrame(
        fork_active=True,
        fork_lane_pairs=(left, right),
        road_branches=det.branches,
        fork_split_source='road_split',
    )
    out_det, out_dbg = collapse_to_selected_lane(det, dbg, 0)
    assert out_det.fork_active is False
    assert out_dbg.fork_active is False
    assert out_det.lane_policy == 'locked'
    assert out_det.active_branch_rank == 0
    assert len(out_dbg.fork_lane_pairs) == 1
    assert int(out_dbg.fork_lane_pairs[0].lateral_rank) == 0
    assert all(int(b.lateral_rank) == 0 for b in out_det.branches)
    assert out_det.white_centerline.shape[0] >= 2


def test_apply_explore_without_rank_sets_policy():
    det = LaneDetections(fork_active=True, branches=(_branch(0, 0.1), _branch(1, -0.1)))
    dbg = LaneDebugFrame(fork_active=True, road_branches=det.branches)
    out_det, out_dbg = apply_active_lane_policy(det, dbg, None)
    assert out_dbg.lane_policy in ('explore', 'ego_only')
    assert out_det.active_branch_rank is None


def test_parse_planner_debug_rank():
    assert parse_selected_rank_from_planner_debug('choice=sign_left rank=0 path=') == 0
    assert parse_selected_rank_from_planner_debug('choice=sign_right rank=1 path=') == 1
    assert parse_selected_rank_from_planner_debug('choice=none rank=- path=') is None


def test_collapse_out_prefers_white_even_if_yellow_conf_higher():
    """Out lock must project onto white, never yellow paint pull."""
    h = 40
    left_center = np.full(h, 80.0, dtype=np.float32)
    left = ForkLanePair(
        lateral_rank=0,
        outer_u=left_center - 20,
        inner_u=left_center + 20,
        center_u=left_center,
        confidence=0.8,
    )
    det = LaneDetections(
        fork_active=True,
        branches=(_branch(0, 0.15), _branch(1, -0.15)),
        white_confidence=0.2,
        yellow_confidence=0.95,
        white_visible=True,
        yellow_visible=True,
    )
    dbg = LaneDebugFrame(
        fork_active=True,
        fork_lane_pairs=(left,),
        road_branches=det.branches,
        fork_split_source='yellow_alt_marks',  # would have fooled legacy
        prefer_yellow=False,
    )
    out_det, _ = collapse_to_selected_lane(det, dbg, 0)
    assert out_det.white_centerline.shape[0] >= 2
    # Out locks onto white channel (conf bumped); yellow conf stays raw high.
    assert out_det.white_confidence == 0.8
    assert out_det.yellow_confidence == 0.95


def test_fork_source_out_rejects_yellow():
    from inference.modules.lane_detection import fork_source_allowed_for_course

    assert fork_source_allowed_for_course(
        'yellow_alt',
        prefer_yellow=False,
        yellow_is_detected=True,
        ego_road_color=None,
    ) is False
    assert fork_source_allowed_for_course(
        'road_split',
        prefer_yellow=False,
        yellow_is_detected=True,
        ego_road_color='white',
    ) is True
    assert fork_source_allowed_for_course(
        'yellow_alt',
        prefer_yellow=True,
        yellow_is_detected=True,
        ego_road_color='yellow',
    ) is True


def test_color_path_out_never_returns_yellow():
    planner = MainPlanner(
        PlannerConfig(route_mode=RouteMode.OUT, prefer_yellow=False, min_points=5)
    )
    yellow = np.array(
        [[0.3, 0.4], [0.5, 0.4], [0.7, 0.4], [0.9, 0.4], [1.1, 0.4]],
        dtype=np.float32,
    )
    white = np.array(
        [[0.3, 0.0], [0.5, 0.0], [0.7, 0.0], [0.9, 0.0], [1.1, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        yellow_centerline=yellow,
        white_centerline=white,
        yellow_confidence=0.99,
        white_confidence=0.5,
    )
    path, source, _ = planner._color_path(lane)
    assert source is PathSource.WHITE_CENTERLINE
    assert np.allclose(path[:, 1], 0.0)


def test_fork_ego_follow_uses_strict_rank_branch_only():
    left_path = np.array(
        [[0.2, 0.15], [0.4, 0.2], [0.6, 0.25], [0.8, 0.3], [1.0, 0.35]],
        dtype=np.float32,
    )
    right_path = left_path.copy()
    right_path[:, 1] *= -1.0
    # Locked / collapsed publish: one branch with lateral_rank, fork_active off.
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
            SimpleNamespace(points=left_path, confidence=0.9, lateral_rank=0),
        ),
        lane_policy='locked',
        active_branch_rank=0,
    )
    planner = MainPlanner(
        PlannerConfig(
            min_points=5,
            require_green_to_start=False,
            stop_on_red=False,
        )
    )
    planner.force_fork_choice(TurnSign.LEFT, state=DrivingState.FORK_TURN)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    debug = LaneDebugFrame()

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, debug),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        output = planner.step(frame, now_sec=0.0)

    assert output.path_source is PathSource.LEFT_BRANCH
    assert output.decision == 'out_fork_ego_follow_rank0'
    assert output.debug['selected_branch_rank'] == 0
    assert output.debug['lane_policy'] == 'locked'
