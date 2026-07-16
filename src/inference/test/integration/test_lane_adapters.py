"""Smoke tests for lane adapters (no ROS msgs required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


from inference.lane_adapters import (  # noqa: E402
    detections_from_module,
    detections_from_msg,
)
from inference.modules.lane_planner import LanePlanner, mock_white_lane  # noqa: E402
from inference.types import LaneMarking  # noqa: E402


def test_detections_from_module_roundtrip_fields():
    pts = np.array([[0.5, 0.2], [1.0, 0.2]], dtype=np.float32)
    marking = SimpleNamespace(
        id=1,
        color=LaneMarking.COLOR_WHITE,
        side_hint=LaneMarking.SIDE_LEFT,
        confidence=0.9,
        length=0.5,
        heading=0.0,
        curvature=0.0,
        points=pts,
    )
    branch = SimpleNamespace(
        lateral_rank=0,
        confidence=0.8,
        width=0.35,
        points=np.array([[0.4, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
    )
    dc = SimpleNamespace(
        lanes=(marking,),
        white_visible=True,
        yellow_visible=False,
        left_visible=True,
        right_visible=False,
        white_confidence=0.9,
        yellow_confidence=0.0,
        left_confidence=0.9,
        right_confidence=0.0,
        white_centerline=pts,
        yellow_centerline=(),
        yellow_crossing_line=False,
        fork_active=True,
        branches=(branch,),
        drivable_area=np.zeros((4, 4), dtype=np.uint8),
    )
    dets = detections_from_module(dc, meters_per_pixel=0.004, x_forward_max=1.5)
    assert dets.white_visible and dets.fork_active
    assert len(dets.lanes) == 1
    assert dets.lanes[0].side_hint == LaneMarking.SIDE_LEFT
    assert len(dets.branches) == 1
    assert dets.meters_per_pixel == 0.004
    assert dets.x_forward_max == 1.5
    assert dets.drivable_area.shape == (4, 4)


def test_detections_from_msg_minimal():
    point = SimpleNamespace(x=0.8, y=0.1, z=0.0)
    marking = SimpleNamespace(
        id=2,
        color=LaneMarking.COLOR_WHITE,
        side_hint=LaneMarking.SIDE_RIGHT,
        confidence=0.7,
        length=0.4,
        heading=0.0,
        curvature=0.0,
        points=[point],
    )
    branch = SimpleNamespace(
        branch_id=1,
        confidence=0.5,
        width=0.3,
        centerline=[point],
    )
    msg = SimpleNamespace(
        lanes=[marking],
        white_visible=True,
        yellow_visible=False,
        left_visible=False,
        right_visible=True,
        white_confidence=0.7,
        yellow_confidence=0.0,
        left_confidence=0.0,
        right_confidence=0.7,
        white_centerline=[point],
        yellow_centerline=[],
        yellow_crossing_line=False,
        fork_active=True,
        branches=[branch],
        drivable_area=SimpleNamespace(height=0, width=0, data=b''),
        meters_per_pixel=0.004,
        x_forward_max=1.5,
    )
    dets = detections_from_msg(msg)
    assert dets.right_visible and dets.fork_active
    assert dets.branches[0].lateral_rank == 1
    assert dets.white_centerline.shape[0] == 1
    assert abs(dets.white_centerline[0, 1] - 0.1) < 1e-6


def test_adapter_output_feeds_planner():
    mock = mock_white_lane(y_left=0.175, y_right=-0.175)
    dets = detections_from_module(
        SimpleNamespace(
            lanes=mock.lanes,
            white_visible=True,
            yellow_visible=False,
            left_visible=True,
            right_visible=True,
            white_confidence=1.0,
            yellow_confidence=0.0,
            left_confidence=1.0,
            right_confidence=1.0,
            white_centerline=(),
            yellow_centerline=(),
            yellow_crossing_line=False,
            fork_active=False,
            branches=(),
            drivable_area=np.empty((0, 0), dtype=np.uint8),
        )
    )
    result = LanePlanner().step(dets)
    assert result.confidence > 0.1
