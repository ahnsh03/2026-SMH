"""Shared data types for perception/planning modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TrafficSignal(str, Enum):
    UNKNOWN = 'unknown'
    GREEN = 'green'
    RED = 'red'


class TurnSign(str, Enum):
    UNKNOWN = 'unknown'
    LEFT = 'left'
    RIGHT = 'right'


class RouteMode(str, Enum):
    """Course selected before launch."""

    OUT = 'out'
    IN = 'in'


class DrivingState(str, Enum):
    """State owned exclusively by the main planner."""

    WAIT_GREEN = 'wait_green'
    NORMAL = 'normal'
    FORK_TURN = 'fork_turn'
    ROUNDABOUT_CIRCLE = 'roundabout_circle'
    ROUNDABOUT_EXIT_READY = 'roundabout_exit_ready'
    ROUNDABOUT_EXIT = 'roundabout_exit'
    FINISHED = 'finished'


class PathSource(str, Enum):
    """Path currently used to produce steering."""

    NONE = 'none'
    WHITE_CENTERLINE = 'white_centerline'
    YELLOW_CENTERLINE = 'yellow_centerline'
    LEFT_BRANCH = 'left_branch'
    RIGHT_BRANCH = 'right_branch'
    HOLD_PREVIOUS = 'hold_previous'
    STOP = 'stop'


@dataclass(frozen=True)
class LaneMarking:
    """Lane marking polyline in base_link (x forward, y left), meters.

    Compatible with Won Tae perception / LaneMarking.msg conventions.
    ``points`` is Nx2 or Nx3 float32; planner uses columns 0:2 (x, y).
    """

    COLOR_UNKNOWN = 0
    COLOR_WHITE = 1
    COLOR_YELLOW = 2

    SIDE_UNKNOWN = 0
    SIDE_LEFT = 1
    SIDE_RIGHT = 2
    SIDE_CENTER = 3

    id: int = 0
    color: int = COLOR_UNKNOWN
    side_hint: int = SIDE_UNKNOWN
    confidence: float = 0.0
    length: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )

    def xy(self) -> np.ndarray:
        """Return Nx2 (x, y) view/copy."""
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float32)
        return pts[:, :2]


@dataclass(frozen=True)
class RoadBranch:
    """Fork / path candidate from perception (lane_msgs/RoadBranch)."""

    lateral_rank: int = 0
    """0 = leftmost branch."""
    confidence: float = 0.0
    width: float = 0.0
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )

    def xy(self) -> np.ndarray:
        pts = np.asarray(self.points, dtype=np.float32)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float32)
        return pts[:, :2]


@dataclass(frozen=True)
class LaneDetections:
    """Perception-only lane output (no steering).

Aligned with ``lane_msgs/LaneDetections`` so adapters can round-trip.
    Planner follows one color at a time (``follow_color``); no white↔yellow auto switch.
    Fork/branches are for mission planners (e.g. yangseojun MainPlanner).
    """

    lanes: tuple[LaneMarking, ...] = ()
    white_visible: bool = False
    yellow_visible: bool = False
    left_visible: bool = False
    right_visible: bool = False
    white_confidence: float = 0.0
    yellow_confidence: float = 0.0
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    white_centerline: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    yellow_centerline: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    yellow_crossing_line: bool = False
    fork_active: bool = False
    branches: tuple[RoadBranch, ...] = ()
    drivable_area: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    meters_per_pixel: float = 0.0
    x_forward_max: float = 0.0

    def marking(
        self,
        *,
        color: int,
        side: int,
    ) -> LaneMarking | None:
        for lane in self.lanes:
            if lane.color == color and lane.side_hint == side and lane.xy().shape[0] > 0:
                return lane
        return None

    def white_left(self) -> LaneMarking | None:
        return self.marking(color=LaneMarking.COLOR_WHITE, side=LaneMarking.SIDE_LEFT)

    def white_right(self) -> LaneMarking | None:
        return self.marking(color=LaneMarking.COLOR_WHITE, side=LaneMarking.SIDE_RIGHT)

    def yellow_left(self) -> LaneMarking | None:
        return self.marking(color=LaneMarking.COLOR_YELLOW, side=LaneMarking.SIDE_LEFT)

    def yellow_right(self) -> LaneMarking | None:
        return self.marking(color=LaneMarking.COLOR_YELLOW, side=LaneMarking.SIDE_RIGHT)

    def pair_for_color(self, color: int) -> tuple[LaneMarking | None, LaneMarking | None]:
        """Left/right markings for a single follow color (white or yellow)."""
        return (
            self.marking(color=color, side=LaneMarking.SIDE_LEFT),
            self.marking(color=color, side=LaneMarking.SIDE_RIGHT),
        )


@dataclass(frozen=True)
class LaneResult:
    """Planner output consumed by pipeline.fuse_control — 담당: 안승현(조향)."""

    steering_offset: float = 0.0
    """-1.0 (left) ~ +1.0 (right), 0 = center. D-Racer: +steering = right."""
    confidence: float = 0.0
    """0.0 ~ 1.0"""
    throttle_scale: float = 1.0
    """0.0 ~ 1.0 multiplier on cruise (e.g. |steer| slowdown)."""


@dataclass(frozen=True)
class TrafficResult:
    """Traffic light & fork sign output — 담당: 장원정."""

    signal: TrafficSignal = TrafficSignal.UNKNOWN
    turn: TurnSign = TurnSign.UNKNOWN


@dataclass(frozen=True)
class ArucoResult:
    """ArUco marker output — 담당: 안승현, 박성준."""

    detected: bool = False
    marker_id: int | None = None
    should_stop: bool = False


@dataclass
class PipelineContext:
    """Legacy control context retained for small unit tests."""

    lane: LaneResult = field(default_factory=LaneResult)
    traffic: TrafficResult = field(default_factory=TrafficResult)
    aruco: ArucoResult = field(default_factory=ArucoResult)


@dataclass(frozen=True)
class ControlCommand:
    """Final steering/throttle sent to /control."""

    steering: float
    throttle: float


@dataclass(frozen=True)
class PlannerOutput:
    """One synchronized perception/planning result for a camera frame."""

    command: ControlCommand
    lane: Any
    traffic: TrafficResult
    aruco: ArucoResult
    state: DrivingState
    path_source: PathSource
    decision: str
    debug: dict[str, Any] = field(default_factory=dict)
