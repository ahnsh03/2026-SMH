"""Shared data types for perception/planning modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TrafficSignal(str, Enum):
    UNKNOWN = 'unknown'
    GREEN = 'green'
    RED = 'red'


class TurnSign(str, Enum):
    UNKNOWN = 'unknown'
    LEFT = 'left'
    RIGHT = 'right'


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
    Phase-2 ``lane_planner`` uses white L/R markings; fork/branches are for
    mission planners (e.g. yangseojun MainPlanner).
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


@dataclass(frozen=True)
class RoundaboutResult:
    """Roundabout planning output — 담당: 양서준."""

    active: bool = False
    """True when roundabout logic should override lane following."""
    steering: float = 0.0
    throttle: float = 0.0


@dataclass
class PipelineContext:
    """Aggregated module outputs passed to fusion logic."""

    lane: LaneResult = field(default_factory=LaneResult)
    traffic: TrafficResult = field(default_factory=TrafficResult)
    aruco: ArucoResult = field(default_factory=ArucoResult)
    roundabout: RoundaboutResult = field(default_factory=RoundaboutResult)


@dataclass(frozen=True)
class ControlCommand:
    """Final steering/throttle sent to /control."""

    steering: float
    throttle: float
