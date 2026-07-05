"""Shared data types for perception/planning modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TrafficSignal(str, Enum):
    UNKNOWN = 'unknown'
    GREEN = 'green'
    RED = 'red'


class TurnSign(str, Enum):
    UNKNOWN = 'unknown'
    LEFT = 'left'
    RIGHT = 'right'


@dataclass(frozen=True)
class LaneResult:
    """Lane detection output — 담당: 장원태."""

    steering_offset: float = 0.0
    """-1.0 (left) ~ +1.0 (right), 0 = center."""
    confidence: float = 0.0
    """0.0 ~ 1.0"""


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
