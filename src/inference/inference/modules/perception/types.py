"""Shared perception wire types (blob + legacy). Avoid importing the 7k legacy module."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LaneBoundary:
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    detected: bool = False
    confidence: float = 0.0


@dataclass(frozen=True)
class LaneMarking:
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
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )


@dataclass(frozen=True)
class RoadBranch:
    lateral_rank: int = 0
    confidence: float = 0.0
    width: float = 0.0
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )


@dataclass(frozen=True)
class ForkLanePair:
    lateral_rank: int
    outer_u: np.ndarray
    inner_u: np.ndarray
    center_u: np.ndarray
    outer_missing: bool = False
    inner_missing: bool = False
    confidence: float = 0.0


@dataclass(frozen=True)
class LaneDetections:
    header: object | None = None
    lanes: tuple[LaneMarking, ...] = ()
    white_visible: bool = False
    yellow_visible: bool = False
    left_visible: bool = False
    right_visible: bool = False
    white_confidence: float = 0.0
    yellow_confidence: float = 0.0
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    drivable_area: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_centerline: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    yellow_centerline: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    yellow_crossing_line: bool = False
    fork_active: bool = False
    branches: tuple[RoadBranch, ...] = ()
    active_branch_rank: int | None = None
    lane_policy: str = 'explore'
    steering_offset: float = 0.0
    confidence: float = 0.0
    # BEV grid → base_link (same contract as lane_msgs/LaneDetections)
    meters_per_pixel: float = 0.0
    x_forward_max: float = 0.0
    # Mask gates for planner judgment (fork/judgment.py) — not L/R geometry.
    out_fork_capture: bool = False
    in_circle_fork_moment: bool = False


@dataclass
class LaneDebugFrame:
    bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )
    white_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    yellow_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    red_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    black_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    road_clean: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    road_raw: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    yellow_dash_points_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    yellow_connected_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_dash_points_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_dash_connected_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    crossing_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_crossing_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_left: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    white_right: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    yellow_left: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    yellow_right: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    road_cells: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    road_branches: tuple = ()
    ego_road_color: str | None = None
    fork_active: bool = False
    yellow_crossing_line: bool = False
    white_crossing_line: bool = False
    red_coverage: float = 0.0
    red_pixel_count: int = 0
    fork_lane_pairs: tuple = ()
    fork_mark_tracks: tuple = ()
    fork_split_source: str = ''
    prefer_yellow: bool | None = None
    active_branch_rank: int | None = None
    lane_policy: str = 'explore'
