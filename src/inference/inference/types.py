"""인지 파이프라인 내부에서 주고받는 순수 파이썬 자료형.

ROS 메시지(lane_msgs)와 분리해 두어, 비전 알고리즘이 ROS에 의존하지 않고
단위 테스트/오프라인 튜닝이 가능하도록 한다. perception_node 가 마지막에
이 자료형을 lane_msgs/LaneDetections 로 변환해 발행한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# base_link 좌표(x 전방, y 왼쪽) polyline 점.
Point = Tuple[float, float]


@dataclass
class LaneMarking:
    color: int = 0          # 0 unknown / 1 white / 2 yellow
    side_hint: int = 0      # 0 unknown / 1 left / 2 right / 3 center
    confidence: float = 0.0
    length: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    points: List[Point] = field(default_factory=list)


@dataclass
class RoadBranch:
    branch_id: int = 0
    confidence: float = 0.0
    width: float = 0.0
    centerline: List[Point] = field(default_factory=list)


@dataclass
class LaneResult:
    """한 프레임 인지 결과 전체 (LaneDetections 로 1:1 매핑)."""
    lanes: List[LaneMarking] = field(default_factory=list)

    white_visible: bool = False
    yellow_visible: bool = False
    left_visible: bool = False
    right_visible: bool = False
    white_confidence: float = 0.0
    yellow_confidence: float = 0.0
    left_confidence: float = 0.0
    right_confidence: float = 0.0

    white_centerline: List[Point] = field(default_factory=list)
    yellow_centerline: List[Point] = field(default_factory=list)

    yellow_crossing_line: bool = False

    fork_active: bool = False
    branches: List[RoadBranch] = field(default_factory=list)

    # 주행가능영역 그리드(mono8 0/255)와 스케일
    drivable_area: Optional["object"] = None   # numpy.ndarray (선택)
    meters_per_pixel: float = 0.0
    x_forward_max: float = 0.0


@dataclass
class SignResult:
    """정지 표지(ArUco) / 방향 표지 인지 결과. 제어 판단의 보조 입력."""
    stop_detected: bool = False
    stop_distance: float = 0.0
    direction: int = 0      # 0 unknown / 1 left / 2 right
    direction_confidence: float = 0.0
