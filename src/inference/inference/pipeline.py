"""인지 파이프라인 오케스트레이터.

프레임 하나를 받아 각 비전 모듈을 순차 실행하고, 결과를 LaneResult /
SignResult 로 합쳐 반환한다. ROS 의존성이 없어 오프라인에서
(scripts/vision_tune 등) 그대로 재사용/튜닝할 수 있다.
"""
from __future__ import annotations

from typing import Tuple

from .modules.aruco_detection import ArucoDetector
from .modules.direction_sign import DirectionSignDetector
from .modules.lane_detection import LaneDetector
from .modules.traffic_sign import TrafficSignDetector
from .types import LaneResult, SignResult


class PerceptionPipeline:
    def __init__(self, params: dict | None = None):
        params = params or {}
        self.lane = LaneDetector(params.get('lane'))
        self.aruco = ArucoDetector(params.get('aruco'))
        self.traffic = TrafficSignDetector(params.get('traffic'))
        self.direction = DirectionSignDetector(params.get('direction'))

    def process(self, frame, now_sec: float) -> Tuple[LaneResult, SignResult]:
        lane_result = self.lane.detect(frame)

        stop, dist = self.aruco.detect(frame)
        direction, dconf = self.direction.detect(frame)
        # traffic 신호는 향후 제어 판단에 함께 실릴 수 있음
        self.traffic.detect(frame)

        sign_result = SignResult(
            stop_detected=stop,
            stop_distance=dist,
            direction=direction,
            direction_confidence=dconf,
        )
        return lane_result, sign_result
