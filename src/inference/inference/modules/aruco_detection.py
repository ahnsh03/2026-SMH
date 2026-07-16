"""정지 표지(ArUco) 인지 모듈 어댑터.

입력  : BGR 프레임
출력  : (should_stop, marker_id)   — PerceptionPipeline/inference_node 용

실제 검출·정지 판단은 modules/aruco_stop.ArucoStopDetector 에 있다
(DICT_6X6_50 ID 3, 벽시계 히스테리시스). 통합 주행노드(lane_drive_node)는
ArucoStopDetector 를 직접 써서 마커 인식 시 throttle=0 으로 정지한다.
"""
from __future__ import annotations

from typing import Tuple

from .aruco_stop import ArucoStopDetector


class ArucoDetector:
    def __init__(self, params: dict | None = None):
        self.impl = ArucoStopDetector(params)

    def detect(self, frame) -> Tuple[bool, float]:
        """(should_stop, marker_id). marker_id 없으면 -1."""
        should_stop, marker_id = self.impl.stop(frame)
        return should_stop, float(marker_id if marker_id is not None else -1)
