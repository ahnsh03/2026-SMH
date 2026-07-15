"""정지 표지(ArUco) 인지 모듈 (스켈레톤).

입력  : BGR 프레임
출력  : (stop_detected, distance)
실제 구현: cv2.aruco 로 마커 검출 → 마커 크기로 거리 추정 → 정지선 로직.
"""
from __future__ import annotations

from typing import Tuple


class ArucoDetector:
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def detect(self, frame) -> Tuple[bool, float]:
        # TODO: cv2.aruco 검출 + 거리 추정
        return False, 0.0
