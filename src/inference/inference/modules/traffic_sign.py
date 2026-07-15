"""신호등/교통표지 색상 인지 모듈 (스켈레톤).

입력  : BGR 프레임
출력  : 신호 상태(예: 정지/진행). 제어 판단의 보조 입력.
실제 구현: ROI + HSV 색상 임계로 적/녹 판정.
"""
from __future__ import annotations


class TrafficSignDetector:
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def detect(self, frame) -> int:
        # TODO: HSV 색상 판정 → 0 unknown / 1 stop / 2 go
        return 0
