"""방향 표지(좌/우 회전) 인지 모듈 (스켈레톤).

입력  : BGR 프레임
출력  : (direction, confidence)  direction: 0 unknown / 1 left / 2 right
실제 구현: 템플릿/특징 매칭 또는 경량 분류기. 갈림길 진입 방향 결정에 사용.
"""
from __future__ import annotations

from typing import Tuple


class DirectionSignDetector:
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def detect(self, frame) -> Tuple[int, float]:
        # TODO: 좌/우 방향 표지 검출
        return 0, 0.0
