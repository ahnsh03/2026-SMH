"""차선 인지 모듈 (스켈레톤).

입력  : BGR 프레임 (numpy)
출력  : LaneResult (센터라인 / 가시성 / 갈림길 / 주행가능영역)

실제 구현 위치:
  - BEV(IPM) 변환 → HSV 색상 마스크(흰/노랑) → 차선 픽셀 추출
  - 좌우 경계 → 센터라인 polyline (base_link 좌표)
  - 갈림길 분기 검출 → RoadBranch 목록
config: bringup/config/perception.yaml 의 lane_vision 블록 참고.
"""
from __future__ import annotations

from ..types import LaneResult


class LaneDetector:
    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def detect(self, frame) -> LaneResult:
        # TODO: BEV → 색상 마스크 → 센터라인/갈림길/주행가능영역 채우기
        return LaneResult()
