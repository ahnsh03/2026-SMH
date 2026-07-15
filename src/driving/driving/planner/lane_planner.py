"""차선 추종 플래너 (Pure Pursuit 스켈레톤).

입력  : LaneDetections (인지 결과)
출력  : ControlCommand (steering/throttle, [-1, 1])

경로선택(path_select) → Pure Pursuit 조향 + EMA 평활 → 명령 생성.
파라미터는 bringup/config/driving.yaml 의 lane_control 블록에서 주입.
"""
from __future__ import annotations

import math

from .path_select import select_path
from ..types import ControlCommand


class LanePlanner:
    def __init__(self, params: dict | None = None):
        params = params or {}
        self.lookahead = float(params.get('lookahead', 0.35))      # m
        self.wheelbase = float(params.get('wheelbase', 0.20))      # m
        self.max_steer = float(params.get('max_steer', 1.0))       # 정규화 한계
        self.base_throttle = float(params.get('base_throttle', 0.25))
        self.steer_ema = float(params.get('steer_ema', 0.5))       # 평활 계수
        self._last_steer = 0.0

    def plan(self, lane_msg) -> ControlCommand:
        path = select_path(lane_msg)
        if not path:
            # 추종 경로 없음 → 상위(control_node)가 안전정지 판단
            return ControlCommand(steering=self._last_steer, throttle=0.0)

        target = self._lookahead_point(path)
        if target is None:
            return ControlCommand(steering=self._last_steer, throttle=0.0)

        steer = self._pure_pursuit(target)
        steer = self._smooth(steer)
        return ControlCommand(steering=steer, throttle=self.base_throttle)

    def _lookahead_point(self, path):
        """lookahead 거리 이상 떨어진 첫 경로점 (base_link, x 전방)."""
        for x, y in path:
            if math.hypot(x, y) >= self.lookahead:
                return (x, y)
        return path[-1] if path else None

    def _pure_pursuit(self, target) -> float:
        x, y = target
        ld2 = x * x + y * y
        if ld2 < 1e-6:
            return 0.0
        # 곡률 kappa = 2*y / Ld^2 → 조향각. 정규화 후 클램프.
        curvature = 2.0 * y / ld2
        steer = math.atan(self.wheelbase * curvature)
        norm = max(-1.0, min(1.0, steer / (math.pi / 4)))
        return max(-self.max_steer, min(self.max_steer, norm))

    def _smooth(self, steer: float) -> float:
        a = self.steer_ema
        self._last_steer = a * steer + (1.0 - a) * self._last_steer
        return self._last_steer
