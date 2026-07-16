"""차선추종 제어기 — Pure Pursuit + 곡률 피드포워드 (실차 튜닝값).

입력  : centerline polyline [(x, y), ...]  (base_link, x 전방+, **y 우측+**, 미터)
        카메라 프레임 기준. 내부에서 후륜축으로 이동(x += camera_to_axle).
출력  : ControlCommand(steering∈[-1,1] +우/-좌, throttle∈[-1,1])

부호 규약(중요, 안전 직결):
  - 인지 centerline y 우측+ (metric_bev 규약). Pure Pursuit 내부 조향은
    δ = +atan(L·2y/d²) 로 "우측=+" 로 계산(부호 반전 없음).
  - 출력은 **차량 규약**으로 변환: 이 D-Racer 는 **오른쪽=음수(−), 왼쪽=양수(+)**.
    따라서 최종 출력에 steer_sign(기본 −1)을 곱한다. 우회전 → 음수, 좌회전 → 양수.
  - 백파일 검증: 사람 /control 은 좌회전 시 +1.00(=좌). 검출 far_y<0(좌)일 때
    사람 + → steer_sign 적용 후 상관 +. 이 규약이 깨지면 실차가 반대로 조향한다.
    test_lane_controller.py 가 고정.

게인은 실차 백파일로 그리드 스윕해 정한 값(사람 수동주행 조향과의 상관·부호
일치를 목적함수). CTE/heading 보정은 인지 centerline 이 이미 양호해 제외 —
순수 PP + 곡률 FF 가 최적.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ..types import ControlCommand

Point = Tuple[float, float]

# 차량 기하 SSOT (docs/vehicle-geometry / 실차)
DEFAULT_WHEELBASE_M = 0.175          # 휠베이스 L
DEFAULT_MAX_STEER_RAD = 0.3054       # δ_max = 17.5°, steering=±1 ↔ ±17.5°
DEFAULT_CAM_TO_AXLE_M = 0.20         # 카메라 → 후륜축


class LaneController:
    """centerline → Control. 상태(EMA/slew) 보유 → reset() 로 초기화."""

    def __init__(self, params: dict | None = None):
        p = params or {}
        # 차량 기하
        self.L = float(p.get("wheelbase_m", DEFAULT_WHEELBASE_M))
        self.max_steer_rad = float(p.get("max_steer_rad", DEFAULT_MAX_STEER_RAD))
        self.cam_to_axle = float(p.get("camera_to_axle_m", DEFAULT_CAM_TO_AXLE_M))
        # 적응형 look-ahead (급커브일수록 짧게 → 커브를 문다)
        self.base_lookahead = float(p.get("base_lookahead_m", 0.85))
        self.curve_lookahead = float(p.get("curve_lookahead_m", 0.45))
        self.curvature_full = float(p.get("curvature_full_scale", 0.60))
        self.curvature_ff_gain = float(p.get("curvature_ff_gain", 0.5))
        # 곡률 추정 3점 (후륜축 기준 전방 거리, m). 후륜축 검출범위 0.42~1.5 안.
        self.kappa_samples = tuple(p.get("kappa_sample_x_m", (0.45, 0.70, 0.95)))
        # 스로틀 스케줄
        self.cruise_throttle = float(p.get("cruise_throttle", 0.22))
        self.curve_throttle = float(p.get("curve_throttle", 0.13))
        # 평활·트림
        self.steer_ema = float(p.get("steer_ema", 0.5))         # 0~1, 클수록 새값 비중↑
        self.steer_rate_limit = float(p.get("steer_rate_limit", 0.25))  # 프레임당 |Δsteer|
        self.lookahead_slew = float(p.get("lookahead_slew_m", 0.15))    # 프레임당 |ΔLd|
        # 차량 조향 부호: 이 D-Racer 는 오른쪽=−, 왼쪽=+ → 내부(우측=+)에 −1 곱해 변환.
        self.steer_sign = float(p.get("steer_sign", -1.0))
        self.steer_trim = float(p.get("steer_trim", 0.10))      # 실차 조향 오프셋(차량 규약)
        self.min_points = int(p.get("min_points", 3))

        self._last_steer = 0.0       # 정규화 조향(EMA/rate-limit 상태)
        self._last_ld = self.base_lookahead

    def reset(self) -> None:
        self._last_steer = 0.0
        self._last_ld = self.base_lookahead

    # ROS 파라미터 이름 → 내부 속성 (라이브 튜닝용)
    _PARAM_ATTR = {
        "wheelbase_m": "L", "max_steer_rad": "max_steer_rad",
        "camera_to_axle_m": "cam_to_axle", "base_lookahead_m": "base_lookahead",
        "curve_lookahead_m": "curve_lookahead", "curvature_full_scale": "curvature_full",
        "curvature_ff_gain": "curvature_ff_gain", "cruise_throttle": "cruise_throttle",
        "curve_throttle": "curve_throttle", "steer_ema": "steer_ema",
        "steer_rate_limit": "steer_rate_limit", "steer_sign": "steer_sign",
        "steer_trim": "steer_trim",
    }

    def update(self, name: str, value) -> bool:
        """단일 게인 라이브 갱신(주행 중). 처리했으면 True."""
        attr = self._PARAM_ATTR.get(name)
        if attr is None:
            return False
        setattr(self, attr, float(value))
        return True

    # ------------------------------------------------------------------ plan
    def plan(self, centerline: List[Point]) -> ControlCommand:
        """centerline(카메라 프레임) → Control. 경로 없으면 조향유지·정지."""
        path = self._to_axle(centerline)
        if len(path) < self.min_points:
            # 경로 소실 → 상위(노드)가 안전정지. 조향은 직전 유지.
            return ControlCommand(steering=self._output(self._last_steer), throttle=0.0)

        kappa = self._signed_curvature(path)
        r = min(1.0, abs(kappa) / self.curvature_full) if self.curvature_full > 0 else 0.0

        # 적응형 look-ahead + slew
        ld_target = self.base_lookahead * (1.0 - r) + self.curve_lookahead * r
        ld = self._slew(self._last_ld, ld_target, self.lookahead_slew)
        self._last_ld = ld

        target = self._lookahead_point(path, ld)
        if target is None:
            return ControlCommand(steering=self._output(self._last_steer), throttle=0.0)

        # Pure Pursuit + 곡률 피드포워드
        x, y = target
        d2 = x * x + y * y
        delta_pp = math.atan(self.L * 2.0 * y / d2) if d2 > 1e-6 else 0.0
        delta_ff = self.curvature_ff_gain * math.atan(self.L * kappa)
        delta = delta_pp + delta_ff

        steer = max(-1.0, min(1.0, delta / self.max_steer_rad))    # 정규화(내부 우측=+)
        steer = self._smooth(steer)

        throttle = self.cruise_throttle * (1.0 - r) + self.curve_throttle * r
        return ControlCommand(steering=self._output(steer), throttle=throttle)

    def _output(self, internal_steer: float) -> float:
        """내부 조향(우측=+) → 차량 규약 출력(steer_sign 적용) + 트림, [-1,1] 클램프."""
        out = self.steer_sign * internal_steer + self.steer_trim
        return max(-1.0, min(1.0, out))     # 서보 입력 범위. 트림 더한 뒤 최종 클램프.

    # ---------------------------------------------------------------- helpers
    def _to_axle(self, centerline: List[Point]) -> List[Point]:
        """카메라 프레임 → 후륜축 프레임(x += cam_to_axle), x 오름차순 정렬."""
        pts = [(float(x) + self.cam_to_axle, float(y)) for x, y in centerline]
        pts.sort(key=lambda p: p[0])
        return pts

    @staticmethod
    def _slew(prev: float, target: float, limit: float) -> float:
        d = target - prev
        if d > limit:
            return prev + limit
        if d < -limit:
            return prev - limit
        return target

    def _sample_y_at_x(self, path: List[Point], x_q: float) -> Optional[float]:
        """x_q 에서 y 선형보간 (path 는 x 오름차순). 범위 밖이면 최근접 클램프."""
        if not path:
            return None
        if x_q <= path[0][0]:
            return path[0][1]
        if x_q >= path[-1][0]:
            return path[-1][1]
        for i in range(1, len(path)):
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            if x0 <= x_q <= x1 and x1 > x0:
                t = (x_q - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return path[-1][1]

    def _signed_curvature(self, path: List[Point]) -> float:
        """전방 3점의 부호있는 곡률 κ. 우측으로 휘면 +(우조향 방향과 일치).

        y' = dy/dx, y'' = d²y/dx² → κ = y'' / (1+y'²)^{3/2}.
        """
        xs = self.kappa_samples
        ys = [self._sample_y_at_x(path, x) for x in xs]
        if any(v is None for v in ys) or len(xs) < 3:
            return 0.0
        x0, x1, x2 = xs[0], xs[1], xs[2]
        y0, y1, y2 = ys
        if (x1 - x0) <= 0 or (x2 - x1) <= 0 or (x2 - x0) <= 0:
            return 0.0
        yp = (y2 - y0) / (x2 - x0)
        ypp = 2.0 * ((y2 - y1) / (x2 - x1) - (y1 - y0) / (x1 - x0)) / (x2 - x0)
        return ypp / (1.0 + yp * yp) ** 1.5

    @staticmethod
    def _lookahead_point(path: List[Point], ld: float) -> Optional[Point]:
        """후륜축에서 유클리드 거리 ld 이상 떨어진 첫 경로점."""
        for x, y in path:
            if x <= 0:
                continue
            if math.hypot(x, y) >= ld:
                return (x, y)
        return path[-1] if path else None

    def _smooth(self, steer: float) -> float:
        """EMA + 프레임당 rate-limit → 서보 안정, 급변 억제."""
        ema = self.steer_ema * steer + (1.0 - self.steer_ema) * self._last_steer
        d = ema - self._last_steer
        if d > self.steer_rate_limit:
            ema = self._last_steer + self.steer_rate_limit
        elif d < -self.steer_rate_limit:
            ema = self._last_steer - self.steer_rate_limit
        self._last_steer = ema
        return ema
