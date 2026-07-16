"""LaneController 단위 테스트 — 부호 규약 + 거동.

핵심(차량 규약): 이 D-Racer 는 **오른쪽=음수(−), 왼쪽=양수(+)**.
내부 PP 는 우측=+ 로 계산하고 steer_sign(기본 −1)으로 출력에서 변환한다.
부호가 깨지면 실차가 반대로 조향한다.
"""
import sys
from pathlib import Path

import pytest

_WS = Path(__file__).resolve().parents[3]        # 2026-SMH-team-new
sys.path.insert(0, str(_WS / "src" / "driving"))

from driving.planner.lane_controller import LaneController  # noqa: E402


def _ctrl(**kw):
    # 평활/트림 제거 → 순간 응답으로 부호 검증
    params = dict(steer_trim=0.0, steer_ema=1.0, steer_rate_limit=1.0, lookahead_slew_m=1.0)
    params.update(kw)
    return LaneController(params)


def _line(slope=0.0, curv=0.0, n=60, x0=0.22, dx=0.02):
    """카메라 프레임 centerline: y = slope·s + curv·s²  (s=x-x0, y 우측+)."""
    pts = []
    for i in range(n):
        x = x0 + dx * i
        s = x - x0
        pts.append((x, slope * s + curv * s * s))
    return pts


def test_straight_zero_steer():
    c = _ctrl()
    cmd = c.plan(_line(0.0, 0.0))
    assert abs(cmd.steering) < 1e-3
    assert cmd.throttle == pytest.approx(c.cruise_throttle, abs=1e-6)


def test_right_curve_steers_right_negative():
    """오른쪽으로 휘는 경로(y>0 전방) → 차량 규약 우조향(음수)."""
    c = _ctrl()      # steer_sign 기본 −1 (차량 규약)
    cmd = c.plan(_line(slope=0.4))
    assert cmd.steering < -0.05


def test_left_curve_steers_left_positive():
    """왼쪽으로 휘는 경로(y<0) → 차량 규약 좌조향(양수)."""
    c = _ctrl()
    cmd = c.plan(_line(slope=-0.4))
    assert cmd.steering > 0.05


def test_steer_sign_flips_output():
    """steer_sign 이 내부(우측=+) 를 차량 규약으로 뒤집는다."""
    right = _line(slope=0.4)
    internal = _ctrl(steer_sign=1.0).plan(right).steering    # 내부 규약: 우향 → +
    vehicle = _ctrl(steer_sign=-1.0).plan(right).steering     # 차량 규약: 우향 → −
    assert internal > 0 and vehicle < 0
    assert vehicle == pytest.approx(-internal, abs=1e-6)


def test_curvature_ff_adds_on_right_bend():
    """우향 곡률 → 곡률 FF 가 우조향(음수)을 더 키운다(FF 켠 값 ≤ 끈 값)."""
    with_ff = _ctrl(curvature_ff_gain=0.6).plan(_line(curv=0.8))
    no_ff = _ctrl(curvature_ff_gain=0.0).plan(_line(curv=0.8))
    assert with_ff.steering < 0.0
    assert with_ff.steering <= no_ff.steering


def test_empty_path_stops():
    c = _ctrl()
    cmd = c.plan([])
    assert cmd.throttle == 0.0


def test_curve_slows_throttle():
    """급커브 스로틀 < 직진 스로틀."""
    c = _ctrl()
    straight = c.plan(_line(0.0, 0.0)).throttle
    c2 = _ctrl()
    sharp = c2.plan(_line(curv=1.2)).throttle
    assert sharp < straight
    assert straight == pytest.approx(c.cruise_throttle, abs=1e-6)


def test_steer_trim_applied():
    """steer_trim 은 출력에 상수로 더해진다(직진 → trim)."""
    c = LaneController(dict(steer_trim=0.1, steer_ema=1.0, steer_rate_limit=1.0))
    cmd = c.plan(_line(0.0, 0.0))
    assert cmd.steering == pytest.approx(0.1, abs=1e-3)


def test_steer_normalized_range():
    """조향 출력은 트림을 더해도 [-1,1] 범위로 클램프(서보 입력 범위)."""
    # 극단적 좌향(내부 steer=-1) + steer_sign −1 → +1, + trim 0.10 = 1.10 → 1.0 로 클램프
    c = LaneController(dict(steer_sign=-1.0, steer_trim=0.10,
                            steer_ema=1.0, steer_rate_limit=1.0))
    cmd = c.plan(_line(slope=-5.0))    # 극단적 좌향
    assert -1.0 <= cmd.steering <= 1.0
    c2 = LaneController(dict(steer_sign=-1.0, steer_trim=0.10,
                             steer_ema=1.0, steer_rate_limit=1.0))
    cmd2 = c2.plan(_line(slope=5.0))   # 극단적 우향
    assert -1.0 <= cmd2.steering <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
