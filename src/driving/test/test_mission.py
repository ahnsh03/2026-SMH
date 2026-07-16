"""MissionController 단위 테스트 — In/Out 모드 + In 상태기계.

Out: 흰선 추종(노랑 무시). In: 노란선 진입 감지 → 회전교차로 1바퀴 → 탈출.
"""
import sys
from pathlib import Path

import pytest

_WS = Path(__file__).resolve().parents[3]        # 2026-SMH-team-new
sys.path.insert(0, str(_WS / "src" / "driving"))
sys.path.insert(0, str(_WS / "src" / "inference"))

from driving.planner.mission import MissionController  # noqa: E402
from inference.types import LaneResult                 # noqa: E402


def _line(slope=0.0, curv=0.0, n=60, x0=0.22, dx=0.02):
    pts = []
    for i in range(n):
        x = x0 + dx * i
        s = x - x0
        pts.append((x, slope * s + curv * s * s))
    return pts


def _lane(white=None, yellow=None):
    r = LaneResult()
    r.white_centerline = white or []
    r.yellow_centerline = yellow or []
    r.white_visible = bool(white)
    r.yellow_visible = bool(yellow)
    return r


def test_out_follows_white_sign():
    """Out 모드: 우향 흰선 → 차량 규약 우조향(음수, steer_sign 기본 −1)."""
    m = MissionController(dict(course_mode="out", steer_trim=0.0,
                               steer_ema=1.0, steer_rate_limit=1.0))
    cmd, st = m.plan(_lane(white=_line(slope=0.4)), 0.05)
    assert cmd.steering < -0.05
    assert st.reason == "out"


def test_out_ignores_yellow():
    """Out 모드: 흰선 직진 + 노란선 급커브여도 흰선만 따른다(조향≈0)."""
    m = MissionController(dict(course_mode="out", steer_trim=0.0,
                               steer_ema=1.0, steer_rate_limit=1.0))
    cmd, _ = m.plan(_lane(white=_line(0.0, 0.0), yellow=_line(slope=1.0)), 0.05)
    assert abs(cmd.steering) < 0.05


def test_in_starts_in_approach_following_white():
    m = MissionController(dict(course_mode="in"))
    cmd, st = m.plan(_lane(white=_line(slope=0.3)), 0.05)
    assert st.reason == "in:approach"


def test_in_entry_detects_yellow():
    """노란 경로가 confirm_frames 연속 감지되면 APPROACH→ENTERING."""
    m = MissionController(dict(course_mode="in", entry_confirm_frames=3,
                               entry_min_yellow_pts=5))
    white = _line(0.0, 0.0)
    yellow = _line(slope=0.3)
    # 노랑 없을 땐 APPROACH 유지
    _, st = m.plan(_lane(white=white), 0.05)
    assert st.reason == "in:approach"
    # 노랑 3프레임 연속 → ENTERING 진입
    last = None
    for _ in range(3):
        _, last = m.plan(_lane(white=white, yellow=yellow), 0.05)
    assert last.reason == "in:entering"


def test_in_full_sequence_reaches_done():
    """APPROACH→ENTERING→ROUNDABOUT→EXITING→DONE 전 상태를 거친다."""
    m = MissionController(dict(
        course_mode="in", entry_confirm_frames=2, entry_min_yellow_pts=5,
        enter_commit_deg=15.0, roundabout_lap_time_s=1.0,
        roundabout_min_time_s=0.3, exit_confirm_frames=3,
        curve_throttle=0.2, roundabout_speed_mps=0.4))
    white = _line(0.0, 0.0)
    yellow = _line(slope=0.6)      # 우향 → 조향+ → heading 누적
    seen = set()
    for _ in range(400):
        _, st = m.plan(_lane(white=white, yellow=yellow), 0.1)
        seen.add(st.reason)
        if st.reason == "in:done":
            break
    for s in ("in:approach", "in:entering", "in:roundabout", "in:exiting", "in:done"):
        assert s in seen, f"상태 {s} 미방문: {sorted(seen)}"


def test_in_roundabout_min_time_guard():
    """회전교차로 체류시간이 min_time 이상이어야 탈출한다(조기 탈출 방지)."""
    dt = 0.1
    m = MissionController(dict(
        course_mode="in", entry_confirm_frames=1, entry_min_yellow_pts=5,
        enter_commit_deg=5.0, roundabout_lap_time_s=0.5,   # lap_time < min_time
        roundabout_min_time_s=3.0, curve_throttle=0.2, roundabout_speed_mps=0.4))
    white = _line(0.0, 0.0)
    yellow = _line(slope=0.6)
    round_time = 0.0
    left_roundabout = False
    for _ in range(80):
        _, st = m.plan(_lane(white=white, yellow=yellow), dt)
        if st.reason == "in:roundabout":
            round_time += dt
        elif round_time > 0:
            left_roundabout = True
            break
    assert left_roundabout, "회전교차로를 벗어나지 못함"
    # lap_time(0.5) 이 아니라 min_time(3.0) 가드가 우선 → 최소 3초 체류
    assert round_time >= 3.0 - dt, f"조기 탈출: {round_time:.2f}s"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
