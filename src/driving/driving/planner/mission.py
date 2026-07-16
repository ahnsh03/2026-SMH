"""미션 모드 판단기 — In 코스 / Out 코스.

인지 결과(LaneResult, inference.types)를 **직접(import)** 받아 추종할 경로를
고르고 LaneController 로 Control 을 만든다. ROS 토픽 왕복 없이 in-process 로
동작해 지연을 없앤다(lane_drive_node 가 프레임마다 호출).

코스 (대회 규정):
  Out 코스: 출발 → S자 → 좌우 갈림길 → 동적 장애물 → 도착  (거의 흰선)
  In  코스: 출발 → 회전 교차로 → 동적 장애물 → 도착        (노란선 진입)

Out 모드: white_centerline 추종(흰선).
In  모드: 상태기계
  APPROACH   흰선 추종. 노란선 진입 감지되면 → ENTERING
  ENTERING   노란선 추종(회전교차로 진입). 진입 각도 확보되면 → ROUNDABOUT
  ROUNDABOUT 노란선 추종. 누적 heading ≥ 임계(1바퀴)면 → EXITING
  EXITING    흰선 재확보(탈출). 흰선 안정되면 → DONE
  DONE       흰선 추종(Out 과 동일)

회전교차로 1바퀴 판정: 오도메트리가 없어 자전거모델로 heading 을 적분한다
  Δψ = (v/L)·tan(δ)·dt,  v ≈ roundabout_speed_mps (커브 스로틀 정속 가정).
  roundabout_lap_time_s > 0 이면 시간 기반으로 대체. **둘 다 실차 튜닝 필요.**
  min/max 시간 가드로 조기·무한 탈출을 막는다(규정: 반드시 1회↑ 회전 후 탈출).

신호등/표지판/ArUco 판단은 이 모듈에 없음(후속). 좌우 갈림길 표지 선택도 후속.
"""
from __future__ import annotations

import math
from typing import Tuple

from ..types import ControlCommand, DriveState
from .lane_controller import LaneController

Point = Tuple[float, float]


class MissionController:
    OUT = "out"
    IN = "in"
    # In 코스 상태
    APPROACH = "approach"
    ENTERING = "entering"
    ROUNDABOUT = "roundabout"
    EXITING = "exiting"
    DONE = "done"

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.mode = str(p.get("course_mode", self.OUT)).lower()
        self.controller = LaneController(p)
        self.L = self.controller.L

        # In 진입 감지: 노란 경로가 이 길이/프레임 이상 지속되면 진입
        self.entry_min_yellow_pts = int(p.get("entry_min_yellow_pts", 5))
        self.entry_confirm_frames = int(p.get("entry_confirm_frames", 3))
        # 상태 전이 각도(도)
        self.enter_commit_deg = float(p.get("enter_commit_deg", 45.0))
        self.roundabout_exit_deg = float(p.get("roundabout_exit_deg", 300.0))
        # 회전교차로 lap 판정
        self.roundabout_speed_mps = float(p.get("roundabout_speed_mps", 0.33))
        self.roundabout_lap_time_s = float(p.get("roundabout_lap_time_s", 0.0))  # >0 이면 시간기반
        self.roundabout_min_time_s = float(p.get("roundabout_min_time_s", 2.0))
        self.roundabout_max_time_s = float(p.get("roundabout_max_time_s", 20.0))
        # 탈출 확정: 흰선이 이 프레임 이상 안정
        self.exit_confirm_frames = int(p.get("exit_confirm_frames", 4))

        self.reset()

    # ROS 파라미터 이름(미션 전용) 집합. 나머지는 controller 로 위임.
    _MISSION_PARAMS = {
        "entry_min_yellow_pts", "entry_confirm_frames", "enter_commit_deg",
        "roundabout_exit_deg", "roundabout_speed_mps", "roundabout_lap_time_s",
        "roundabout_min_time_s", "roundabout_max_time_s", "exit_confirm_frames",
    }

    def update(self, name: str, value) -> bool:
        """단일 파라미터 라이브 갱신(주행 중). 처리했으면 True."""
        if name == "course_mode":
            self.mode = str(value).lower()
            return True
        if name in self._MISSION_PARAMS:
            cur = getattr(self, name)
            setattr(self, name, type(cur)(value))
            return True
        return self.controller.update(name, value)

    def reset(self) -> None:
        self.state = self.APPROACH
        self.controller.reset()
        self._heading = 0.0        # 상태 진입 후 누적 heading(rad)
        self._t_state = 0.0        # 현재 상태 경과(s)
        self._yellow_run = 0       # 노란 경로 연속 감지 프레임
        self._white_run = 0        # 흰 경로 연속 안정 프레임
        self._last_cmd = ControlCommand()

    # ------------------------------------------------------------------ plan
    def plan(self, lane, dt: float) -> Tuple[ControlCommand, DriveState]:
        """LaneResult + dt → (Control, DriveState). dt 는 프레임 간격(s)."""
        white = list(getattr(lane, "white_centerline", []) or [])
        yellow = list(getattr(lane, "yellow_centerline", []) or [])
        yellow_ok = len(yellow) >= self.entry_min_yellow_pts

        if self.mode != self.IN:
            cmd = self.controller.plan(white)
            self._last_cmd = cmd
            return cmd, DriveState(stopped=(cmd.throttle == 0.0), reason="out")

        return self._plan_in(white, yellow, yellow_ok, dt)

    # -------------------------------------------------------------- In 상태기계
    def _plan_in(self, white, yellow, yellow_ok, dt) -> Tuple[ControlCommand, DriveState]:
        self._t_state += dt

        if self.state == self.APPROACH:
            # 흰선 추종하며 노란 진입 대기
            self._yellow_run = self._yellow_run + 1 if yellow_ok else 0
            path = white
            if self._yellow_run >= self.entry_confirm_frames:
                self._enter_state(self.ENTERING)
                path = yellow

        elif self.state == self.ENTERING:
            # 노란선 추종하며 회전교차로로 진입. 진입각 확보 시 ROUNDABOUT.
            path = yellow if yellow_ok else white
            if abs(math.degrees(self._heading)) >= self.enter_commit_deg:
                self._enter_state(self.ROUNDABOUT)

        elif self.state == self.ROUNDABOUT:
            # 노란 원형 차로 추종. 1바퀴(누적 heading/시간) 채우면 EXITING.
            path = yellow if yellow_ok else white
            if self._lap_complete():
                self._enter_state(self.EXITING)

        elif self.state == self.EXITING:
            # 흰선 재확보(탈출). 흰선 안정되면 DONE.
            path = white if white else yellow
            white_ok = len(white) >= self.controller.min_points
            self._white_run = self._white_run + 1 if white_ok else 0
            if self._white_run >= self.exit_confirm_frames:
                self._enter_state(self.DONE)

        else:  # DONE
            path = white

        cmd = self.controller.plan(path)
        self._last_cmd = cmd
        # 자전거모델로 heading 적분 (다음 판정용)
        self._heading += self._yaw_rate(cmd) * dt
        return cmd, DriveState(stopped=(cmd.throttle == 0.0), reason=f"in:{self.state}")

    def _enter_state(self, state: str) -> None:
        self.state = state
        self._t_state = 0.0
        self._heading = 0.0
        self._white_run = 0

    def _yaw_rate(self, cmd: ControlCommand) -> float:
        """자전거모델 yaw rate = (v/L)·tan(δ).  δ=조향정규화×δ_max, v=스로틀×속도계수."""
        delta = (cmd.steering - self.controller.steer_trim) * self.controller.max_steer_rad
        v = abs(cmd.throttle) * self._speed_scale()
        return (v / self.L) * math.tan(delta)

    def _speed_scale(self) -> float:
        # throttle 단위당 m/s. roundabout_speed_mps 를 커브 스로틀 기준으로 환산.
        curve_thr = max(1e-3, self.controller.curve_throttle)
        return self.roundabout_speed_mps / curve_thr

    def _lap_complete(self) -> bool:
        if self._t_state < self.roundabout_min_time_s:
            return False
        if self._t_state >= self.roundabout_max_time_s:
            return True                                  # 안전: 무한 회전 방지
        if self.roundabout_lap_time_s > 0.0:
            return self._t_state >= self.roundabout_lap_time_s
        return abs(math.degrees(self._heading)) >= self.roundabout_exit_deg
