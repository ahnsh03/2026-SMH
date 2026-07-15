"""제어 계층 내부 자료형. ROS 비의존 → 단위 테스트/튜닝 용이."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlCommand:
    """최종 주행 명령. [-1, 1] 정규화 (D-Racer control_node 가 서보/스로틀로 변환)."""
    steering: float = 0.0
    throttle: float = 0.0


@dataclass
class DriveState:
    """제어 판단 상태 (정지/주행/갈림길 등). 디버깅·미션 로직용."""
    stopped: bool = False
    reason: str = ''
