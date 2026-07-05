"""
Perception/planning pipeline — module fusion.

담당자는 modules/*.py 만 수정하세요.
우선순위·통합 로직 변경은 팀장(또는 통합 담당) PR 로만 수정합니다.
"""

from __future__ import annotations

import numpy as np

from inference.modules import aruco_detection, lane_detection, roundabout, traffic_sign
from inference.types import ControlCommand, PipelineContext, TrafficSignal


def run_perception(frame: np.ndarray) -> PipelineContext:
    """Call each team module. Modules do not depend on each other."""
    return PipelineContext(
        lane=lane_detection.detect(frame),
        traffic=traffic_sign.detect(frame),
        aruco=aruco_detection.detect(frame),
        roundabout=roundabout.plan(frame),
    )


def fuse_control(
    ctx: PipelineContext,
    *,
    steer_trim: float = 0.0,
    default_throttle: float = 0.0,
    cruise_throttle: float = 0.35,
) -> ControlCommand:
    """
    Merge module outputs into a single control command.

    Priority (highest first):
      1. ArUco stop
      2. Red traffic light stop
      3. Roundabout override (when active)
      4. Lane following (default)
    """
    if ctx.aruco.should_stop:
        return ControlCommand(steering=steer_trim, throttle=0.0)

    if ctx.traffic.signal == TrafficSignal.RED:
        return ControlCommand(steering=steer_trim, throttle=0.0)

    if ctx.roundabout.active:
        return ControlCommand(
            steering=ctx.roundabout.steering,
            throttle=ctx.roundabout.throttle,
        )

    steering = steer_trim + ctx.lane.steering_offset
    steering = float(np.clip(steering, -1.0, 1.0))
    throttle = cruise_throttle if ctx.lane.confidence > 0.1 else default_throttle
    return ControlCommand(steering=steering, throttle=throttle)
