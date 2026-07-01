"""Traffic light & sign detection — 담당: 장원정"""

from __future__ import annotations

import numpy as np


class TrafficState:
    UNKNOWN = 'unknown'
    GREEN = 'green'
    RED = 'red'
    LEFT = 'left'
    RIGHT = 'right'


def detect(frame: np.ndarray) -> dict:
    """
    Returns dict with keys:
      signal: TrafficState (GREEN / RED / UNKNOWN)
      turn: TrafficState (LEFT / RIGHT / UNKNOWN) for fork mission
    """
    _ = frame
    return {'signal': TrafficState.UNKNOWN, 'turn': TrafficState.UNKNOWN}
