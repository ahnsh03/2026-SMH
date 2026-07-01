"""Roundabout planning — 담당: 양서준 (ArUco 완료 후 안승현·박성준 합류)"""

from __future__ import annotations

import numpy as np


def plan(frame: np.ndarray) -> tuple[float, float]:
    """
    Returns (steering, throttle) for roundabout segment.
    """
    _ = frame
    return 0.0, 0.0
