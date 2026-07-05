"""Lane detection module — 담당: 장원태"""

from __future__ import annotations

import numpy as np

from inference.types import LaneResult


def detect(frame: np.ndarray) -> LaneResult:
    """
    Detect lane position from camera frame.

    Returns LaneResult with steering_offset (-1.0 ~ +1.0) and confidence (0.0 ~ 1.0).
    """
    _ = frame
    return LaneResult(steering_offset=0.0, confidence=0.0)
