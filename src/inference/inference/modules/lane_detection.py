"""Lane detection module — 담당: 장원태"""

from __future__ import annotations

import numpy as np


def detect(frame: np.ndarray) -> tuple[float, float]:
    """
    Returns (steering_offset, confidence).
    steering_offset: -1.0 (left) ~ +1.0 (right), 0 = center.
    """
    _ = frame
    return 0.0, 0.0
