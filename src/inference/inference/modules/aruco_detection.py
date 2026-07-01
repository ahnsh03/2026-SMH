"""ArUco marker detection for dynamic obstacle — 담당: 안승현, 박성준"""

from __future__ import annotations

import numpy as np


def detect(frame: np.ndarray) -> dict:
    """
    Returns dict with keys:
      detected: bool
      marker_id: int | None
      should_stop: bool
    """
    _ = frame
    return {'detected': False, 'marker_id': None, 'should_stop': False}
