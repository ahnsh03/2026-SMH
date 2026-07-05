"""ArUco marker detection — 담당: 안승현"""

from __future__ import annotations

import numpy as np


def detect_markers(frame: np.ndarray) -> list[int]:
    """
    Detect ArUco marker IDs visible in the frame.

    Returns a list of detected marker IDs (empty if none).
    """
    _ = frame
    return []
