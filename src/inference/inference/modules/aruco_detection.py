"""ArUco facade — do not edit unless coordinating detector + stop_logic."""

from __future__ import annotations

import numpy as np

from inference.modules.aruco import detect_markers, should_stop_for_markers
from inference.types import ArucoResult


def detect(frame: np.ndarray) -> ArucoResult:
    """Run detector + stop logic and return a unified ArucoResult."""
    marker_ids = detect_markers(frame)
    should_stop, marker_id = should_stop_for_markers(marker_ids)
    return ArucoResult(
        detected=bool(marker_ids),
        marker_id=marker_id,
        should_stop=should_stop,
    )
