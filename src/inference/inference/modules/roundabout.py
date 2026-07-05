"""Roundabout planning — 담당: 양서준 (ArUco 완료 후 안승현·박성준 합류)"""

from __future__ import annotations

import numpy as np

from inference.types import RoundaboutResult


def plan(frame: np.ndarray) -> RoundaboutResult:
    """
    Plan steering/throttle for the roundabout segment.

    Set active=True when roundabout logic should override lane following.
    """
    _ = frame
    return RoundaboutResult(active=False, steering=0.0, throttle=0.0)
