"""Traffic light & sign detection — 담당: 장원정"""

from __future__ import annotations

import numpy as np

from inference.types import TrafficResult, TrafficSignal, TurnSign


def detect(frame: np.ndarray) -> TrafficResult:
    """
    Detect traffic light color and fork turn sign.

    Returns TrafficResult with signal (GREEN/RED/UNKNOWN) and turn (LEFT/RIGHT/UNKNOWN).
    """
    _ = frame
    return TrafficResult(
        signal=TrafficSignal.UNKNOWN,
        turn=TurnSign.UNKNOWN,
    )
