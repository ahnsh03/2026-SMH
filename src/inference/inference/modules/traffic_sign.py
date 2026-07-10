"""Traffic light & sign detection facade — 담당: 장원정."""

from __future__ import annotations

import numpy as np

from inference.modules.trafficsign import detect_signal
from inference.types import TrafficResult, TurnSign


def detect(frame: np.ndarray) -> TrafficResult:
    """
    Detect traffic light color and fork turn sign.

    Returns TrafficResult with signal (GREEN/RED/UNKNOWN) and turn (LEFT/RIGHT/UNKNOWN).
    Turn-sign detection is not implemented yet — always UNKNOWN for now.
    """
    return TrafficResult(
        signal=detect_signal(frame),
        turn=TurnSign.UNKNOWN,
    )
