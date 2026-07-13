"""Traffic light & sign detection facade — 담당: 장원정."""

from __future__ import annotations

import numpy as np

from inference.modules.trafficsign import detect_signal
from inference.types import TrafficResult, TurnSign

_direction_detector_available = True


def _detect_turn_safely(frame: np.ndarray) -> TurnSign:
    """Use the optional ONNX sign model without breaking the driving stack."""
    global _direction_detector_available
    if not _direction_detector_available:
        return TurnSign.UNKNOWN
    try:
        # Lazy import keeps traffic-light-only deployments usable without ONNX.
        from inference.modules.direction_sign import detect_turn

        return detect_turn(frame)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError):
        _direction_detector_available = False
        return TurnSign.UNKNOWN


def detect(frame: np.ndarray) -> TrafficResult:
    """
    Detect traffic light color and fork turn sign.

    Returns TrafficResult with signal (GREEN/RED/UNKNOWN) and turn (LEFT/RIGHT/UNKNOWN).
    Direction detection is optional; unavailable weights/runtime return UNKNOWN.
    """
    return TrafficResult(
        signal=detect_signal(frame),
        turn=_detect_turn_safely(frame),
    )
