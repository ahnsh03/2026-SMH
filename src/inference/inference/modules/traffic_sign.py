"""Traffic light & sign detection facade — 담당: 장원정."""

from __future__ import annotations

import numpy as np

from inference.modules.trafficsign import detect_signal
from inference.types import TrafficResult, TrafficSignal, TurnSign

_direction_detector_available = True


def _detect_turn_and_signal_safely(frame: np.ndarray) -> tuple[TurnSign, TrafficSignal]:
    """Use the optional ONNX sign+light model without breaking the driving stack."""
    global _direction_detector_available
    if not _direction_detector_available:
        return TurnSign.UNKNOWN, TrafficSignal.UNKNOWN
    try:
        # Lazy import keeps traffic-light-only deployments usable without ONNX.
        from inference.modules.direction_sign import detect_turn_and_signal

        return detect_turn_and_signal(frame)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError):
        _direction_detector_available = False
        return TurnSign.UNKNOWN, TrafficSignal.UNKNOWN


def detect(frame: np.ndarray) -> TrafficResult:
    """
    Detect traffic light color and fork turn sign from a single YOLO pass.

    Returns TrafficResult with signal (GREEN/RED/UNKNOWN) and turn (LEFT/RIGHT/UNKNOWN).
    Sign and light are classes of the same ONNX model (one forward pass per
    frame instead of two separate models); if the model is unavailable or
    doesn't find a light in this frame, the HSV rule-based color_detector
    fills in the signal so red/green detection still degrades gracefully.
    """
    turn, signal = _detect_turn_and_signal_safely(frame)
    if signal is TrafficSignal.UNKNOWN:
        signal = detect_signal(frame)
    return TrafficResult(signal=signal, turn=turn)
