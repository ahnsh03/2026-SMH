"""Traffic light & sign detection facade — 담당: 장원정 / 박성준(YOLO light A/B).

Board CPU budget (D3-G): **at most one YOLO forward per frame**.

* **Signs (required):** ``direction_sign`` + ``weights/sign_best.onnx`` (2-class)
* **Lights (default):** OpenCV HSV — cheap, no second ONNX
* **Lights YOLO (optional A/B only):** Sungjun ``sign_light_best_v5b`` classes 2/3
  via ``TRAFFIC_LIGHT_BACKEND=yolo|…`` — do **not** enable in race laps while
  the sign model is also running.

Backend (env ``TRAFFIC_LIGHT_BACKEND``, default ``opencv``):

* ``opencv`` — HSV only ← race default (1 YOLO = signs)
* ``yolo`` / ``yolo_then_opencv`` / ``opencv_then_yolo`` — second YOLO; bench only
"""

from __future__ import annotations

import os

import numpy as np

from inference.modules.trafficsign import detect_signal_opencv, detect_signal_yolo
from inference.types import TrafficResult, TrafficSignal, TurnSign

_direction_detector_available = True


def _light_backend() -> str:
    # Default opencv: keep the sole YOLO slot for mandatory direction signs.
    raw = os.environ.get('TRAFFIC_LIGHT_BACKEND', 'opencv').strip().lower()
    allowed = {'opencv', 'yolo', 'yolo_then_opencv', 'opencv_then_yolo'}
    return raw if raw in allowed else 'opencv'


def _uses_light_yolo(backend: str | None = None) -> bool:
    return (backend or _light_backend()) != 'opencv'


def _detect_turn_safely(frame: np.ndarray) -> TurnSign:
    """Board 2-class sign ONNX (not Sungjun sign classes)."""
    global _direction_detector_available
    if not _direction_detector_available:
        return TurnSign.UNKNOWN
    try:
        from inference.modules.direction_sign import detect_turn

        return detect_turn(frame)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError):
        _direction_detector_available = False
        return TurnSign.UNKNOWN


def detect_signal(frame: np.ndarray) -> TrafficSignal:
    """Resolve traffic-light color with the configured backend."""
    backend = _light_backend()
    if backend == 'opencv':
        return detect_signal_opencv(frame)
    if backend == 'yolo':
        return detect_signal_yolo(frame)
    if backend == 'opencv_then_yolo':
        signal = detect_signal_opencv(frame)
        if signal is TrafficSignal.UNKNOWN:
            signal = detect_signal_yolo(frame)
        return signal
    # yolo_then_opencv
    signal = detect_signal_yolo(frame)
    if signal is TrafficSignal.UNKNOWN:
        signal = detect_signal_opencv(frame)
    return signal


def detect_signal_both(frame: np.ndarray) -> dict[str, object]:
    """Run both light backends for offline/webcam A/B (expensive — not for race)."""
    return {
        'opencv': detect_signal_opencv(frame),
        'yolo': detect_signal_yolo(frame),
        'selected': detect_signal(frame),
        'mode': _light_backend(),
        'two_yolo_risk': _uses_light_yolo(),
    }


def detect(frame: np.ndarray) -> TrafficResult:
    """Detect traffic light color and fork turn sign."""
    return TrafficResult(
        signal=detect_signal(frame),
        turn=_detect_turn_safely(frame),
    )
