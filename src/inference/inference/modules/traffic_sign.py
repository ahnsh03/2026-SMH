"""Traffic light & sign detection facade — 담당: 장원정 / 박성준.

* **Lights:** disabled (OpenCV + YOLO both off — unreliable on track).
  Planner uses ``green_wait_timeout_sec`` then assumes green.
* **Signs (required):** ``direction_sign`` + ``weights/sign_best.onnx`` (2-class)
"""

from __future__ import annotations

import numpy as np

from inference.types import TrafficResult, TrafficSignal, TurnSign

_direction_detector_available = True


def _light_backend() -> str:
    return 'off'


def _uses_light_yolo(backend: str | None = None) -> bool:
    del backend
    return False


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
    """Traffic lights disabled — always UNKNOWN (timeout assume-green handles start)."""
    del frame
    return TrafficSignal.UNKNOWN


def detect_signal_both(frame: np.ndarray) -> dict[str, object]:
    """Offline helper — runtime lights are off."""
    del frame
    return {
        'opencv': TrafficSignal.UNKNOWN,
        'yolo': TrafficSignal.UNKNOWN,
        'selected': TrafficSignal.UNKNOWN,
        'mode': 'off',
        'two_yolo_risk': False,
        'opencv_disabled': True,
        'yolo_disabled': True,
    }


def detect(frame: np.ndarray) -> TrafficResult:
    """Detect fork turn sign only (traffic light path is off)."""
    return TrafficResult(
        signal=detect_signal(frame),
        turn=_detect_turn_safely(frame),
    )
