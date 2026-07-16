"""Traffic light & sign detection facade — 담당: 장원정 / 박성준(YOLO light).

* **Lights (only):** team-new YOLO ONNX ``sign_light_best_v5b`` classes 2/3
* **Signs (required):** ``direction_sign`` + ``weights/sign_best.onnx`` (2-class)

OpenCV HSV light detection is **disabled** (false positives on nearby colored
objects). Env ``TRAFFIC_LIGHT_BACKEND`` is ignored for runtime lights.
"""

from __future__ import annotations

import numpy as np

from inference.modules.trafficsign import detect_signal_yolo
from inference.types import TrafficResult, TrafficSignal, TurnSign

_direction_detector_available = True


def _light_backend() -> str:
    return 'yolo'


def _uses_light_yolo(backend: str | None = None) -> bool:
    del backend
    return True


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
    """YOLO-only traffic-light color (OpenCV HSV path removed)."""
    return detect_signal_yolo(frame)


def detect_signal_both(frame: np.ndarray) -> dict[str, object]:
    """Offline A/B helper — runtime still YOLO-only (opencv field unused)."""
    yolo = detect_signal_yolo(frame)
    return {
        'opencv': TrafficSignal.UNKNOWN,
        'yolo': yolo,
        'selected': yolo,
        'mode': 'yolo',
        'two_yolo_risk': True,
        'opencv_disabled': True,
    }


def detect(frame: np.ndarray) -> TrafficResult:
    """Detect traffic light color and fork turn sign."""
    return TrafficResult(
        signal=detect_signal(frame),
        turn=_detect_turn_safely(frame),
    )
