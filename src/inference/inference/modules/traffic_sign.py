"""Traffic light & sign detection facade — 담당: 장원정 / 박성준(YOLO light).

Turn signs: ``direction_sign`` + ``weights/sign_best.onnx`` (board / 원정 2-class).
Traffic lights: OpenCV HSV and/or Sungjun ``sign_light_best_v5b.onnx`` (classes 2/3 only).

Backend (env ``TRAFFIC_LIGHT_BACKEND``, default ``yolo_then_opencv``):

* ``opencv`` — HSV only
* ``yolo`` — Sungjun YOLO lights only
* ``yolo_then_opencv`` — YOLO, then HSV if UNKNOWN
* ``opencv_then_yolo`` — HSV, then YOLO if UNKNOWN
"""

from __future__ import annotations

import os

import numpy as np

from inference.modules.trafficsign import detect_signal_opencv, detect_signal_yolo
from inference.types import TrafficResult, TrafficSignal, TurnSign

_direction_detector_available = True


def _light_backend() -> str:
    raw = os.environ.get('TRAFFIC_LIGHT_BACKEND', 'yolo_then_opencv').strip().lower()
    allowed = {'opencv', 'yolo', 'yolo_then_opencv', 'opencv_then_yolo'}
    return raw if raw in allowed else 'yolo_then_opencv'


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
    # yolo_then_opencv (default for A/B on track)
    signal = detect_signal_yolo(frame)
    if signal is TrafficSignal.UNKNOWN:
        signal = detect_signal_opencv(frame)
    return signal


def detect_signal_both(frame: np.ndarray) -> dict[str, object]:
    """Run both backends for track-side A/B (does not change driving)."""
    return {
        'opencv': detect_signal_opencv(frame),
        'yolo': detect_signal_yolo(frame),
        'selected': detect_signal(frame),
        'mode': _light_backend(),
    }


def detect(frame: np.ndarray) -> TrafficResult:
    """Detect traffic light color and fork turn sign."""
    return TrafficResult(
        signal=detect_signal(frame),
        turn=_detect_turn_safely(frame),
    )
