"""Unit tests for traffic light color detection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `pytest` without an installed ament package overlay.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

cv2 = pytest.importorskip('cv2')

from inference.modules.trafficsign import detect_signal  # noqa: E402
from inference.types import TrafficSignal  # noqa: E402


def _light_frame(color_bgr: tuple[int, int, int], size: int = 200, radius: int = 40) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(frame, (size // 2, size // 2), radius, color_bgr, -1)
    return frame


def test_detect_red():
    frame = _light_frame((0, 0, 255))
    assert detect_signal(frame) == TrafficSignal.RED


def test_detect_green():
    frame = _light_frame((0, 255, 0))
    assert detect_signal(frame) == TrafficSignal.GREEN


def test_detect_unlit_lens_is_unknown():
    frame = _light_frame((20, 20, 20))
    assert detect_signal(frame) == TrafficSignal.UNKNOWN


def test_detect_non_round_green_blob_is_unknown():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 90), (180, 115), (0, 255, 0), -1)
    assert detect_signal(frame) == TrafficSignal.UNKNOWN


def test_detect_empty_frame():
    assert detect_signal(np.zeros((0, 0), dtype=np.uint8)) == TrafficSignal.UNKNOWN
    assert detect_signal(np.full((64, 64, 3), 255, dtype=np.uint8)) == TrafficSignal.UNKNOWN


def test_facade_wraps_signal():
    from inference.modules import traffic_sign
    from inference.modules.trafficsign.debounce import _CONFIRM_FRAMES

    frame = _light_frame((0, 0, 255))
    # The facade debounces across frames (see debounce.py) — a signal must repeat
    # for _CONFIRM_FRAMES consecutive frames before it is reported.
    for _ in range(_CONFIRM_FRAMES):
        result = traffic_sign.detect(frame)
    assert result.signal == TrafficSignal.RED
