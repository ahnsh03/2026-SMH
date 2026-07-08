"""Unit tests for ArUco detect + stop hysteresis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `pytest` without an installed ament package overlay.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.aruco import detect_markers, should_stop_for_markers  # noqa: E402
from inference.modules.aruco.stop_logic import (  # noqa: E402
    STOP_MARKER_IDS,
    reset_stop_logic,
)


@pytest.fixture(autouse=True)
def _reset_debouncer():
    reset_stop_logic()
    yield
    reset_stop_logic()


def test_stop_requires_enter_duration():
    # First sighting: arm timer, expose ID, do not stop yet.
    assert should_stop_for_markers([3], now=0.0) == (False, 3)
    # Still inside enter window (0.15s).
    assert should_stop_for_markers([3], now=0.10) == (False, 3)
    # Past ENTER (0.15s) continuous presence → stop.
    assert should_stop_for_markers([3], now=0.15) == (True, 3)


def test_non_stop_ids_ignored():
    assert should_stop_for_markers([0, 1, 2], now=0.0) == (False, None)
    assert should_stop_for_markers([0, 1, 2], now=1.0) == (False, None)


def test_brief_dropout_does_not_release():
    should_stop_for_markers([3], now=0.0)
    should_stop_for_markers([3], now=0.20)
    # Flicker for under 1.5s must keep stop.
    assert should_stop_for_markers([], now=0.30) == (True, 3)
    assert should_stop_for_markers([], now=1.70) == (True, 3)
    # Marker visible again resets absent timer.
    assert should_stop_for_markers([3], now=1.80) == (True, 3)
    assert should_stop_for_markers([], now=1.90) == (True, 3)
    assert should_stop_for_markers([], now=3.39) == (True, 3)


def test_exit_after_absent_duration():
    should_stop_for_markers([3], now=0.0)
    should_stop_for_markers([3], now=0.20)
    assert should_stop_for_markers([], now=0.30) == (True, 3)
    # 1.5s continuous absence → release.
    assert should_stop_for_markers([], now=1.80) == (False, None)


def test_stop_marker_id_constant():
    assert STOP_MARKER_IDS == frozenset({3})


cv2 = pytest.importorskip('cv2')


def _competition_stop_frame(size: int = 400, border: int = 40) -> np.ndarray:
    """Synthetic frame matching data/ArUco_stop.png (DICT_6X6_50 ID 3)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    try:
        marker = cv2.aruco.generateImageMarker(dictionary, 3, size)
    except AttributeError:
        marker = cv2.aruco.drawMarker(dictionary, 3, size)
    canvas = np.full((size + 2 * border, size + 2 * border), 255, dtype=np.uint8)
    canvas[border : border + size, border : border + size] = marker
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def test_detect_markers_on_competition_id():
    frame = _competition_stop_frame()
    assert detect_markers(frame) == [3]


def test_detect_empty_frame():
    assert detect_markers(np.zeros((0, 0), dtype=np.uint8)) == []
    assert detect_markers(np.full((64, 64, 3), 255, dtype=np.uint8)) == []


def test_facade_detect_pipeline(monkeypatch):
    from inference.modules import aruco_detection
    from inference.modules.aruco import stop_logic

    clock = {'t': 0.0}

    def fake_monotonic():
        return clock['t']

    monkeypatch.setattr(stop_logic.time, 'monotonic', fake_monotonic)
    reset_stop_logic()
    frame = _competition_stop_frame()

    clock['t'] = 0.0
    first = aruco_detection.detect(frame)
    assert first.detected is True
    assert first.should_stop is False
    assert first.marker_id == 3

    clock['t'] = 0.20
    second = aruco_detection.detect(frame)
    assert second.detected is True
    assert second.should_stop is True
    assert second.marker_id == 3
