"""ArucoStopDetector 테스트 — 히스테리시스(시간 주입) + 실제 마커 검출."""
import sys
from pathlib import Path

import numpy as np
import pytest

_WS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_WS / "src" / "inference"))

import cv2  # noqa: E402
from inference.modules.aruco_stop import ArucoStopDetector  # noqa: E402


# ---------------------------------------------------------------- 히스테리시스
def test_enter_needs_sustained_detection():
    """정지 마커가 enter_seconds 이상 연속 검출돼야 정지."""
    d = ArucoStopDetector({"enter_stop_seconds": 0.15, "exit_stop_seconds": 1.5})
    s, _ = d.update_ids([3], now=0.0)
    assert s is False                      # 방금 봄 → 아직 정지 아님
    s, mid = d.update_ids([3], now=0.2)    # 0.2s 연속 → 정지
    assert s is True and mid == 3


def test_exit_needs_sustained_absence():
    """정지 후 exit_seconds 이상 연속 미검출돼야 재출발."""
    d = ArucoStopDetector({"enter_stop_seconds": 0.15, "exit_stop_seconds": 1.5})
    d.update_ids([3], now=0.0)
    s, _ = d.update_ids([3], now=0.2)
    assert s is True
    s, _ = d.update_ids([], now=0.5)       # 잠깐 놓침(0.3s) → 정지 유지
    assert s is True
    s, _ = d.update_ids([], now=2.0)       # 1.5s+ 미검출 → 재출발
    assert s is False


def test_non_stop_marker_ignored():
    """정지 대상(ID 3)이 아닌 마커는 정지시키지 않는다."""
    d = ArucoStopDetector({"enter_stop_seconds": 0.1})
    d.update_ids([7], now=0.0)
    s, _ = d.update_ids([7, 5], now=0.5)
    assert s is False


def test_flicker_absence_does_not_resume_early():
    """짧게 놓쳐도(<exit_seconds) 정지 유지."""
    d = ArucoStopDetector({"enter_stop_seconds": 0.1, "exit_stop_seconds": 1.0})
    d.update_ids([3], now=0.0)
    assert d.update_ids([3], now=0.2)[0] is True
    for t in (0.3, 0.5, 0.9):              # 0.7s 미검출(1.0 미만)
        assert d.update_ids([], now=t)[0] is True


# ------------------------------------------------------------ 실제 마커 검출
def _marker_frame(marker_id=3, size=160, canvas=(240, 320)):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    try:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    except AttributeError:
        marker = cv2.aruco.drawMarker(dictionary, marker_id, size)
    frame = np.full((canvas[0], canvas[1]), 255, np.uint8)   # 흰 배경(quiet zone)
    y = (canvas[0] - size) // 2
    x = (canvas[1] - size) // 2
    frame[y:y + size, x:x + size] = marker
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_detects_real_marker_id3():
    """합성 DICT_6X6_50 ID3 마커를 실제로 검출한다."""
    d = ArucoStopDetector()
    ids = d.detect_ids(_marker_frame(3))
    assert 3 in ids


def test_real_marker_triggers_stop():
    d = ArucoStopDetector({"enter_stop_seconds": 0.1})
    frame = _marker_frame(3)
    d.stop(frame, now=0.0)
    s, mid = d.stop(frame, now=0.3)
    assert s is True and mid == 3


def test_min_marker_px_gating():
    """min_marker_px 하한보다 작은 마커는 무시(거리 게이팅)."""
    d = ArucoStopDetector({"min_marker_px": 400.0})   # 160px 마커보다 큼
    ids = d.detect_ids(_marker_frame(3, size=160))
    assert 3 not in ids
    d2 = ArucoStopDetector({"min_marker_px": 50.0})   # 160px 마커보다 작음
    assert 3 in d2.detect_ids(_marker_frame(3, size=160))


def test_empty_frame_no_stop():
    d = ArucoStopDetector({"enter_stop_seconds": 0.1})
    blank = np.zeros((180, 320, 3), np.uint8)
    d.stop(blank, now=0.0)
    assert d.stop(blank, now=0.5)[0] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
