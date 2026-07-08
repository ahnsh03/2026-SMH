"""ArUco marker detection — 담당: 안승현 (박성준 공동 개발)."""

from __future__ import annotations

import cv2
import numpy as np

# data/ArUco_stop.png 실측: DICT_6X6_50 ID 3 (pixel-exact vs generateImageMarker).
# 6X6_100/250/1000도 ID 3은 동일 비트이지만 최소 사전으로 고정한다.
_ARUCO_DICT_ID = cv2.aruco.DICT_6X6_50


def _build_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(_ARUCO_DICT_ID)
    try:
        parameters = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        # OpenCV < 4.7 (e.g. Ubuntu 22.04 apt python3-opencv) legacy API.
        parameters = cv2.aruco.DetectorParameters_create()
        return (dictionary, parameters)


_DETECTOR = _build_detector()


def _detect_with(detector, gray: np.ndarray) -> list[int]:
    if isinstance(detector, tuple):
        dictionary, parameters = detector
        _corners, ids, _rejected = cv2.aruco.detectMarkers(
            gray, dictionary, parameters=parameters
        )
    else:
        _corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return []
    return sorted({int(marker_id) for marker_id in ids.flatten()})


def detect_markers(frame: np.ndarray) -> list[int]:
    """
    Detect ArUco marker IDs visible in the frame.

    Returns a sorted list of unique marker IDs (empty if none).
    """
    if frame is None or getattr(frame, 'size', 0) == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return _detect_with(_DETECTOR, gray)
