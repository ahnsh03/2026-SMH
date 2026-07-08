"""ArUco marker detection — 담당: 안승현"""

from __future__ import annotations

import cv2
import numpy as np

# 대회 측이 실제 사용하는 dictionary가 공식 문서/저장소에 명시되어 있지 않아,
# 동적 장애물 미션에서는 "마커 존재 여부"만 중요하다는 전제로 흔히 쓰이는
# 4x4/5x5/6x6 계열을 순차 조회한다. 실물 마커(ArUco_stop.png)로 실제
# dictionary가 확인되면 아래 후보를 하나로 좁혀 검출 비용을 줄일 것.
_CANDIDATE_DICT_IDS = (
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_6X6_250,
)


def _build_detector(dict_id: int):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    try:
        parameters = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    except AttributeError:
        # OpenCV < 4.7 (e.g. Ubuntu 22.04 apt python3-opencv) legacy API.
        parameters = cv2.aruco.DetectorParameters_create()
        return (dictionary, parameters)


_DETECTORS = [_build_detector(dict_id) for dict_id in _CANDIDATE_DICT_IDS]


def _detect_with(detector, gray: np.ndarray) -> list[int]:
    if isinstance(detector, tuple):
        dictionary, parameters = detector
        _corners, ids, _rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    else:
        _corners, ids, _rejected = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [int(marker_id) for marker_id in ids.flatten()]


def detect_markers(frame: np.ndarray) -> list[int]:
    """
    Detect ArUco marker IDs visible in the frame.

    Returns a list of detected marker IDs (empty if none).
    """
    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    found: set[int] = set()
    for detector in _DETECTORS:
        found.update(_detect_with(detector, gray))
    return sorted(found)
