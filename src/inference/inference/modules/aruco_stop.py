"""ArUco 정지 마커 인식 → 정지 판단.

참조본(2026-SMH .../modules/aruco: detector + stop_logic)을 team-new 로 이식·통합.
대회 정지 마커: data/ArUco_stop.png = **DICT_6X6_50 ID 3**.

동작
----
- cv2.aruco 로 마커 ID 검출 (OpenCV<4.7 legacy / >=4.7 신 API 모두 지원).
- **벽시계(초) 히스테리시스**로 깜빡임 억제 (FPS 무관):
    ENTER: 짧게(0.15s) — 오탐만 거르고 빠르게 정지
    EXIT : 길게(1.5s)  — 기울기·조명으로 수 프레임 놓쳐도 재출발 안 함
- (옵션) 마커 변 길이(px) 하한으로 **거리 게이팅** — 가까이 왔을 때만 정지.

사용:
    aruco = ArucoStopDetector()
    should_stop, marker_id = aruco.stop(frame)   # should_stop=True 면 정지
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# 대회 정지 마커
DEFAULT_STOP_IDS = (3,)
DEFAULT_DICT_ID = 6  # cv2.aruco.DICT_6X6_50 (상수값). 아래서 cv2 로 재확인.


def _resolve_dict_id(dict_id) -> int:
    if cv2 is not None and hasattr(cv2, "aruco"):
        return int(getattr(cv2.aruco, "DICT_6X6_50", dict_id))
    return int(dict_id)


def _build_detector(dict_id: int):
    """OpenCV 버전에 맞는 detector 반환 (신 API 객체 또는 (dict, params) 튜플)."""
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    try:
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params)
    except AttributeError:
        # OpenCV < 4.7 (Ubuntu 22.04 apt python3-opencv 4.5.x) legacy API.
        params = cv2.aruco.DetectorParameters_create()
        return (dictionary, params)


def _max_side_px(corner: np.ndarray) -> float:
    """마커 4모서리 → 최대 변 길이(px). 거리 게이팅용(가까울수록 큼)."""
    pts = np.asarray(corner, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return 0.0
    d = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
    return max(d)


class ArucoStopDetector:
    """마커 검출 + 히스테리시스 정지 판단. 상태 보유 → reset() 로 초기화."""

    def __init__(self, params: dict | None = None):
        p = params or {}
        ids = p.get("stop_marker_ids", DEFAULT_STOP_IDS)
        self.stop_ids = frozenset(int(i) for i in ids)
        self.enter_seconds = float(p.get("enter_stop_seconds", 0.15))
        self.exit_seconds = float(p.get("exit_stop_seconds", 1.5))
        # 마커 변 길이(px) 하한. 0 이면 거리 무관(감지 즉시 판단). 실차 튜닝값.
        self.min_marker_px = float(p.get("min_marker_px", 0.0))
        self._detector = None
        if cv2 is not None and hasattr(cv2, "aruco"):
            self._detector = _build_detector(_resolve_dict_id(DEFAULT_DICT_ID))
        self.reset()

    def reset(self) -> None:
        self._stopped = False
        self._last_id: Optional[int] = None
        self._present_since: Optional[float] = None
        self._absent_since: Optional[float] = None

    # ------------------------------------------------------------------ detect
    def detect_ids(self, frame) -> List[int]:
        """프레임에서 (거리 게이팅 통과한) 정렬된 마커 ID 목록."""
        if self._detector is None or frame is None or getattr(frame, "size", 0) == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        det = self._detector
        if isinstance(det, tuple):
            dictionary, params = det
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        else:
            corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            return []
        out = []
        for mid, corner in zip(ids.flatten(), corners):
            if self.min_marker_px > 0 and _max_side_px(corner) < self.min_marker_px:
                continue
            out.append(int(mid))
        return sorted(set(out))

    # ------------------------------------------------------------------ stop
    def stop(self, frame, now: float | None = None) -> Tuple[bool, Optional[int]]:
        """프레임 → (should_stop, marker_id). now 는 테스트용(단조 초)."""
        ids = self.detect_ids(frame)
        stop_ids = [i for i in ids if i in self.stop_ids]
        return self._debounce(stop_ids, time.monotonic() if now is None else now)

    def update_ids(self, marker_ids: List[int], now: float | None = None):
        """이미 검출된 ID 목록으로 정지 판단(테스트/외부 검출 재사용)."""
        stop_ids = [i for i in marker_ids if i in self.stop_ids]
        return self._debounce(stop_ids, time.monotonic() if now is None else now)

    def _debounce(self, stop_ids: List[int], now: float) -> Tuple[bool, Optional[int]]:
        if stop_ids:
            self._last_id = stop_ids[0]
            self._absent_since = None
            if self._present_since is None:
                self._present_since = now
            if not self._stopped and (now - self._present_since) >= self.enter_seconds:
                self._stopped = True
            return self._stopped, self._last_id
        # 미검출
        self._present_since = None
        if self._absent_since is None:
            self._absent_since = now
        if self._stopped and (now - self._absent_since) >= self.exit_seconds:
            self._stopped = False
            self._last_id = None
            self._absent_since = None
        return self._stopped, (self._last_id if self._stopped else None)
