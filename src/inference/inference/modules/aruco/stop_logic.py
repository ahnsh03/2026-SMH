"""ArUco stop decision — 담당: 박성준 (안승현 공동 개발)."""

from __future__ import annotations

import time

# 대회 정지 마커: data/ArUco_stop.png → DICT_6X6_50 ID 3.
STOP_MARKER_IDS: frozenset[int] = frozenset({3})

# ---------------------------------------------------------------------------
# [실차 튜닝] TUNABLE — 보드/트랙 테스트 때 여기만 조정하면 됨.
# 히스테리시스는 벽시계(초) 기준 (FPS 무관).
#   ENTER: 짧게 — 깜빡 오탐만 걸러내고 빨리 정지
#   EXIT : 길게 — 기울기·조명으로 수 프레임 놓쳐도 출발하지 않음
#          (정지 인지 후 스탑워치 중지 → 해제 여유 두는 편이 유리)
# 후보: EXIT 1.0(공격) / 1.5(기본) / 2.0(안전)
# ---------------------------------------------------------------------------
_ENTER_STOP_SECONDS = 0.15  # [실차 튜닝] 정지 진입에 필요한 연속 검출 시간(초)
_EXIT_STOP_SECONDS = 1.5  # [실차 튜닝] 재출발에 필요한 연속 미검출 시간(초)


class _StopDebouncer:
    """Wall-clock hysteresis over stop-relevant marker IDs."""

    def __init__(self, enter_seconds: float, exit_seconds: float) -> None:
        self._enter_seconds = enter_seconds
        self._exit_seconds = exit_seconds
        self.reset()

    def reset(self) -> None:
        self._stopped = False
        self._last_marker_id: int | None = None
        self._present_since: float | None = None
        self._absent_since: float | None = None

    def update(
        self,
        marker_ids: list[int],
        *,
        now: float | None = None,
    ) -> tuple[bool, int | None]:
        now = time.monotonic() if now is None else now
        stop_ids = [mid for mid in marker_ids if mid in STOP_MARKER_IDS]

        if stop_ids:
            self._last_marker_id = stop_ids[0]
            self._absent_since = None
            if self._present_since is None:
                self._present_since = now
            if (
                not self._stopped
                and (now - self._present_since) >= self._enter_seconds
            ):
                self._stopped = True
            # 아직 should_stop 전이어도 검출 중이면 ID를 노출 (보드 /debug/aruco 확인용)
            return self._stopped, self._last_marker_id

        self._present_since = None
        if self._absent_since is None:
            self._absent_since = now
        if (
            self._stopped
            and (now - self._absent_since) >= self._exit_seconds
        ):
            self._stopped = False
            self._last_marker_id = None
            self._absent_since = None

        return self._stopped, (self._last_marker_id if self._stopped else None)


_debouncer = _StopDebouncer(_ENTER_STOP_SECONDS, _EXIT_STOP_SECONDS)


def reset_stop_logic() -> None:
    """Clear hysteresis state (tests / node restart)."""
    _debouncer.reset()


def should_stop_for_markers(
    marker_ids: list[int],
    *,
    now: float | None = None,
) -> tuple[bool, int | None]:
    """
    Decide whether the vehicle should stop for detected markers.

    Only STOP_MARKER_IDS (competition stop pole) trigger a stop.
    Returns (should_stop, primary_marker_id).

    ``now`` is for tests (injected monotonic seconds); production leaves it None.
    """
    return _debouncer.update(marker_ids, now=now)
