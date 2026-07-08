"""ArUco stop decision — 담당: 박성준"""

from __future__ import annotations

# 동적 장애물 폴대가 눕고 서는 과도기에 검출이 한두 프레임 흔들릴 수 있어
# 히스테리시스를 둔다. 정지 진입은 빠르게(오탐 노이즈만 걸러내는 수준),
# 정지 해제는 더 보수적으로 — 판정 정확도가 속도보다 중요하고, 정지가
# 확인되면 대회 스탑워치가 멈추므로 해제를 서두를 이유가 없다.
_ENTER_STOP_FRAMES = 2
_EXIT_STOP_FRAMES = 5


class _StopDebouncer:
    """Frame-to-frame hysteresis over detected marker IDs."""

    def __init__(self, enter_frames: int, exit_frames: int) -> None:
        self._enter_frames = enter_frames
        self._exit_frames = exit_frames
        self._present_streak = 0
        self._absent_streak = 0
        self._stopped = False
        self._last_marker_id: int | None = None

    def update(self, marker_ids: list[int]) -> tuple[bool, int | None]:
        if marker_ids:
            self._present_streak += 1
            self._absent_streak = 0
            self._last_marker_id = marker_ids[0]
        else:
            self._absent_streak += 1
            self._present_streak = 0

        if not self._stopped and self._present_streak >= self._enter_frames:
            self._stopped = True
        elif self._stopped and self._absent_streak >= self._exit_frames:
            self._stopped = False
            self._last_marker_id = None

        return self._stopped, (self._last_marker_id if self._stopped else None)


_debouncer = _StopDebouncer(_ENTER_STOP_FRAMES, _EXIT_STOP_FRAMES)


def should_stop_for_markers(marker_ids: list[int]) -> tuple[bool, int | None]:
    """
    Decide whether the vehicle should stop for detected markers.

    Returns (should_stop, primary_marker_id).
    """
    return _debouncer.update(marker_ids)
