"""Traffic signal debounce — persistence filter to reject brief false positives."""

from __future__ import annotations

from inference.types import TrafficSignal

# Real lit lenses hold a stable color for well over a second once in frame (45+
# consecutive raw frames measured on real bags). Brief color-alike false positives
# (clothing, road paint, lane markings glimpsed while driving past) only sustain a
# handful of frames (<=24 measured). Requiring the same raw reading to repeat for
# this many consecutive frames absorbs those without delaying a real light noticeably.
_CONFIRM_FRAMES = 15


class _SignalDebouncer:
    """Only report a raw signal once it repeats for _confirm_frames consecutive frames."""

    def __init__(self, confirm_frames: int) -> None:
        self._confirm_frames = confirm_frames
        self._candidate = TrafficSignal.UNKNOWN
        self._streak = 0
        self._confirmed = TrafficSignal.UNKNOWN

    def update(self, raw_signal: TrafficSignal) -> TrafficSignal:
        if raw_signal == self._candidate:
            self._streak += 1
        else:
            self._candidate = raw_signal
            self._streak = 1

        if self._streak >= self._confirm_frames:
            self._confirmed = self._candidate
        return self._confirmed


_debouncer = _SignalDebouncer(_CONFIRM_FRAMES)


def debounce_signal(raw_signal: TrafficSignal) -> TrafficSignal:
    """Smooth a raw per-frame TrafficSignal into a persistence-confirmed one."""
    return _debouncer.update(raw_signal)
