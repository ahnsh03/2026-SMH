"""ArUco stop decision — 담당: 박성준"""

from __future__ import annotations


def should_stop_for_markers(marker_ids: list[int]) -> tuple[bool, int | None]:
    """
    Decide whether the vehicle should stop for detected markers.

    Returns (should_stop, primary_marker_id).
    """
    _ = marker_ids
    return False, None
