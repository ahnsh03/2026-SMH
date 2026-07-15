#!/usr/bin/env python3
"""Keep OpenCV tuner windows inside the visible desktop (WSL / multi-monitor)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Sequence

import cv2

_ORIGIN_X = 48
_ORIGIN_Y = 48
_GAP = 12
_FALLBACK_W = 1280
_FALLBACK_H = 720


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, '').strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def visible_work_area() -> tuple[int, int, int, int]:
    ox = _env_int('VISION_TUNE_ORIGIN_X')
    oy = _env_int('VISION_TUNE_ORIGIN_Y')
    mw = _env_int('VISION_TUNE_MAX_W')
    mh = _env_int('VISION_TUNE_MAX_H')
    screen_w, screen_h = _probe_screen_size()
    if ox is None:
        ox = _ORIGIN_X
    if oy is None:
        oy = _ORIGIN_Y
    if mw is None:
        mw = max(640, min(screen_w - ox - 24, screen_w - 96))
    if mh is None:
        mh = max(360, min(screen_h - oy - 48, screen_h - 96))
    return ox, oy, mw, mh


def _probe_screen_size() -> tuple[int, int]:
    if shutil.which('xdpyinfo'):
        try:
            out = subprocess.check_output(
                ['xdpyinfo'],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.5,
            )
            m = re.search(r'dimensions:\s+(\d+)x(\d+)', out)
            if m:
                return int(m.group(1)), int(m.group(2))
        except (OSError, subprocess.SubprocessError):
            pass
    return _FALLBACK_W, _FALLBACK_H


def place_window(name: str, x: int, y: int) -> None:
    try:
        cv2.moveWindow(name, int(x), int(y))
    except cv2.error:
        pass


def map_and_place_window(
    name: str,
    frame: object | None = None,
    *,
    x: int | None = None,
    y: int | None = None,
    pumps: int = 3,
) -> None:
    """Show (optional) frame then re-place — WSLg often ignores move entirely.

    On multi-monitor WSLg the compositor places new GTK/OpenCV windows on the
    secondary half of a wide virtual desktop (e.g. x≈2000 on a 4000-wide
    screen). ``moveWindow`` / xdotool ConfigureRequests are dropped, so we
    only retry briefly and rely on :func:`diagnose_window_placement`.
    """

    ox, oy, _, _ = visible_work_area()
    if x is None:
        x = ox
    if y is None:
        y = oy
    if frame is not None:
        cv2.imshow(name, frame)  # type: ignore[arg-type]
    for _ in range(max(1, int(pumps))):
        cv2.waitKey(1)
        place_window(name, int(x), int(y))
        _xdotool_move(name, int(x), int(y))


def window_geometry(title_substr: str) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) via xdotool — more reliable than xwininfo on WSLg."""

    if not shutil.which('xdotool'):
        return None
    try:
        search = subprocess.run(
            ['xdotool', 'search', '--name', _xdotool_search_pattern(title_substr)],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        wids = [ln.strip() for ln in search.stdout.splitlines() if ln.strip()]
        if not wids:
            return None
        geo = subprocess.run(
            ['xdotool', 'getwindowgeometry', wids[0]],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        # Position: 2087,126 (screen: 0) / Geometry: 1268x400
        pm = re.search(r'Position:\s*(-?\d+),(-?\d+)', geo.stdout)
        gm = re.search(r'Geometry:\s*(\d+)x(\d+)', geo.stdout)
        if not pm or not gm:
            return None
        return int(pm.group(1)), int(pm.group(2)), int(gm.group(1)), int(gm.group(2))
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def diagnose_window_placement(title_substr: str) -> str:
    """Human-readable placement note for WSLg multi-monitor quirks."""

    screen_w, screen_h = _probe_screen_size()
    geo = window_geometry(title_substr)
    if geo is None:
        return (
            f'window "{title_substr}" not found yet '
            f'(display {screen_w}x{screen_h})'
        )
    x, y, w, h = geo
    # WSLg exposes all Windows monitors as one wide X screen. New OpenCV
    # windows often land near / past the mid-point (second monitor).
    primary_guess = screen_w // 2 if screen_w >= 3000 else screen_w
    left_overlap = max(0, min(x + w, primary_guess) - max(x, 0))
    mostly_off_primary = left_overlap < (w * 0.45)
    line = (
        f'window at xdotool=({x},{y}) size={w}x{h} '
        f'display={screen_w}x{screen_h}'
    )
    if mostly_off_primary or x < -100 or y < -100:
        line += (
            ' — OFF PRIMARY / hard to see. '
            'WSLg ignores programmatic move; press Win+Shift+Left '
            'or drag from the Windows taskbar. Live PNG fallback is safer.'
        )
    return line


def _xdotool_search_pattern(title_substr: str) -> str:
    """Escape regex metacharacters so titles like 'a (b | c)' still match."""

    return re.escape(title_substr)


def _xdotool_move(title_substr: str, x: int, y: int) -> None:
    if not shutil.which('xdotool'):
        return
    try:
        # Never use --sync: WSLg often never acks ConfigureRequest (hangs).
        subprocess.run(
            [
                'xdotool',
                'search',
                '--name',
                _xdotool_search_pattern(title_substr),
                'windowmove',
                str(x),
                str(y),
                'windowactivate',
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.5,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def place_windows(
    names: Sequence[str],
    *,
    widths: Sequence[int] | None = None,
    heights: Sequence[int] | None = None,
    origin_x: int | None = None,
    origin_y: int | None = None,
    max_row_width: int | None = None,
) -> None:
    ox, oy, default_mw, max_h = visible_work_area()
    if origin_x is None:
        origin_x = ox
    if origin_y is None:
        origin_y = oy
    if max_row_width is None:
        max_row_width = default_mw
    x = origin_x
    y = origin_y
    row_h = 0
    for i, name in enumerate(names):
        w = int(widths[i]) if widths is not None and i < len(widths) else 640
        h = int(heights[i]) if heights is not None and i < len(heights) else 360
        if x > origin_x and x + w > origin_x + max_row_width:
            x = origin_x
            y += row_h + _GAP
            row_h = 0
        place_window(name, x, min(y, origin_y + max(0, max_h - 80)))
        x += w + _GAP
        row_h = max(row_h, h)
