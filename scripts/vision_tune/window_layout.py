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
