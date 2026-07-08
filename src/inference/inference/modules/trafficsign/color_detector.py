"""Traffic light color detection — 담당: 장원정."""

from __future__ import annotations

import cv2
import numpy as np

from inference.types import TrafficSignal

# HSV thresholds tuned for lit traffic-light lenses (bright, saturated pixels).
# Red wraps around hue 0, so it needs two ranges.
_RED_RANGES = (
    ((0, 158, 125), (18, 255, 255)),
    ((169, 158, 125), (180, 255, 255)),
)
_GREEN_RANGE = ((40, 80, 120), (90, 255, 255))

_MIN_RED_PIXELS = 50
_MIN_GREEN_PIXELS = 50
_MORPH_KERNEL = np.ones((5, 5), dtype=np.uint8)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Join fragmented light pixels and remove tiny speckles."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)


def _mask_pixels(mask: np.ndarray) -> int:
    """Return the number of selected pixels in a binary mask."""
    return int(cv2.countNonZero(mask))


def detect_signal(frame: np.ndarray) -> TrafficSignal:
    """
    Detect the lit traffic-light color in a BGR frame.

    Looks for enough bright, saturated pixels matching the red or green lens
    color. An unlit lens (low saturation/brightness) or no light in frame both
    yield UNKNOWN.
    """
    if frame is None or getattr(frame, 'size', 0) == 0:
        return TrafficSignal.UNKNOWN

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in _RED_RANGES:
        red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    green_mask = cv2.inRange(hsv, np.array(_GREEN_RANGE[0]), np.array(_GREEN_RANGE[1]))
    red_mask = _clean_mask(red_mask)
    green_mask = _clean_mask(green_mask)

    red_pixels = _mask_pixels(red_mask)
    green_pixels = _mask_pixels(green_mask)

    red_ok = red_pixels >= _MIN_RED_PIXELS
    green_ok = green_pixels >= _MIN_GREEN_PIXELS

    if red_ok:
        return TrafficSignal.RED
    if green_ok:
        return TrafficSignal.GREEN
    return TrafficSignal.UNKNOWN
