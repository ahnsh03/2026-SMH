"""Traffic light color detection — 담당: 장원정."""

from __future__ import annotations

import cv2
import numpy as np

from inference.types import TrafficSignal

# HSV thresholds tuned for lit traffic-light lenses (bright, saturated pixels).
# Red wraps around hue 0, so it needs two ranges.
# Retuned against bag_20260715_230145 / bag_20260715_230316 (2026-07-16, low-light).
_RED_RANGES = (
    ((0, 110, 100), (8, 255, 255)),
    ((172, 110, 100), (180, 255, 255)),
)
_GREEN_RANGE = ((60, 100, 100), (88, 255, 255))

_MIN_RED_PIXELS = 40
_MIN_GREEN_PIXELS = 40
_MIN_CIRCULARITY = 0.7
_MAX_ASPECT_RATIO = 1.5
_MORPH_KERNEL = np.ones((5, 5), dtype=np.uint8)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Join fragmented light pixels and remove tiny speckles."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)


def _mask_pixels(mask: np.ndarray) -> int:
    """Return the number of selected pixels in a binary mask."""
    return int(cv2.countNonZero(mask))


def _has_loose_round_blob(mask: np.ndarray) -> bool:
    """Return True when the largest mask blob is plausibly traffic-light shaped."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > best_area:
            best_area = area
            best_contour = contour
    if best_contour is None:
        return False

    perimeter = cv2.arcLength(best_contour, True)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * best_area / (perimeter**2)

    _, _, width, height = cv2.boundingRect(best_contour)
    if width == 0 or height == 0:
        return False
    aspect_ratio = max(width, height) / min(width, height)

    return circularity >= _MIN_CIRCULARITY and aspect_ratio <= _MAX_ASPECT_RATIO


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

    red_ok = red_pixels >= _MIN_RED_PIXELS and _has_loose_round_blob(red_mask)
    green_ok = green_pixels >= _MIN_GREEN_PIXELS and _has_loose_round_blob(green_mask)

    if red_ok:
        return TrafficSignal.RED
    if green_ok:
        return TrafficSignal.GREEN
    return TrafficSignal.UNKNOWN
