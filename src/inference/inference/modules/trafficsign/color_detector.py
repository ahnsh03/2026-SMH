"""Traffic light color detection — 담당: 장원정."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

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
_MIN_CIRCULARITY = 0.35
_MAX_ASPECT_RATIO = 3.0
_MORPH_KERNEL = np.ones((5, 5), dtype=np.uint8)


@dataclass(frozen=True)
class BlobInfo:
    """One connected component in a red/green mask."""

    color: str
    area: float
    circularity: float
    aspect_ratio: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    centroid: tuple[float, float]
    shape_ok: bool
    is_largest: bool


@dataclass
class SignalInspect:
    """Full OpenCV traffic-light diagnostics for one camera frame."""

    signal: TrafficSignal
    red_mask: np.ndarray
    green_mask: np.ndarray
    red_pixels: int
    green_pixels: int
    red_ok: bool
    green_ok: bool
    blobs: list[BlobInfo] = field(default_factory=list)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Join fragmented light pixels and remove tiny speckles."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)


def _mask_pixels(mask: np.ndarray) -> int:
    """Return the number of selected pixels in a binary mask."""
    return int(cv2.countNonZero(mask))


def _blob_infos(mask: np.ndarray, *, color: str) -> list[BlobInfo]:
    """Score every external contour (largest first)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    infos: list[BlobInfo] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 1.0:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            (4.0 * np.pi * area / (perimeter**2)) if perimeter > 0.0 else 0.0
        )
        x, y, width, height = cv2.boundingRect(contour)
        if width == 0 or height == 0:
            continue
        aspect_ratio = max(width, height) / min(width, height)
        m = cv2.moments(contour)
        if m['m00'] > 0:
            cx = float(m['m10'] / m['m00'])
            cy = float(m['m01'] / m['m00'])
        else:
            cx = float(x + width / 2)
            cy = float(y + height / 2)
        shape_ok = (
            circularity >= _MIN_CIRCULARITY and aspect_ratio <= _MAX_ASPECT_RATIO
        )
        infos.append(
            BlobInfo(
                color=color,
                area=area,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                bbox=(int(x), int(y), int(width), int(height)),
                centroid=(cx, cy),
                shape_ok=shape_ok,
                is_largest=False,
            )
        )
    infos.sort(key=lambda b: b.area, reverse=True)
    if infos:
        top = infos[0]
        infos[0] = BlobInfo(
            color=top.color,
            area=top.area,
            circularity=top.circularity,
            aspect_ratio=top.aspect_ratio,
            bbox=top.bbox,
            centroid=top.centroid,
            shape_ok=top.shape_ok,
            is_largest=True,
        )
    return infos


def _has_loose_round_blob(mask: np.ndarray) -> bool:
    """Return True when the largest mask blob is plausibly traffic-light shaped."""
    infos = _blob_infos(mask, color='any')
    if not infos:
        return False
    return infos[0].shape_ok


def inspect_signal(
    frame: np.ndarray,
    *,
    red_ranges: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]] | None = None,
    green_range: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    min_red_pixels: int | None = None,
    min_green_pixels: int | None = None,
) -> SignalInspect:
    """
    Run the same pipeline as ``detect_signal`` and return masks + blob stats.

    Optional HSV / pixel overrides are for offline trackbar tuning only.
    """
    empty = np.zeros((1, 1), dtype=np.uint8)
    if frame is None or getattr(frame, 'size', 0) == 0:
        return SignalInspect(
            signal=TrafficSignal.UNKNOWN,
            red_mask=empty,
            green_mask=empty,
            red_pixels=0,
            green_pixels=0,
            red_ok=False,
            green_ok=False,
        )

    red_ranges = tuple(red_ranges) if red_ranges is not None else _RED_RANGES
    green_range = green_range if green_range is not None else _GREEN_RANGE
    min_red = int(min_red_pixels if min_red_pixels is not None else _MIN_RED_PIXELS)
    min_green = int(
        min_green_pixels if min_green_pixels is not None else _MIN_GREEN_PIXELS
    )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in red_ranges:
        red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    green_mask = cv2.inRange(
        hsv, np.array(green_range[0]), np.array(green_range[1])
    )
    red_mask = _clean_mask(red_mask)
    green_mask = _clean_mask(green_mask)

    red_pixels = _mask_pixels(red_mask)
    green_pixels = _mask_pixels(green_mask)

    red_blobs = _blob_infos(red_mask, color='red')
    green_blobs = _blob_infos(green_mask, color='green')

    # Match detect_signal: total pixels + largest-blob shape only.
    red_ok = red_pixels >= min_red and bool(red_blobs and red_blobs[0].shape_ok)
    green_ok = green_pixels >= min_green and bool(
        green_blobs and green_blobs[0].shape_ok
    )

    if red_ok:
        signal = TrafficSignal.RED
    elif green_ok:
        signal = TrafficSignal.GREEN
    else:
        signal = TrafficSignal.UNKNOWN

    return SignalInspect(
        signal=signal,
        red_mask=red_mask,
        green_mask=green_mask,
        red_pixels=red_pixels,
        green_pixels=green_pixels,
        red_ok=red_ok,
        green_ok=green_ok,
        blobs=red_blobs + green_blobs,
    )


def detect_signal(frame: np.ndarray) -> TrafficSignal:
    """
    Detect the lit traffic-light color in a BGR frame.

    Looks for enough bright, saturated pixels matching the red or green lens
    color. An unlit lens (low saturation/brightness) or no light in frame both
    yield UNKNOWN.
    """
    return inspect_signal(frame).signal
