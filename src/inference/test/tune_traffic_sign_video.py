"""Interactive HSV tuner for traffic-light videos.

The video loops like a bag replay. It shows the red mask and detector output
as separate windows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.types import TrafficSignal  # noqa: E402
from inference.modules.trafficsign.color_detector import (  # noqa: E402
    _MIN_RED_PIXELS,
    _RED_RANGES,
)


WINDOW = 'detect result'
MASK_WINDOW = 'red mask'
MORPH_KERNEL = np.ones((5, 5), dtype=np.uint8)


def _noop(_: int) -> None:
    pass


def _make_controls() -> None:
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

    red1_lower, red1_upper = _RED_RANGES[0]
    red2_lower, _ = _RED_RANGES[1]
    controls = {
        'red1_h_hi': red1_upper[0],
        'red2_h_lo': red2_lower[0],
        'sat_min': red1_lower[1],
        'val_min': red1_lower[2],
        'min_pixels': _MIN_RED_PIXELS,
        'roi_top_pct': 0,
        'roi_bottom_pct': 100,
    }
    limits = {
        'red1_h_hi': 180,
        'red2_h_lo': 180,
        'sat_min': 255,
        'val_min': 255,
        'min_pixels': 20000,
        'roi_top_pct': 100,
        'roi_bottom_pct': 100,
    }
    for name, value in controls.items():
        cv2.createTrackbar(name, WINDOW, value, limits[name], _noop)


def _get_controls() -> dict[str, int]:
    names = (
        'red1_h_hi',
        'red2_h_lo',
        'sat_min',
        'val_min',
        'min_pixels',
        'roi_top_pct',
        'roi_bottom_pct',
    )
    return {name: cv2.getTrackbarPos(name, WINDOW) for name in names}


def _best_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_area = 0.0
    best_box = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= best_area:
            continue
        best_area = area
        best_box = cv2.boundingRect(contour)
    return best_box


def _detect_red(frame: np.ndarray, controls: dict[str, int]) -> tuple[TrafficSignal, np.ndarray, int, tuple[int, int, int, int] | None]:
    height = frame.shape[0]
    top = int(height * controls['roi_top_pct'] / 100)
    bottom = int(height * controls['roi_bottom_pct'] / 100)
    bottom = max(top + 1, min(height, bottom))

    roi = frame[top:bottom]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat_min = controls['sat_min']
    val_min = controls['val_min']
    red1_h_hi = controls['red1_h_hi']
    red2_h_lo = controls['red2_h_lo']

    mask1 = cv2.inRange(hsv, np.array((0, sat_min, val_min)), np.array((red1_h_hi, 255, 255)))
    mask2 = cv2.inRange(hsv, np.array((red2_h_lo, sat_min, val_min)), np.array((180, 255, 255)))
    roi_mask = mask1 | mask2
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[top:bottom] = roi_mask

    red_pixels = int(cv2.countNonZero(mask))
    signal = TrafficSignal.RED if red_pixels >= controls['min_pixels'] else TrafficSignal.UNKNOWN
    return signal, mask, red_pixels, _best_box(mask)


def _draw(
    frame: np.ndarray,
    signal: TrafficSignal,
    red_pixels: int,
    box: tuple[int, int, int, int] | None,
    frame_idx: int,
    controls: dict[str, int],
) -> np.ndarray:
    annotated = frame.copy()
    color = (0, 0, 255) if signal == TrafficSignal.RED else (180, 180, 180)
    if box is not None:
        x, y, w, h = box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    label = 'RED' if signal == TrafficSignal.RED else 'UNKNOWN'
    cv2.putText(annotated, label, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
    detail = f'frame={frame_idx} red_pixels={red_pixels}'
    cv2.putText(annotated, detail, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    pixel_status = 'OK' if red_pixels >= controls['min_pixels'] else 'FAIL'
    reason = f'pixels {pixel_status} >= {controls["min_pixels"]}'
    cv2.putText(annotated, reason, (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return annotated


def _print_constants(controls: dict[str, int]) -> None:
    print('Candidate constants:')
    print('_RED_RANGES = (')
    print(f'    ((0, {controls["sat_min"]}, {controls["val_min"]}), ({controls["red1_h_hi"]}, 255, 255)),')
    print(f'    (({controls["red2_h_lo"]}, {controls["sat_min"]}, {controls["val_min"]}), (180, 255, 255)),')
    print(')')
    print(f'_MIN_RED_PIXELS = {controls["min_pixels"]}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Loop a video while tuning red traffic-light HSV thresholds.')
    parser.add_argument('--video', required=True, help='Input video path.')
    parser.add_argument('--delay-ms', type=int, default=20, help='Playback delay between frames.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.video}')

    _make_controls()
    frame_idx = 0
    paused = False
    last_controls: dict[str, int] | None = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

        controls = _get_controls()
        signal, mask, red_pixels, box = _detect_red(frame, controls)
        annotated = _draw(frame, signal, red_pixels, box, frame_idx, controls)
        cv2.imshow(MASK_WINDOW, mask)
        cv2.imshow(WINDOW, annotated)

        key = cv2.waitKey(args.delay_ms) & 0xFF
        if key == ord('q') or key == 27:
            last_controls = controls
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('p'):
            _print_constants(controls)

        if not paused:
            frame_idx += 1

    if last_controls is not None:
        _print_constants(last_controls)
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
