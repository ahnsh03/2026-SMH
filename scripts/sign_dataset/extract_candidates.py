#!/usr/bin/env python3
"""Extract left/right sign candidate frames from rosbag2 recordings and
draft-label them as a YOLO detection dataset.

This is a standalone data-prep tool. It intentionally does NOT import or
modify `inference.modules.direction_sign.detector` (owned by the traffic-sign
module maintainer) — the blue-circle/white-arrow candidate search below is a
local, throwaway re-implementation used only to generate DRAFT bounding boxes
that a human should review before training. It is not part of the runtime
inference stack.

Usage:
    python3 scripts/sign_dataset/extract_candidates.py \
        --bags-root /mnt/c \
        --bags bag_20260711_144948 bag_20260711_150234 bev_bag bev_bag2 camera_only_01 camera_only_02 \
        --out dataset

Requires: rosbags, opencv-python(-headless), numpy (see scripts/sign_dataset/README.md).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

# ---------------------------------------------------------------------------
# Draft candidate search (local copy — see module docstring for why this is
# not imported from inference.modules.direction_sign.detector).
# ---------------------------------------------------------------------------

_BLUE_LOWER = np.array([90, 70, 35], dtype=np.uint8)
_BLUE_UPPER = np.array([140, 255, 255], dtype=np.uint8)
_WHITE_LOWER = np.array([0, 0, 170], dtype=np.uint8)
_WHITE_UPPER = np.array([179, 80, 255], dtype=np.uint8)
_MIN_BLUE_AREA_RATIO = 0.001
_MIN_ARROW_PIXELS = 15
_MIN_ARROW_OFFSET_RATIO = 0.012

# Traffic-light candidate search thresholds. These mirror the ranges used by
# the runtime HSV fallback (inference.modules.trafficsign.color_detector) but
# are re-implemented locally for the same reason as the sign search above:
# this is a throwaway draft-labeler, not the runtime detector.
_RED_LOWER1 = np.array([0, 158, 125], dtype=np.uint8)
_RED_UPPER1 = np.array([18, 255, 255], dtype=np.uint8)
_RED_LOWER2 = np.array([169, 158, 125], dtype=np.uint8)
_RED_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)
_GREEN_LOWER = np.array([40, 80, 120], dtype=np.uint8)
_GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)
_MIN_LIGHT_AREA_RATIO = 0.0008
_MIN_LIGHT_CIRCULARITY = 0.55
_MAX_LIGHT_ASPECT = 1.4
_MIN_LIGHT_CORE_RATIO = 0.12

LEFT, RIGHT, RED_LIGHT, GREEN_LIGHT = 0, 1, 2, 3
CLASS_NAMES = ('Left Sign', 'Right Sign', 'Red Light', 'Green Light')


@dataclass
class Candidate:
    cls: int  # LEFT=0, RIGHT=1
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


def find_sign_candidates(frame: np.ndarray) -> list[Candidate]:
    """Blue-circle + white-arrow search, returning ALL plausible boxes
    (not just the best one) so a human can pick/reject during review.
    """
    if frame is None or frame.size == 0:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, _BLUE_LOWER, _BLUE_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame.shape[0] * frame.shape[1])
    white = cv2.inRange(hsv, _WHITE_LOWER, _WHITE_UPPER)

    results: list[Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * _MIN_BLUE_AREA_RATIO:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 8 or height < 8:
            continue
        aspect = width / float(height)
        if not 0.55 <= aspect <= 1.45:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-6)
        if circularity < 0.35:
            continue

        region = np.zeros_like(blue)
        cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)
        erode_px = max(1, int(round(min(width, height) * 0.03)))
        region = cv2.erode(
            region,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)),
        )

        band_top = y + int(round(0.25 * height))
        band_bottom = y + int(round(0.58 * height))
        arrow_mask = cv2.bitwise_and(white, region)
        band = arrow_mask[band_top:band_bottom, x:x + width]
        _, columns = np.nonzero(band)
        if columns.size < _MIN_ARROW_PIXELS:
            continue
        offset = float(np.mean(columns) - (width - 1) / 2.0) / float(width)
        if abs(offset) < _MIN_ARROW_OFFSET_RATIO:
            continue

        cls = LEFT if offset < 0.0 else RIGHT
        score = area * max(circularity, 0.01) * abs(offset)
        # Pad the box slightly: the blue-circle contour hugs the disc tightly,
        # a learned detector benefits from a little context margin.
        pad = int(round(0.08 * max(width, height)))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(frame.shape[1], x + width + pad)
        y2 = min(frame.shape[0], y + height + pad)
        results.append(Candidate(cls, x1, y1, x2, y2, score))

    results.sort(key=lambda c: c.score, reverse=True)
    return results


def _light_blobs(mask: np.ndarray, value: np.ndarray, frame_area: float, cls: int) -> list[Candidate]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: list[Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * _MIN_LIGHT_AREA_RATIO:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 6 or height < 6:
            continue
        aspect = width / float(height)
        if aspect > _MAX_LIGHT_ASPECT or (1.0 / aspect) > _MAX_LIGHT_ASPECT:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-6)
        if circularity < _MIN_LIGHT_CIRCULARITY:
            continue

        # A real lit LED/lamp blows out to a bright near-white core at its
        # center (specular highlight); flat-colored cloth/plastic clutter
        # does not. Require a minimum fraction of very-high-V pixels inside
        # the blob to reject those false positives.
        region = np.zeros_like(mask)
        cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)
        core = cv2.bitwise_and(value, value, mask=region)
        hot = core > 230
        core_ratio = float(np.count_nonzero(hot)) / max(float(np.count_nonzero(region)), 1.0)
        if core_ratio < _MIN_LIGHT_CORE_RATIO:
            continue

        # The colored HSV mask often includes a dim diffuse glow/reflection
        # trail (e.g. on the floor below the lamp) attached to the actual
        # bulb. Anchor the box on the hot (near-white) core pixels only —
        # that is the physical lamp housing, not its reflection — instead of
        # the full contour's bounding rect.
        hot_ys, hot_xs = np.nonzero(hot)
        hx, hy = int(hot_xs.min()), int(hot_ys.min())
        hw, hh = int(hot_xs.max() - hx + 1), int(hot_ys.max() - hy + 1)
        x, y, width, height = hx, hy, hw, hh

        score = area * max(circularity, 0.01) * (1.0 + core_ratio)
        pad = int(round(0.6 * max(width, height))) + 3
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2 = min(mask.shape[1], x + width + pad)
        y2 = min(mask.shape[0], y + height + pad)
        results.append(Candidate(cls, x1, y1, x2, y2, score))
    return results


def find_light_candidates(frame: np.ndarray) -> list[Candidate]:
    """Red/green lit-blob search, returning ALL plausible boxes for human
    review — same draft-only role as find_sign_candidates() above.
    """
    if frame is None or frame.size == 0:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    frame_area = float(frame.shape[0] * frame.shape[1])

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, _RED_LOWER1, _RED_UPPER1),
        cv2.inRange(hsv, _RED_LOWER2, _RED_UPPER2),
    )
    green_mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)

    results = (
        _light_blobs(red_mask, value, frame_area, RED_LIGHT)
        + _light_blobs(green_mask, value, frame_area, GREEN_LIGHT)
    )
    results.sort(key=lambda c: c.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Bag iteration
# ---------------------------------------------------------------------------


def iter_bag_frames(bag_dir: Path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic == '/camera/image/compressed']
        if not conns:
            return
        for _conn, timestamp, rawdata in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(rawdata, conns[0].msgtype)
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                yield timestamp, frame


def to_yolo_label(c: Candidate, img_w: int, img_h: int) -> str:
    cx = (c.x1 + c.x2) / 2.0 / img_w
    cy = (c.y1 + c.y2) / 2.0 / img_h
    w = (c.x2 - c.x1) / img_w
    h = (c.y2 - c.y1) / img_h
    return f'{c.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}'


def draw_overlay(frame: np.ndarray, candidates: list[Candidate]) -> np.ndarray:
    vis = frame.copy()
    colors = {LEFT: (0, 165, 255), RIGHT: (0, 255, 0), RED_LIGHT: (0, 0, 255), GREEN_LIGHT: (255, 200, 0)}
    for c in candidates:
        cv2.rectangle(vis, (c.x1, c.y1), (c.x2, c.y2), colors[c.cls], 2)
        cv2.putText(
            vis, f'{CLASS_NAMES[c.cls]} {c.score:.0f}', (c.x1, max(0, c.y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[c.cls], 1, cv2.LINE_AA,
        )
    return vis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bags-root', type=Path, default=Path('/mnt/c'))
    parser.add_argument('--bags', nargs='+', required=True)
    parser.add_argument('--out', type=Path, default=Path('dataset'))
    parser.add_argument(
        '--min-gap-frames', type=int, default=5,
        help='minimum raw-frame gap between two kept sign candidates within one bag '
             '(de-duplicates near-identical consecutive sightings)',
    )
    parser.add_argument(
        '--light-min-gap-frames', type=int, default=45,
        help='minimum raw-frame gap between two kept light candidates within one bag '
             '(a lit lamp stays on for many consecutive frames, so this is much larger '
             'than --min-gap-frames to avoid near-duplicate reviews)',
    )
    args = parser.parse_args()

    images_dir = args.out / 'images'
    labels_dir = args.out / 'labels'
    preview_dir = args.out / 'preview'
    for d in (images_dir, labels_dir, preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    total_kept = 0
    per_bag_counts: dict[str, int] = {}
    for bag_name in args.bags:
        bag_dir = args.bags_root / bag_name
        if not bag_dir.exists():
            print(f'[skip] {bag_name}: not found at {bag_dir}', file=sys.stderr)
            continue

        kept = 0
        frame_idx = 0
        last_kept_sign_idx = -10**9
        last_kept_light_idx = -10**9
        for _timestamp, frame in iter_bag_frames(bag_dir):
            sign_candidates = find_sign_candidates(frame)
            light_candidates = find_light_candidates(frame)
            take_sign = bool(sign_candidates) and (frame_idx - last_kept_sign_idx) >= args.min_gap_frames
            take_light = bool(light_candidates) and (frame_idx - last_kept_light_idx) >= args.light_min_gap_frames
            if take_sign or take_light:
                candidates = (sign_candidates if take_sign else []) + (light_candidates if take_light else [])
                stem = f'{bag_name}_{frame_idx:05d}'
                img_h, img_w = frame.shape[:2]
                cv2.imwrite(str(images_dir / f'{stem}.png'), frame)
                label_lines = [to_yolo_label(c, img_w, img_h) for c in candidates]
                (labels_dir / f'{stem}.txt').write_text('\n'.join(label_lines) + '\n')
                cv2.imwrite(str(preview_dir / f'{stem}.png'), draw_overlay(frame, candidates))
                kept += 1
                if take_sign:
                    last_kept_sign_idx = frame_idx
                if take_light:
                    last_kept_light_idx = frame_idx
            frame_idx += 1

        per_bag_counts[bag_name] = kept
        total_kept += kept
        print(f'{bag_name}: {frame_idx} frames scanned, {kept} candidates kept')

    data_yaml = args.out / 'data.yaml'
    data_yaml.write_text(
        'path: ' + str(args.out.resolve()) + '\n'
        'train: images\n'
        'val: images\n'
        'names:\n' + ''.join(f'  {i}: {name}\n' for i, name in enumerate(CLASS_NAMES))
    )

    print()
    print(f'Total kept: {total_kept} (from {sum(per_bag_counts.values())} across bags)')
    print(f'Dataset written to: {args.out.resolve()}')
    print(f'Preview overlays (for human review) in: {preview_dir.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
