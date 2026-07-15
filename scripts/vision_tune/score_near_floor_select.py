#!/usr/bin/env python3
"""Score black keep_near_floor_blob: near-band pixels vs old total-area winner.

Example (docker /workspace)::

  python3 scripts/vision_tune/score_near_floor_select.py \\
    --from-bev data/captures/bev_videos/out.mp4 --start 1412 --end 1491

Flags SUSPECT when chosen centroid is high while a lower near-CC exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parents[1]
_INFERENCE = _ROOT / 'src' / 'inference'
for p in (_SCRIPT, _INFERENCE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hsv import default_config_path, load_hsv_ranges, make_mask  # noqa: E402
from viz_raw_hsv_masks import _bin, _keep_near_floor_blob  # noqa: E402


def _old_total_area_pick(
    labels: np.ndarray,
    stats: np.ndarray,
    near_labs: set[int],
    min_near_area: int,
) -> int:
    near_ok = [
        (lab, int(stats[lab, cv2.CC_STAT_AREA]))
        for lab in near_labs
        if int(stats[lab, cv2.CC_STAT_AREA]) >= int(min_near_area)
    ]
    if not near_ok:
        return 0
    return max(near_ok, key=lambda t: t[1])[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-bev', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument('--start', type=int, default=1412, help='1-based frame')
    ap.add_argument('--end', type=int, default=1491, help='1-based inclusive')
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--near-band', type=float, default=0.35)
    args = ap.parse_args(argv)

    ranges = load_hsv_ranges(args.config)
    cap = cv2.VideoCapture(str(args.from_bev.expanduser().resolve()))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open {args.from_bev}')

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(1, int(args.start))
    end = min(n_frames, int(args.end))
    stride = max(1, int(args.stride))

    changed = 0
    suspect_new = 0
    total = 0

    print(
        f'near_band={args.near_band}  frames {start}–{end} stride={stride}  '
        f'(score=near_area vs old=total_area)'
    )
    print(
        f'{"f":>5}  {"old":>6}  {"new":>6}  {"old_cy":>6}  {"new_cy":>6}  '
        f'{"nearA":>6}  {"totA":>6}  note'
    )

    for f1 in range(start, end + 1, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f1 - 1)
        ok, bev = cap.read()
        if not ok:
            continue
        black_raw = _bin(make_mask(bev, ranges['black_road'], morph=False))
        h, w = black_raw.shape
        near_h = max(2, int(round(h * float(args.near_band))))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(
            (black_raw > 0).astype(np.uint8), connectivity=8
        )
        if n <= 1:
            continue
        near_slice = labels[h - near_h :, :]
        near_labs = {int(lab) for lab in np.unique(near_slice) if int(lab) > 0}

        old_lab = _old_total_area_pick(labels, stats, near_labs, 80)
        chosen = _keep_near_floor_blob(
            black_raw, near_band_ratio=float(args.near_band)
        )
        new_lab = 0
        for lab in range(1, n):
            if np.any((labels == lab) & (chosen > 0)):
                new_lab = lab
                break

        total += 1
        old_cy = float(cents[old_lab][1]) / h if old_lab else 0.0
        new_cy = float(cents[new_lab][1]) / h if new_lab else 0.0
        near_a = (
            int(np.count_nonzero(near_slice == new_lab)) if new_lab else 0
        )
        tot_a = int(stats[new_lab, cv2.CC_STAT_AREA]) if new_lab else 0

        note = ''
        if old_lab != new_lab:
            changed += 1
            note = 'CHANGED'
        # Still suspect if new pick has high mass above midline vs rival
        rivals = []
        for lab in near_labs:
            if lab == new_lab:
                continue
            a = int(stats[lab, cv2.CC_STAT_AREA])
            if a < 2000:
                continue
            cy = float(cents[lab][1]) / h
            if cy > 0.55:
                rivals.append(lab)
        if new_cy < 0.50 and rivals:
            suspect_new += 1
            note = (note + ' ' if note else '') + '!!SUSPECT'

        print(
            f'{f1:5d}  lab{old_lab:3d}  lab{new_lab:3d}  {old_cy:6.2f}  '
            f'{new_cy:6.2f}  {near_a:6d}  {tot_a:6d}  {note}'
        )

    cap.release()
    print(
        f'\nsummary: {total} frames, {changed} selection CHANGED, '
        f'{suspect_new} still !!SUSPECT'
    )
    print(
        'visual check:\n'
        f'  python3 scripts/vision_tune/play_bag_drivable.py out '
        f'--from-bev {args.from_bev} --start {start - 1}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
