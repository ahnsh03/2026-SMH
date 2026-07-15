#!/usr/bin/env python3
"""Cyan A/B mosaics: BEV | ego without cyan | ego +cyan | cyan | gained.

Default: ``data/captures/from_bag/out_glare`` → ``data/captures/raw_hsv_masks/cyan_ab/``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parents[1]
if str(_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT))

import viz_raw_hsv_masks as V  # noqa: E402
from hsv import default_config_path, load_hsv_ranges, make_mask  # noqa: E402
from metric_ipm import load_metric_ipm, warp_metric_ipm  # noqa: E402

FROM_BAG = {
    'out_glare': _ROOT / 'data' / 'captures' / 'from_bag' / 'out_glare',
    'out': _ROOT / 'data' / 'captures' / 'from_bag' / 'out',
    'in': _ROOT / 'data' / 'captures' / 'from_bag' / 'in',
}
OUT_ROOT = _ROOT / 'data' / 'captures' / 'raw_hsv_masks' / 'cyan_ab'


def _pipe(
    frame: np.ndarray,
    *,
    ranges,
    ipm,
    use_cyan: bool,
    open_k: int,
    close_k: int,
    open_iters: int,
    close_iters: int,
    max_hole_px: int,
):
    bev = warp_metric_ipm(frame, ipm)
    white = V._bin(make_mask(bev, ranges['white']))
    yellow = V._bin(make_mask(bev, ranges['yellow']))
    black = V._bin(make_mask(bev, ranges['black_road']))
    red = V._bin(make_mask(bev, ranges['red_road']))
    cyan = (
        V._bin(make_mask(bev, ranges['black_cyan']))
        if 'black_cyan' in ranges
        else np.zeros_like(black)
    )
    road = V._or2(black, red)
    if use_cyan:
        road = V._or2(road, cyan)
    lane_road = V._or2(V._or2(white, road), yellow)
    cleaned = V._clean_lane_mask(
        lane_road,
        open_k=open_k,
        close_k=close_k,
        open_iters=open_iters,
        close_iters=close_iters,
        max_hole_px=max_hole_px,
    )
    ego = V._keep_bottom_ego_blob(cleaned)
    return bev, cleaned, ego, cyan


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-bag', choices=tuple(FROM_BAG), default='out_glare')
    ap.add_argument('--folder', type=Path, default=None)
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--open-k', type=int, default=5)
    ap.add_argument('--close-k', type=int, default=17)
    args = ap.parse_args(argv)

    folder = args.folder.expanduser().resolve() if args.folder else FROM_BAG[args.from_bag]
    bag = args.from_bag if args.folder is None else folder.name
    dest = OUT_ROOT / bag
    if args.clean and dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    if not paths:
        raise SystemExit(f'No images in {folder}')

    ranges = load_hsv_ranges(args.config)
    ipm = load_metric_ipm(args.config)
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        bev, _f0, ego0, cyan = _pipe(
            frame,
            ranges=ranges,
            ipm=ipm,
            use_cyan=False,
            open_k=args.open_k,
            close_k=args.close_k,
            open_iters=1,
            close_iters=2,
            max_hole_px=5000,
        )
        _, _f1, ego1, _ = _pipe(
            frame,
            ranges=ranges,
            ipm=ipm,
            use_cyan=True,
            open_k=args.open_k,
            close_k=args.close_k,
            open_iters=1,
            close_iters=2,
            max_hole_px=5000,
        )
        gained = V._bin(cv2.bitwise_and(ego1, cv2.bitwise_not(ego0)))
        gained_bgr = np.zeros((*ego0.shape, 3), dtype=np.uint8)
        gained_bgr[ego0 > 0] = (40, 40, 40)
        gained_bgr[gained > 0] = (0, 255, 0)
        gained_bgr[(ego1 > 0) & (gained == 0)] = (200, 200, 200)

        p0 = V._fit_bgr(bev)
        V._title(p0, '1 BEV')
        p1 = V._panel_bin(ego0, '2 ego NO cyan')
        p2 = V._panel_bin(ego1, '3 ego +cyan')
        p3 = V._panel_bin(cyan, '4 cyan mask')
        p4 = V._fit_bgr(gained_bgr)
        V._title(p4, f'5 gained {V._cov(gained):.1f}%')
        gap = np.full((V.PANEL_H, V.GAP, 3), 28, dtype=np.uint8)
        row = np.hstack([p0, gap, p1, gap, p2, gap, p3, gap, p4])
        footer = np.full((36, row.shape[1], 3), 16, dtype=np.uint8)
        cv2.putText(
            footer,
            f'[{i + 1}/{len(paths)}] {path.name}  SSOT: black|red|cyan → morph → ego',
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        mosaic = np.vstack([row, footer])
        name = f'{i + 1:04d}_{path.stem}_cyan_ab.png'
        cv2.imwrite(str(dest / name), mosaic)
        print(f'Wrote {dest.name}/{name}', flush=True)

    print(f'Dir: {dest}  (n={len(paths)})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
