#!/usr/bin/env python3
"""Export side-by-side BEV | ego_blob videos for black trial #1 vs #2 × IN/OUT.

Writes under ``data/captures/bev_videos/black_trials/``::

  in_trial1_near.mp4
  in_trial2_top_drop.mp4
  out_trial1_near.mp4
  out_trial2_top_drop.mp4

Example::

  python3 scripts/vision_tune/export_black_trial_videos.py
  python3 scripts/vision_tune/export_black_trial_videos.py --stride 2
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

from hsv import default_config_path  # noqa: E402
from viz_raw_hsv_masks import extract_five_from_bev  # noqa: E402

PANEL_W = 480
PANEL_H = 360
GAP = 8
OUT_DIR = _ROOT / 'data' / 'captures' / 'bev_videos' / 'black_trials'


def _fit(img: np.ndarray, w: int, h: int, *, nearest: bool = False) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None or img.size == 0:
        return out
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = r
    return out


def _label(panel: np.ndarray, text: str) -> None:
    cv2.putText(
        panel,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _compose(
    bev: np.ndarray,
    ego: np.ndarray,
    *,
    title: str,
    index: int,
    n: int,
) -> np.ndarray:
    left = _fit(bev, PANEL_W, PANEL_H)
    _label(left, 'BEV')
    right = _fit(ego, PANEL_W, PANEL_H, nearest=True)
    cov = 100.0 * float(np.count_nonzero(ego)) / float(max(ego.size, 1))
    _label(right, f'ego {cov:.1f}%')
    gap = np.full((PANEL_H, GAP, 3), 28, dtype=np.uint8)
    row = np.hstack([left, gap, right])
    footer = np.full((36, row.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(
        footer,
        f'{title}  [{index + 1}/{n}]',
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([row, footer])


def export_one(
    *,
    bev_path: Path,
    out_path: Path,
    course: str,
    black_mode: str,
    config: Path,
    open_k: int,
    close_k: int,
    close_iters: int,
    max_hole_px: int,
    stride: int,
    fps: float | None,
) -> Path:
    cap = cv2.VideoCapture(str(bev_path))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open {bev_path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    use_fps = float(fps) if fps and fps > 0 else max(1.0, src_fps / max(1, stride))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = f'{course} black_mode={black_mode} open={open_k} close={close_k}'
    writer = None
    written = 0
    idx = 0
    while True:
        ok, bev = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        five = extract_five_from_bev(
            bev,
            config,
            open_k=open_k,
            close_k=close_k,
            open_iters=1,
            close_iters=close_iters,
            max_hole_px=max_hole_px,
            course=course,
            black_mode=black_mode,
        )
        frame = _compose(
            five['bev'],
            five['ego_blob'],
            title=title,
            index=idx,
            n=n,
        )
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, use_fps, (w, h))
            if not writer.isOpened():
                raise SystemExit(f'VideoWriter failed: {out_path}')
        writer.write(frame)
        written += 1
        if written % 200 == 0:
            print(f'  … {out_path.name}: {written} frames (src idx {idx}/{n})')
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    print(f'wrote {out_path} ({written} frames @ {use_fps:.2f} fps)')
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument('--in-bev', type=Path, default=_ROOT / 'data/captures/bev_videos/in.mp4')
    ap.add_argument('--out-bev', type=Path, default=_ROOT / 'data/captures/bev_videos/out.mp4')
    ap.add_argument('--outdir', type=Path, default=OUT_DIR)
    ap.add_argument('--open-k', type=int, default=5)
    ap.add_argument('--close-k', type=int, default=17)
    ap.add_argument('--close-iters', type=int, default=2)
    ap.add_argument('--max-hole-px', type=int, default=5000)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--fps', type=float, default=0.0, help='0 = source_fps/stride')
    args = ap.parse_args(argv)

    jobs = [
        ('in', args.in_bev, 'near', 'in_trial1_near.mp4'),
        ('in', args.in_bev, 'top_drop', 'in_trial2_top_drop.mp4'),
        ('out', args.out_bev, 'near', 'out_trial1_near.mp4'),
        ('out', args.out_bev, 'top_drop', 'out_trial2_top_drop.mp4'),
    ]

    print(
        f'export black trials → {args.outdir}  '
        f'morph open={args.open_k} close={args.close_k} '
        f'iters={args.close_iters} stride={args.stride}'
    )
    for course, src, mode, name in jobs:
        src = Path(src).expanduser().resolve()
        if not src.is_file():
            raise SystemExit(f'missing BEV video: {src}')
        dest = Path(args.outdir).expanduser().resolve() / name
        print(f'\n[{course} / {mode}] {src.name} → {dest.name}')
        export_one(
            bev_path=src,
            out_path=dest,
            course=course,
            black_mode=mode,
            config=args.config,
            open_k=int(args.open_k),
            close_k=int(args.close_k),
            close_iters=int(args.close_iters),
            max_hole_px=int(args.max_hole_px),
            stride=max(1, int(args.stride)),
            fps=float(args.fps) if args.fps else None,
        )
    print('\ndone. default runtime/player remains trial #1 (black_mode=near).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
