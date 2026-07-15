#!/usr/bin/env python3
"""Export 5 OUT BEV|ego videos: pre-retune HSV + one delta each.

Baseline = HSV before 2026-07-16 retune (black H17 Vmax140, red Smin155,
cyan S190-220 V200-230). Each clip applies **only one** retune knob:

  1. black_road H  (h_min 17→14)
  2. black_road V  (v_max 140→180)
  3. red_road S    (s_min 155→110)
  4. black_cyan S  (190-220 → 200-215)
  5. black_cyan V  (200-230 → 190-238)

SSOT pipeline: black_mode=near, morph open5/close17.

Example::

  python3 scripts/vision_tune/export_hsv_single_deltas.py
"""

from __future__ import annotations

import argparse
import copy
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

from hsv import (  # noqa: E402
    CHANNEL_NAMES,
    HsvRange,
    default_config_path,
    load_hsv_ranges,
)
from viz_raw_hsv_masks import extract_five_from_bev  # noqa: E402

PANEL_W = 480
PANEL_H = 360
GAP = 8
OUT_DIR = _ROOT / 'data' / 'captures' / 'bev_videos' / 'hsv_single_deltas'

# Pre-retune real_car road channels (white/yellow/cyan2 unchanged in retune).
_BASELINE_OVERRIDES: dict[str, dict[str, int]] = {
    'black_road': {
        'h_min': 17,
        'h_max': 70,
        's_min': 0,
        's_max': 255,
        'v_min': 15,
        'v_max': 140,
    },
    'red_road': {
        'h_min': 0,
        'h_max': 9,
        's_min': 155,
        's_max': 255,
        'v_min': 120,
        'v_max': 255,
    },
    'black_cyan': {
        'h_min': 90,
        'h_max': 100,
        's_min': 190,
        's_max': 220,
        'v_min': 200,
        'v_max': 230,
    },
}


def _baseline_ranges(config: Path) -> dict[str, HsvRange]:
    ranges = load_hsv_ranges(config)
    out = {k: copy.deepcopy(v) for k, v in ranges.items()}
    for name, block in _BASELINE_OVERRIDES.items():
        out[name] = HsvRange.from_dict(block, block)
    return out


def _with_patch(
    base: dict[str, HsvRange],
    channel: str,
    patch: dict[str, int],
) -> dict[str, HsvRange]:
    out = {k: copy.deepcopy(v) for k, v in base.items()}
    d = out[channel].to_dict()
    d.update(patch)
    out[channel] = HsvRange.from_dict(d, d)
    return out


VARIANTS: list[tuple[str, str, dict[str, int]]] = [
    ('01_black_H_only', 'black_road', {'h_min': 14}),
    ('02_black_V_only', 'black_road', {'v_max': 180}),
    ('03_red_S_only', 'red_road', {'s_min': 110}),
    (
        '04_cyan_S_only',
        'black_cyan',
        {'s_min': 200, 's_max': 215},
    ),
    (
        '05_cyan_V_only',
        'black_cyan',
        {'v_min': 190, 'v_max': 238},
    ),
]


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
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([row, footer])


def export_variant(
    *,
    bev_path: Path,
    out_path: Path,
    ranges: dict[str, HsvRange],
    title: str,
    config: Path,
    open_k: int,
    close_k: int,
    close_iters: int,
    max_hole_px: int,
    stride: int,
) -> Path:
    cap = cv2.VideoCapture(str(bev_path))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open {bev_path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
    use_fps = max(1.0, src_fps / max(1, stride))

    out_path.parent.mkdir(parents=True, exist_ok=True)
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
            course='out',
            black_mode='near',
            ranges=ranges,
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
        if written % 400 == 0:
            print(f'  … {out_path.name}: {written} frames', flush=True)
        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    print(f'wrote {out_path} ({written} frames @ {use_fps:.2f} fps)', flush=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument(
        '--from-bev',
        type=Path,
        default=_ROOT / 'data/captures/bev_videos/out.mp4',
    )
    ap.add_argument('--outdir', type=Path, default=OUT_DIR)
    ap.add_argument('--open-k', type=int, default=5)
    ap.add_argument('--close-k', type=int, default=17)
    ap.add_argument('--close-iters', type=int, default=2)
    ap.add_argument('--max-hole-px', type=int, default=5000)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args(argv)

    base = _baseline_ranges(args.config)
    print(
        'baseline = pre-retune HSV; SSOT near + morph '
        f'{args.open_k}/{args.close_k}; source={args.from_bev}',
        flush=True,
    )
    for name in CHANNEL_NAMES:
        if name in _BASELINE_OVERRIDES:
            print(f'  {name}: {base[name].to_dict()}', flush=True)

    for tag, channel, patch in VARIANTS:
        ranges = _with_patch(base, channel, patch)
        patched = {k: ranges[channel].to_dict()[k] for k in patch}
        title = f'out near | {tag} | {channel} {patched}'
        dest = Path(args.outdir).expanduser().resolve() / f'{tag}.mp4'
        print(f'\n[{tag}] {channel} {patch}', flush=True)
        export_variant(
            bev_path=Path(args.from_bev).expanduser().resolve(),
            out_path=dest,
            ranges=ranges,
            title=title,
            config=args.config,
            open_k=int(args.open_k),
            close_k=int(args.close_k),
            close_iters=int(args.close_iters),
            max_hole_px=int(args.max_hole_px),
            stride=max(1, int(args.stride)),
        )

    print(f'\ndone → {args.outdir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
