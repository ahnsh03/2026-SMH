#!/usr/bin/env python3
"""Offline score for OUT ego-blob fork *stretch* (Y-shape on 6_ego_blob).

Uses ``inference.modules.perception.fork.ego_shape.score_out_ego_fork_shape``.
Sibling to ``score_out_fork_moment.py`` (white+road moment); does not replace it.

Docs: ``docs/out-ego-fork-shape.md`` · ``docs/fork-moment-detection.md`` §3.5

Examples:

  # scan saved OUT BEV video (recommended)
  PYTHONPATH=scripts/vision_tune:src/inference python3 \\
    scripts/vision_tune/score_out_ego_fork_shape.py \\
    --from-bev data/captures/bev_videos/out.mp4 \\
    --label-from 1690 --label-to 1783 --stride 5

  # optional CSV
  ... --csv data/captures/raw_hsv_masks/out_ego_fork_shape_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parents[1]
_INFERENCE = _ROOT / 'src' / 'inference'
for p in (_SCRIPT, str(_INFERENCE)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hsv import default_config_path  # noqa: E402
from inference.modules.perception.fork.ego_shape import (  # noqa: E402
    score_out_ego_fork_shape,
)
from viz_raw_hsv_masks import extract_five_from_bev  # noqa: E402


def _cluster(frames: list[int], gap: int = 15) -> list[tuple[int, int]]:
    if not frames:
        return []
    out: list[tuple[int, int]] = []
    s = p = frames[0]
    for x in frames[1:]:
        if x - p <= gap:
            p = x
        else:
            out.append((s, p))
            s = p = x
    out.append((s, p))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--from-bev',
        type=Path,
        default=_ROOT / 'data' / 'captures' / 'bev_videos' / 'out.mp4',
    )
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument('--course', default='out')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--label-from', type=int, default=1690)
    ap.add_argument('--label-to', type=int, default=1783)
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument(
        '--recall-dense',
        action='store_true',
        help='also densify-scan the label window for recall',
    )
    args = ap.parse_args(argv)

    path = args.from_bev.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f'Missing BEV video: {path}')

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open {path}')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    rows: list[dict] = []
    hard_hits: list[int] = []
    soft_hits: list[int] = []

    for f in range(1, n_frames + 1, max(args.stride, 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, bev = cap.read()
        if not ok:
            continue
        ego = extract_five_from_bev(
            bev, args.config, course=args.course, black_mode='near'
        )['ego_blob']
        s = score_out_ego_fork_shape(ego)
        row = {
            'frame': f,
            'cov': round(s.cov, 2),
            'w_far': round(s.w_far, 1),
            'w_near': round(s.w_near, 1),
            'wr_fn': round(s.wr_fn, 3),
            'dual_far': s.dual_far,
            'dual_near': s.dual_near,
            'mean_sep_far': round(s.mean_sep_far, 1),
            'max_sep': round(s.max_sep, 1),
            'max_run': s.max_run,
            'solid': round(s.solid, 3),
            'hard': int(s.hard),
            'soft': int(s.soft),
            'in_label': int(args.label_from <= f <= args.label_to),
        }
        rows.append(row)
        if s.hard:
            hard_hits.append(f)
        if s.soft:
            soft_hits.append(f)

    def _summary(name: str, hits: list[int]) -> None:
        inside = [h for h in hits if args.label_from <= h <= args.label_to]
        outside = [h for h in hits if not (args.label_from <= h <= args.label_to)]
        print(
            f'{name}: hits={len(hits)} in_label={len(inside)} '
            f'outside_label={len(outside)}'
        )
        print(f'  clusters={_cluster(hits)}')
        if outside:
            print(f'  outside frames={outside}')

    print(f'BEV={path} n≈{n_frames} stride={args.stride}')
    print(f'label=[{args.label_from},{args.label_to}]')
    _summary('hard(GateC)', hard_hits)
    _summary('soft(GateB)', soft_hits)

    if args.recall_dense:
        for gname, key in (('hard', 'hard'), ('soft', 'soft')):
            hit = tot = 0
            for f in range(args.label_from, args.label_to + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
                ok, bev = cap.read()
                if not ok:
                    continue
                ego = extract_five_from_bev(
                    bev, args.config, course=args.course, black_mode='near'
                )['ego_blob']
                s = score_out_ego_fork_shape(ego)
                tot += 1
                if getattr(s, key):
                    hit += 1
            print(f'label recall {gname}: {hit}/{tot}')

    cap.release()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f'wrote {args.csv}')

    # Distant FP check: any hard hit far from label±40 is a real FP
    margin = 40
    lo, hi = args.label_from - margin, args.label_to + margin
    distant = [h for h in hard_hits if h < lo or h > hi]
    if distant:
        print(f'FAIL distant hard FP: {distant}')
        return 1
    print('OK distant hard FP=0 (approach/exit bleed within ±40 allowed)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
