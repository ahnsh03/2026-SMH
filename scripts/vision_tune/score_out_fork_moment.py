#!/usr/bin/env python3
"""Offline score for OUT course fork moment.

Uses ``inference.modules.perception.fork.moment.score_out_fork_moment``.
Docs: ``docs/fork-moment-detection.md`` · ``docs/lane-occlusion-fork-strategy.md`` §5.1.3

Examples:

  PYTHONPATH=scripts/vision_tune:src/inference python3 \\
    scripts/vision_tune/score_out_fork_moment.py --folder data/captures/from_bag/out
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parents[1]
_INFERENCE = _ROOT / 'src' / 'inference'
for p in (_SCRIPT, str(_INFERENCE)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hsv import default_config_path, load_hsv_ranges, make_mask  # noqa: E402
from metric_ipm import load_metric_ipm, warp_metric_ipm  # noqa: E402

from inference.modules.perception.fork.moment import (  # noqa: E402
    score_out_fork_moment,
)
from viz_raw_hsv_masks import extract_five_from_bev  # noqa: E402

EXPECTED_HARD_POS = frozenset(
    {
        'frame_20260715_045046_248624_1758',  # 0011
        'frame_20260715_045053_939644_1784',  # 0012
    }
)
EXPECTED_NEAR_FORK = frozenset(
    {
        'frame_20260715_045104_991720_1976',  # 0013 post-apex
    }
)


@dataclass(frozen=True)
class Row:
    stem: str
    index: int
    far_dualW: float
    mid_dualW: float
    sepW: float
    far_dualRoad: float
    span_road: float
    road_pct: float
    hard: bool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--folder',
        type=Path,
        default=_ROOT / 'data' / 'captures' / 'from_bag' / 'out',
    )
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument('--csv', type=Path, default=None)
    args = ap.parse_args(argv)

    folder = args.folder.expanduser().resolve()
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    if not paths:
        raise SystemExit(f'No images in {folder}')

    ranges = load_hsv_ranges(args.config)
    ipm = load_metric_ipm(args.config)
    rows: list[Row] = []
    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        bev = warp_metric_ipm(frame, ipm)
        white = make_mask(bev, ranges['white'])
        road = extract_five_from_bev(bev, args.config, course='out')['road']
        s = score_out_fork_moment(white, road)
        rows.append(
            Row(
                stem=path.stem,
                index=i + 1,
                far_dualW=s.far_dual_white,
                mid_dualW=s.mid_dual_white,
                sepW=s.sep_white,
                far_dualRoad=s.far_dual_road,
                span_road=s.span_road,
                road_pct=s.road_pct,
                hard=s.hard,
            )
        )

    print(
        f'{"idx":>3} {"hard":>4} {"farW":>6} {"midW":>6} {"sepW":>6} '
        f'{"farRo":>6} {"span":>5} {"rp":>5}  stem'
    )
    for r in rows:
        mark = ''
        if r.stem in EXPECTED_HARD_POS:
            mark = '  <<POS'
        elif r.stem in EXPECTED_NEAR_FORK:
            mark = '  <<NEAR'
        print(
            f'{r.index:3d} {int(r.hard):4d} {r.far_dualW:6.1f} {r.mid_dualW:6.1f} '
            f'{r.sepW:6.1f} {r.far_dualRoad:6.1f} {r.span_road:5.2f} '
            f'{r.road_pct:5.1f}  {r.stem}{mark}'
        )

    hard_hits = [r for r in rows if r.hard]
    print(f'\nSummary: n={len(rows)} hard={len(hard_hits)}')
    print('Hard hits:', ', '.join(f'{r.index:04d}' for r in hard_hits) or '(none)')
    pos_ok = all(
        any(r.stem == s and r.hard for r in rows) for s in EXPECTED_HARD_POS
    )
    extra = [
        r
        for r in hard_hits
        if r.stem not in EXPECTED_HARD_POS and r.stem not in EXPECTED_NEAR_FORK
    ]
    print(f'Labeled POS (0011/12): {"PASS" if pos_ok else "FAIL"}')
    print('Extra hard:', ', '.join(f'{r.index:04d}' for r in extra) or '(none)')
    neg_ok = len(extra) == 0

    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
        print(f'CSV → {args.csv}')

    return 0 if pos_ok and neg_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
