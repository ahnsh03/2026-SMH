#!/usr/bin/env python3
"""Offline score for IN circle keep/exit fork moment.

Uses ``inference.modules.perception.fork.moment.score_in_circle_fork_moment``.
Docs: ``docs/fork-moment-detection.md`` · ``docs/lane-occlusion-fork-strategy.md`` §5.1.2

Examples (inside 2026-smh-sim after workspace PYTHONPATH):

  PYTHONPATH=scripts/vision_tune:src/inference python3 \\
    scripts/vision_tune/score_in_fork_moment.py --folder data/captures/from_bag/in
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
    score_in_circle_fork_moment,
)
from viz_raw_hsv_masks import extract_five_from_bev  # noqa: E402

EXPECTED_HARD_POS = frozenset(
    {
        'frame_20260715_045830_994784_0714',  # 0008
        'frame_20260715_045837_397668_0734',  # 0009
        'frame_20260715_045902_029012_1174',  # 0019
    }
)
EXPECTED_EARLY_ARM = frozenset(
    {
        'frame_20260715_045828_850604_0694',  # 0007
    }
)
EXPECTED_HARD_NEG = frozenset(
    {
        'frame_20260715_045810_176698_0377',  # 0002
        'frame_20260715_045817_608691_0484',  # 0004
        'frame_20260715_045842_060994_0762',  # 0010
        'frame_20260715_045853_435977_1005',  # 0015
        'frame_20260715_045857_520267_1084',  # 0017
        'frame_20260715_045859_632914_1117',  # 0018
        'frame_20260715_045907_321283_1268',  # 0020
    }
)


@dataclass(frozen=True)
class Row:
    stem: str
    index: int
    far_dualY: float
    mid_dualY: float
    far_dualF: float
    far_sep: float
    span_ratio: float
    ya2_ratio: float
    top_dualFree: float
    hard_base: bool
    hard: bool
    boosted: bool


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--folder',
        type=Path,
        default=_ROOT / 'data' / 'captures' / 'from_bag' / 'in',
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
        yellow = make_mask(bev, ranges['yellow'])
        road = extract_five_from_bev(bev, args.config, course='in')['road']
        s = score_in_circle_fork_moment(yellow, road)
        rows.append(
            Row(
                stem=path.stem,
                index=i + 1,
                far_dualY=s.far_dual_yellow,
                mid_dualY=s.mid_dual_yellow,
                far_dualF=s.far_dual_free,
                far_sep=s.far_sep_yellow,
                span_ratio=s.span_ratio,
                ya2_ratio=s.ya2_ratio,
                top_dualFree=s.top_dual_free,
                hard_base=s.hard_base,
                hard=s.hard,
                boosted=s.boosted,
            )
        )

    print(
        f'{"idx":>3} {"base":>4} {"hard":>4} {"boost":>5} '
        f'{"farY":>6} {"farF":>6} {"midY":>6} {"sep":>6} {"span":>5}  stem'
    )
    for r in rows:
        mark = ''
        if r.stem in EXPECTED_HARD_POS:
            mark = '  <<POS'
        elif r.stem in EXPECTED_EARLY_ARM:
            mark = '  <<EARLY'
        elif r.stem in EXPECTED_HARD_NEG:
            mark = '  <<NEG'
        print(
            f'{r.index:3d} {int(r.hard_base):4d} {int(r.hard):4d} {int(r.boosted):5d} '
            f'{r.far_dualY:6.1f} {r.far_dualF:6.1f} {r.mid_dualY:6.1f} '
            f'{r.far_sep:6.1f} {r.span_ratio:5.2f}  {r.stem}{mark}'
        )

    hard_hits = [r for r in rows if r.hard]
    print(f'\nSummary: n={len(rows)} hard={len(hard_hits)}')
    print('Hard hits:', ', '.join(f'{r.index:04d}' for r in hard_hits) or '(none)')
    pos_ok = all(
        any(r.stem == s and r.hard for r in rows) for s in EXPECTED_HARD_POS
    )
    neg_ok = all(
        not any(r.stem == s and r.hard for r in rows) for s in EXPECTED_HARD_NEG
    )
    print(f'Labeled POS (0008/09/19): {"PASS" if pos_ok else "FAIL"}')
    print(f'Labeled NEG reject: {"PASS" if neg_ok else "FAIL"}')
    extra = [r for r in hard_hits if r.stem not in EXPECTED_HARD_POS]
    print(
        'Extra hard:',
        ', '.join(f'{r.index:04d}' for r in extra) or '(none)',
    )

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
