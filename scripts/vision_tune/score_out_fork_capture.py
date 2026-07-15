#!/usr/bin/env python3
"""Validate integrated OUT fork capture (moment tip + ego stretch) from bag.

Sources (pick one):
  --from-bag out     rosbag2 under bags/out_course (JPEG → Metric IPM BEV)
  --from-bev PATH    pre-exported BEV mp4 (same bag, faster repeat)
  --folder PATH      stills under data/captures/from_bag/out

Docs: docs/out-ego-fork-shape.md · docs/fork-moment-detection.md §3.5

Examples:

  PYTHONPATH=scripts/vision_tune:src/inference python3 \\
    scripts/vision_tune/score_out_fork_capture.py --from-bag out --stride 5

  PYTHONPATH=scripts/vision_tune:src/inference python3 \\
    scripts/vision_tune/score_out_fork_capture.py \\
    --from-bev data/captures/bev_videos/out.mp4 --stride 5 --recall-dense

  PYTHONPATH=scripts/vision_tune:src/inference python3 \\
    scripts/vision_tune/score_out_fork_capture.py \\
    --folder data/captures/from_bag/out
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parents[1]
_INFERENCE = _ROOT / 'src' / 'inference'
for p in (_SCRIPT, str(_INFERENCE)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hsv import default_config_path, load_hsv_ranges, make_mask  # noqa: E402
from metric_ipm import load_metric_ipm, warp_metric_ipm  # noqa: E402
from inference.modules.perception.fork.capture import (  # noqa: E402
    score_out_fork_capture,
)
from viz_raw_hsv_masks import extract_five_from_bev  # noqa: E402

# Labeled stills (from_bag/out) — tip moment POS / near / expected tip reject
EXPECTED_TIP_POS = frozenset(
    {
        'frame_20260715_045046_248624_1758',  # 0011
        'frame_20260715_045053_939644_1784',  # 0012
    }
)
EXPECTED_TIP_NEAR = frozenset(
    {
        'frame_20260715_045104_991720_1976',  # 0013 post-apex
    }
)

_STEM_FRAME_RE = re.compile(r'_(\d+)$')


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


def _frame_from_stem(stem: str) -> int | None:
    m = _STEM_FRAME_RE.search(stem)
    return int(m.group(1)) if m else None


def _score_bev(bev: np.ndarray, config: Path, course: str):
    five = extract_five_from_bev(bev, config, course=course, black_mode='near')
    ranges = load_hsv_ranges(config)
    white = make_mask(bev, ranges['white'])
    return score_out_fork_capture(white, five['road'], five['ego_blob'])


def _summarize(name: str, hits: list[int], lo: int, hi: int) -> None:
    inside = [h for h in hits if lo <= h <= hi]
    outside = [h for h in hits if not (lo <= h <= hi)]
    print(
        f'{name}: hits={len(hits)} in_label={len(inside)} '
        f'outside_label={len(outside)} clusters={_cluster(hits)}'
    )
    if outside:
        print(f'  outside={outside[:60]}{"…" if len(outside) > 60 else ""}')


def run_bev_video(
    path: Path,
    *,
    config: Path,
    course: str,
    stride: int,
    label_from: int,
    label_to: int,
    recall_dense: bool,
    csv_path: Path | None,
) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open {path}')
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    rows: list[dict] = []
    stretch: list[int] = []
    tip: list[int] = []
    tip_ctx: list[int] = []
    capture: list[int] = []

    for f in range(1, n_frames + 1, max(stride, 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, bev = cap.read()
        if not ok:
            continue
        s = _score_bev(bev, config, course)
        rows.append(
            {
                'frame': f,
                'capture': int(s.capture),
                'in_stretch': int(s.in_stretch),
                'tip': int(s.tip),
                'tip_in_context': int(s.tip_in_context),
                'ego_wr': round(s.ego.wr_fn, 3),
                'ego_sep_far': round(s.ego.mean_sep_far, 1),
                'ego_solid': round(s.ego.solid, 3),
                'sepW': round(s.moment.sep_white, 1),
                'span_road': round(s.moment.span_road, 2),
                'in_label': int(label_from <= f <= label_to),
            }
        )
        if s.in_stretch:
            stretch.append(f)
        if s.tip:
            tip.append(f)
        if s.tip_in_context:
            tip_ctx.append(f)
        if s.capture:
            capture.append(f)

    print(f'source=BEV {path} n≈{n_frames} stride={stride}')
    print(f'label=[{label_from},{label_to}]')
    _summarize('stretch(ego.hard)', stretch, label_from, label_to)
    _summarize('tip(moment.hard)', tip, label_from, label_to)
    _summarize('tip_in_context', tip_ctx, label_from, label_to)
    _summarize('capture', capture, label_from, label_to)

    # Tip frames should sit inside / near stretch cluster
    tip_orphan = []
    stretch_set = set(stretch)
    for t in tip:
        near = any(abs(t - s) <= 40 for s in stretch_set) or any(
            abs(t - x) <= 40 for x in range(label_from, label_to + 1)
        )
        if not near and t not in stretch_set:
            tip_orphan.append(t)
    print(f'tip orphan (far from stretch/label): {tip_orphan or "none"}')

    if recall_dense:
        hit = tot = 0
        tip_hit = tip_tot = 0
        for f in range(label_from, label_to + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
            ok, bev = cap.read()
            if not ok:
                continue
            s = _score_bev(bev, config, course)
            tot += 1
            if s.capture:
                hit += 1
            tip_tot += 1
            if s.tip:
                tip_hit += 1
        print(f'label capture recall: {hit}/{tot}')
        print(f'label tip hits (dense): {tip_hit}/{tip_tot}')

    cap.release()

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f'wrote {csv_path}')

    margin = 40
    lo, hi = label_from - margin, label_to + margin
    distant = [h for h in capture if h < lo or h > hi]
    # allow exit bleed only slightly beyond hi (ego exit ~1811)
    distant = [h for h in distant if h < lo or h > hi + 40]
    if distant:
        print(f'FAIL distant capture FP: {distant}')
        return 1
    if tip_orphan:
        print('FAIL tip fired far from stretch')
        return 1
    print('OK distant capture FP≈0; tip tied to stretch')
    return 0


def run_folder(folder: Path, *, config: Path, course: str, csv_path: Path | None) -> int:
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    if not paths:
        raise SystemExit(f'No images in {folder}')

    ipm = load_metric_ipm(config)
    rows: list[dict] = []
    tip_hits: list[str] = []
    stretch_hits: list[str] = []
    capture_hits: list[str] = []

    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        bev = warp_metric_ipm(frame, ipm)
        s = _score_bev(bev, config, course)
        fr = _frame_from_stem(path.stem)
        rows.append(
            {
                'stem': path.stem,
                'frame': fr if fr is not None else -1,
                'capture': int(s.capture),
                'in_stretch': int(s.in_stretch),
                'tip': int(s.tip),
                'tip_in_context': int(s.tip_in_context),
                'ego_wr': round(s.ego.wr_fn, 3),
                'ego_sep_far': round(s.ego.mean_sep_far, 1),
                'sepW': round(s.moment.sep_white, 1),
                'span_road': round(s.moment.span_road, 2),
                'road_pct': round(s.moment.road_pct, 1),
            }
        )
        if s.tip:
            tip_hits.append(path.stem)
        if s.in_stretch:
            stretch_hits.append(path.stem)
        if s.capture:
            capture_hits.append(path.stem)

    print(f'source=folder {folder} n={len(paths)}')
    print(f'tip hits ({len(tip_hits)}): {tip_hits}')
    print(f'stretch hits ({len(stretch_hits)}): {stretch_hits}')
    print(f'capture hits ({len(capture_hits)}): {capture_hits}')

    missing_pos = EXPECTED_TIP_POS - set(tip_hits)
    extra_tip = set(tip_hits) - EXPECTED_TIP_POS - EXPECTED_TIP_NEAR
    # near-fork tip after apex should usually miss
    print(f'missing tip POS: {sorted(missing_pos) or "none"}')
    print(f'extra tip (nontarget): {sorted(extra_tip) or "none"}')

    # POS frames should be capture via tip_in_context or stretch
    pos_capture_miss = []
    by_stem = {r['stem']: r for r in rows}
    for stem in EXPECTED_TIP_POS:
        r = by_stem.get(stem)
        if r is None or not r['capture']:
            pos_capture_miss.append(stem)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f'wrote {csv_path}')

    ok = not missing_pos and not extra_tip and not pos_capture_miss
    if pos_capture_miss:
        print(f'FAIL POS not in capture: {pos_capture_miss}')
    if ok:
        print('OK from_bag tip POS + nontarget tip FP=0 + POS⊂capture')
        return 0
    print('FAIL folder checks')
    return 1


def run_bag(
    course: str,
    *,
    bag: Path | None,
    config: Path,
    stride: int,
    label_from: int,
    label_to: int,
    topic: str,
    csv_path: Path | None,
    max_frames: int,
) -> int:
    from capture_from_bag import decode_jpeg, load_camera_jpegs, resolve_bag

    bag_dir = resolve_bag(course, bag)
    print(f'Loading bag {bag_dir} …', flush=True)
    jpegs, _stamps = load_camera_jpegs(bag_dir, topic)
    ipm = load_metric_ipm(config)
    n = len(jpegs)
    if max_frames > 0:
        n = min(n, max_frames)

    stretch: list[int] = []
    tip: list[int] = []
    tip_ctx: list[int] = []
    capture: list[int] = []
    rows: list[dict] = []

    for i in range(0, n, max(stride, 1)):
        frame = decode_jpeg(jpegs[i])
        if frame is None:
            continue
        f = i + 1  # 1-based index matching stem suffix convention
        bev = warp_metric_ipm(frame, ipm)
        s = _score_bev(bev, config, course)
        rows.append(
            {
                'frame': f,
                'capture': int(s.capture),
                'in_stretch': int(s.in_stretch),
                'tip': int(s.tip),
                'tip_in_context': int(s.tip_in_context),
                'ego_wr': round(s.ego.wr_fn, 3),
                'ego_sep_far': round(s.ego.mean_sep_far, 1),
                'sepW': round(s.moment.sep_white, 1),
            }
        )
        if s.in_stretch:
            stretch.append(f)
        if s.tip:
            tip.append(f)
        if s.tip_in_context:
            tip_ctx.append(f)
        if s.capture:
            capture.append(f)
        if (i // stride) % 50 == 0:
            print(f'  … frame {f}/{n}', flush=True)

    print(f'source=bag {bag_dir} n={n} stride={stride}')
    print(f'label=[{label_from},{label_to}]')
    _summarize('stretch(ego.hard)', stretch, label_from, label_to)
    _summarize('tip(moment.hard)', tip, label_from, label_to)
    _summarize('tip_in_context', tip_ctx, label_from, label_to)
    _summarize('capture', capture, label_from, label_to)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f'wrote {csv_path}')

    margin = 40
    lo, hi = label_from - margin, label_to + margin + 40
    distant = [h for h in capture if h < lo or h > hi]
    tip_in_window = [t for t in tip if label_from - 20 <= t <= label_to + 50]
    print(f'tip near mission: {tip_in_window}')

    # Expect at least one tip near labeled POS frames (1758/1784) when stride allows
    tip_near_pos = any(abs(t - 1758) <= stride * 2 or abs(t - 1784) <= stride * 2 for t in tip)
    if stride <= 5 and not tip_near_pos and tip:
        # soft warn if stride skipped exact POS
        print('WARN: no tip within ±stride of 1758/1784 (stride may skip)')

    if distant:
        print(f'FAIL distant capture FP: {distant}')
        return 1
    if not stretch:
        print('FAIL no stretch hits on bag')
        return 1
    print('OK bag: stretch present, distant capture FP≈0')
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--from-bag', choices=('out', 'in'), default=None)
    src.add_argument(
        '--from-bev',
        type=Path,
        default=None,
        help='BEV mp4 (default used if no --from-bag/--folder)',
    )
    src.add_argument('--folder', type=Path, default=None)
    ap.add_argument('--bag', type=Path, default=None, help='explicit rosbag2 dir')
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument('--course', default='out')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--label-from', type=int, default=1690)
    ap.add_argument('--label-to', type=int, default=1783)
    ap.add_argument('--recall-dense', action='store_true')
    ap.add_argument('--csv', type=Path, default=None)
    ap.add_argument(
        '--topic',
        default='/camera/image/compressed',
    )
    ap.add_argument(
        '--max-frames',
        type=int,
        default=0,
        help='cap bag frames (0=all); useful for smoke',
    )
    args = ap.parse_args(argv)

    if args.folder is not None:
        return run_folder(
            args.folder.expanduser().resolve(),
            config=args.config,
            course=args.course,
            csv_path=args.csv,
        )
    if args.from_bag is not None or args.bag is not None:
        course = args.from_bag or args.course
        return run_bag(
            course,
            bag=args.bag,
            config=args.config,
            stride=args.stride,
            label_from=args.label_from,
            label_to=args.label_to,
            topic=args.topic,
            csv_path=args.csv,
            max_frames=args.max_frames,
        )

    bev = args.from_bev
    if bev is None:
        bev = _ROOT / 'data' / 'captures' / 'bev_videos' / 'out.mp4'
    return run_bev_video(
        bev.expanduser().resolve(),
        config=args.config,
        course=args.course,
        stride=args.stride,
        label_from=args.label_from,
        label_to=args.label_to,
        recall_dense=args.recall_dense,
        csv_path=args.csv,
    )


if __name__ == '__main__':
    raise SystemExit(main())
