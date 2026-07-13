#!/usr/bin/env python3
"""Headless fork 4-line / 2+2 pair auto-tune (P3) with capture logging.

Grabs /camera/image/compressed (or --frame), sweeps fork_track_* params,
scores marking-based L/R pairs + branch centers, writes ranked captures under
data/captures/lane_tune_logs/auto_fork/.

Example (inside 2026-smh-sim):

  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 scripts/vision_tune/auto_tune_fork.py --top 8 --apply-best
  python3 scripts/vision_tune/auto_tune_fork.py --frame path/to/frame.png
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_INFER = _REPO_ROOT / 'src' / 'inference'
if str(_INFER) not in sys.path:
    sys.path.insert(0, str(_INFER))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

OUT_ROOT = _REPO_ROOT / 'data' / 'captures' / 'lane_tune_logs' / 'auto_fork'
# Scene-scoped layout (keeps failed runs; never deletes):
#   auto_fork/<scene>/runs/<stamp>/
#   auto_fork/<scene>/verify/current.png
#   auto_fork/INDEX.md
SCENE_ALIASES = {
    'out': 'out_fork',
    'out_fork': 'out_fork',
    'out_fork_v2': 'out_fork',
    'exit': 'in_roundabout_exit',
    'exit_v2': 'in_roundabout_exit',
    'in_roundabout_exit': 'in_roundabout_exit',
    'in': 'in_roundabout_exit',
}


def resolve_scene(label: str) -> str:
    key = (label or 'unlabeled').strip().lower()
    return SCENE_ALIASES.get(key, key.replace(' ', '_'))


@dataclass(frozen=True)
class ForkParams:
    assoc_m: float
    min_rows: int
    pair_width_m: float
    far_zone: float
    max_row_gap: int
    near_zone: float


@dataclass
class ForkScore:
    params: ForkParams
    score: float
    n_tracks: int
    n_far_tracks: int
    n_pairs: int
    fork_active: bool
    split_source: str
    center_sep_m: float
    mean_pair_width_m: float
    center_on_road: float
    notes: str


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def grab_compressed_frame(topic: str, timeout_sec: float = 8.0) -> np.ndarray:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    frame_box: dict[str, np.ndarray | None] = {'frame': None}

    class Grab(Node):
        def __init__(self) -> None:
            super().__init__('auto_tune_fork_grab')
            self.create_subscription(CompressedImage, topic, self._cb, qos)

        def _cb(self, msg: CompressedImage) -> None:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                frame_box['frame'] = img

    rclpy.init()
    node = Grab()
    try:
        t0 = datetime.now().timestamp()
        while frame_box['frame'] is None:
            rclpy.spin_once(node, timeout_sec=0.05)
            if datetime.now().timestamp() - t0 > timeout_sec:
                raise TimeoutError(f'no frame on {topic} within {timeout_sec}s')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    assert frame_box['frame'] is not None
    return frame_box['frame']


def _pair_width_m(pair: Any, mpp: float) -> float:
    both = ~np.isnan(pair.outer_u) & ~np.isnan(pair.inner_u)
    if not np.any(both):
        return float('nan')
    return float(np.nanmedian(np.abs(pair.outer_u[both] - pair.inner_u[both]))) * mpp


def _center_on_road_ratio(pairs: list[Any], road_clean: np.ndarray) -> float:
    if road_clean.size == 0 or not pairs:
        return 0.0
    h, w = road_clean.shape[:2]
    hits = 0
    total = 0
    for pair in pairs:
        rows = np.flatnonzero(~np.isnan(pair.center_u))
        for row in rows[::3]:
            u = int(round(float(pair.center_u[row])))
            if 0 <= row < h and 0 <= u < w:
                total += 1
                if road_clean[row, u] > 0:
                    hits += 1
    return float(hits) / float(max(1, total))


def score_fork_result(
    ld: Any, frame: np.ndarray, params: ForkParams
) -> tuple[ForkScore, Any, np.ndarray]:
    ld.apply_detect_tune(
        fork_track_assoc_m=params.assoc_m,
        fork_track_min_rows=params.min_rows,
        fork_pair_width_m=params.pair_width_m,
        fork_far_zone_ratio=params.far_zone,
        fork_track_max_row_gap=params.max_row_gap,
        fork_near_zone_ratio=params.near_zone,
    )
    _dets, debug = ld.detect_with_debug(frame)

    yellow = debug.yellow_connected_bev
    white = debug.white_dash_connected_bev
    if yellow.size == 0:
        yellow = np.zeros((ld.BEV_HEIGHT, ld.BEV_WIDTH), dtype=np.uint8)
    if white.size == 0:
        white = np.zeros((ld.BEV_HEIGHT, ld.BEV_WIDTH), dtype=np.uint8)

    # Score the same preference path as runtime (yellow then white).
    pairs = list(debug.fork_lane_pairs)
    tracks = list(debug.fork_mark_tracks)
    if not pairs:
        pairs, tracks = ld.extract_marking_fork_lane_pairs(yellow)
    if not pairs:
        pairs, tracks = ld.extract_marking_fork_lane_pairs(white)

    n_tracks = len(tracks)
    far_end = max(1, int(round(ld.BEV_HEIGHT * params.far_zone)))
    n_far = sum(1 for t in tracks if np.any(~np.isnan(t[:far_end])))
    n_pairs = len(pairs)

    center_sep_m = 0.0
    mean_width = float('nan')
    if n_pairs >= 2:
        c0 = pairs[0].center_u
        c1 = pairs[1].center_u
        both = ~np.isnan(c0) & ~np.isnan(c1)
        if np.any(both):
            center_sep_m = abs(float(np.nanmedian(c0[both] - c1[both]))) * ld.METERS_PER_PIXEL
        widths = [_pair_width_m(p, ld.METERS_PER_PIXEL) for p in pairs]
        widths = [w for w in widths if np.isfinite(w)]
        if widths:
            mean_width = float(np.mean(widths))

    on_road = _center_on_road_ratio(pairs, debug.road_clean)

    score = 0.0
    # Ideal: 4 far strands, exactly 2 path pairs.
    score += 14.0 * np.exp(-0.5 * ((n_far - 4) / 1.25) ** 2)
    score += 10.0 * np.exp(-0.5 * ((n_tracks - 4) / 1.5) ** 2)
    if n_pairs >= 2:
        score += 16.0
    elif n_pairs == 1:
        score += 3.0
    else:
        score -= 8.0

    # Path centers should be ~ one lane apart (not fused, not opposite courses).
    if center_sep_m > 0.0:
        score += 8.0 * np.exp(-0.5 * ((center_sep_m - 0.35) / 0.18) ** 2)
        if center_sep_m < 0.10:
            score -= 6.0

    if np.isfinite(mean_width):
        score += 6.0 * np.exp(-0.5 * ((mean_width - params.pair_width_m) / 0.08) ** 2)

    score += 7.0 * on_road
    if debug.fork_active and n_pairs >= 2:
        score += 4.0
    if str(debug.fork_split_source).endswith('marks'):
        score += 3.0

    notes = (
        f'far={n_far} tracks={n_tracks} pairs={n_pairs} '
        f'sep={center_sep_m:.3f}m w={mean_width if np.isfinite(mean_width) else -1:.3f} '
        f'onRoad={on_road:.2f} src={debug.fork_split_source or "-"}'
    )
    result = ForkScore(
        params=params,
        score=float(score),
        n_tracks=n_tracks,
        n_far_tracks=n_far,
        n_pairs=n_pairs,
        fork_active=bool(debug.fork_active),
        split_source=str(debug.fork_split_source or ''),
        center_sep_m=float(center_sep_m),
        mean_pair_width_m=float(mean_width) if np.isfinite(mean_width) else -1.0,
        center_on_road=float(on_road),
        notes=notes,
    )
    preview = ld.make_fork_lane_pair_preview(debug, focus='all')
    if n_pairs >= 2 and not list(debug.fork_lane_pairs):
        # Preview with scored pairs when debug did not keep them.
        debug.fork_lane_pairs = tuple(pairs)
        debug.fork_mark_tracks = tuple(tracks)
        debug.road_branches = tuple(ld.fork_lane_pairs_to_road_branches(pairs))
        preview = ld.make_fork_lane_pair_preview(debug, focus='all')
    return result, debug, preview


def save_candidate(
    out_dir: Path,
    frame: np.ndarray,
    preview: np.ndarray,
    debug: Any,
    result: ForkScore,
    rank: int,
) -> Path:
    name = (
        f'r{rank:02d}_s{result.score:05.1f}_'
        f'a{result.params.assoc_m:.3f}_m{result.params.min_rows:02d}_'
        f'w{result.params.pair_width_m:.2f}_f{result.params.far_zone:.2f}'
    )
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / 'frame.png'), frame)
    cv2.imwrite(str(d / 'preview.png'), preview)
    if debug.yellow_connected_bev.size:
        cv2.imwrite(str(d / 'yellow_connected.png'), debug.yellow_connected_bev)
    if debug.white_dash_connected_bev.size:
        cv2.imwrite(str(d / 'white_connected.png'), debug.white_dash_connected_bev)
    if debug.road_clean.size:
        cv2.imwrite(str(d / 'road_clean.png'), debug.road_clean)
    meta = {
        'rank': rank,
        'score': result.score,
        'params': asdict(result.params),
        'metrics': {
            'n_tracks': result.n_tracks,
            'n_far_tracks': result.n_far_tracks,
            'n_pairs': result.n_pairs,
            'fork_active': result.fork_active,
            'split_source': result.split_source,
            'center_sep_m': result.center_sep_m,
            'mean_pair_width_m': result.mean_pair_width_m,
            'center_on_road': result.center_on_road,
            'notes': result.notes,
        },
    }
    (d / 'meta.yaml').write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )
    return d


def build_grid() -> list[ForkParams]:
    assocs = [0.05, 0.06, 0.08, 0.10, 0.12]
    min_rows = [10, 14, 18, 22]
    widths = [0.32, 0.35, 0.38]
    fars = [0.35, 0.45, 0.55]
    gaps = [8, 12, 16]
    nears = [0.22, 0.28, 0.35]
    grid: list[ForkParams] = []
    # Full Cartesian is huge; use a structured sparse grid (~180).
    for a in assocs:
        for m in min_rows:
            for w in widths:
                for f in fars:
                    for g in gaps:
                        for n in nears:
                            # Skip redundant extreme combos to keep runtime sane.
                            if a >= 0.12 and m >= 22:
                                continue
                            if f == 0.35 and n == 0.35 and g == 8:
                                continue
                            grid.append(
                                ForkParams(
                                    assoc_m=a,
                                    min_rows=m,
                                    pair_width_m=w,
                                    far_zone=f,
                                    max_row_gap=g,
                                    near_zone=n,
                                )
                            )
    # Still too many (~5*4*3*3*3*3 ≈ 1620). Subsample systematically.
    if len(grid) > 220:
        step = max(1, len(grid) // 200)
        grid = grid[::step]
    return grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--frame', type=Path, default=None)
    parser.add_argument('--top', type=int, default=8)
    parser.add_argument('--apply-best', action='store_true')
    parser.add_argument('--label', default='', help='optional spawn/scene tag')
    args = parser.parse_args()

    from inference.modules import lane_detection as ld

    ld.VISUALIZE = False
    ld.VISUALIZE_MODE = ld.VISUALIZE_OFF

    if args.frame is not None:
        frame = cv2.imread(str(args.frame))
        if frame is None:
            raise SystemExit(f'cannot read {args.frame}')
        print(f'[auto-fork] using frame {args.frame}')
    else:
        print(f'[auto-fork] grabbing {args.topic} ...')
        frame = grab_compressed_frame(args.topic)
        print(f'[auto-fork] got frame {frame.shape}')

    stamp = _stamp()
    scene = resolve_scene(args.label)
    run_dir = OUT_ROOT / scene / 'runs' / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    verify_dir = OUT_ROOT / scene / 'verify'
    verify_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / 'source_frame.png'), frame)
    (run_dir / 'scene.txt').write_text(scene + '\n', encoding='utf-8')

    grid = build_grid()
    print(f'[auto-fork] sweeping {len(grid)} param sets → {run_dir}')
    results: list[tuple[ForkScore, Any, np.ndarray]] = []
    for i, params in enumerate(grid):
        scored, debug, preview = score_fork_result(ld, frame, params)
        results.append((scored, debug, preview))
        if (i + 1) % 20 == 0 or i == 0:
            print(
                f'  [{i + 1}/{len(grid)}] score={scored.score:.2f} '
                f'a={params.assoc_m:.3f} m={params.min_rows} '
                f'w={params.pair_width_m:.2f} | {scored.notes}'
            )

    results.sort(key=lambda item: item[0].score, reverse=True)
    top = results[: max(1, args.top)]
    ranking = []
    for rank, (scored, debug, preview) in enumerate(top, start=1):
        path = save_candidate(run_dir, frame, preview, debug, scored, rank)
        ranking.append(
            {
                'rank': rank,
                'dir': path.name,
                'score': scored.score,
                'params': asdict(scored.params),
                'notes': scored.notes,
            }
        )
        print(
            f'[top {rank}] score={scored.score:.2f} '
            f'a={scored.params.assoc_m:.3f} m={scored.params.min_rows} '
            f'w={scored.params.pair_width_m:.2f} f={scored.params.far_zone:.2f} '
            f'g={scored.params.max_row_gap} n={scored.params.near_zone:.2f} '
            f'| {scored.notes} → {path.name}'
        )

    (run_dir / 'ranking.json').write_text(
        json.dumps(ranking, indent=2), encoding='utf-8'
    )
    best = top[0][0]
    # Scene + global latest pointers (relative to repo).
    rel = str(run_dir.relative_to(_REPO_ROOT))
    (OUT_ROOT / scene / 'LATEST_RUN.txt').write_text(rel + '\n', encoding='utf-8')
    (OUT_ROOT / 'LATEST_RUN.txt').write_text(rel + '\n', encoding='utf-8')
    # Copy best preview to scene verify/ for quick glance.
    best_preview = top[0][2]
    cv2.imwrite(str(verify_dir / 'current.png'), best_preview)
    cv2.imwrite(str(verify_dir / f'{stamp}.png'), best_preview)
    # Append INDEX (never wipe history).
    index_path = OUT_ROOT / 'INDEX.md'
    note = ranking[0]['notes'] if ranking else ''
    line = (
        f'| {stamp} | {scene} | {best.score:.1f} | pairs={best.n_pairs} | '
        f'`{rel}` | {note} |\n'
    )
    if not index_path.exists():
        index_path.write_text(
            '# auto_fork run index\n\n'
            '| stamp | scene | score | pairs | path | notes |\n'
            '|-------|-------|------:|-------|------|-------|\n',
            encoding='utf-8',
        )
    with index_path.open('a', encoding='utf-8') as fh:
        fh.write(line)

    if args.apply_best:
        if best.n_pairs < 2 or best.score < 20.0:
            print(
                f'[auto-fork] skip apply-best: score={best.score:.2f} '
                f'pairs={best.n_pairs} (need pairs>=2 and score>=20)'
            )
        else:
            cfg = _REPO_ROOT / 'config' / 'lane_vision.yaml'
            data = yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}
            block = dict(data.get('detect_tune') or {})
            block['fork_track_assoc_m'] = float(best.params.assoc_m)
            block['fork_track_min_rows'] = int(best.params.min_rows)
            # Keep track-width SSOT (0.35); auto width is only a soft preference.
            block['fork_pair_width_m'] = 0.35
            block['fork_far_zone_ratio'] = float(best.params.far_zone)
            block['fork_track_max_row_gap'] = int(best.params.max_row_gap)
            block['fork_near_zone_ratio'] = float(best.params.near_zone)
            block['note'] = (
                f'auto_tune_fork best score={best.score:.2f} '
                f'pairs={best.n_pairs} sep={best.center_sep_m:.3f}m'
                + (f' label={args.label}' if args.label else '')
            )
            data['detect_tune'] = block
            cfg.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                encoding='utf-8',
            )
            print(f'[auto-fork] applied best → {cfg}')

    print(f'[auto-fork] done. scene={scene} run={run_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
