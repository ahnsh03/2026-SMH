#!/usr/bin/env python3
"""Headless dash-connect auto-tune against a live camera (or saved frame).

Grabs /camera/image/compressed, sweeps gap/lat/head (+ optional max lateral
jump), scores Phase-A fork quality, writes ranked captures under
data/captures/lane_tune_logs/auto_dash/.

Example (inside 2026-smh-sim):

  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 scripts/vision_tune/auto_tune_dash.py
  python3 scripts/vision_tune/auto_tune_dash.py --frame data/captures/lane_tune_logs/.../frame.png
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

OUT_ROOT = _REPO_ROOT / 'data' / 'captures' / 'lane_tune_logs' / 'auto_dash'


@dataclass(frozen=True)
class DashParams:
    gap_m: float
    lat_m: float
    head_deg: float
    area_px: int = 12
    max_lat_jump_m: float = 0.18


@dataclass
class DashScore:
    params: DashParams
    score: float
    n_tracks: int
    n_far_tracks: int
    n_pairs: int
    raw_px: int
    link_px: int
    conn_px: int
    cross_rows: int
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
            super().__init__('auto_tune_dash_grab')
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


def _count_row_segments(mask_row: np.ndarray, min_w: int = 2) -> int:
    cols = np.flatnonzero(mask_row)
    if cols.size == 0:
        return 0
    breaks = np.flatnonzero(np.diff(cols) > 1)
    starts = cols[np.concatenate(([0], breaks + 1))]
    ends = cols[np.concatenate((breaks, [cols.size - 1]))]
    return sum(1 for a, b in zip(starts, ends) if b - a + 1 >= min_w)


def score_dash_result(ld: Any, frame: np.ndarray, params: DashParams) -> tuple[DashScore, Any, np.ndarray]:
    ld.apply_detect_tune(
        dash_max_forward_gap_m=params.gap_m,
        dash_max_lateral_error_m=params.lat_m,
        dash_max_heading_diff_deg=params.head_deg,
        dash_min_component_area_px=params.area_px,
    )
    # Optional: stash jump gate if module exposes it.
    if hasattr(ld, 'DASH_MAX_LATERAL_JUMP_M'):
        ld.DASH_MAX_LATERAL_JUMP_M = float(params.max_lat_jump_m)

    _dets, debug = ld.detect_with_debug(frame)
    yellow_pts = debug.yellow_dash_points_bev
    yellow_conn = debug.yellow_connected_bev
    if yellow_pts.size == 0:
        yellow_pts = np.zeros(debug.bev.shape[:2], dtype=np.uint8)
    if yellow_conn.size == 0:
        yellow_conn = yellow_pts

    raw_px = int(np.count_nonzero(yellow_pts))
    conn_px = int(np.count_nonzero(yellow_conn))
    link_px = int(np.count_nonzero((yellow_conn > 0) & (yellow_pts == 0)))

    tracks = ld.track_marking_polylines(yellow_conn)
    n_tracks = len(tracks)
    far_end = max(1, int(round(ld.BEV_HEIGHT * 0.45)))
    n_far = sum(1 for t in tracks if np.any(~np.isnan(t[:far_end])))
    pairs, _ = ld.extract_yellow_fork_lane_pairs(yellow_conn)
    n_pairs = len(pairs)

    # Cross-merge detector: mid band rows with a single wide yellow blob
    # spanning more than ~0.5 m (likely L+R inners fused).
    mid0 = int(ld.BEV_HEIGHT * 0.35)
    mid1 = int(ld.BEV_HEIGHT * 0.75)
    wide_px = int(round(0.45 / ld.METERS_PER_PIXEL))
    cross_rows = 0
    for row in range(mid0, mid1):
        segs = []
        cols = np.flatnonzero(yellow_conn[row] > 0)
        if cols.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(cols) > 1)
        starts = cols[np.concatenate(([0], breaks + 1))]
        ends = cols[np.concatenate((breaks, [cols.size - 1]))]
        for a, b in zip(starts.tolist(), ends.tolist()):
            segs.append(b - a + 1)
        if len(segs) == 1 and segs[0] >= wide_px:
            cross_rows += 1
        elif len(segs) >= 2:
            # Adjacent segment centers closer than 0.12 m → sticky
            centers = []
            for a, b in zip(starts.tolist(), ends.tolist()):
                centers.append(0.5 * (a + b))
            centers.sort()
            for i in range(len(centers) - 1):
                if abs(centers[i + 1] - centers[i]) * ld.METERS_PER_PIXEL < 0.10:
                    cross_rows += 1
                    break

    # Far-zone strand count preference: 4 is ideal at exit fork.
    far_seg_counts = [
        _count_row_segments(yellow_conn[r])
        for r in range(0, far_end, 3)
    ]
    far_seg_counts = [c for c in far_seg_counts if c > 0]
    median_far_segs = float(np.median(far_seg_counts)) if far_seg_counts else 0.0

    link_ratio = link_px / max(1, conn_px)

    # Score: prefer 3–5 far tracks / ~4 median segments, some link fill,
    # strongly penalize cross merges, mild prefer pairs==2.
    score = 0.0
    score += 12.0 * np.exp(-0.5 * ((n_far - 4) / 1.2) ** 2)
    score += 8.0 * np.exp(-0.5 * ((median_far_segs - 4.0) / 1.0) ** 2)
    score += 6.0 * min(1.0, link_ratio / 0.08)  # reward connect fill up to ~8%
    if n_pairs >= 2:
        score += 5.0
    score -= 0.35 * float(cross_rows)
    if n_far < 2:
        score -= 10.0
    if link_ratio > 0.25:
        score -= 4.0  # over-connected soup

    notes = (
        f'far={n_far} medSeg={median_far_segs:.1f} pairs={n_pairs} '
        f'link%={100 * link_ratio:.1f} crossRows={cross_rows}'
    )
    result = DashScore(
        params=params,
        score=float(score),
        n_tracks=n_tracks,
        n_far_tracks=n_far,
        n_pairs=n_pairs,
        raw_px=raw_px,
        link_px=link_px,
        conn_px=conn_px,
        cross_rows=int(cross_rows),
        notes=notes,
    )
    preview = ld.make_dash_preview(debug, focus='all')
    return result, debug, preview


def save_candidate(
    out_dir: Path,
    frame: np.ndarray,
    preview: np.ndarray,
    debug: Any,
    result: DashScore,
    rank: int,
) -> Path:
    name = (
        f'r{rank:02d}_s{result.score:05.1f}_'
        f'g{result.params.gap_m:.2f}_l{result.params.lat_m:.3f}_'
        f'h{result.params.head_deg:.0f}'
    )
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / 'frame.png'), frame)
    cv2.imwrite(str(d / 'preview.png'), preview)
    if debug.yellow_dash_points_bev.size:
        cv2.imwrite(str(d / 'yellow_dash_points.png'), debug.yellow_dash_points_bev)
    if debug.yellow_connected_bev.size:
        cv2.imwrite(str(d / 'yellow_connected.png'), debug.yellow_connected_bev)
    if debug.road_clean.size:
        cv2.imwrite(str(d / 'road_clean.png'), debug.road_clean)
    link = (
        (debug.yellow_connected_bev > 0)
        & (debug.yellow_dash_points_bev == 0)
    ).astype(np.uint8) * 255
    cv2.imwrite(str(d / 'link_only.png'), link)
    meta = {
        'rank': rank,
        'score': result.score,
        'params': asdict(result.params),
        'metrics': {
            'n_tracks': result.n_tracks,
            'n_far_tracks': result.n_far_tracks,
            'n_pairs': result.n_pairs,
            'raw_px': result.raw_px,
            'link_px': result.link_px,
            'conn_px': result.conn_px,
            'cross_rows': result.cross_rows,
            'notes': result.notes,
        },
    }
    (d / 'meta.yaml').write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding='utf-8',
    )
    return d


def build_grid() -> list[DashParams]:
    gaps = [0.15, 0.20, 0.25, 0.30, 0.35]
    lats = [0.03, 0.04, 0.05, 0.06, 0.08]
    heads = [20, 27, 35]
    jumps = [0.14, 0.18, 0.22]
    grid: list[DashParams] = []
    for g in gaps:
        for la in lats:
            for h in heads:
                for j in jumps:
                    grid.append(
                        DashParams(
                            gap_m=g, lat_m=la, head_deg=float(h), max_lat_jump_m=j
                        )
                    )
    return grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--frame', type=Path, default=None)
    parser.add_argument('--top', type=int, default=8)
    parser.add_argument('--apply-best', action='store_true')
    args = parser.parse_args()

    from inference.modules import lane_detection as ld

    ld.VISUALIZE = False
    ld.VISUALIZE_MODE = ld.VISUALIZE_OFF

    if args.frame is not None:
        frame = cv2.imread(str(args.frame))
        if frame is None:
            raise SystemExit(f'cannot read {args.frame}')
        print(f'[auto] using frame {args.frame}')
    else:
        print(f'[auto] grabbing {args.topic} ...')
        frame = grab_compressed_frame(args.topic)
        print(f'[auto] got frame {frame.shape}')

    run_dir = OUT_ROOT / _stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / 'source_frame.png'), frame)

    grid = build_grid()
    print(f'[auto] sweeping {len(grid)} param sets → {run_dir}')
    results: list[tuple[DashScore, Any, np.ndarray]] = []
    for i, params in enumerate(grid):
        scored, debug, preview = score_dash_result(ld, frame, params)
        results.append((scored, debug, preview))
        if (i + 1) % 25 == 0 or i == 0:
            print(
                f'  [{i + 1}/{len(grid)}] score={scored.score:.2f} '
                f'g={params.gap_m:.2f} l={params.lat_m:.3f} h={params.head_deg:.0f} '
                f'j={params.max_lat_jump_m:.2f} | {scored.notes}'
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
            f'g={scored.params.gap_m:.2f} l={scored.params.lat_m:.3f} '
            f'h={scored.params.head_deg:.0f} j={scored.params.max_lat_jump_m:.2f} '
            f'| {scored.notes} → {path.name}'
        )

    (run_dir / 'ranking.json').write_text(
        json.dumps(ranking, indent=2), encoding='utf-8'
    )
    latest = OUT_ROOT / 'LATEST_RUN.txt'
    latest.write_text(str(run_dir.relative_to(_REPO_ROOT)) + '\n', encoding='utf-8')

    best = top[0][0]
    if args.apply_best:
        cfg = _REPO_ROOT / 'config' / 'lane_vision.yaml'
        data = yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}
        block = dict(data.get('detect_tune') or {})
        block['dash_max_forward_gap_m'] = float(best.params.gap_m)
        block['dash_max_lateral_error_m'] = float(best.params.lat_m)
        block['dash_max_heading_diff_deg'] = float(best.params.head_deg)
        block['dash_min_component_area_px'] = int(best.params.area_px)
        block['note'] = (
            f'auto_tune_dash best score={best.score:.2f} '
            f'jump={best.params.max_lat_jump_m:.2f}m'
        )
        data['detect_tune'] = block
        cfg.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding='utf-8',
        )
        print(f'[auto] applied best → {cfg}')

    print(f'[auto] done. LATEST_RUN → {latest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
