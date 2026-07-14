#!/usr/bin/env python3
"""Live NORMAL-section mask-steer / PP A-B bench with logging.

Design
------
- Owns /control (do **not** run sim-auto / inference_node at the same time).
- Teleports to spawn presets, drives for ``duration``, logs CSV + metrics.
- Variants override tracker + speed gains via dataclasses.replace.

Metrics (lower is smoother / safer unless noted)
------------------------------------------------
- steer_rms, steer_jerk_mean   : command smoothness
- cte_abs_mean / cte_abs_p95   : lane-center lateral error while driving
- mask_valid_ratio             : fraction of mask_p successes (mask only)
- distance_m                   : odom progress (higher = faster/further OK)
- fail_ratio                   : frames with |cte| > fail_cte_m or path lost
- score                        : composite for ranking (higher better)

Example (inside 2026-smh-sim, bringup only)::

  python3 scripts/drive_test/mask_steer_bench.py \\
      --segments start,out_in_merge --duration 7 --repeat 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_INFER = _ROOT / 'src' / 'inference'
if str(_INFER) not in sys.path:
    sys.path.insert(0, str(_INFER))

from inference.pipeline import MainPlanner, PlannerConfig, load_planner_config
from inference.modules import lane_detection as ld

OUT_ROOT = _ROOT / 'data' / 'captures' / 'mask_steer_logs'


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _teleport(spawn: str) -> None:
    import subprocess

    subprocess.check_call(
        [sys.executable, str(_ROOT / 'scripts' / 'teleport_spawn_pose.py'), spawn],
        cwd=str(_ROOT),
    )


VARIANTS: dict[str, dict[str, Any]] = {
    'pp_base': {
        'normal_tracker': 'pp',
        'cruise_throttle': 0.17,
        'curve_throttle': 0.08,
    },
    'pp_fast': {
        'normal_tracker': 'pp',
        'cruise_throttle': 0.28,
        'curve_throttle': 0.12,
    },
    'mask_p_base': {
        'normal_tracker': 'mask_p',
        'cruise_throttle': 0.17,
        'mask_steer_k': 2.0,
        'mask_steer_alpha': 0.4,
        'mask_near_band_ratio': 0.55,
        'mask_curve_speed_scale': 0.80,
    },
    'mask_p_fast': {
        'normal_tracker': 'mask_p',
        'cruise_throttle': 0.28,
        'mask_steer_k': 1.6,
        'mask_steer_alpha': 0.35,
        'mask_near_band_ratio': 0.50,
        'mask_curve_speed_scale': 0.75,
    },
    'mask_p_soft': {
        'normal_tracker': 'mask_p',
        'cruise_throttle': 0.22,
        'mask_steer_k': 1.3,
        'mask_steer_alpha': 0.25,
        'mask_near_band_ratio': 0.45,
        'mask_curve_speed_scale': 0.70,
        'steering_rate_limit_per_sec': 3.5,
    },
    'mask_p_tuned': {
        # Round-1 winner (mask_p_fast) + soft smoothing on merge.
        'normal_tracker': 'mask_p',
        'cruise_throttle': 0.26,
        'mask_steer_k': 1.45,
        'mask_steer_alpha': 0.30,
        'mask_near_band_ratio': 0.48,
        'mask_curve_speed_scale': 0.72,
        'steering_rate_limit_per_sec': 4.0,
    },
}


def build_planner(variant: str, route: str) -> MainPlanner:
    base = load_planner_config(route_mode=route)
    overrides = VARIANTS[variant]
    cfg = replace(base, **overrides)
    return MainPlanner(cfg)


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    fail_cte_m: float,
) -> dict[str, Any]:
    if not rows:
        return {
            'n_rows': 0,
            'score': -1e9,
            'ok': False,
            'reason': 'no_rows',
        }

    steers = np.array([float(r['steering']) for r in rows], dtype=np.float64)
    ctes = np.array(
        [float(r['cte_m']) for r in rows if r.get('cte_m') is not None],
        dtype=np.float64,
    )
    jerks = np.abs(np.diff(steers)) if steers.size > 1 else np.array([0.0])
    path_lost = sum(1 for r in rows if r.get('path_lost'))
    mask_ok = sum(1 for r in rows if r.get('path_source') == 'mask_drivable')
    mask_frames = sum(
        1
        for r in rows
        if str(r.get('tracker', '')).startswith('mask')
        or r.get('path_source') in ('mask_drivable', 'white_centerline', 'yellow_centerline')
    )
    fail = sum(
        1
        for r in rows
        if r.get('path_lost')
        or (
            r.get('cte_m') is not None
            and abs(float(r['cte_m'])) > fail_cte_m
        )
    )
    dist = 0.0
    xs = [r.get('odom_x') for r in rows if r.get('odom_x') is not None]
    ys = [r.get('odom_y') for r in rows if r.get('odom_y') is not None]
    if len(xs) >= 2:
        for i in range(1, len(xs)):
            dist += math.hypot(float(xs[i]) - float(xs[i - 1]), float(ys[i]) - float(ys[i - 1]))

    cte_abs_mean = float(np.mean(np.abs(ctes))) if ctes.size else float('nan')
    cte_abs_p95 = float(np.percentile(np.abs(ctes), 95)) if ctes.size else float('nan')
    steer_rms = float(np.sqrt(np.mean(steers**2)))
    steer_jerk = float(np.mean(jerks))
    fail_ratio = fail / len(rows)
    mask_valid_ratio = (mask_ok / len(rows)) if rows else 0.0

    # Higher score is better: reward distance, penalize fail/jerk/CTE.
    score = (
        3.0 * dist
        - 8.0 * fail_ratio
        - 2.5 * (cte_abs_mean if math.isfinite(cte_abs_mean) else 1.0)
        - 4.0 * steer_jerk
        - 1.0 * steer_rms
        + 0.5 * mask_valid_ratio
    )
    return {
        'n_rows': len(rows),
        'distance_m': round(dist, 3),
        'steer_rms': round(steer_rms, 4),
        'steer_jerk_mean': round(steer_jerk, 4),
        'cte_abs_mean': None if not math.isfinite(cte_abs_mean) else round(cte_abs_mean, 4),
        'cte_abs_p95': None if not math.isfinite(cte_abs_p95) else round(cte_abs_p95, 4),
        'mask_valid_ratio': round(mask_valid_ratio, 3),
        'path_lost_frames': path_lost,
        'fail_frames': fail,
        'fail_ratio': round(fail_ratio, 3),
        'score': round(score, 3),
        'ok': fail_ratio < 0.35 and dist > 0.15,
    }


def run_live(
    *,
    segment: str,
    variant: str,
    route: str,
    duration_sec: float,
    out_dir: Path,
    camera_topic: str,
    fail_cte_m: float,
    settle_sec: float,
) -> dict[str, Any]:
    import rclpy
    from cv_bridge import CvBridge
    from control_msgs.msg import Control
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    _teleport(segment)
    time.sleep(settle_sec)

    planner = build_planner(variant, route)
    ld.VISUALIZE = False
    ld._apply_detect_tune_from_yaml()

    rclpy.init()
    node = Node('mask_steer_bench')
    bridge = CvBridge()
    latest: dict[str, Any] = {
        'frame': None,
        'odom_x': None,
        'odom_y': None,
    }

    def _cb_raw(msg: Image) -> None:
        latest['frame'] = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _cb_compressed(msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        latest['frame'] = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _cb_odom(msg: Odometry) -> None:
        latest['odom_x'] = float(msg.pose.pose.position.x)
        latest['odom_y'] = float(msg.pose.pose.position.y)

    if camera_topic.endswith('compressed'):
        node.create_subscription(CompressedImage, camera_topic, _cb_compressed, 10)
    else:
        node.create_subscription(Image, camera_topic, _cb_raw, 10)
    node.create_subscription(Odometry, '/odom', _cb_odom, 10)
    control_pub = node.create_publisher(Control, '/control', 10)

    run_dir = _ensure(out_dir / f'{variant}__{segment}')
    csv_path = run_dir / 'drive.csv'
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    last_step = t0
    first_frame = None

    try:
        while time.time() - t0 < duration_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = latest['frame']
            if frame is None:
                continue
            now = time.time()
            dt = max(0.02, now - last_step)
            last_step = now
            # Keep NORMAL so we compare trackers without fork FSM noise.
            from inference.types import DrivingState

            planner.state = DrivingState.NORMAL
            output = planner.step(frame, now_sec=now)
            cmd = output.command
            msg = Control()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.steering = float(cmd.steering)
            msg.throttle = float(cmd.throttle)
            control_pub.publish(msg)

            if first_frame is None:
                first_frame = frame.copy()

            dbg = output.debug or {}
            path_lost = str(output.path_source.value).startswith('hold')
            row = {
                't': round(now - t0, 3),
                'variant': variant,
                'segment': segment,
                'tracker': dbg.get('normal_tracker'),
                'state': str(output.state.value),
                'decision': output.decision,
                'path_source': str(output.path_source.value),
                'steering': float(cmd.steering),
                'throttle': float(cmd.throttle),
                'cte_m': dbg.get('cross_track_error_m'),
                'raw_steering': dbg.get('raw_steering'),
                'curve_ratio': dbg.get('curve_ratio'),
                'path_lost': path_lost,
                'odom_x': latest['odom_x'],
                'odom_y': latest['odom_y'],
                'dt': round(dt, 4),
            }
            rows.append(row)
            time.sleep(0.01)
    finally:
        # Always stop the car between variants.
        stop = Control()
        stop.header.stamp = node.get_clock().now().to_msg()
        stop.header.frame_id = 'base_link'
        stop.steering = 0.0
        stop.throttle = 0.0
        for _ in range(5):
            control_pub.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.02)
        node.destroy_node()
        rclpy.shutdown()

    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if first_frame is not None:
        cv2.imwrite(str(run_dir / 'frame0.png'), first_frame)

    metrics = summarize_rows(rows, fail_cte_m=fail_cte_m)
    summary = {
        'variant': variant,
        'segment': segment,
        'route': route,
        'duration_sec': duration_sec,
        'config': VARIANTS[variant],
        'csv': str(csv_path),
        **metrics,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--segments',
        default='start,out_in_merge',
        help='comma spawn ids from spawn_poses.yaml',
    )
    parser.add_argument(
        '--variants',
        default='pp_base,mask_p_base,mask_p_soft,mask_p_fast,pp_fast',
        help='comma variant ids',
    )
    parser.add_argument('--route', default='out', choices=('out', 'in'))
    parser.add_argument('--duration', type=float, default=7.0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--settle', type=float, default=0.9)
    parser.add_argument('--fail-cte-m', type=float, default=0.22)
    parser.add_argument('--camera-topic', default='/camera/image/compressed')
    args = parser.parse_args()

    segments = [s.strip() for s in args.segments.split(',') if s.strip()]
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    for name in variants:
        if name not in VARIANTS:
            print(f'unknown variant: {name}', file=sys.stderr)
            print('known:', ', '.join(VARIANTS), file=sys.stderr)
            return 2

    stamp = _stamp()
    out_root = _ensure(OUT_ROOT / stamp)
    index: list[dict[str, Any]] = []

    meta = {
        'stamp': stamp,
        'segments': segments,
        'variants': variants,
        'route': args.route,
        'duration_sec': args.duration,
        'repeat': args.repeat,
        'fail_cte_m': args.fail_cte_m,
        'note': 'inference_node / sim-auto must be OFF; this script publishes /control',
    }
    (out_root / 'META.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    for rep in range(args.repeat):
        for segment in segments:
            for variant in variants:
                label = f'r{rep:02d}_{variant}__{segment}'
                print(f'=== {label}')
                try:
                    summary = run_live(
                        segment=segment,
                        variant=variant,
                        route=args.route,
                        duration_sec=args.duration,
                        out_dir=_ensure(out_root / f'rep{rep:02d}'),
                        camera_topic=args.camera_topic,
                        fail_cte_m=args.fail_cte_m,
                        settle_sec=args.settle,
                    )
                except Exception as exc:  # noqa: BLE001 — log and continue sweep
                    summary = {
                        'variant': variant,
                        'segment': segment,
                        'ok': False,
                        'score': -1e9,
                        'error': str(exc),
                    }
                    print(f'ERROR {label}: {exc}', file=sys.stderr)
                index.append(summary)
                print(json.dumps(summary, ensure_ascii=False))
                time.sleep(0.4)

    (out_root / 'INDEX.json').write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    # Rank by segment then overall.
    lines = [
        f'# Mask steer bench {stamp}',
        '',
        f"route=`{args.route}` duration={args.duration}s fail_cte={args.fail_cte_m}",
        '',
        '| variant | segment | score | dist_m | cte_mean | jerk | fail_ratio | ok |',
        '|---|---|---:|---:|---:|---:|---:|---|',
    ]
    ranked = sorted(
        [r for r in index if 'score' in r],
        key=lambda r: float(r.get('score', -1e9)),
        reverse=True,
    )
    for row in ranked:
        lines.append(
            f"| {row.get('variant')} | {row.get('segment')} | "
            f"{row.get('score')} | {row.get('distance_m')} | "
            f"{row.get('cte_abs_mean')} | {row.get('steer_jerk_mean')} | "
            f"{row.get('fail_ratio')} | {row.get('ok')} |"
        )

    by_variant: dict[str, list[float]] = {}
    for row in index:
        if 'score' not in row or row.get('error'):
            continue
        by_variant.setdefault(str(row['variant']), []).append(float(row['score']))
    lines.extend(['', '## Mean score by variant', ''])
    means = sorted(
        ((k, float(np.mean(v))) for k, v in by_variant.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for name, mean in means:
        lines.append(f'- **{name}**: {mean:.3f}')
    winner = means[0][0] if means else None
    lines.extend(['', f'**Winner (mean score): `{winner}`**', ''])
    (out_root / 'INDEX.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote {out_root}')
    print(f'Winner: {winner}')
    return 0 if any(r.get('ok') for r in index) else 1


if __name__ == '__main__':
    raise SystemExit(main())
