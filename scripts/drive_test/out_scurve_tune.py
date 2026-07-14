#!/usr/bin/env python3
"""OUT S-curve tune: inout_fork → out_fork with live visualization.

Symptom target
--------------
After raising δ_max / steer rate, corners **cut inside** (over-early / over-
aggressive steering). Sweep softer mask_p gains while keeping speed usable.

Metrics (higher score better)
-----------------------------
- reached_out_fork     : odom within ``goal_radius_m`` of out_fork
- fail_ratio           : |CTE| > fail or path lost
- cte_abs_mean / jerk  : smoothness
- steer_abs_p95        : over-command (cutting often shows early high |steer|)
- distance_m / time    : progress (prefer reaching goal, then faster ok)

Visualization (default ON for monitoring)
-----------------------------------------
``--viz control`` → OpenCV ``Lane drive`` (what control follows).
``--viz on`` → + HSV masks. ``--viz off`` disables windows.

Example (sim-auto OFF, bringup ON)::

  PYTHONUNBUFFERED=1 python3 scripts/drive_test/out_scurve_tune.py \\
      --max-sec 35 --viz control
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_INFER = _ROOT / 'src' / 'inference'
_DRIVE = Path(__file__).resolve().parent
for p in (_INFER, _DRIVE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from inference.pipeline import MainPlanner, load_planner_config  # noqa: E402
from inference.modules import lane_detection as ld  # noqa: E402
from viz_util import apply_lane_viz  # noqa: E402

OUT_ROOT = _ROOT / 'data' / 'captures' / 'out_scurve_tune'
SPAWN_YAML = _ROOT / 'src' / 'dracer_sim' / 'config' / 'spawn_poses.yaml'
START_SPAWN = 'inout_fork'
GOAL_SPAWN = 'out_fork'


# Gated hybrid SSOT vs prior winners.
CANDIDATES: dict[str, dict[str, Any]] = {
    'pp_ssot': {
        # Matches config/main_planner.yaml simple PP SSOT.
        'normal_tracker': 'pp',
        'cruise_throttle': 0.38,
        'curve_throttle': 0.24,
        'lookahead_m': 0.75,
        'curve_lookahead_m': 0.50,
        'cte_gain': 0.08,
        'heading_gain': 0.16,
        'steer_command_deadband': 0.04,
        'steering_rate_limit_per_sec': 8.0,
        'max_steering_command': 0.90,
        'error_speed_cte_full_m': 0.22,
        'error_speed_steer_full': 0.65,
        'error_speed_min_scale': 0.72,
    },
    'bal_fast': {
        # Previous S-curve winner (near-only COM).
        'normal_tracker': 'mask_p',
        'cruise_throttle': 0.38,
        'curve_throttle': 0.24,
        'mask_steer_k': 1.45,
        'mask_steer_alpha': 0.28,
        'mask_near_band_ratio': 0.55,
        'mask_far_band_ratio': 0.90,
        'mask_far_blend': 0.0,
        'mask_use_path_correction': False,
        'mask_curve_steer_threshold': 0.30,
        'mask_curve_speed_scale': 0.75,
        'steering_rate_limit_per_sec': 6.5,
        'max_steering_command': 0.78,
    },
    'mask_hard_wide': {
        # Lap-proven mask_policy winner (aggressive).
        'normal_tracker': 'mask_p',
        'cruise_throttle': 0.33,
        'curve_throttle': 0.20,
        'mask_steer_k': 2.0,
        'mask_steer_alpha': 0.40,
        'mask_near_band_ratio': 0.85,
        'mask_far_band_ratio': 0.90,
        'mask_far_blend': 0.0,
        'mask_use_path_correction': False,
        'mask_corridor_mode': 'hard',
        'mask_corridor_half_width_m': 0.38,
        'mask_fork_force_pp': True,
        'mask_curve_steer_threshold': 0.35,
        'mask_curve_speed_scale': 0.80,
        'steering_rate_limit_per_sec': 12.0,
        'max_steering_command': 1.0,
    },
}


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


def load_spawn_xy() -> dict[str, tuple[float, float]]:
    data = yaml.safe_load(SPAWN_YAML.read_text(encoding='utf-8'))
    poses = data.get('poses') or {}
    out: dict[str, tuple[float, float]] = {}
    for name, entry in poses.items():
        out[str(name)] = (float(entry['x']), float(entry['y']))
    return out


def build_planner(overrides: dict[str, Any]) -> MainPlanner:
    base = load_planner_config(route_mode='out')
    return MainPlanner(replace(base, **overrides))


def score_run(
    rows: list[dict[str, Any]],
    *,
    reached: bool,
    fail_cte_m: float,
    elapsed_sec: float,
    goal_dist_final: float,
) -> dict[str, Any]:
    if not rows:
        return {
            'n_rows': 0,
            'score': -1e9,
            'ok': False,
            'reached_out_fork': False,
            'reason': 'no_rows',
        }

    steers = np.asarray([float(r['steering']) for r in rows], dtype=np.float64)
    ctes = np.asarray(
        [float(r['cte_m']) for r in rows if r.get('cte_m') is not None],
        dtype=np.float64,
    )
    jerks = np.abs(np.diff(steers)) if steers.size > 1 else np.asarray([0.0])
    path_lost = sum(1 for r in rows if r.get('path_lost'))
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
    xs = [r['odom_x'] for r in rows if r.get('odom_x') is not None]
    ys = [r['odom_y'] for r in rows if r.get('odom_y') is not None]
    if len(xs) >= 2:
        for i in range(1, len(xs)):
            dist += math.hypot(float(xs[i]) - float(xs[i - 1]), float(ys[i]) - float(ys[i - 1]))

    cte_abs_mean = float(np.mean(np.abs(ctes))) if ctes.size else float('nan')
    cte_abs_p95 = float(np.percentile(np.abs(ctes), 95)) if ctes.size else float('nan')
    steer_abs_p95 = float(np.percentile(np.abs(steers), 95))
    steer_jerk = float(np.mean(jerks))
    fail_ratio = fail / len(rows)

    # Primary: reach out_fork without dying. Then lane quality / no cutting.
    score = (
        (45.0 if reached else 0.0)
        + 2.0 * dist
        - 12.0 * fail_ratio
        - 4.0 * (cte_abs_mean if math.isfinite(cte_abs_mean) else 1.0)
        - 6.0 * steer_jerk
        - 3.0 * max(0.0, steer_abs_p95 - 0.55)  # penalize near-full lock overuse
        - 0.8 * float(goal_dist_final)
        - (0.15 * elapsed_sec if reached else 0.0)
        - 0.02 * path_lost
    )
    ok = bool(reached and fail_ratio < 0.25)
    return {
        'n_rows': len(rows),
        'reached_out_fork': reached,
        'ok': ok,
        'score': round(score, 3),
        'distance_m': round(dist, 3),
        'elapsed_sec': round(elapsed_sec, 2),
        'goal_dist_final_m': round(goal_dist_final, 3),
        'fail_ratio': round(fail_ratio, 4),
        'path_lost_frames': path_lost,
        'cte_abs_mean': None if not math.isfinite(cte_abs_mean) else round(cte_abs_mean, 4),
        'cte_abs_p95': None if not math.isfinite(cte_abs_p95) else round(cte_abs_p95, 4),
        'steer_abs_p95': round(steer_abs_p95, 4),
        'steer_jerk_mean': round(steer_jerk, 4),
    }


def run_one(
    *,
    name: str,
    overrides: dict[str, Any],
    out_dir: Path,
    catalog: dict[str, tuple[float, float]],
    camera_topic: str,
    max_sec: float,
    settle_sec: float,
    fail_cte_m: float,
    goal_radius_m: float,
    viz: str,
) -> dict[str, Any]:
    import rclpy
    from control_msgs.msg import Control
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    from inference.types import DrivingState
    import cv2

    print(f'\n=== START variant={name} viz={viz}', flush=True)
    print(f'    overrides={overrides}', flush=True)

    _teleport(START_SPAWN)
    time.sleep(settle_sec)

    planner = build_planner(overrides)
    chosen = apply_lane_viz(viz)
    print(f'    Lane viz mode → {chosen} (watch OpenCV ``Lane drive``)', flush=True)
    ld._apply_detect_tune_from_yaml()

    gx, gy = catalog[GOAL_SPAWN]
    rclpy.init()
    node = Node(f'out_scurve_{name}')
    bridge = CvBridge()
    latest: dict[str, Any] = {'frame': None, 'odom_x': None, 'odom_y': None}

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

    run_dir = _ensure(out_dir / name)
    csv_path = run_dir / 'drive.csv'
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    last_step = t0
    reached = False

    try:
        while time.time() - t0 < max_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = latest['frame']
            if frame is None:
                continue
            now = time.time()
            dt = max(0.02, now - last_step)
            last_step = now
            planner.state = DrivingState.NORMAL
            output = planner.step(frame, now_sec=now)
            cmd = output.command
            msg = Control()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.steering = float(cmd.steering)
            msg.throttle = float(cmd.throttle)
            control_pub.publish(msg)

            dbg = output.debug or {}
            ox, oy = latest.get('odom_x'), latest.get('odom_y')
            goal_d = (
                math.hypot(float(ox) - gx, float(oy) - gy)
                if ox is not None and oy is not None
                else float('nan')
            )
            if math.isfinite(goal_d) and goal_d <= goal_radius_m:
                reached = True

            path_lost = str(output.path_source.value).startswith('hold')
            rows.append(
                {
                    't': round(now - t0, 3),
                    'steering': float(cmd.steering),
                    'throttle': float(cmd.throttle),
                    'cte_m': dbg.get('cross_track_error_m'),
                    'path_source': output.path_source.value,
                    'path_lost': path_lost,
                    'odom_x': ox,
                    'odom_y': oy,
                    'goal_dist_m': None if not math.isfinite(goal_d) else round(goal_d, 3),
                    'steer_k': overrides.get('mask_steer_k'),
                }
            )
            if reached:
                break
    finally:
        for _ in range(6):
            stop = Control()
            stop.header.stamp = node.get_clock().now().to_msg()
            stop.header.frame_id = 'base_link'
            stop.steering = 0.0
            stop.throttle = 0.0
            control_pub.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.02)
        node.destroy_node()
        rclpy.shutdown()

    elapsed = time.time() - t0
    last = rows[-1] if rows else {}
    goal_final = float(last.get('goal_dist_m') or 99.0)
    metrics = score_run(
        rows,
        reached=reached,
        fail_cte_m=fail_cte_m,
        elapsed_sec=elapsed,
        goal_dist_final=goal_final,
    )

    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        'variant': name,
        'overrides': overrides,
        'csv': str(csv_path),
        'start': START_SPAWN,
        'goal': GOAL_SPAWN,
        'viz': viz,
        **metrics,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps({k: summary[k] for k in summary if k != 'overrides'}, ensure_ascii=False), flush=True)
    time.sleep(0.6)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--candidates',
        default=','.join(CANDIDATES.keys()),
        help='comma candidate ids',
    )
    parser.add_argument('--max-sec', type=float, default=35.0)
    parser.add_argument('--settle', type=float, default=1.0)
    parser.add_argument('--fail-cte-m', type=float, default=0.28)
    parser.add_argument('--goal-radius-m', type=float, default=1.2)
    parser.add_argument('--camera-topic', default='/camera/image/compressed')
    parser.add_argument(
        '--viz',
        default='control',
        help='OpenCV perception windows: off|control|on (default control — monitor SSOT)',
    )
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    names = [n.strip() for n in args.candidates.split(',') if n.strip()]
    for n in names:
        if n not in CANDIDATES:
            raise SystemExit(f'unknown candidate {n}; known={list(CANDIDATES)}')

    catalog = load_spawn_xy()
    run_root = _ensure(args.out or (OUT_ROOT / _stamp()))
    meta = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'start': START_SPAWN,
        'goal': GOAL_SPAWN,
        'candidates': names,
        'viz': args.viz,
        'max_sec': args.max_sec,
        'goal_radius_m': args.goal_radius_m,
        'note': 'sim-auto OFF; script owns /control; Lane drive windows open by default',
    }
    (run_root / 'meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(f'=== out_scurve_tune → {run_root} (viz={args.viz})', flush=True)

    results: list[dict[str, Any]] = []
    for name in names:
        try:
            summary = run_one(
                name=name,
                overrides=CANDIDATES[name],
                out_dir=run_root,
                catalog=catalog,
                camera_topic=args.camera_topic,
                max_sec=args.max_sec,
                settle_sec=args.settle,
                fail_cte_m=args.fail_cte_m,
                goal_radius_m=args.goal_radius_m,
                viz=args.viz,
            )
        except Exception as exc:  # noqa: BLE001
            summary = {
                'variant': name,
                'ok': False,
                'reached_out_fork': False,
                'score': -1e9,
                'error': str(exc),
            }
            print(f'ERROR {name}: {exc}', file=sys.stderr, flush=True)
        results.append(summary)

    ranked = sorted(results, key=lambda r: float(r.get('score', -1e9)), reverse=True)
    (run_root / 'ranking.json').write_text(
        json.dumps(ranked, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    lines = [
        f'# OUT S-curve tune {run_root.name}',
        '',
        f'start=`{START_SPAWN}` goal=`{GOAL_SPAWN}` viz=`{args.viz}` max_sec={args.max_sec}',
        '',
        '| variant | score | reached | ok | dist_m | cte_mean | steer_p95 | jerk | fail |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in ranked:
        lines.append(
            f"| {row.get('variant')} | {row.get('score')} | {row.get('reached_out_fork')} | "
            f"{row.get('ok')} | {row.get('distance_m')} | {row.get('cte_abs_mean')} | "
            f"{row.get('steer_abs_p95')} | {row.get('steer_jerk_mean')} | {row.get('fail_ratio')} |"
        )
    winner = ranked[0] if ranked else None
    if winner is not None:
        lines.extend(
            [
                '',
                f"**Winner: `{winner.get('variant')}`** score={winner.get('score')}",
                '',
                '```json',
                json.dumps(winner.get('overrides') or CANDIDATES.get(str(winner.get('variant')), {}), indent=2),
                '```',
                '',
            ]
        )
    (run_root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote {run_root}', flush=True)
    if winner:
        print(f"Winner: {winner.get('variant')} score={winner.get('score')}", flush=True)
    return 0 if any(r.get('ok') for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
