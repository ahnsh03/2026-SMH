#!/usr/bin/env python3
"""Full-course IN/OUT start→lap evaluation (robustness + speed).

Unlike short segment teleports, every trial:
1. Teleports to ``start``
2. Drives the full ring for ``route_mode`` out|in with FSM enabled
3. Recovers via nearest-course checkpoint teleport on sustained lane loss
4. Logs perception health + control metrics for later analysis

Repeat each (mode × policy) several times and rank by:
- Robustness: lap_ok rate, farthest checkpoint, lost/skip events, CTE, path_lost
- Speed: successful-lap elapsed time and mean odom speed
- Perception: white/yellow visibility & confidence, path_source mix, fork ratio

Owns ``/control`` — do **not** run sim-auto concurrently.

Example::

  PYTHONUNBUFFERED=1 python3 scripts/drive_test/course_mode_bench.py \\
      --modes out,in --repeats 2 --max-lap-sec-out 180 --max-lap-sec-in 240
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_INFER = _ROOT / 'src' / 'inference'
if str(_INFER) not in sys.path:
    sys.path.insert(0, str(_INFER))

from inference.pipeline import MainPlanner, load_planner_config
from inference.modules import lane_detection as ld
from inference.types import DrivingState, TurnSign

OUT_ROOT = _ROOT / 'data' / 'captures' / 'course_mode_logs'
SPAWN_YAML = _ROOT / 'src' / 'dracer_sim' / 'config' / 'spawn_poses.yaml'

OUT_CHECKPOINTS: list[str] = [
    'start',
    'inout_fork',
    'out_fork',
    'out_fork_merge_left',
    'out_in_merge',
    'obstacle',
]

IN_CHECKPOINTS: list[str] = [
    'start',
    'inout_fork',
    'in_roundabout_entry',
    'in_roundabout_exit',
    'in_out_merge',
    'obstacle',
]

_MASK_SPEED = {
    'cruise_throttle': 0.33,
    'mask_steer_k': 2.0,
    'mask_steer_alpha': 0.40,
    'mask_near_band_ratio': 0.85,
    'mask_curve_speed_scale': 0.80,
    'steering_rate_limit_per_sec': 12.0,
}

POLICIES: dict[str, dict[str, Any]] = {
    'pp_ref': {
        'normal_tracker': 'pp',
        'cruise_throttle': 0.31,
        'curve_throttle': 0.16,
        'lookahead_m': 0.70,
        'curve_lookahead_m': 0.38,
        'steering_rate_limit_per_sec': 12.0,
        'cte_gain': 0.10,
        'heading_gain': 0.30,
        'heading_preview_m': 0.45,
    },
    'mask_raw': {
        **_MASK_SPEED,
        'normal_tracker': 'mask_p',
        'mask_corridor_mode': 'off',
        'mask_fork_force_pp': False,
        'mask_require_color_path': False,
    },
    'mask_fork_pp': {
        **_MASK_SPEED,
        'normal_tracker': 'mask_p',
        'mask_corridor_mode': 'off',
        'mask_fork_force_pp': True,
        'mask_require_color_path': False,
    },
    'mask_hard_fork_pp': {
        **_MASK_SPEED,
        'normal_tracker': 'mask_p',
        'mask_corridor_mode': 'hard',
        'mask_corridor_half_width_m': 0.28,
        'mask_fork_force_pp': True,
        'mask_require_color_path': True,
    },
    'mask_soft_fork_pp': {
        **_MASK_SPEED,
        'normal_tracker': 'mask_p',
        'mask_corridor_mode': 'soft',
        'mask_path_weight_sigma_m': 0.20,
        'mask_fork_force_pp': True,
        'mask_require_color_path': True,
    },
    'mask_hard_wide': {
        **_MASK_SPEED,
        'normal_tracker': 'mask_p',
        'mask_corridor_mode': 'hard',
        'mask_corridor_half_width_m': 0.38,
        'mask_fork_force_pp': True,
        'mask_require_color_path': True,
    },
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _teleport(spawn: str, *, retries: int = 4, pause_sec: float = 0.8) -> None:
    import subprocess

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    str(_ROOT / 'scripts' / 'teleport_spawn_pose.py'),
                    spawn,
                ],
                cwd=str(_ROOT),
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(pause_sec * (attempt + 1))
    raise RuntimeError(f'teleport failed after {retries} tries: {spawn}: {last_exc}')


def load_spawn_xy() -> dict[str, tuple[float, float]]:
    data = yaml.safe_load(SPAWN_YAML.read_text(encoding='utf-8')) or {}
    poses = data.get('poses') or {}
    out: dict[str, tuple[float, float]] = {}
    for name, pose in poses.items():
        out[name] = (float(pose['x']), float(pose['y']))
    return out


def checkpoints_for(mode: str) -> list[str]:
    return list(IN_CHECKPOINTS if mode == 'in' else OUT_CHECKPOINTS)


def nearest_spawn(
    x: float,
    y: float,
    names: list[str],
    catalog: dict[str, tuple[float, float]],
) -> tuple[str, int, float]:
    best_name = names[0]
    best_i = 0
    best_d = 1e9
    for i, name in enumerate(names):
        sx, sy = catalog[name]
        d = math.hypot(x - sx, y - sy)
        if d < best_d:
            best_d = d
            best_name = name
            best_i = i
    return best_name, best_i, best_d


def build_planner(mode: str, overrides: dict[str, Any]) -> MainPlanner:
    base = load_planner_config(route_mode=mode)
    clean = {k: v for k, v in overrides.items() if not str(k).startswith('_')}
    cfg = replace(base, **clean)
    planner = MainPlanner(cfg)
    # Prefer LEFT at forks, but do NOT arm fork perception all lap.
    # OUT: forks arm only when a turn sign is seen (route.out_fork_require_sign).
    planner.desired_turn = TurnSign.LEFT
    planner._sign_candidate = TurnSign.LEFT
    planner._sign_candidate_frames = max(1, int(cfg.sign_confirm_frames))
    # Leave _forced_turn UNKNOWN so white-follow stays clean until a sign.
    return planner


def is_lane_lost(output: Any, *, fail_cte_m: float) -> bool:
    decision = str(getattr(output, 'decision', '') or '')
    if decision in ('aruco_stop', 'red_signal_stop', 'wait_green'):
        return False
    dbg = output.debug or {}
    if bool(dbg.get('aruco_stop')):
        return False
    path_src = str(output.path_source.value)
    if path_src == 'stop':
        return False
    if path_src.startswith('hold'):
        return True
    cte = dbg.get('cross_track_error_m')
    if cte is not None and abs(float(cte)) > fail_cte_m:
        return True
    if path_src in ('none',):
        return True
    return False


def summarize_rows(rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            'n_rows': 0,
            'distance_m': 0.0,
            'cte_abs_mean': None,
            'steer_jerk_mean': 0.0,
            'path_lost_ratio': 1.0,
            'white_vis_ratio': 0.0,
            'yellow_vis_ratio': 0.0,
            'white_conf_mean': None,
            'yellow_conf_mean': None,
            'fork_active_ratio': 0.0,
            'path_sources': {},
            'mean_speed_mps': 0.0,
        }

    steers = np.array([float(r['steering']) for r in rows], dtype=np.float64)
    jerks = np.abs(np.diff(steers)) if steers.size > 1 else np.array([0.0])
    ctes = np.array(
        [float(r['cte_m']) for r in rows if r.get('cte_m') is not None],
        dtype=np.float64,
    )
    dist = 0.0
    xs = [r['odom_x'] for r in rows if r.get('odom_x') is not None]
    ys = [r['odom_y'] for r in rows if r.get('odom_y') is not None]
    if len(xs) >= 2:
        for i in range(1, len(xs)):
            dist += math.hypot(float(xs[i]) - float(xs[i - 1]), float(ys[i]) - float(ys[i - 1]))
    elapsed = max(float(rows[-1]['t']) - float(rows[0]['t']), 1e-3)
    path_lost_ratio = sum(1 for r in rows if r.get('path_lost')) / len(rows)
    white_vis = sum(1 for r in rows if r.get('white_visible')) / len(rows)
    yellow_vis = sum(1 for r in rows if r.get('yellow_visible')) / len(rows)
    wconf = [float(r['white_confidence']) for r in rows if r.get('white_confidence') is not None]
    yconf = [float(r['yellow_confidence']) for r in rows if r.get('yellow_confidence') is not None]
    fork_ratio = sum(1 for r in rows if r.get('fork_active')) / len(rows)
    sources = Counter(str(r.get('path_source')) for r in rows)
    lost_events = sum(1 for e in events if e.get('event') == 'lane_lost_recover')
    skip_events = sum(
        1
        for e in events
        if str(e.get('event', '')).startswith('skip')
    )
    cte_mean = float(np.mean(np.abs(ctes))) if ctes.size else float('nan')
    return {
        'n_rows': len(rows),
        'distance_m': round(dist, 3),
        'elapsed_drive_sec': round(elapsed, 2),
        'mean_speed_mps': round(dist / elapsed, 4),
        'cte_abs_mean': None if not math.isfinite(cte_mean) else round(cte_mean, 4),
        'steer_jerk_mean': round(float(np.mean(jerks)), 4),
        'path_lost_ratio': round(path_lost_ratio, 4),
        'white_vis_ratio': round(white_vis, 4),
        'yellow_vis_ratio': round(yellow_vis, 4),
        'white_conf_mean': None if not wconf else round(float(np.mean(wconf)), 4),
        'yellow_conf_mean': None if not yconf else round(float(np.mean(yconf)), 4),
        'fork_active_ratio': round(fork_ratio, 4),
        'path_sources': dict(sources),
        'lost_events': lost_events,
        'skip_events': skip_events,
    }


def run_one_lap(
    *,
    mode: str,
    policy: str,
    overrides: dict[str, Any],
    repeat: int,
    out_dir: Path,
    catalog: dict[str, tuple[float, float]],
    camera_topic: str,
    max_lap_sec: float,
    lost_hold_sec: float,
    max_retries: int,
    fail_cte_m: float,
    lap_min_distance_m: float,
    start_radius_m: float,
    settle_sec: float,
) -> dict[str, Any]:
    import rclpy
    from control_msgs.msg import Control
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    cps = checkpoints_for(mode)
    run_dir = _ensure(out_dir / mode / policy / f'r{repeat:02d}')
    csv_path = run_dir / 'drive.csv'
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    from viz_util import apply_lane_viz

    planner = build_planner(mode, overrides)
    # Perception windows: off | control (Lane drive) | on/all (+ HSV masks).
    apply_lane_viz(str(overrides.get('_viz_mode') or 'control'))
    ld._apply_detect_tune_from_yaml()

    farthest_idx = 0
    retries = 0
    passed_obstacle = False
    lap_ok = False

    # Always from start — user contract for this bench.
    _teleport('start')
    time.sleep(settle_sec)

    rclpy.init()
    node = Node(f'course_mode_{mode}_{policy}_{repeat}')
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

    def publish_cmd(steering: float, throttle: float) -> None:
        msg = Control()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.steering = float(steering)
        msg.throttle = float(throttle)
        control_pub.publish(msg)

    def stop_car() -> None:
        for _ in range(8):
            publish_cmd(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.02)

    t0 = time.time()
    last_step = t0
    lost_since: float | None = None
    recover_cooldown_until = 0.0
    sx0, sy0 = catalog['start']
    prev_xy: tuple[float, float] | None = None
    odom_distance = 0.0
    first_frame = None

    try:
        while time.time() - t0 < max_lap_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = latest['frame']
            ox, oy = latest['odom_x'], latest['odom_y']
            if frame is None or ox is None or oy is None:
                continue

            now = time.time()
            last_step = now
            if first_frame is None:
                first_frame = frame.copy()
            if prev_xy is not None:
                odom_distance += math.hypot(ox - prev_xy[0], oy - prev_xy[1])
            prev_xy = (ox, oy)

            planner.desired_turn = TurnSign.LEFT
            planner._forced_turn = TurnSign.LEFT
            output = planner.step(frame, now_sec=now)
            if str(output.decision) == 'aruco_stop':
                publish_cmd(
                    output.command.steering,
                    max(0.12, float(overrides.get('cruise_throttle', 0.2)) * 0.5),
                )
            else:
                publish_cmd(output.command.steering, output.command.throttle)

            near_name, near_i, near_d = nearest_spawn(ox, oy, cps, catalog)
            expected = min(farthest_idx + 1, len(cps) - 1)
            sx, sy = catalog[cps[expected]]
            if math.hypot(ox - sx, oy - sy) < 1.35:
                farthest_idx = expected
            if near_i > farthest_idx and near_d < 1.0:
                farthest_idx = near_i
            if cps[farthest_idx] == 'obstacle' or (near_name == 'obstacle' and near_d < 1.5):
                passed_obstacle = True

            d_start = math.hypot(ox - sx0, oy - sy0)
            elapsed = now - t0
            if (
                passed_obstacle
                and farthest_idx >= len(cps) - 2
                and d_start < start_radius_m
                and odom_distance >= lap_min_distance_m
                and elapsed > 25.0
            ):
                lap_ok = True
                events.append(
                    {
                        't': round(elapsed, 2),
                        'event': 'lap_complete',
                        'x': ox,
                        'y': oy,
                        'odom_distance': round(odom_distance, 3),
                        'farthest': cps[farthest_idx],
                    }
                )
                break

            lost = False
            if now >= recover_cooldown_until:
                lost = is_lane_lost(output, fail_cte_m=fail_cte_m)
            if lost:
                if lost_since is None:
                    lost_since = now
            else:
                lost_since = None

            event = None
            if lost_since is not None and (now - lost_since) >= lost_hold_sec:
                stop_car()
                recover_name, recover_i, _ = nearest_spawn(ox, oy, cps, catalog)
                if recover_i < farthest_idx:
                    recover_i = farthest_idx
                    recover_name = cps[recover_i]
                retries += 1
                event = 'lane_lost_recover'
                if retries > max_retries:
                    if recover_i >= len(cps) - 1:
                        recover_i = 0
                        recover_name = 'start'
                        passed_obstacle = True
                        farthest_idx = len(cps) - 1
                        event = 'skip_past_obstacle_to_start'
                    else:
                        recover_i = (recover_i + 1) % len(cps)
                        recover_name = cps[recover_i]
                        farthest_idx = max(farthest_idx, recover_i)
                        event = 'skip_checkpoint'
                    retries = 0
                events.append(
                    {
                        't': round(now - t0, 2),
                        'event': event,
                        'spawn': recover_name,
                        'retries': retries,
                        'farthest': cps[farthest_idx],
                        'decision': str(output.decision),
                        'path_source': str(output.path_source.value),
                    }
                )
                _teleport(recover_name)
                time.sleep(settle_sec)
                planner = build_planner(mode, overrides)
                lost_since = None
                recover_cooldown_until = time.time() + 1.5

            dbg = output.debug or {}
            rows.append(
                {
                    't': round(now - t0, 3),
                    'mode': mode,
                    'policy': policy,
                    'repeat': repeat,
                    'state': str(output.state.value),
                    'decision': output.decision,
                    'path_source': str(output.path_source.value),
                    'steering': float(output.command.steering),
                    'throttle': float(output.command.throttle),
                    'cte_m': dbg.get('cross_track_error_m'),
                    'white_visible': bool(dbg.get('white_visible')),
                    'yellow_visible': bool(dbg.get('yellow_visible')),
                    'white_confidence': dbg.get('white_confidence'),
                    'yellow_confidence': dbg.get('yellow_confidence'),
                    'fork_active': bool(dbg.get('fork_active')),
                    'branch_count': dbg.get('branch_count'),
                    'mask_corridor_mode': dbg.get('mask_corridor_mode'),
                    'mask_forkish': dbg.get('mask_forkish'),
                    'prefer_yellow': dbg.get('prefer_yellow'),
                    'path_lost': str(output.path_source.value).startswith('hold'),
                    'odom_x': ox,
                    'odom_y': oy,
                    'farthest_idx': farthest_idx,
                    'farthest': cps[farthest_idx],
                    'event': event or '',
                }
            )
            _ = last_step
    finally:
        stop_car()
        node.destroy_node()
        rclpy.shutdown()

    metrics = summarize_rows(rows, events)
    if lap_ok and metrics['distance_m'] < lap_min_distance_m * 0.8:
        lap_ok = False
        events.append(
            {
                'event': 'lap_rejected_short_distance',
                'distance_m': metrics['distance_m'],
                'need': lap_min_distance_m,
            }
        )

    if rows:
        with csv_path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if first_frame is not None:
        cv2.imwrite(str(run_dir / 'frame0.png'), first_frame)

    # Ranking primitive for one lap.
    max_idx = max(len(cps) - 1, 1)
    robust = (
        (25.0 if lap_ok else 0.0)
        + 10.0 * (farthest_idx / max_idx)
        - 4.0 * metrics['lost_events']
        - 3.0 * metrics['skip_events']
        - 8.0 * metrics['path_lost_ratio']
        - 2.0 * (metrics['cte_abs_mean'] or 0.25)
        - 2.0 * metrics['steer_jerk_mean']
    )
    # Faster successful laps score higher; failed laps get speed from mean_speed only.
    if lap_ok:
        speed = 40.0 / max(float(metrics['elapsed_drive_sec']), 30.0) * 40.0
        speed += 8.0 * float(metrics['mean_speed_mps'])
    else:
        speed = 5.0 * float(metrics['mean_speed_mps'])
    # Perception health bonus (course-appropriate color).
    if mode == 'out':
        percep = 4.0 * metrics['white_vis_ratio'] + 1.5 * (metrics['white_conf_mean'] or 0.0)
    else:
        percep = (
            2.5 * metrics['yellow_vis_ratio']
            + 2.0 * metrics['white_vis_ratio']
            + 1.0 * (metrics['yellow_conf_mean'] or 0.0)
            + 1.0 * (metrics['white_conf_mean'] or 0.0)
        )
    score = robust + speed + percep

    summary = {
        'mode': mode,
        'policy': policy,
        'repeat': repeat,
        'overrides': overrides,
        'checkpoints': cps,
        'lap_ok': lap_ok,
        'farthest_checkpoint': cps[farthest_idx],
        'farthest_idx': farthest_idx,
        'wall_elapsed_sec': round(time.time() - t0, 2),
        'csv': str(csv_path),
        'events': events,
        'robust_score': round(robust, 3),
        'speed_score': round(speed, 3),
        'percep_score': round(percep, 3),
        'score': round(score, 3),
        **metrics,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    (run_dir / 'events.json').write_text(
        json.dumps(events, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps({k: summary[k] for k in (
        'mode', 'policy', 'repeat', 'lap_ok', 'farthest_checkpoint',
        'wall_elapsed_sec', 'distance_m', 'mean_speed_mps', 'cte_abs_mean',
        'lost_events', 'skip_events', 'white_vis_ratio', 'yellow_vis_ratio',
        'score',
    )}, ensure_ascii=False), flush=True)
    return summary


def aggregate(mode: str, policy: str, laps: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(laps), 1)
    ok = sum(1 for L in laps if L.get('lap_ok'))
    ok_laps = [L for L in laps if L.get('lap_ok')]
    mean_score = float(np.mean([float(L['score']) for L in laps]))
    mean_far = float(np.mean([int(L['farthest_idx']) for L in laps]))
    mean_lost = float(np.mean([int(L['lost_events']) for L in laps]))
    mean_skip = float(np.mean([int(L['skip_events']) for L in laps]))
    ctes = [float(L['cte_abs_mean']) for L in laps if L.get('cte_abs_mean') is not None]
    mean_cte = float(np.mean(ctes)) if ctes else float('nan')
    mean_white = float(np.mean([float(L['white_vis_ratio']) for L in laps]))
    mean_yellow = float(np.mean([float(L['yellow_vis_ratio']) for L in laps]))
    if ok_laps:
        mean_lap_sec = float(np.mean([float(L['wall_elapsed_sec']) for L in ok_laps]))
        mean_ok_speed = float(np.mean([float(L['mean_speed_mps']) for L in ok_laps]))
    else:
        mean_lap_sec = float('nan')
        mean_ok_speed = float(np.mean([float(L['mean_speed_mps']) for L in laps]))
    # Final rank: prefer high lap_ok, then high score, then faster ok laps.
    rank_key = (
        ok / n,
        mean_score,
        -(mean_lap_sec if math.isfinite(mean_lap_sec) else 1e6),
        mean_ok_speed,
    )
    return {
        'mode': mode,
        'policy': policy,
        'n_repeats': len(laps),
        'lap_ok_rate': round(ok / n, 3),
        'lap_ok_count': ok,
        'mean_score': round(mean_score, 3),
        'mean_farthest_idx': round(mean_far, 3),
        'mean_lost_events': round(mean_lost, 3),
        'mean_skip_events': round(mean_skip, 3),
        'mean_cte_abs': None if not math.isfinite(mean_cte) else round(mean_cte, 4),
        'mean_white_vis': round(mean_white, 4),
        'mean_yellow_vis': round(mean_yellow, 4),
        'mean_ok_lap_sec': None if not math.isfinite(mean_lap_sec) else round(mean_lap_sec, 2),
        'mean_ok_speed_mps': round(mean_ok_speed, 4),
        'rank_key': list(rank_key),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', default='out,in')
    parser.add_argument(
        '--policies',
        default='pp_ref,mask_fork_pp,mask_hard_wide,mask_soft_fork_pp,mask_hard_fork_pp,mask_raw',
    )
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--max-lap-sec-out', type=float, default=180.0)
    parser.add_argument('--max-lap-sec-in', type=float, default=240.0)
    parser.add_argument('--lost-hold-sec', type=float, default=1.2)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--fail-cte-m', type=float, default=0.35)
    parser.add_argument('--lap-min-distance-m', type=float, default=12.0)
    parser.add_argument('--start-radius-m', type=float, default=1.4)
    parser.add_argument('--settle', type=float, default=1.0)
    parser.add_argument('--camera-topic', default='/camera/image/compressed')
    parser.add_argument(
        '--viz',
        default='control',
        help='Perception OpenCV windows: off|control|all (default control = Lane drive only)',
    )
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    modes = [m.strip().lower() for m in args.modes.split(',') if m.strip()]
    policies = [p.strip() for p in args.policies.split(',') if p.strip()]
    viz_mode = str(args.viz).strip().lower()
    for m in modes:
        if m not in ('out', 'in'):
            raise SystemExit(f'bad mode: {m}')
    for p in policies:
        if p not in POLICIES:
            raise SystemExit(f'unknown policy {p}; known={list(POLICIES)}')

    catalog = load_spawn_xy()
    run_root = _ensure(args.out or (OUT_ROOT / _stamp()))
    meta = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'modes': modes,
        'policies': policies,
        'repeats': args.repeats,
        'viz': viz_mode,
        'max_lap_sec_out': args.max_lap_sec_out,
        'max_lap_sec_in': args.max_lap_sec_in,
        'start_policy': 'always teleport to start each lap',
        'scoring': {
            'robust': 'lap_ok, farthest, lost/skip, path_lost, cte, jerk',
            'speed': 'ok-lap wall time + mean_speed_mps',
            'perception': 'white/yellow visibility & confidence by mode',
            'rank': 'lap_ok_rate → mean_score → faster ok lap → speed',
        },
        'policy_overrides': {p: POLICIES[p] for p in policies},
        'out_checkpoints': OUT_CHECKPOINTS,
        'in_checkpoints': IN_CHECKPOINTS,
    }
    (run_root / 'META.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(f'=== course_mode_bench → {run_root}', flush=True)

    all_laps: list[dict[str, Any]] = []
    for mode in modes:
        max_sec = args.max_lap_sec_in if mode == 'in' else args.max_lap_sec_out
        for policy in policies:
            for rep in range(args.repeats):
                print(
                    f'\n=== MODE={mode} POLICY={policy} REP={rep}/{args.repeats-1} '
                    f'max={max_sec}s',
                    flush=True,
                )
                overrides = dict(POLICIES[policy])
                overrides['_viz_mode'] = viz_mode
                summary = run_one_lap(
                    mode=mode,
                    policy=policy,
                    overrides=overrides,
                    repeat=rep,
                    out_dir=run_root,
                    catalog=catalog,
                    camera_topic=args.camera_topic,
                    max_lap_sec=max_sec,
                    lost_hold_sec=args.lost_hold_sec,
                    max_retries=args.max_retries,
                    fail_cte_m=args.fail_cte_m,
                    lap_min_distance_m=args.lap_min_distance_m,
                    start_radius_m=args.start_radius_m,
                    settle_sec=args.settle,
                )
                all_laps.append(summary)

    with (run_root / 'all_laps.csv').open('w', newline='', encoding='utf-8') as fh:
        fields = [
            'mode', 'policy', 'repeat', 'lap_ok', 'farthest_checkpoint',
            'wall_elapsed_sec', 'distance_m', 'mean_speed_mps', 'cte_abs_mean',
            'lost_events', 'skip_events', 'path_lost_ratio',
            'white_vis_ratio', 'yellow_vis_ratio',
            'white_conf_mean', 'yellow_conf_mean',
            'robust_score', 'speed_score', 'percep_score', 'score',
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for L in all_laps:
            writer.writerow({k: L.get(k) for k in fields})

    rankings: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        aggs = []
        for policy in policies:
            laps = [L for L in all_laps if L['mode'] == mode and L['policy'] == policy]
            aggs.append(aggregate(mode, policy, laps))
        aggs.sort(
            key=lambda a: tuple(a['rank_key']),
            reverse=True,
        )
        rankings[mode] = aggs
        print(f'\n=== RANKING mode={mode} ===', flush=True)
        for i, a in enumerate(aggs):
            print(
                f"{i+1:2d}. {a['policy']:18s} ok={a['lap_ok_rate']:.2f} "
                f"score={a['mean_score']:6.2f} lap_sec={a['mean_ok_lap_sec']} "
                f"spd={a['mean_ok_speed_mps']:.3f} "
                f"Wvis={a['mean_white_vis']:.2f} Yvis={a['mean_yellow_vis']:.2f} "
                f"lost={a['mean_lost_events']:.1f}",
                flush=True,
            )

    # Overall: mean of per-mode ranks (lower rank number better).
    overall: list[dict[str, Any]] = []
    for policy in policies:
        ranks = []
        ok_rates = []
        scores = []
        for mode in modes:
            ordered = [a['policy'] for a in rankings[mode]]
            ranks.append(ordered.index(policy) + 1)
            row = next(a for a in rankings[mode] if a['policy'] == policy)
            ok_rates.append(row['lap_ok_rate'])
            scores.append(row['mean_score'])
        overall.append(
            {
                'policy': policy,
                'mean_rank': round(float(np.mean(ranks)), 3),
                'mean_lap_ok_rate': round(float(np.mean(ok_rates)), 3),
                'mean_score': round(float(np.mean(scores)), 3),
                'per_mode_rank': {m: rankings[m].index(
                    next(a for a in rankings[m] if a['policy'] == policy)
                ) + 1 for m in modes},
            }
        )
    overall.sort(key=lambda r: (r['mean_rank'], -r['mean_lap_ok_rate'], -r['mean_score']))

    report = {
        'run_root': str(run_root),
        'rankings': rankings,
        'overall': overall,
        'recommendation': overall[0]['policy'] if overall else None,
    }
    (run_root / 'REPORT.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )

    lines = [
        '# Course mode bench (IN/OUT full laps from start)',
        '',
        f'- run: `{run_root}`',
        f'- repeats: {args.repeats}',
        '',
    ]
    for mode in modes:
        lines.append(f'## Mode `{mode}`')
        lines.append('')
        for i, a in enumerate(rankings[mode]):
            lines.append(
                f"{i+1}. **{a['policy']}** — ok_rate={a['lap_ok_rate']}, "
                f"score={a['mean_score']}, ok_lap_sec={a['mean_ok_lap_sec']}, "
                f"speed={a['mean_ok_speed_mps']}, "
                f"Wvis={a['mean_white_vis']}, Yvis={a['mean_yellow_vis']}, "
                f"lost={a['mean_lost_events']}, skip={a['mean_skip_events']}"
            )
        lines.append('')
    lines.extend(['## Overall', ''])
    for i, a in enumerate(overall):
        lines.append(
            f"{i+1}. **{a['policy']}** — mean_rank={a['mean_rank']}, "
            f"ok={a['mean_lap_ok_rate']}, score={a['mean_score']}, "
            f"ranks={a['per_mode_rank']}"
        )
    if overall:
        lines.extend(
            [
                '',
                '## Recommendation',
                '',
                f"**{overall[0]['policy']}** — best mean rank across IN/OUT "
                '(robust lap_ok first, then score/speed).',
            ]
        )
    (run_root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nWrote {run_root / "REPORT.md"}', flush=True)
    print(f'Recommendation: {overall[0]["policy"] if overall else None}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
