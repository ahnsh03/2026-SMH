#!/usr/bin/env python3
"""Mask course-color / fork-guard policy A/B bench with rich logging.

Goal
----
Compare NORMAL trackers under OUT course stress (forks + merges) so we can
pick which guard combination actually stops unwanted course pull.

Policies (combinatorial, explicit names)
---------------------------------------
- pp_ref                 : Pure Pursuit baseline
- mask_raw               : mask COM, corridor=off, fork_force=false  (legacy risk)
- mask_fork_pp           : corridor=off, fork_force=true
- mask_hard              : hard corridor, fork_force=false
- mask_soft              : soft weighted COM, fork_force=false
- mask_hard_fork_pp      : hard + fork_force (recommended candidate)
- mask_soft_fork_pp      : soft + fork_force
- mask_hard_narrow       : hard half_width=0.20 + fork_force
- mask_hard_wide         : hard half_width=0.38 + fork_force

Stress segments (teleport)
--------------------------
start, inout_fork, out_fork, out_fork_merge_left, out_in_merge

Scoring (higher better)
-----------------------
Primary: stay on white OUT centerline (cte), avoid path lost, make distance.
Secondary: steer jerk, fork frames that still used raw mask (bad),
mask→PP fallback rate, optional wrong-course odom drift heuristics.

Owns ``/control`` — do **not** run sim-auto / inference_node concurrently.

Example::

  python3 scripts/drive_test/mask_policy_bench.py \\
      --phase segments --duration 8 --repeat 1
  python3 scripts/drive_test/mask_policy_bench.py \\
      --phase top_lap --top-n 3 --max-lap-sec 150
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

OUT_ROOT = _ROOT / 'data' / 'captures' / 'mask_policy_logs'
SPAWN_YAML = _ROOT / 'src' / 'dracer_sim' / 'config' / 'spawn_poses.yaml'

STRESS_SEGMENTS: list[str] = [
    'start',
    'inout_fork',
    'out_fork',
    'out_fork_merge_left',
    'out_in_merge',
]

OUT_CHECKPOINTS: list[str] = [
    'start',
    'inout_fork',
    'out_fork',
    'out_fork_merge_left',
    'out_in_merge',
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
    'mask_hard': {
        **_MASK_SPEED,
        'normal_tracker': 'mask_p',
        'mask_corridor_mode': 'hard',
        'mask_corridor_half_width_m': 0.28,
        'mask_fork_force_pp': False,
        'mask_require_color_path': True,
    },
    'mask_soft': {
        **_MASK_SPEED,
        'normal_tracker': 'mask_p',
        'mask_corridor_mode': 'soft',
        'mask_path_weight_sigma_m': 0.20,
        'mask_fork_force_pp': False,
        'mask_require_color_path': True,
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
    'mask_hard_narrow': {
        **_MASK_SPEED,
        'normal_tracker': 'mask_p',
        'mask_corridor_mode': 'hard',
        'mask_corridor_half_width_m': 0.20,
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


def build_planner(overrides: dict[str, Any]) -> MainPlanner:
    base = load_planner_config(route_mode='out')
    cfg = replace(base, **overrides)
    planner = MainPlanner(cfg)
    planner.force_fork_choice(TurnSign.LEFT, state=DrivingState.NORMAL)
    planner._forced_turn = TurnSign.LEFT
    planner.desired_turn = TurnSign.LEFT
    return planner


def summarize_segment(
    rows: list[dict[str, Any]],
    *,
    fail_cte_m: float,
    segment: str,
    catalog: dict[str, tuple[float, float]],
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
    mask_frames = sum(1 for r in rows if r.get('path_source') == 'mask_drivable')
    fork_pp_frames = sum(
        1 for r in rows if 'mask_fork_pp' in str(r.get('decision', ''))
    )
    fallback_pp = sum(
        1 for r in rows if 'mask_fallback_pp' in str(r.get('decision', ''))
    )
    bad_fork_mask = sum(
        1
        for r in rows
        if bool(r.get('fork_active'))
        and r.get('path_source') == 'mask_drivable'
        and not bool(r.get('mask_fork_force_pp', True))
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
            dist += math.hypot(
                float(xs[i]) - float(xs[i - 1]), float(ys[i]) - float(ys[i - 1])
            )

    spawn_xy = catalog.get(segment)
    drift_penalty = 0.0
    if spawn_xy is not None and xs and ys:
        dy = float(ys[-1]) - float(spawn_xy[1])
        if segment == 'inout_fork':
            drift_penalty = max(0.0, dy - 0.35) * 2.5
        elif segment == 'out_in_merge':
            drift_penalty = max(0.0, -dy - 0.45) * 2.0
        elif segment == 'out_fork':
            drift_penalty = max(0.0, -dy - 0.55) * 1.5

    cte_abs_mean = float(np.mean(np.abs(ctes))) if ctes.size else float('nan')
    cte_abs_p95 = float(np.percentile(np.abs(ctes), 95)) if ctes.size else float('nan')
    steer_rms = float(np.sqrt(np.mean(steers**2)))
    steer_jerk = float(np.mean(jerks))
    fail_ratio = fail / max(len(rows), 1)
    mask_ratio = mask_frames / max(len(rows), 1)
    bad_fork_ratio = bad_fork_mask / max(len(rows), 1)

    score = (
        4.0 * dist
        - 10.0 * fail_ratio
        - 3.0 * (cte_abs_mean if math.isfinite(cte_abs_mean) else 1.0)
        - 5.0 * steer_jerk
        - 1.0 * steer_rms
        - 6.0 * bad_fork_ratio
        - drift_penalty
        - 0.5 * (fallback_pp / max(len(rows), 1))
        + 0.3 * mask_ratio
        + 0.2 * (fork_pp_frames / max(len(rows), 1))
    )
    return {
        'n_rows': len(rows),
        'distance_m': round(dist, 3),
        'steer_rms': round(steer_rms, 4),
        'steer_jerk_mean': round(steer_jerk, 4),
        'cte_abs_mean': None if not math.isfinite(cte_abs_mean) else round(cte_abs_mean, 4),
        'cte_abs_p95': None if not math.isfinite(cte_abs_p95) else round(cte_abs_p95, 4),
        'mask_frame_ratio': round(mask_ratio, 3),
        'fork_pp_frames': fork_pp_frames,
        'fallback_pp_frames': fallback_pp,
        'bad_fork_mask_frames': bad_fork_mask,
        'path_lost_frames': path_lost,
        'fail_frames': fail,
        'fail_ratio': round(fail_ratio, 3),
        'drift_penalty': round(drift_penalty, 3),
        'score': round(score, 3),
        'ok': fail_ratio < 0.30 and dist > 0.20 and bad_fork_ratio < 0.15,
    }


def run_segment(
    *,
    policy: str,
    segment: str,
    overrides: dict[str, Any],
    duration_sec: float,
    out_dir: Path,
    camera_topic: str,
    fail_cte_m: float,
    settle_sec: float,
    catalog: dict[str, tuple[float, float]],
    force_fsm_normal: bool,
) -> dict[str, Any]:
    import rclpy
    from control_msgs.msg import Control
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    _teleport(segment)
    time.sleep(settle_sec)

    planner = build_planner(overrides)
    ld.VISUALIZE = False
    ld._apply_detect_tune_from_yaml()

    rclpy.init()
    node = Node(f'mask_policy_{policy}_{segment}')
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

    run_dir = _ensure(out_dir / policy / f'{segment}')
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
            last_step = now
            if force_fsm_normal:
                # Isolate NORMAL tracker (still sees fork_active flags).
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
                'policy': policy,
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
                'white_visible': dbg.get('white_visible'),
                'yellow_visible': dbg.get('yellow_visible'),
                'fork_active': dbg.get('fork_active'),
                'branch_count': dbg.get('branch_count'),
                'mask_corridor_mode': dbg.get('mask_corridor_mode'),
                'mask_fork_force_pp': dbg.get('mask_fork_force_pp'),
                'mask_forkish': dbg.get('mask_forkish'),
                'mask_area_px': dbg.get('mask_area_px'),
                'mask_error_norm': dbg.get('mask_error_norm'),
                'mask_color_path_points': dbg.get('mask_color_path_points'),
                'path_lost': path_lost,
                'odom_x': latest['odom_x'],
                'odom_y': latest['odom_y'],
            }
            rows.append(row)
            _ = last_step
    finally:
        for _ in range(6):
            msg = Control()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.steering = 0.0
            msg.throttle = 0.0
            control_pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(0.02)
        node.destroy_node()
        rclpy.shutdown()

    if rows:
        with csv_path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if first_frame is not None:
        cv2.imwrite(str(run_dir / 'first_frame.png'), first_frame)

    summary = summarize_segment(
        rows, fail_cte_m=fail_cte_m, segment=segment, catalog=catalog
    )
    summary.update(
        {
            'policy': policy,
            'segment': segment,
            'overrides': overrides,
            'csv': str(csv_path),
            'n_logged': len(rows),
            'duration_sec': duration_sec,
            'force_fsm_normal': force_fsm_normal,
        }
    )
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def nearest_out_spawn(
    x: float, y: float, catalog: dict[str, tuple[float, float]]
) -> tuple[str, int, float]:
    best_name = OUT_CHECKPOINTS[0]
    best_i = 0
    best_d = 1e9
    for i, name in enumerate(OUT_CHECKPOINTS):
        sx, sy = catalog[name]
        d = math.hypot(x - sx, y - sy)
        if d < best_d:
            best_d = d
            best_name = name
            best_i = i
    return best_name, best_i, best_d


def is_lane_lost(output, *, fail_cte_m: float) -> bool:
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


def run_recovery_lap(
    *,
    policy: str,
    overrides: dict[str, Any],
    out_dir: Path,
    catalog: dict[str, tuple[float, float]],
    camera_topic: str,
    max_lap_sec: float,
    lost_hold_sec: float,
    max_retries: int,
    fail_cte_m: float,
    lap_min_distance_m: float,
    settle_sec: float,
) -> dict[str, Any]:
    """Short OUT ring lap with teleport recovery (same contract as out_lap_bench)."""
    import rclpy
    from control_msgs.msg import Control
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    run_dir = _ensure(out_dir / 'laps' / policy)
    csv_path = run_dir / 'drive.csv'
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    planner = build_planner(overrides)
    ld.VISUALIZE = False
    ld._apply_detect_tune_from_yaml()

    active_idx = 0
    retries = 0
    farthest_idx = 0
    _teleport(OUT_CHECKPOINTS[active_idx])
    time.sleep(settle_sec)

    rclpy.init()
    node = Node(f'mask_policy_lap_{policy}')
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
    visited_after_start = False
    lap_ok = False
    dist_acc = 0.0
    prev_xy: tuple[float, float] | None = None

    try:
        while time.time() - t0 < max_lap_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = latest['frame']
            if frame is None:
                continue
            now = time.time()
            last_step = now
            output = planner.step(frame, now_sec=now)
            publish_cmd(float(output.command.steering), float(output.command.throttle))

            ox, oy = latest['odom_x'], latest['odom_y']
            if ox is not None and oy is not None:
                if prev_xy is not None:
                    dist_acc += math.hypot(ox - prev_xy[0], oy - prev_xy[1])
                prev_xy = (float(ox), float(oy))
                name, idx, _ = nearest_out_spawn(float(ox), float(oy), catalog)
                if idx > farthest_idx:
                    farthest_idx = idx
                if idx >= 2:
                    visited_after_start = True
                if (
                    visited_after_start
                    and idx == 0
                    and dist_acc >= lap_min_distance_m
                    and name == 'start'
                ):
                    lap_ok = True
                    break

            dbg = output.debug or {}
            rows.append(
                {
                    't': round(now - t0, 3),
                    'policy': policy,
                    'decision': output.decision,
                    'path_source': str(output.path_source.value),
                    'steering': float(output.command.steering),
                    'throttle': float(output.command.throttle),
                    'cte_m': dbg.get('cross_track_error_m'),
                    'fork_active': dbg.get('fork_active'),
                    'mask_corridor_mode': dbg.get('mask_corridor_mode'),
                    'mask_forkish': dbg.get('mask_forkish'),
                    'odom_x': ox,
                    'odom_y': oy,
                    'farthest_idx': farthest_idx,
                    'event': '',
                }
            )

            lost = is_lane_lost(output, fail_cte_m=fail_cte_m)
            if now < recover_cooldown_until:
                lost_since = None
                continue
            if lost:
                if lost_since is None:
                    lost_since = now
                elif now - lost_since >= lost_hold_sec:
                    stop_car()
                    if ox is None or oy is None:
                        lost_since = None
                        continue
                    recover_name, recover_i, _ = nearest_out_spawn(
                        float(ox), float(oy), catalog
                    )
                    if recover_i == active_idx:
                        retries += 1
                    else:
                        active_idx = recover_i
                        retries = 1
                    if retries > max_retries:
                        active_idx = (active_idx + 1) % len(OUT_CHECKPOINTS)
                        retries = 0
                        recover_name = OUT_CHECKPOINTS[active_idx]
                        event = 'skip_checkpoint'
                    else:
                        event = 'lane_lost_recover'
                    events.append(
                        {
                            't': now - t0,
                            'event': event,
                            'spawn': recover_name,
                            'retries': retries,
                        }
                    )
                    if rows:
                        rows[-1]['event'] = event
                    _teleport(recover_name)
                    time.sleep(settle_sec)
                    planner = build_planner(overrides)
                    lost_since = None
                    recover_cooldown_until = time.time() + 1.5
            else:
                lost_since = None
            _ = last_step
    finally:
        stop_car()
        node.destroy_node()
        rclpy.shutdown()

    if rows:
        with csv_path.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    steers = np.array([float(r['steering']) for r in rows], dtype=np.float64) if rows else np.array([0.0])
    jerks = np.abs(np.diff(steers)) if steers.size > 1 else np.array([0.0])
    ctes = np.array(
        [float(r['cte_m']) for r in rows if r.get('cte_m') is not None],
        dtype=np.float64,
    )
    lost_events = sum(1 for e in events if e['event'] == 'lane_lost_recover')
    skip_events = sum(1 for e in events if e['event'] == 'skip_checkpoint')
    cte_mean = float(np.mean(np.abs(ctes))) if ctes.size else float('nan')
    score = (
        12.0 * float(farthest_idx)
        + 0.6 * dist_acc
        - 4.0 * lost_events
        - 3.0 * skip_events
        - 2.5 * float(np.mean(jerks))
        - (1.5 * cte_mean if math.isfinite(cte_mean) else 0.5)
        + (8.0 if lap_ok else 0.0)
    )
    summary = {
        'policy': policy,
        'overrides': overrides,
        'lap_ok': lap_ok,
        'farthest_checkpoint': OUT_CHECKPOINTS[farthest_idx],
        'farthest_idx': farthest_idx,
        'elapsed_sec': round(time.time() - t0, 2),
        'csv': str(csv_path),
        'n_rows': len(rows),
        'distance_m': round(dist_acc, 3),
        'steer_jerk_mean': round(float(np.mean(jerks)), 4),
        'cte_abs_mean': None if not math.isfinite(cte_mean) else round(cte_mean, 4),
        'lost_events': lost_events,
        'skip_events': skip_events,
        'events': events,
        'score': round(score, 3),
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def aggregate_policy(segment_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not segment_rows:
        return {'score': -1e9, 'ok': False}
    scores = [float(r['score']) for r in segment_rows]
    oks = [bool(r.get('ok')) for r in segment_rows]
    by_seg = {r['segment']: r['score'] for r in segment_rows}
    # Stress segments weigh more in ranking.
    weights = {
        'start': 0.8,
        'inout_fork': 1.4,
        'out_fork': 1.5,
        'out_fork_merge_left': 1.2,
        'out_in_merge': 1.4,
    }
    wsum = 0.0
    total = 0.0
    for seg, sc in by_seg.items():
        w = float(weights.get(seg, 1.0))
        total += w * float(sc)
        wsum += w
    weighted = total / max(wsum, 1e-6)
    return {
        'n_segments': len(segment_rows),
        'mean_score': round(float(np.mean(scores)), 3),
        'weighted_score': round(weighted, 3),
        'ok_ratio': round(sum(oks) / max(len(oks), 1), 3),
        'by_segment': {k: round(float(v), 3) for k, v in by_seg.items()},
        'ok': all(oks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--phase',
        choices=('segments', 'top_lap', 'all'),
        default='all',
    )
    parser.add_argument(
        '--policies',
        default=','.join(POLICIES.keys()),
        help='comma policy ids',
    )
    parser.add_argument(
        '--segments',
        default=','.join(STRESS_SEGMENTS),
    )
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--settle', type=float, default=1.0)
    parser.add_argument('--fail-cte-m', type=float, default=0.35)
    parser.add_argument('--camera-topic', default='/camera/image_raw')
    parser.add_argument(
        '--force-fsm-normal',
        action='store_true',
        default=True,
        help='keep planner in NORMAL so mask policies are comparable',
    )
    parser.add_argument('--no-force-fsm-normal', action='store_false', dest='force_fsm_normal')
    parser.add_argument('--top-n', type=int, default=3)
    parser.add_argument('--max-lap-sec', type=float, default=150.0)
    parser.add_argument('--lost-hold-sec', type=float, default=1.2)
    parser.add_argument('--max-retries', type=int, default=2)
    parser.add_argument('--lap-min-distance-m', type=float, default=10.0)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(',') if p.strip()]
    segments = [s.strip() for s in args.segments.split(',') if s.strip()]
    for p in policies:
        if p not in POLICIES:
            raise SystemExit(f'unknown policy: {p}. known={list(POLICIES)}')

    catalog = load_spawn_xy()
    run_root = _ensure(args.out or (OUT_ROOT / _stamp()))
    meta = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'phase': args.phase,
        'policies': policies,
        'segments': segments,
        'duration_sec': args.duration,
        'repeat': args.repeat,
        'fail_cte_m': args.fail_cte_m,
        'force_fsm_normal': args.force_fsm_normal,
        'scoring': {
            'primary': [
                'distance_m',
                'fail_ratio (cte/path_lost)',
                'cte_abs_mean vs white OUT path',
            ],
            'fork_guards': [
                'bad_fork_mask_frames (fork_active but still mask_drivable without force)',
                'drift_penalty at inout_fork / out_in_merge / out_fork',
            ],
            'secondary': ['steer_jerk_mean', 'fallback_pp', 'mask_frame_ratio'],
            'aggregate': 'weighted mean favoring fork/merge segments',
        },
        'policy_overrides': {p: POLICIES[p] for p in policies},
    }
    (run_root / 'meta.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(f'=== mask_policy_bench → {run_root}', flush=True)

    segment_results: list[dict[str, Any]] = []
    if args.phase in ('segments', 'all'):
        for rep in range(args.repeat):
            for policy in policies:
                for segment in segments:
                    print(
                        f'\n=== seg policy={policy} segment={segment} rep={rep}',
                        flush=True,
                    )
                    summary = run_segment(
                        policy=policy if args.repeat == 1 else f'{policy}_r{rep}',
                        segment=segment,
                        overrides=POLICIES[policy],
                        duration_sec=args.duration,
                        out_dir=run_root / 'segments',
                        camera_topic=args.camera_topic,
                        fail_cte_m=args.fail_cte_m,
                        settle_sec=args.settle,
                        catalog=catalog,
                        force_fsm_normal=args.force_fsm_normal,
                    )
                    summary['base_policy'] = policy
                    summary['rep'] = rep
                    segment_results.append(summary)

        by_policy: dict[str, list[dict[str, Any]]] = {}
        for row in segment_results:
            by_policy.setdefault(str(row['base_policy']), []).append(row)
        ranking = []
        for policy, rows in by_policy.items():
            agg = aggregate_policy(rows)
            ranking.append({'policy': policy, **agg, 'overrides': POLICIES[policy]})
        ranking.sort(key=lambda r: float(r['weighted_score']), reverse=True)
        (run_root / 'segment_ranking.json').write_text(
            json.dumps(ranking, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        with (run_root / 'segment_results.csv').open('w', newline='', encoding='utf-8') as fh:
            fields = [
                'policy',
                'segment',
                'rep',
                'score',
                'ok',
                'distance_m',
                'cte_abs_mean',
                'fail_ratio',
                'steer_jerk_mean',
                'bad_fork_mask_frames',
                'drift_penalty',
                'mask_frame_ratio',
            ]
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in segment_results:
                writer.writerow({k: r.get(k) for k in fields})
        print('\n=== SEGMENT RANKING ===', flush=True)
        for i, row in enumerate(ranking):
            print(
                f"{i+1:2d}. {row['policy']:20s} weighted={row['weighted_score']:7.3f} "
                f"ok_ratio={row['ok_ratio']:.2f} mean={row['mean_score']:.3f}",
                flush=True,
            )
    else:
        ranking = [
            {'policy': p, 'weighted_score': 0.0, 'overrides': POLICIES[p]}
            for p in policies
        ]

    lap_results: list[dict[str, Any]] = []
    if args.phase in ('top_lap', 'all'):
        top = ranking[: max(1, args.top_n)]
        if args.phase == 'top_lap':
            top = [{'policy': p, 'overrides': POLICIES[p]} for p in policies[: args.top_n]]
        for row in top:
            policy = str(row['policy'])
            print(f'\n=== LAP policy={policy}', flush=True)
            lap = run_recovery_lap(
                policy=policy,
                overrides=dict(row.get('overrides') or POLICIES[policy]),
                out_dir=run_root,
                catalog=catalog,
                camera_topic=args.camera_topic,
                max_lap_sec=args.max_lap_sec,
                lost_hold_sec=args.lost_hold_sec,
                max_retries=args.max_retries,
                fail_cte_m=args.fail_cte_m,
                lap_min_distance_m=args.lap_min_distance_m,
                settle_sec=args.settle,
            )
            lap_results.append(lap)
        lap_results.sort(key=lambda r: float(r['score']), reverse=True)
        (run_root / 'lap_ranking.json').write_text(
            json.dumps(lap_results, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )

    report = {
        'run_root': str(run_root),
        'segment_ranking': ranking if args.phase in ('segments', 'all') else [],
        'lap_ranking': lap_results,
        'recommendation': (ranking[0]['policy'] if ranking else None),
    }
    (run_root / 'REPORT.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    # Markdown summary for humans.
    lines = [
        '# Mask policy bench report',
        '',
        f'- run: `{run_root}`',
        f'- phase: `{args.phase}`',
        '',
        '## Segment ranking (weighted)',
        '',
    ]
    if ranking and args.phase in ('segments', 'all'):
        for i, row in enumerate(ranking):
            lines.append(
                f"{i+1}. **{row['policy']}** — weighted={row['weighted_score']}, "
                f"ok_ratio={row['ok_ratio']}, by_segment={row.get('by_segment')}"
            )
    else:
        lines.append('_skipped_')
    lines.extend(['', '## Lap ranking', ''])
    if lap_results:
        for i, row in enumerate(lap_results):
            lines.append(
                f"{i+1}. **{row['policy']}** — lap_ok={row['lap_ok']}, "
                f"score={row['score']}, farthest={row['farthest_checkpoint']}, "
                f"lost={row['lost_events']}, skip={row['skip_events']}"
            )
    else:
        lines.append('_skipped_')
    if ranking:
        lap_winner = lap_results[0]['policy'] if lap_results else None
        seg_winner = ranking[0]['policy']
        pick = lap_winner or seg_winner
        lines.extend(
            [
                '',
                '## Recommendation',
                '',
                f'- Stress segments: **{seg_winner}**',
                (
                    f'- Recovery lap: **{lap_winner}**'
                    if lap_winner
                    else '- Recovery lap: _skipped_'
                ),
                f'- Default pick: **{pick}** '
                '(prefer lap winner when `lap_ok`, else segment winner).',
            ]
        )
        report['recommendation'] = pick
        report['recommendation_segment'] = seg_winner
        report['recommendation_lap'] = lap_winner
    (run_root / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nWrote {run_root / "REPORT.md"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
