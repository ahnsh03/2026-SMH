#!/usr/bin/env python3
"""OUT-course full-lap control A/B bench with recovery teleports.

Policy (user contract)
----------------------
1. ``route_mode=out`` (white-priority) — one control family at a time.
2. Drive continuously until a sustained lane loss, then teleport to the
   nearest OUT spawn along the CW checkpoint ring and retry.
3. If the same checkpoint exceeds ``max_retries``, skip forward to the
   next checkpoint and continue (log a skip).
4. When progress returns near ``start`` after covering the ring → lap done
   for this parameter set; then advance to the next param trial / family.
5. Inside one family, mutate parameters from failure signals (develop loop).

Owns ``/control`` — do **not** run sim-auto / inference_node concurrently.

Example::

  python3 scripts/drive_test/out_lap_bench.py \\
      --families pp,mask_p --max-param-trials 3 --max-lap-sec 180
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from copy import deepcopy
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

OUT_ROOT = _ROOT / 'data' / 'captures' / 'out_lap_logs'
SPAWN_YAML = _ROOT / 'src' / 'dracer_sim' / 'config' / 'spawn_poses.yaml'

# CW outer ring checkpoints (OUT white course). Recovery + skip order.
OUT_CHECKPOINTS: list[str] = [
    'start',
    'inout_fork',
    'out_fork',
    'out_fork_merge_left',
    'out_in_merge',
    'obstacle',
]

# Seed parameter sets per family (overrides for PlannerConfig.replace).
FAMILY_SEEDS: dict[str, dict[str, Any]] = {
    # Trusted out-lap 20260714_120800 winners (steer joint 4 rad/s).
    'pp': {
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
    'mask_p': {
        'normal_tracker': 'mask_p',
        'cruise_throttle': 0.33,
        'mask_steer_k': 2.0,
        'mask_steer_alpha': 0.40,
        'mask_near_band_ratio': 0.85,
        'mask_curve_speed_scale': 0.80,
        'steering_rate_limit_per_sec': 12.0,
    },
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_spawn_xy() -> dict[str, tuple[float, float]]:
    data = yaml.safe_load(SPAWN_YAML.read_text(encoding='utf-8')) or {}
    poses = data.get('poses') or {}
    out: dict[str, tuple[float, float]] = {}
    for name in OUT_CHECKPOINTS:
        pose = poses.get(name) or {}
        out[name] = (float(pose['x']), float(pose['y']))
    return out


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


def build_planner(overrides: dict[str, Any]) -> MainPlanner:
    base = load_planner_config(route_mode='out')
    cfg = replace(base, **overrides)
    planner = MainPlanner(cfg)
    # Consistent OUT left branch at fork so the lap is reproducible.
    planner.force_fork_choice(TurnSign.LEFT, state=DrivingState.NORMAL)
    planner._forced_turn = TurnSign.LEFT
    planner.desired_turn = TurnSign.LEFT
    return planner


def mutate_params(
    family: str,
    current: dict[str, Any],
    *,
    lost_events: int,
    skip_events: int,
    steer_jerk: float,
    distance_m: float,
    lap_ok: bool,
    trial: int,
) -> dict[str, Any]:
    """Develop next parameter set from the last lap attempt signals."""
    nxt = deepcopy(current)
    if lap_ok:
        # Successful lap → carefully raise cruise for speed.
        nxt['cruise_throttle'] = float(
            np.clip(float(nxt.get('cruise_throttle', 0.2)) + 0.03, 0.12, 0.34)
        )
        return nxt

    # Failure → stabilize.
    if family == 'mask_p':
        nxt['cruise_throttle'] = float(
            np.clip(float(nxt.get('cruise_throttle', 0.22)) - 0.03, 0.12, 0.34)
        )
        nxt['mask_steer_k'] = float(
            np.clip(float(nxt.get('mask_steer_k', 1.35)) - 0.15, 0.8, 2.5)
        )
        nxt['mask_steer_alpha'] = float(
            np.clip(float(nxt.get('mask_steer_alpha', 0.28)) - 0.04, 0.12, 0.6)
        )
        if lost_events + skip_events >= 2:
            nxt['mask_near_band_ratio'] = float(
                np.clip(float(nxt.get('mask_near_band_ratio', 0.46)) - 0.05, 0.30, 0.70)
            )
        if steer_jerk > 0.04:
            nxt['steering_rate_limit_per_sec'] = float(
                np.clip(
                    float(nxt.get('steering_rate_limit_per_sec', 4.0)) - 0.5,
                    2.0,
                    8.0,
                )
            )
    else:  # pp
        nxt['cruise_throttle'] = float(
            np.clip(float(nxt.get('cruise_throttle', 0.2)) - 0.03, 0.10, 0.34)
        )
        nxt['curve_throttle'] = float(
            np.clip(float(nxt.get('curve_throttle', 0.09)) - 0.01, 0.05, 0.20)
        )
        if lost_events >= 1:
            nxt['lookahead_m'] = float(
                np.clip(float(nxt.get('lookahead_m', 0.80)) + 0.08, 0.45, 1.20)
            )
        if steer_jerk > 0.04:
            nxt['cte_gain'] = float(
                np.clip(float(nxt.get('cte_gain', 0.08)) - 0.02, 0.0, 0.25)
            )
            nxt['heading_gain'] = float(
                np.clip(float(nxt.get('heading_gain', 0.25)) - 0.05, 0.0, 0.5)
            )
            nxt['steering_rate_limit_per_sec'] = float(
                np.clip(
                    float(nxt.get('steering_rate_limit_per_sec', 5.0)) - 0.5,
                    2.0,
                    8.0,
                )
            )
        if distance_m < 3.0 and trial > 0:
            nxt['curve_lookahead_m'] = float(
                np.clip(float(nxt.get('curve_lookahead_m', 0.45)) + 0.05, 0.30, 0.80)
            )
    return nxt


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    farthest_idx: int = 0,
) -> dict[str, Any]:
    if not rows:
        return {
            'n_rows': 0,
            'distance_m': 0.0,
            'steer_jerk_mean': 0.0,
            'cte_abs_mean': None,
            'lost_events': 0,
            'skip_events': 0,
            'score': -1e9,
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
    lost_events = sum(1 for r in rows if r.get('event') == 'lane_lost_recover')
    skip_events = sum(1 for r in rows if r.get('event') == 'skip_checkpoint')
    cte_mean = float(np.mean(np.abs(ctes))) if ctes.size else float('nan')
    # Prefer ring progress over raw odom wander.
    score = (
        12.0 * float(farthest_idx)
        + 0.6 * dist
        - 4.0 * lost_events
        - 3.0 * skip_events
        - 2.5 * float(np.mean(jerks))
        - (1.5 * cte_mean if math.isfinite(cte_mean) else 0.5)
    )
    return {
        'n_rows': len(rows),
        'distance_m': round(dist, 3),
        'steer_jerk_mean': round(float(np.mean(jerks)), 4),
        'cte_abs_mean': None if not math.isfinite(cte_mean) else round(cte_mean, 4),
        'lost_events': lost_events,
        'skip_events': skip_events,
        'score': round(score, 3),
    }


def is_lane_lost(output, *, fail_cte_m: float, family: str) -> bool:
    """True when lane following has meaningfully failed (not mission stops)."""
    decision = str(getattr(output, 'decision', '') or '')
    # ArUco / traffic mission stops must NOT trigger teleport recovery — that
    # created an infinite obstacle-zone loop (red lane + stop marker).
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
    if family == 'mask_p':
        if path_src == 'mask_drivable':
            return False
        if path_src.endswith('fallback_pp'):
            return False
    cte = dbg.get('cross_track_error_m')
    if cte is not None and abs(float(cte)) > fail_cte_m:
        return True
    if not bool(dbg.get('white_visible', True)) and path_src in (
        'none',
        'hold_previous',
    ):
        return True
    if path_src in ('none',) and family == 'pp':
        return True
    if family == 'mask_p' and path_src in ('none', 'hold_previous'):
        return True
    return False


def next_checkpoint_index(idx: int) -> int:
    """Ring advance: after obstacle wrap to start (final stretch)."""
    return (idx + 1) % len(OUT_CHECKPOINTS)

def run_one_lap(
    *,
    family: str,
    overrides: dict[str, Any],
    trial: int,
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
    viz: str = 'control',
) -> dict[str, Any]:
    import rclpy
    from control_msgs.msg import Control
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    from viz_util import apply_lane_viz

    run_dir = _ensure(out_dir / f'{family}_t{trial:02d}')
    csv_path = run_dir / 'drive.csv'
    events: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    planner = build_planner(overrides)
    apply_lane_viz(viz)
    ld._apply_detect_tune_from_yaml()

    # Start at first checkpoint.
    active_idx = 0
    retries = 0
    farthest_idx = 0
    _teleport(OUT_CHECKPOINTS[active_idx])
    time.sleep(settle_sec)

    rclpy.init()
    node = Node(f'out_lap_bench_{family}_{trial}')
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
    passed_obstacle = False
    lap_ok = False
    first_frame = None
    sx0, sy0 = catalog['start']
    prev_xy: tuple[float, float] | None = None
    odom_distance = 0.0

    try:
        while time.time() - t0 < max_lap_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = latest['frame']
            ox, oy = latest['odom_x'], latest['odom_y']
            if frame is None or ox is None or oy is None:
                continue

            now = time.time()
            dt = max(0.02, now - last_step)
            last_step = now
            if first_frame is None:
                first_frame = frame.copy()
            if prev_xy is not None:
                odom_distance += math.hypot(ox - prev_xy[0], oy - prev_xy[1])
            prev_xy = (ox, oy)

            # Keep forced left preference alive; allow FSM states otherwise.
            planner.desired_turn = TurnSign.LEFT
            planner._forced_turn = TurnSign.LEFT
            output = planner.step(frame, now_sec=now)
            # During OUT white-follow bench, ignore ArUco hard-stop (mission)
            # so we can drive through the red-lane zone toward start.
            if str(output.decision) == 'aruco_stop':
                publish_cmd(output.command.steering, max(0.12, float(
                    overrides.get('cruise_throttle', 0.2)
                ) * 0.5))
            else:
                publish_cmd(output.command.steering, output.command.throttle)

            near_name, near_i, near_d = nearest_out_spawn(ox, oy, catalog)
            # Ordered progress: accept a checkpoint only at/after the next expected.
            expected = min(farthest_idx + 1, len(OUT_CHECKPOINTS) - 1)
            sx, sy = catalog[OUT_CHECKPOINTS[expected]]
            if math.hypot(ox - sx, oy - sy) < 1.35:
                farthest_idx = expected
            if near_i > farthest_idx and near_d < 1.0:
                farthest_idx = near_i
            if (
                OUT_CHECKPOINTS[farthest_idx] == 'obstacle'
                or (near_name == 'obstacle' and near_d < 1.5)
            ):
                passed_obstacle = True

            d_start = math.hypot(ox - sx0, oy - sy0)
            elapsed = now - t0
            if (
                passed_obstacle
                and farthest_idx >= len(OUT_CHECKPOINTS) - 2
                and d_start < start_radius_m
                and odom_distance >= lap_min_distance_m
                and elapsed > 20.0
            ):
                lap_ok = True
                events.append(
                    {
                        't': round(elapsed, 2),
                        'event': 'lap_complete',
                        'x': ox,
                        'y': oy,
                        'odom_distance': round(odom_distance, 3),
                        'farthest': OUT_CHECKPOINTS[farthest_idx],
                    }
                )
                break

            lost = False
            if now >= recover_cooldown_until:
                lost = is_lane_lost(output, fail_cte_m=fail_cte_m, family=family)
            if lost:
                if lost_since is None:
                    lost_since = now
            else:
                lost_since = None

            event = None
            if lost_since is not None and (now - lost_since) >= lost_hold_sec:
                stop_car()
                recover_name, recover_i, _ = nearest_out_spawn(ox, oy, catalog)
                if recover_i < farthest_idx:
                    recover_i = farthest_idx
                    recover_name = OUT_CHECKPOINTS[recover_i]
                # Red-lane / ArUco mission at obstacle: do not respawn onto
                # obstacle forever — jump to start final stretch after retries.
                retries += 1
                event = 'lane_lost_recover'
                ev = {
                    't': round(now - t0, 2),
                    'event': event,
                    'spawn': recover_name,
                    'retries': retries,
                    'farthest': OUT_CHECKPOINTS[farthest_idx],
                    'x': ox,
                    'y': oy,
                    'decision': str(output.decision),
                    'path_source': str(output.path_source.value),
                }
                if retries > max_retries:
                    if recover_i >= len(OUT_CHECKPOINTS) - 1:
                        # Leave obstacle zone → start (lap finish attempt).
                        recover_i = 0
                        recover_name = 'start'
                        passed_obstacle = True
                        farthest_idx = len(OUT_CHECKPOINTS) - 1
                        event = 'skip_past_obstacle_to_start'
                    else:
                        recover_i = next_checkpoint_index(recover_i)
                        recover_name = OUT_CHECKPOINTS[recover_i]
                        farthest_idx = max(farthest_idx, recover_i)
                        event = 'skip_checkpoint'
                    retries = 0
                    ev['event'] = event
                    ev['spawn'] = recover_name
                events.append(ev)
                active_idx = recover_i
                _teleport(recover_name)
                time.sleep(settle_sec)
                recover_cooldown_until = time.time() + max(1.5, settle_sec + 0.5)
                planner.neutralize_steering()
                lost_since = None
                planner.force_fork_choice(TurnSign.LEFT, state=DrivingState.NORMAL)
                planner._forced_turn = TurnSign.LEFT

            dbg = output.debug or {}
            rows.append(
                {
                    't': round(now - t0, 3),
                    'family': family,
                    'trial': trial,
                    'steering': float(output.command.steering),
                    'throttle': float(output.command.throttle),
                    'decision': output.decision,
                    'path_source': str(output.path_source.value),
                    'state': str(output.state.value),
                    'cte_m': dbg.get('cross_track_error_m'),
                    'white_visible': dbg.get('white_visible'),
                    'odom_x': ox,
                    'odom_y': oy,
                    'near_spawn': near_name,
                    'farthest_idx': farthest_idx,
                    'event': event,
                    'dt': round(dt, 4),
                }
            )
            time.sleep(0.01)
    finally:
        stop_car()
        node.destroy_node()
        rclpy.shutdown()

    metrics = summarize_rows(rows, farthest_idx=farthest_idx)
    # Distance gate for lap_ok.
    if lap_ok and metrics['distance_m'] < lap_min_distance_m:
        lap_ok = False
        events.append(
            {
                'event': 'lap_rejected_short_distance',
                'distance_m': metrics['distance_m'],
                'need': lap_min_distance_m,
            }
        )
    if first_frame is not None:
        cv2.imwrite(str(run_dir / 'frame0.png'), first_frame)

    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        'family': family,
        'trial': trial,
        'overrides': overrides,
        'lap_ok': lap_ok,
        'farthest_checkpoint': OUT_CHECKPOINTS[farthest_idx],
        'farthest_idx': farthest_idx,
        'elapsed_sec': round(time.time() - t0, 2),
        'events': events,
        'csv': str(csv_path),
        **metrics,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    (run_dir / 'events.json').write_text(
        json.dumps(events, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', default='pp,mask_p')
    parser.add_argument('--max-param-trials', type=int, default=3)
    parser.add_argument('--max-lap-sec', type=float, default=180.0)
    parser.add_argument('--lost-hold-sec', type=float, default=1.2)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--fail-cte-m', type=float, default=0.28)
    parser.add_argument('--lap-min-distance-m', type=float, default=12.0)
    parser.add_argument('--start-radius-m', type=float, default=1.4)
    parser.add_argument('--settle', type=float, default=1.0)
    parser.add_argument('--camera-topic', default='/camera/image/compressed')
    parser.add_argument(
        '--viz',
        default='control',
        help='Perception windows: off|control|on (default control = Lane drive)',
    )
    parser.add_argument(
        '--stop-on-first-lap',
        action='store_true',
        help='Within a family, stop param trials after first successful lap',
    )
    args = parser.parse_args()

    families = [f.strip() for f in args.families.split(',') if f.strip()]
    for fam in families:
        if fam not in FAMILY_SEEDS:
            print(f'unknown family: {fam}', file=sys.stderr)
            return 2

    catalog = load_spawn_xy()
    stamp = _stamp()
    out_root = _ensure(OUT_ROOT / stamp)
    index: list[dict[str, Any]] = []
    meta = {
        'stamp': stamp,
        'families': families,
        'checkpoints': OUT_CHECKPOINTS,
        'max_param_trials': args.max_param_trials,
        'max_lap_sec': args.max_lap_sec,
        'viz': args.viz,
        'note': 'sim-auto OFF; this script owns /control',
    }
    (out_root / 'META.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    for family in families:
        params = deepcopy(FAMILY_SEEDS[family])
        best: dict[str, Any] | None = None
        for trial in range(args.max_param_trials):
            print(f'\n=== FAMILY={family} trial={trial} params={params}')
            try:
                summary = run_one_lap(
                    family=family,
                    overrides=params,
                    trial=trial,
                    out_dir=_ensure(out_root / family),
                    catalog=catalog,
                    camera_topic=args.camera_topic,
                    max_lap_sec=args.max_lap_sec,
                    lost_hold_sec=args.lost_hold_sec,
                    max_retries=args.max_retries,
                    fail_cte_m=args.fail_cte_m,
                    lap_min_distance_m=args.lap_min_distance_m,
                    start_radius_m=args.start_radius_m,
                    settle_sec=args.settle,
                    viz=args.viz,
                )
            except Exception as exc:  # noqa: BLE001
                summary = {
                    'family': family,
                    'trial': trial,
                    'overrides': params,
                    'lap_ok': False,
                    'error': str(exc),
                    'score': -1e9,
                    'distance_m': 0.0,
                    'lost_events': 0,
                    'skip_events': 0,
                    'steer_jerk_mean': 0.0,
                }
                print(f'ERROR: {exc}', file=sys.stderr)
            index.append(summary)
            print(json.dumps({k: summary[k] for k in summary if k != 'events'}, ensure_ascii=False))
            if best is None or float(summary.get('score', -1e9)) > float(
                best.get('score', -1e9)
            ):
                best = summary
            if summary.get('lap_ok') and args.stop_on_first_lap:
                break
            params = mutate_params(
                family,
                params,
                lost_events=int(summary.get('lost_events', 0)),
                skip_events=int(summary.get('skip_events', 0)),
                steer_jerk=float(summary.get('steer_jerk_mean') or 0.0),
                distance_m=float(summary.get('distance_m') or 0.0),
                lap_ok=bool(summary.get('lap_ok')),
                trial=trial,
            )
            time.sleep(0.5)
        if best is not None:
            (out_root / family / 'BEST.json').write_text(
                json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8'
            )

    (out_root / 'INDEX.json').write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    lines = [
        f'# OUT lap bench {stamp}',
        '',
        '| family | trial | lap_ok | dist_m | lost | skip | jerk | score | farthest |',
        '|---|---:|---|---:|---:|---:|---:|---:|---|',
    ]
    for row in index:
        lines.append(
            f"| {row.get('family')} | {row.get('trial')} | {row.get('lap_ok')} | "
            f"{row.get('distance_m')} | {row.get('lost_events')} | {row.get('skip_events')} | "
            f"{row.get('steer_jerk_mean')} | {row.get('score')} | {row.get('farthest_checkpoint')} |"
        )
    # Per-family winner
    lines.extend(['', '## Best by family', ''])
    for family in families:
        cands = [r for r in index if r.get('family') == family and 'score' in r]
        if not cands:
            continue
        win = max(cands, key=lambda r: float(r.get('score', -1e9)))
        lines.append(
            f"- **{family}**: trial={win.get('trial')} score={win.get('score')} "
            f"lap_ok={win.get('lap_ok')} params=`{win.get('overrides')}`"
        )
    (out_root / 'INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nWrote {out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
