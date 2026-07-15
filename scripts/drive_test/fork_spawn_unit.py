#!/usr/bin/env python3
"""Spawn-segment fork drive unit tests with logging (Out fork + In exit).

Contract
--------
- route_mode IN  → yellow priority (WonTae single-lane) + yellow fork layers
- route_mode OUT → white centerline + white/road_split fork layers
- LEFT  → branch rank **0**
- RIGHT → branch rank **1**
- While fork-selected: PP follows **only** the chosen layer

Modes
-----
offline : replay a saved frame through perception + MainPlanner (no Gazebo)
live    : teleport spawn → camera loop → log CSV (needs 2026-smh-sim + bringup)

Examples (inside container)::

  # Offline smoke (saved frames)
  python3 scripts/drive_test/fork_spawn_unit.py --mode offline --scenario all

  # Live: teleport + drive log (auto_driving already running optional;
  # this script runs its own MainPlanner.step on /camera/... )
  python3 scripts/drive_test/fork_spawn_unit.py --mode live \\
      --scenario out_left --duration 8 --repeat 2

  ./scripts/dev_container.sh teleport out_fork   # manual teleport if needed
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
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

from inference.pipeline import (  # noqa: E402
    MainPlanner,
    load_planner_config,
)
from inference.types import DrivingState, TurnSign  # noqa: E402
from inference.modules import lane_detection as ld  # noqa: E402

OUT_ROOT = _ROOT / 'data' / 'captures' / 'fork_drive_logs'

# Default frames from fork verify sources (offline).
_DEFAULT_FRAMES = {
    'out': _ROOT
    / 'data/captures/lane_tune_logs/auto_fork/out_fork/runs/20260713_152900/source_frame.png',
    'exit': _ROOT
    / 'data/captures/lane_tune_logs/auto_fork/in_roundabout_exit/runs/20260713_152921/source_frame.png',
}


@dataclass(frozen=True)
class Scenario:
    name: str
    spawn: str
    route: str  # in|out
    turn: str  # left|right
    rank: int
    state: str  # FORK_TURN | ROUNDABOUT_EXIT
    frame_key: str  # out|exit


SCENARIOS: dict[str, Scenario] = {
    'out_left': Scenario(
        'out_left', 'out_fork', 'out', 'left', 0, 'FORK_TURN', 'out'
    ),
    'out_right': Scenario(
        'out_right', 'out_fork', 'out', 'right', 1, 'FORK_TURN', 'out'
    ),
    'in_exit_left': Scenario(
        'in_exit_left',
        'in_roundabout_exit',
        'in',
        'left',
        0,
        'ROUNDABOUT_EXIT',
        'exit',
    ),
    'in_exit_right': Scenario(
        'in_exit_right',
        'in_roundabout_exit',
        'in',
        'right',
        1,
        'ROUNDABOUT_EXIT',
        'exit',
    ),
}


def _turn(s: str) -> TurnSign:
    return TurnSign.LEFT if s.lower() == 'left' else TurnSign.RIGHT


def _state(s: str) -> DrivingState:
    return DrivingState[s]


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_planner(route: str) -> MainPlanner:
    cfg = load_planner_config(route_mode=route)
    # Unit tests: arm fork without a camera sign at the spawn pose.
    if route == 'out':
        from dataclasses import replace

        cfg = replace(cfg, out_fork_forced_turn_arms=True)
    return MainPlanner(cfg)


def run_offline_once(scenario: Scenario, frame_path: Path, out_dir: Path) -> dict[str, Any]:
    """One-shot perception + forced fork choice on a still frame."""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise FileNotFoundError(frame_path)

    ld.VISUALIZE = False
    ld._apply_detect_tune_from_yaml()
    planner = build_planner(scenario.route)
    planner.force_fork_choice(_turn(scenario.turn), state=_state(scenario.state))

    prefer_yellow = scenario.route == 'in'
    # Bypass sign/aruco: call lane detect then drive decision path via step patches
    lane = ld.detect(frame, prefer_yellow=prefer_yellow)
    # Manually exercise selected layer (same as FORK_TURN body).
    rank = int(planner._fork_selected_rank)
    path, source, conf = planner._selected_layer_path(lane, rank)
    pursuit = planner._pure_pursuit(path, dt_sec=0.1)

    preview = None
    try:
        _, dbg = ld.detect_with_debug(frame, prefer_yellow=prefer_yellow)
        preview = ld.make_fork_lane_pair_preview(dbg, focus='all')
        focus = 'left' if rank == 0 else 'right'
        focused = ld.make_fork_lane_pair_preview(dbg, focus=focus)
    except Exception:
        focused = None

    result = {
        'scenario': scenario.name,
        'route': scenario.route,
        'turn': scenario.turn,
        'expected_rank': scenario.rank,
        'selected_rank': rank,
        'fork_active': bool(lane.fork_active),
        'n_branches': len(getattr(lane, 'branches', ()) or ()),
        'path_source': getattr(source, 'value', str(source)),
        'path_points': int(path.shape[0]),
        'path_confidence': float(conf),
        'pursuit_valid': bool(pursuit.valid),
        'steering': float(getattr(pursuit, 'steering', 0.0) or 0.0)
        if pursuit.valid
        else None,
        'rank_ok': rank == scenario.rank,
        'layer_ok': bool(lane.fork_active)
        and len(getattr(lane, 'branches', ()) or ()) >= 2
        and path.shape[0] >= planner.config.min_points,
        'frame': str(frame_path),
    }

    run_dir = _ensure_dir(out_dir / scenario.name)
    (run_dir / 'result.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8'
    )
    cv2.imwrite(str(run_dir / 'frame.png'), frame)
    if preview is not None:
        cv2.imwrite(str(run_dir / 'fork_all.png'), preview)
    if focused is not None:
        cv2.imwrite(str(run_dir / f'fork_rank{rank}.png'), focused)
    return result


def _teleport(spawn: str) -> None:
    import subprocess

    cmd = [
        sys.executable,
        str(_ROOT / 'scripts' / 'teleport_spawn_pose.py'),
        spawn,
    ]
    subprocess.check_call(cmd, cwd=str(_ROOT))


def run_live_once(
    scenario: Scenario,
    *,
    duration_sec: float,
    out_dir: Path,
    camera_topic: str,
    viz: str = 'control',
    drive: bool = True,
    rclpy_owned: bool = False,
) -> dict[str, Any]:
    """Teleport, force fork choice, step planner on live camera, log + viz.

    Opens OpenCV ``Lane drive`` / fork pair preview when ``viz`` is not off.
    With ``drive``, publishes ``/control`` so the vehicle follows the chosen branch.
    """
    import rclpy
    from control_msgs.msg import Control
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    _DRIVE = Path(__file__).resolve().parent
    if str(_DRIVE) not in sys.path:
        sys.path.insert(0, str(_DRIVE))
    from viz_util import apply_lane_viz  # noqa: E402

    _teleport(scenario.spawn)
    time.sleep(0.8)

    planner = build_planner(scenario.route)
    planner.force_fork_choice(_turn(scenario.turn), state=_state(scenario.state))
    apply_lane_viz(viz)
    ld._apply_detect_tune_from_yaml()

    own_rclpy = False
    if not rclpy.ok():
        rclpy.init()
        own_rclpy = True
    node = Node(f'fork_spawn_{scenario.name}')
    bridge = CvBridge()
    latest: dict[str, Any] = {'frame': None, 't': 0.0}

    def _cb_raw(msg: Image) -> None:
        latest['frame'] = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        latest['t'] = time.time()

    def _cb_compressed(msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        latest['frame'] = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        latest['t'] = time.time()

    if camera_topic.endswith('compressed'):
        node.create_subscription(CompressedImage, camera_topic, _cb_compressed, 10)
    else:
        node.create_subscription(Image, camera_topic, _cb_raw, 10)
    control_pub = node.create_publisher(Control, '/control', 10) if drive else None

    run_dir = _ensure_dir(out_dir / scenario.name)
    csv_path = run_dir / 'drive.csv'
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    ok_frames = 0
    snap_i = 0
    last_snap = 0.0
    try:
        while time.time() - t0 < duration_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = latest['frame']
            if frame is None:
                continue
            now = time.time()
            planner.desired_turn = _turn(scenario.turn)
            planner._fork_selected_rank = scenario.rank
            planner.state = _state(scenario.state)
            output = planner.step(frame, now_sec=now)
            if control_pub is not None and output.command is not None:
                msg = Control()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.header.frame_id = 'base_link'
                msg.steering = float(output.command.steering)
                msg.throttle = float(output.command.throttle)
                control_pub.publish(msg)

            rank = output.debug.get('selected_branch_rank')
            path_src = str(output.path_source.value)
            row = {
                't': round(now - t0, 3),
                'state': str(
                    output.state.value if hasattr(output.state, 'value') else output.state
                ),
                'decision': output.decision,
                'path_source': path_src,
                'selected_rank': rank,
                'fork_active': output.debug.get('fork_active'),
                'steering': float(output.command.steering)
                if output.command is not None
                else None,
                'throttle': float(output.command.throttle)
                if output.command is not None
                else None,
            }
            rows.append(row)
            if rank == scenario.rank and path_src.endswith('branch'):
                ok_frames += 1

            if now - last_snap >= 1.0:
                last_snap = now
                try:
                    prefer_yellow = scenario.route == 'in'
                    _, dbg = ld.detect_with_debug(
                        frame,
                        prefer_yellow=prefer_yellow,
                        enable_fork=True,
                        active_branch_rank=scenario.rank,
                    )
                    focus = 'left' if scenario.rank == 0 else 'right'
                    sheet = ld.make_fork_lane_pair_preview(dbg, focus=focus)
                    if sheet is not None:
                        label = (
                            f'{scenario.name} rank={scenario.rank} '
                            f'src={path_src} ste={row["steering"]}'
                        )
                        cv2.putText(
                            sheet,
                            label,
                            (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        cv2.imwrite(
                            str(run_dir / f'snap_{snap_i:02d}_{focus}.png'), sheet
                        )
                        cv2.imshow('Fork select', sheet)
                        snap_i += 1
                except Exception as exc:  # noqa: BLE001
                    print(f'[viz] snap skip: {exc}', flush=True)
                cv2.waitKey(1)
            time.sleep(0.02)
    finally:
        if control_pub is not None:
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
        if own_rclpy and not rclpy_owned and rclpy.ok():
            try:
                cv2.destroyWindow('Fork select')
            except Exception:
                pass
            rclpy.shutdown()

    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        'scenario': scenario.name,
        'spawn': scenario.spawn,
        'route': scenario.route,
        'turn': scenario.turn,
        'expected_rank': scenario.rank,
        'duration_sec': duration_sec,
        'viz': viz,
        'drive': drive,
        'n_rows': len(rows),
        'ok_branch_frames': ok_frames,
        'ok_ratio': (ok_frames / len(rows)) if rows else 0.0,
        'snaps': snap_i,
        'csv': str(csv_path),
        'ok': bool(rows) and (ok_frames / max(len(rows), 1)) >= 0.3,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode', choices=('offline', 'live'), default='offline'
    )
    parser.add_argument(
        '--scenario',
        default='all',
        help='out_left|out_right|in_exit_left|in_exit_right|all',
    )
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument(
        '--camera-topic',
        default='/camera/image/compressed',
    )
    parser.add_argument(
        '--frame-out',
        type=Path,
        default=None,
        help='Override offline out_fork frame',
    )
    parser.add_argument(
        '--frame-exit',
        type=Path,
        default=None,
        help='Override offline exit frame',
    )
    parser.add_argument(
        '--viz',
        default='control',
        help='off|control|on — Lane drive + Fork select (live)',
    )
    parser.add_argument(
        '--drive',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Publish /control while live (default on)',
    )
    args = parser.parse_args()

    names = (
        list(SCENARIOS)
        if args.scenario == 'all'
        else [s.strip() for s in args.scenario.split(',')]
    )
    for name in names:
        if name not in SCENARIOS:
            print(f'unknown scenario: {name}', file=sys.stderr)
            return 2

    stamp = _stamp()
    out_root = _ensure_dir(OUT_ROOT / stamp)
    index: list[dict[str, Any]] = []

    frames = {
        'out': args.frame_out or _DEFAULT_FRAMES['out'],
        'exit': args.frame_exit or _DEFAULT_FRAMES['exit'],
    }

    print(
        f'=== fork_spawn_unit mode={args.mode} viz={args.viz} '
        f'drive={args.drive} → {out_root}',
        flush=True,
    )

    for rep in range(args.repeat):
        for name in names:
            sc = SCENARIOS[name]
            rep_dir = _ensure_dir(out_root / f'r{rep:02d}_{name}')
            print(f'=== [{args.mode}] rep={rep} {name} → {rep_dir}', flush=True)
            if args.mode == 'offline':
                frame = frames[sc.frame_key]
                result = run_offline_once(sc, Path(frame), rep_dir)
            else:
                result = run_live_once(
                    sc,
                    duration_sec=args.duration,
                    out_dir=rep_dir,
                    camera_topic=args.camera_topic,
                    viz=args.viz,
                    drive=args.drive,
                )
            index.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    (out_root / 'INDEX.json').write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    lines = [
        f'# Fork spawn unit {stamp}',
        '',
        f'Mode: `{args.mode}` · viz=`{args.viz}` · drive={args.drive} · repeat={args.repeat}',
        '',
        '| scenario | result | notes |',
        '|----------|--------|-------|',
    ]
    for row in index:
        if 'rank_ok' in row:
            lines.append(
                f"| {row['scenario']} | rank={row['rank_ok']} layer={row['layer_ok']} "
                f"| forks={row.get('n_branches')} pts={row.get('path_points')} |"
            )
        else:
            lines.append(
                f"| {row['scenario']} | ok={row.get('ok')} ratio={row.get('ok_ratio', 0):.2f} "
                f"| rows={row.get('n_rows')} snaps={row.get('snaps')} |"
            )
    (out_root / 'INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nWrote {out_root}', flush=True)
    fails = [
        r
        for r in index
        if ('rank_ok' in r and (not r['rank_ok'] or not r['layer_ok']))
        or ('ok_ratio' in r and r['ok_ratio'] < 0.3)
    ]
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
