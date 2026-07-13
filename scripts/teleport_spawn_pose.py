#!/usr/bin/env python3
"""Teleport the sim robot to a mission spawn preset without relaunching Gazebo.

Uses /set_entity_state (gazebo_ros) and config/spawn_poses.yaml as SSOT.

Examples (inside 2026-smh-sim, bringup already running):

  python3 scripts/teleport_spawn_pose.py --list
  python3 scripts/teleport_spawn_pose.py in_roundabout_exit
  python3 scripts/teleport_spawn_pose.py custom --x 0.0 --y -3.6 --yaw 1.57

From the host:

  ./scripts/dev_container.sh teleport in_roundabout_exit
  ./scripts/dev_container.sh teleport --list
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import rclpy
    from gazebo_msgs.msg import EntityState
    from gazebo_msgs.srv import SetEntityState
    from geometry_msgs.msg import Pose, Quaternion, Twist, Vector3
except ImportError as exc:  # pragma: no cover
    print(
        '[teleport] ROS 2 / gazebo_msgs import failed. '
        'source /opt/ros/humble/setup.bash && source install/setup.bash\n'
        f'  detail: {exc}',
        file=sys.stderr,
    )
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = ROOT / 'src' / 'dracer_sim' / 'config' / 'spawn_poses.yaml'
DEFAULT_ENTITY = 'limo'
SET_ENTITY_STATE = '/set_entity_state'


def _resolve_yaml(path: Path | None) -> Path:
    if path is not None:
        return path
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory('dracer_sim'))
        candidate = share / 'config' / 'spawn_poses.yaml'
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    return DEFAULT_YAML


def load_catalog(yaml_path: Path) -> dict[str, Any]:
    if not yaml_path.is_file():
        raise FileNotFoundError(f'spawn poses yaml not found: {yaml_path}')
    data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
    poses = data.get('poses') or {}
    if not isinstance(poses, dict) or not poses:
        raise ValueError(f'no poses in {yaml_path}')
    return {
        'default': str(data.get('default') or 'start'),
        'poses': poses,
        'path': yaml_path,
    }


def yaw_to_quat(yaw: float) -> Quaternion:
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def resolve_pose(
    catalog: dict[str, Any],
    pose_id: str,
    *,
    x: float | None,
    y: float | None,
    z: float | None,
    yaw: float | None,
) -> tuple[str, str, float, float, float, float]:
    poses = catalog['poses']
    pose_id = (pose_id or catalog['default']).strip()

    if pose_id == 'custom':
        if x is None or y is None or yaw is None:
            raise ValueError('custom requires --x --y --yaw (and optional --z)')
        return 'custom', 'manual', float(x), float(y), float(z if z is not None else 0.15), float(yaw)

    if pose_id not in poses:
        known = ', '.join(sorted(poses) + ['custom'])
        raise KeyError(f"unknown pose '{pose_id}'. known: {known}")

    pose = poses[pose_id]
    return (
        pose_id,
        str(pose.get('label') or pose_id),
        float(x if x is not None else pose['x']),
        float(y if y is not None else pose['y']),
        float(z if z is not None else pose.get('z', 0.15)),
        float(yaw if yaw is not None else pose['yaw']),
    )


def print_list(catalog: dict[str, Any]) -> None:
    print(f"SSOT: {catalog['path']}")
    print(f"default: {catalog['default']}")
    print('')
    for name in sorted(catalog['poses']):
        p = catalog['poses'][name]
        label = p.get('label') or name
        print(
            f"  {name:22s}  x={float(p['x']):7.3f}  y={float(p['y']):7.3f}  "
            f"z={float(p.get('z', 0.15)):5.2f}  yaw={float(p['yaw']):7.3f}  # {label}"
        )
    print('  custom                  # --x --y --yaw [--z]')


def teleport(entity: str, x: float, y: float, z: float, yaw: float, timeout_sec: float) -> None:
    rclpy.init()
    node = rclpy.create_node('teleport_spawn_pose')
    client = node.create_client(SetEntityState, SET_ENTITY_STATE)
    try:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(
                f'{SET_ENTITY_STATE} unavailable (is sim-bringup running in this ROS domain?)'
            )

        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = entity
        req.state.reference_frame = 'world'
        req.state.pose = Pose()
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = z
        req.state.pose.orientation = yaw_to_quat(yaw)
        # Zero twist so the robot does not keep sliding after the jump.
        req.state.twist = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=0.0),
        )

        future = client.call_async(req)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        if not future.done():
            raise RuntimeError(f'{SET_ENTITY_STATE} call timed out')
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(
                f'{SET_ENTITY_STATE} failed for entity={entity!r} '
                '(wrong --entity? limo vs dracer_sim)'
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Teleport limo/dracer_sim to a spawn_poses.yaml preset (Gazebo runtime).',
    )
    p.add_argument(
        'pose',
        nargs='?',
        default=None,
        help='Preset name from spawn_poses.yaml, or custom',
    )
    p.add_argument('--list', '-l', action='store_true', help='List presets and exit')
    p.add_argument('--entity', default=DEFAULT_ENTITY, help=f'Gazebo model name (default: {DEFAULT_ENTITY})')
    p.add_argument('--yaml', type=Path, default=None, help='Override spawn_poses.yaml path')
    p.add_argument('--x', type=float, default=None, help='Override / custom x')
    p.add_argument('--y', type=float, default=None, help='Override / custom y')
    p.add_argument('--z', type=float, default=None, help='Override / custom z')
    p.add_argument('--yaw', type=float, default=None, help='Override / custom yaw (rad)')
    p.add_argument('--timeout', type=float, default=5.0, help='Service wait/call timeout (s)')
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = load_catalog(_resolve_yaml(args.yaml))

    if args.list or args.pose is None:
        if args.pose is None and not args.list:
            print('[teleport] pose name required (or use --list)\n', file=sys.stderr)
            print_list(catalog)
            return 2
        print_list(catalog)
        return 0

    try:
        pose_id, label, x, y, z, yaw = resolve_pose(
            catalog, args.pose, x=args.x, y=args.y, z=args.z, yaw=args.yaw
        )
    except (KeyError, ValueError) as exc:
        print(f'[teleport] {exc}', file=sys.stderr)
        return 2

    print(
        f'[teleport] {args.entity} → {pose_id} ({label}) '
        f'x={x:.4g} y={y:.4g} z={z:.4g} yaw={yaw:.4g}'
    )
    try:
        teleport(args.entity, x, y, z, yaw, timeout_sec=args.timeout)
    except Exception as exc:
        print(f'[teleport] FAILED: {exc}', file=sys.stderr)
        return 1

    print('[teleport] ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
