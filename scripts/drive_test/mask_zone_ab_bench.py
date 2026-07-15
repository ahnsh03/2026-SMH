#!/usr/bin/env python3
"""OUT eval-zone A/B for mask_p controllers (drivable-area only).

Loads ``reference_segments.json`` evaluation.zones, teleports to each zone
entry, drives with mask_p variants, scores CTE/heading vs pass_criteria and
transit time. Waypoints are for grading only — control uses road_clean mask.

Owns ``/control`` — keep sim-auto / inference_node OFF.

Example (inside 2026-smh-sim, bringup only)::

  python3 scripts/drive_test/mask_zone_ab_bench.py \\
      --variants atan_row,atan_ridge,atan_area,atan_fast,img_row \\
      --viz off
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

_ROOT = Path(__file__).resolve().parents[2]
_INFER = _ROOT / 'src' / 'inference'
if str(_INFER) not in sys.path:
    sys.path.insert(0, str(_INFER))

from inference.pipeline import MainPlanner, load_planner_config
from inference.modules import lane_detection as ld

OUT_ROOT = _ROOT / 'data' / 'captures' / 'mask_zone_ab_logs'
_REF_CANDIDATES = (
    _ROOT / 'data' / 'out_best_route' / 'reference_segments.json',
    _ROOT / 'data' / 'captures' / 'out_best_route' / 'reference_segments.json',
    _ROOT.parent / 'data' / 'out_best_route' / 'reference_segments.json',
    Path('/mnt/wslg/distro/home/aim06/projects/2026-seame-hackathon/data/out_best_route/reference_segments.json'),
)

# Soft corridor + mask free-space shared by all candidates.
_MASK_COMMON: dict[str, Any] = {
    'normal_tracker': 'mask_p',
    'mask_corridor_mode': 'soft',
    'mask_corridor_half_width_m': 0.42,
    'mask_path_weight_sigma_m': 0.28,
    'mask_use_path_correction': False,
    'mask_min_area_px': 80.0,
    'mask_lane_width_m': 0.35,
    'mask_require_color_path': False,
    'mask_occlusion_hold_frames': 24,
    'mask_near_band_ratio': 0.78,
    'mask_far_blend': 0.18,
    'mask_curve_speed_scale': 0.50,
    'error_speed_steer_full': 0.55,
    'error_speed_min_scale': 0.55,
}

VARIANTS: dict[str, dict[str, Any]] = {
    'atan_row': {
        **_MASK_COMMON,
        'cruise_throttle': 0.28,
        'curve_throttle': 0.18,
        'mask_steer_law': 'lateral_atan',
        'mask_steer_k': 1.80,
        'mask_steer_alpha': 0.35,
        'mask_center_mode': 'row_mid',
        'mask_erode_px': 2,
        'steering_rate_limit_per_sec': 12.0,
    },
    'atan_ridge': {
        **_MASK_COMMON,
        'cruise_throttle': 0.28,
        'curve_throttle': 0.18,
        'mask_steer_law': 'lateral_atan',
        'mask_steer_k': 1.80,
        'mask_steer_alpha': 0.35,
        'mask_center_mode': 'dist_ridge',
        'mask_erode_px': 2,
        'steering_rate_limit_per_sec': 12.0,
    },
    'atan_area': {
        **_MASK_COMMON,
        'cruise_throttle': 0.28,
        'curve_throttle': 0.18,
        'mask_steer_law': 'lateral_atan',
        'mask_steer_k': 1.80,
        'mask_steer_alpha': 0.35,
        'mask_center_mode': 'area',
        'mask_erode_px': 0,
        'steering_rate_limit_per_sec': 12.0,
    },
    'atan_fast': {
        **_MASK_COMMON,
        'cruise_throttle': 0.34,
        'curve_throttle': 0.20,
        'mask_steer_law': 'lateral_atan',
        'mask_steer_k': 2.00,
        'mask_steer_alpha': 0.38,
        'mask_center_mode': 'row_mid',
        'mask_erode_px': 2,
        'steering_rate_limit_per_sec': 14.0,
    },
    'img_row': {
        **_MASK_COMMON,
        'cruise_throttle': 0.28,
        'curve_throttle': 0.18,
        'mask_steer_law': 'image_p',
        'mask_steer_k': 1.55,
        'mask_steer_alpha': 0.28,
        'mask_center_mode': 'row_mid',
        'mask_erode_px': 2,
        'steering_rate_limit_per_sec': 8.0,
    },
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _find_reference() -> Path:
    for p in _REF_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError(
        'reference_segments.json not found; tried:\n'
        + '\n'.join(str(p) for p in _REF_CANDIDATES)
    )


def load_zones(ref_path: Path) -> list[dict[str, Any]]:
    data = json.loads(ref_path.read_text(encoding='utf-8'))
    zones = list((data.get('evaluation') or {}).get('zones') or [])
    if not zones:
        raise ValueError(f'no evaluation.zones in {ref_path}')
    return zones


def zone_polyline(zone: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (Nx2 xy meters, cumulative arc s along zone)."""
    wps = zone['waypoints']
    xy = np.array([[float(w['x_m']), float(w['y_m'])] for w in wps], dtype=np.float64)
    if xy.shape[0] < 2:
        raise ValueError(f'zone {zone.get("id")} has <2 waypoints')
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return xy, s


def entry_pose(xy: np.ndarray, *, back_m: float = 0.25) -> tuple[float, float, float]:
    """Spawn just before first waypoint facing along the zone tangent."""
    d = xy[1] - xy[0]
    yaw = math.atan2(float(d[1]), float(d[0]))
    ux, uy = math.cos(yaw), math.sin(yaw)
    x = float(xy[0, 0] - back_m * ux)
    y = float(xy[0, 1] - back_m * uy)
    return x, y, yaw


def project_cte_heading(
    xy: np.ndarray,
    s_poly: np.ndarray,
    x: float,
    y: float,
    yaw: float,
) -> tuple[float, float, float]:
    """Signed CTE (left +), heading err (rad), arc progress s along zone."""
    p = np.array([x, y], dtype=np.float64)
    best_d2 = float('inf')
    best_cte = 0.0
    best_s = 0.0
    best_tang = 0.0
    for i in range(xy.shape[0] - 1):
        a = xy[i]
        b = xy[i + 1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-12:
            continue
        t = float(np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0))
        proj = a + t * ab
        dlt = p - proj
        d2 = float(np.dot(dlt, dlt))
        if d2 >= best_d2:
            continue
        best_d2 = d2
        L = math.sqrt(L2)
        tx, ty = float(ab[0] / L), float(ab[1] / L)
        # Left-positive lateral (robot left of path → +CTE).
        best_cte = float(tx * dlt[1] - ty * dlt[0])
        best_s = float(s_poly[i] + t * L)
        best_tang = math.atan2(ty, tx)
    herr = (yaw - best_tang + math.pi) % (2.0 * math.pi) - math.pi
    return best_cte, herr, best_s


def quat_yaw(q) -> float:
    # nav_msgs Quaternion → yaw (ENU / Gazebo).
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _teleport_custom(x: float, y: float, yaw: float, *, retries: int = 4) -> None:
    import subprocess

    cmd = [
        sys.executable,
        str(_ROOT / 'scripts' / 'teleport_spawn_pose.py'),
        'custom',
        '--x',
        f'{x:.6f}',
        '--y',
        f'{y:.6f}',
        '--yaw',
        f'{yaw:.6f}',
    ]
    last: Exception | None = None
    for attempt in range(retries):
        try:
            subprocess.check_call(cmd, cwd=str(_ROOT))
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.6 + 0.3 * attempt)
    raise RuntimeError(f'teleport failed after {retries}: {last}')


def build_planner(variant: str, route: str) -> MainPlanner:
    base = load_planner_config(route_mode=route)
    return MainPlanner(replace(base, **VARIANTS[variant]))


def score_zone_rows(
    rows: list[dict[str, Any]],
    *,
    zone: dict[str, Any],
    length_m: float,
    finish_frac: float,
) -> dict[str, Any]:
    crit = zone.get('pass_criteria') or {}
    cte_lim_rms = float(crit.get('cte_rms_m_max', 0.06))
    cte_lim_abs = float(crit.get('cte_abs_max_m', 0.12))
    h_lim_rms = float(crit.get('heading_err_rms_rad_max', 0.25))
    maneuver = str(zone.get('maneuver') or 'corner')

    if not rows:
        return {
            'n_rows': 0,
            'completed': False,
            'pass': False,
            'transit_s': float('nan'),
            'cte_rms_m': float('nan'),
            'cte_abs_max_m': float('nan'),
            'heading_err_rms_rad': float('nan'),
            'path_lost_frames': 0,
            'mean_throttle': 0.0,
            'max_progress_m': 0.0,
            'maneuver': maneuver,
        }

    ctes = np.array([float(r['cte_m']) for r in rows], dtype=np.float64)
    herrs = np.array([float(r['heading_err_rad']) for r in rows], dtype=np.float64)
    progress = float(max(float(r['s_m']) for r in rows))
    completed = progress >= finish_frac * length_m
    path_lost = sum(1 for r in rows if r.get('path_lost'))
    # Prefer time of first sample past finish threshold.
    transit = float(rows[-1]['t'])
    for r in rows:
        if float(r['s_m']) >= finish_frac * length_m:
            transit = float(r['t'])
            break

    cte_rms = float(np.sqrt(np.mean(ctes * ctes)))
    cte_abs_max = float(np.max(np.abs(ctes)))
    h_rms = float(np.sqrt(np.mean(herrs * herrs)))
    mean_thr = float(np.mean([float(r['throttle']) for r in rows]))

    ok_cte = cte_rms <= cte_lim_rms and cte_abs_max <= cte_lim_abs
    ok_h = h_rms <= h_lim_rms
    # Soft path_lost tolerance: short occlusion holds OK if finished cleanly.
    ok_lost = path_lost < max(8, int(0.15 * len(rows)))
    passed = bool(completed and ok_cte and ok_h and ok_lost)

    return {
        'n_rows': len(rows),
        'completed': completed,
        'pass': passed,
        'pass_cte': ok_cte,
        'pass_heading': ok_h,
        'pass_lost': ok_lost,
        'transit_s': round(transit, 3),
        'cte_rms_m': round(cte_rms, 4),
        'cte_abs_max_m': round(cte_abs_max, 4),
        'heading_err_rms_rad': round(h_rms, 4),
        'path_lost_frames': path_lost,
        'mean_throttle': round(mean_thr, 3),
        'max_progress_m': round(progress, 3),
        'length_m': round(length_m, 3),
        'maneuver': maneuver,
        'criteria': {
            'cte_rms_m_max': cte_lim_rms,
            'cte_abs_max_m': cte_lim_abs,
            'heading_err_rms_rad_max': h_lim_rms,
        },
    }


def rank_variants(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-variant, sort by pass_count, CTE, transit."""
    by_v: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        by_v.setdefault(str(r['variant']), []).append(r)

    ranked: list[dict[str, Any]] = []
    for variant, runs in by_v.items():
        n_pass = sum(1 for r in runs if r.get('pass'))
        n_zones = len(runs)
        # Weighted CTE: straight ×1.2
        w_sum = 0.0
        cte_acc = 0.0
        transit_sum = 0.0
        complete_n = 0
        for r in runs:
            w = 1.2 if r.get('maneuver') == 'straight' else 1.0
            cte = float(r.get('cte_rms_m') or 1.0)
            if not math.isfinite(cte):
                cte = 1.0
            cte_acc += w * cte
            w_sum += w
            if r.get('completed') and math.isfinite(float(r.get('transit_s', float('nan')))):
                transit_sum += float(r['transit_s'])
                complete_n += 1
            else:
                transit_sum += 120.0  # penalty
                complete_n += 1
        cte_w = cte_acc / max(w_sum, 1e-6)
        transit_tot = transit_sum
        # Sort key helpers (higher better for score display).
        score = (
            1000.0 * n_pass
            - 100.0 * cte_w
            - 1.0 * transit_tot
            - 50.0 * sum(1 for r in runs if not r.get('completed'))
        )
        ranked.append(
            {
                'variant': variant,
                'n_pass': n_pass,
                'n_zones': n_zones,
                'weighted_cte_rms': round(cte_w, 4),
                'total_transit_s': round(transit_tot, 3),
                'score': round(score, 3),
                'config': VARIANTS[variant],
                'zones': [
                    {
                        'id': r['zone_id'],
                        'pass': r.get('pass'),
                        'cte_rms_m': r.get('cte_rms_m'),
                        'transit_s': r.get('transit_s'),
                        'completed': r.get('completed'),
                    }
                    for r in runs
                ],
            }
        )

    ranked.sort(
        key=lambda d: (
            -int(d['n_pass']),
            float(d['weighted_cte_rms']),
            float(d['total_transit_s']),
        )
    )
    return ranked


def run_zone(
    *,
    zone: dict[str, Any],
    variant: str,
    route: str,
    out_dir: Path,
    camera_topic: str,
    settle_sec: float,
    timeout_sec: float,
    finish_frac: float,
    viz: str,
) -> dict[str, Any]:
    import rclpy
    from control_msgs.msg import Control
    from cv_bridge import CvBridge
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    from viz_util import apply_lane_viz

    xy, s_poly = zone_polyline(zone)
    length_m = float(zone.get('length_m') or s_poly[-1])
    sx, sy, syaw = entry_pose(xy)
    zone_id = str(zone['id'])

    _teleport_custom(sx, sy, syaw)
    time.sleep(settle_sec)

    planner = build_planner(variant, route)
    apply_lane_viz(viz)
    ld._apply_detect_tune_from_yaml()

    rclpy.init()
    node = Node('mask_zone_ab_bench')
    bridge = CvBridge()
    latest: dict[str, Any] = {
        'frame': None,
        'odom_x': None,
        'odom_y': None,
        'odom_yaw': None,
    }

    def _cb_raw(msg: Image) -> None:
        latest['frame'] = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _cb_compressed(msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        latest['frame'] = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def _cb_odom(msg: Odometry) -> None:
        latest['odom_x'] = float(msg.pose.pose.position.x)
        latest['odom_y'] = float(msg.pose.pose.position.y)
        latest['odom_yaw'] = quat_yaw(msg.pose.pose.orientation)

    if camera_topic.endswith('compressed'):
        node.create_subscription(CompressedImage, camera_topic, _cb_compressed, 10)
    else:
        node.create_subscription(Image, camera_topic, _cb_raw, 10)
    node.create_subscription(Odometry, '/odom', _cb_odom, 10)
    control_pub = node.create_publisher(Control, '/control', 10)

    run_dir = _ensure(out_dir / f'{variant}__{zone_id}')
    csv_path = run_dir / 'drive.csv'
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    last_step = t0
    finished = False

    try:
        while time.time() - t0 < timeout_sec:
            rclpy.spin_once(node, timeout_sec=0.05)
            frame = latest['frame']
            ox, oy, oyaw = latest['odom_x'], latest['odom_y'], latest['odom_yaw']
            if frame is None or ox is None or oy is None or oyaw is None:
                continue
            now = time.time()
            dt = max(0.02, now - last_step)
            last_step = now

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

            cte, herr, s_m = project_cte_heading(xy, s_poly, ox, oy, oyaw)
            path_lost = str(output.path_source.value).startswith('hold')
            rows.append(
                {
                    't': round(now - t0, 3),
                    'variant': variant,
                    'zone_id': zone_id,
                    'decision': output.decision,
                    'path_source': str(output.path_source.value),
                    'steering': float(cmd.steering),
                    'throttle': float(cmd.throttle),
                    'cte_m': round(cte, 4),
                    'heading_err_rad': round(herr, 4),
                    's_m': round(s_m, 4),
                    'path_lost': path_lost,
                    'odom_x': ox,
                    'odom_y': oy,
                    'odom_yaw': round(oyaw, 4),
                    'dt': round(dt, 4),
                }
            )
            if s_m >= finish_frac * length_m:
                finished = True
                break
            # Bail if clearly off track early.
            if len(rows) > 20 and abs(cte) > 0.35:
                break
            time.sleep(0.01)
    finally:
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

    metrics = score_zone_rows(
        rows, zone=zone, length_m=length_m, finish_frac=finish_frac
    )
    summary = {
        'variant': variant,
        'zone_id': zone_id,
        'maneuver': zone.get('maneuver'),
        'label': zone.get('label'),
        'route': route,
        'timeout_sec': timeout_sec,
        'finished_early': finished,
        'spawn': {'x': sx, 'y': sy, 'yaw': syaw},
        'config': VARIANTS[variant],
        'csv': str(csv_path),
        **metrics,
    }
    (run_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return summary


def write_index(out_root: Path, ranked: list[dict[str, Any]], results: list[dict[str, Any]]) -> None:
    lines = [
        '# Mask zone A/B',
        '',
        f'Winner: **{ranked[0]["variant"]}**' if ranked else 'Winner: (none)',
        '',
        '| variant | pass | wCTE_rms | transit_s | score |',
        '|---|---:|---:|---:|---:|',
    ]
    for r in ranked:
        lines.append(
            f'| {r["variant"]} | {r["n_pass"]}/{r["n_zones"]} | '
            f'{r["weighted_cte_rms"]} | {r["total_transit_s"]} | {r["score"]} |'
        )
    lines.extend(['', '## Per zone', ''])
    for r in results:
        flag = 'PASS' if r.get('pass') else 'FAIL'
        lines.append(
            f'- `{r["variant"]}` / `{r["zone_id"]}`: {flag} '
            f'cte_rms={r.get("cte_rms_m")} max={r.get("cte_abs_max_m")} '
            f'h_rms={r.get("heading_err_rms_rad")} t={r.get("transit_s")}s '
            f'prog={r.get("max_progress_m")}/{r.get("length_m")}'
        )
    (out_root / 'INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--variants',
        default='atan_row,atan_ridge,atan_area,atan_fast,img_row',
        help='comma variant ids',
    )
    parser.add_argument(
        '--zones',
        default='',
        help='comma zone ids (default: all evaluation.zones)',
    )
    parser.add_argument('--route', default='out', choices=('out', 'in'))
    parser.add_argument('--settle', type=float, default=1.2)
    parser.add_argument(
        '--timeout-scale',
        type=float,
        default=2.8,
        help='timeout_sec ≈ scale * length_m / 0.22',
    )
    parser.add_argument('--finish-frac', type=float, default=0.92)
    parser.add_argument('--camera-topic', default='/camera/image/compressed')
    parser.add_argument('--viz', default='off', help='off|control|on')
    parser.add_argument(
        '--ref',
        default='',
        help='override path to reference_segments.json',
    )
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    for name in variants:
        if name not in VARIANTS:
            print(f'unknown variant: {name}', file=sys.stderr)
            print('known:', ', '.join(VARIANTS), file=sys.stderr)
            return 2

    ref_path = Path(args.ref) if args.ref else _find_reference()
    all_zones = load_zones(ref_path)
    if args.zones.strip():
        want = {z.strip() for z in args.zones.split(',') if z.strip()}
        zones = [z for z in all_zones if z['id'] in want]
        missing = want - {z['id'] for z in zones}
        if missing:
            print(f'unknown zones: {sorted(missing)}', file=sys.stderr)
            return 2
    else:
        zones = all_zones

    stamp = _stamp()
    out_root = _ensure(OUT_ROOT / stamp)
    meta = {
        'stamp': stamp,
        'ref': str(ref_path),
        'variants': variants,
        'zones': [z['id'] for z in zones],
        'route': args.route,
        'viz': args.viz,
        'note': 'mask_p only; waypoints for scoring; inference_node must be OFF',
    }
    (out_root / 'META.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f'ref={ref_path}')
    print(f'out={out_root}')

    results: list[dict[str, Any]] = []
    for variant in variants:
        for zone in zones:
            length_m = float(zone.get('length_m') or 3.0)
            timeout = max(8.0, args.timeout_scale * length_m / 0.22)
            label = f'{variant}__{zone["id"]}'
            print(f'=== {label} (timeout={timeout:.1f}s len={length_m:.2f})')
            try:
                summary = run_zone(
                    zone=zone,
                    variant=variant,
                    route=args.route,
                    out_dir=out_root,
                    camera_topic=args.camera_topic,
                    settle_sec=args.settle,
                    timeout_sec=timeout,
                    finish_frac=args.finish_frac,
                    viz=args.viz,
                )
            except Exception as exc:  # noqa: BLE001 — bench must continue
                print(f'  ERROR: {exc}')
                summary = {
                    'variant': variant,
                    'zone_id': zone['id'],
                    'maneuver': zone.get('maneuver'),
                    'pass': False,
                    'completed': False,
                    'cte_rms_m': 1.0,
                    'transit_s': 120.0,
                    'error': str(exc),
                }
            results.append(summary)
            print(
                f'  pass={summary.get("pass")} cte_rms={summary.get("cte_rms_m")} '
                f't={summary.get("transit_s")} prog={summary.get("max_progress_m")}'
            )

    ranked = rank_variants(results)
    (out_root / 'summary.json').write_text(
        json.dumps(
            {'ranked': ranked, 'results': results, 'winner': ranked[0] if ranked else None},
            indent=2,
            ensure_ascii=False,
        ),
        encoding='utf-8',
    )
    write_index(out_root, ranked, results)

    if ranked:
        w = ranked[0]
        print(f'\nWinner: {w["variant"]}  pass={w["n_pass"]}/{w["n_zones"]} '
              f'wCTE={w["weighted_cte_rms"]} transit={w["total_transit_s"]}')
    print(f'Wrote {out_root}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
