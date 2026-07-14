#!/usr/bin/env python3
"""Offline mask_p vs stanley A/B on synthetic paths (no Gazebo).

Reports steer_rms / steer_rms_straight so a winner can be chosen without a
live lap. Board SSOT stays ``mask_p`` until a live ``out_lap_bench`` beats it.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_INFER = _ROOT / 'src' / 'inference'
if str(_INFER) not in sys.path:
    sys.path.insert(0, str(_INFER))

from inference.pipeline import MainPlanner, load_planner_config  # noqa: E402


def _synthetic_straight(n: int = 40, y0: float = 0.04) -> np.ndarray:
    xs = np.linspace(0.25, 1.6, n, dtype=np.float32)
    ys = np.full(n, y0, dtype=np.float32)
    return np.column_stack((xs, ys))


def _synthetic_curve(n: int = 50) -> np.ndarray:
    xs = np.linspace(0.2, 1.7, n, dtype=np.float32)
    # Gentle left bend (positive y).
    ys = (0.12 * (xs - 0.2) ** 2).astype(np.float32)
    return np.column_stack((xs, ys))


def _run_tracker(name: str, path: np.ndarray, *, steps: int = 30) -> dict:
    base = load_planner_config(route_mode='out')
    cfg = replace(
        base,
        normal_tracker=name,
        min_points=3,
        steering_rate_limit_per_sec=100.0,
        mask_steer_alpha=1.0,
        stanley_steer_alpha=1.0,
        track_err_alpha=0.5,
        track_enable_path_hold=True,
    )
    planner = MainPlanner(cfg)
    steers: list[float] = []
    ctes: list[float] = []
    for i in range(steps):
        # Inject small noise / occasional jump to stress track_state.
        noisy = path.copy()
        noisy[:, 1] += 0.01 * math.sin(0.4 * i)
        if i == 12:
            noisy[:, 1] += 0.25  # jump (should be rejected)
        if name == 'stanley':
            result = planner._stanley_pursuit(noisy, dt_sec=0.1)
        else:
            # mask_p needs a lane mask; approximate centered blob from path.
            import cv2
            from inference.modules import lane_detection as ld

            h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
            mask = np.zeros((h, w), dtype=np.uint8)
            for x, y in noisy:
                u, v = ld.vehicle_xy_to_bev_uv(float(x), float(y))
                cv2.circle(mask, (int(u), int(v)), 10, 255, -1)
            lane = type(
                'L',
                (),
                {
                    'drivable_area': mask,
                    'meters_per_pixel': float(ld.METERS_PER_PIXEL),
                    'x_forward_max': 1.5,
                },
            )()
            result = planner._mask_com_pursuit(lane, dt_sec=0.1, color_path=noisy)
        if not result.valid:
            continue
        steers.append(float(result.steering))
        ctes.append(float(result.cross_track_error_m))
    arr = np.asarray(steers, dtype=np.float64)
    if arr.size == 0:
        return {'tracker': name, 'ok': False, 'reason': 'no_valid'}
    rms = float(np.sqrt(np.mean(arr**2)))
    # After jump frame, treat low-|cte| samples as straight metric proxy.
    straight = [s for s, c in zip(steers, ctes) if abs(c) < 0.08]
    rms_s = float(np.sqrt(np.mean(np.square(straight)))) if straight else rms
    return {
        'tracker': name,
        'ok': True,
        'n': int(arr.size),
        'steer_rms': round(rms, 4),
        'steer_rms_straight': round(rms_s, 4),
        'cte_abs_mean': round(float(np.mean(np.abs(ctes))), 4) if ctes else None,
    }


def main() -> int:
    cases = {
        'straight': _synthetic_straight(),
        'curve': _synthetic_curve(),
    }
    report: dict = {'cases': {}}
    for case_name, path in cases.items():
        rows = [_run_tracker(t, path) for t in ('mask_p', 'stanley')]
        report['cases'][case_name] = rows
        print(f'=== {case_name} ===')
        for row in rows:
            print(json.dumps(row, ensure_ascii=False))

    # Winner policy: keep mask_p SSOT unless stanley clearly smoother on straight.
    straight_rows = {r['tracker']: r for r in report['cases']['straight']}
    mask_s = straight_rows.get('mask_p', {}).get('steer_rms_straight', 1e9)
    stan_s = straight_rows.get('stanley', {}).get('steer_rms_straight', 1e9)
    if (
        straight_rows.get('stanley', {}).get('ok')
        and straight_rows.get('mask_p', {}).get('ok')
        and stan_s < 0.85 * mask_s
    ):
        winner = 'stanley'
        note = 'offline straight rms favors stanley — confirm with out_lap before YAML flip'
    else:
        winner = 'mask_p'
        note = 'keep board SSOT mask_p; stanley remains A/B'
    report['recommended_normal'] = winner
    report['note'] = note
    print('=== recommendation ===')
    print(json.dumps({'recommended_normal': winner, 'note': note}, ensure_ascii=False))

    out = _ROOT / 'data' / 'captures' / 'tracker_ab_offline'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'latest.json'
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
