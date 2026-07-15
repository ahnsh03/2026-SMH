#!/usr/bin/env python3
"""Compare control families on OUT ``eval_s_curve`` (reference CTE grading).

Runs mask_p / Pure Pursuit / Stanley / hybrid candidates from the zone
entry teleport, scores against ``reference_segments.json`` pass_criteria,
writes ranked ANALYSIS + applies nothing by itself (see ``--apply``).

Owns ``/control`` — keep sim-auto / inference_node OFF.

Example (inside 2026-smh-sim, bringup only)::

  PYTHONUNBUFFERED=1 python3 scripts/drive_test/scurve_control_compare.py \\
      --viz off --apply
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import mask_zone_ab_bench as zb  # noqa: E402

OUT_ROOT = _ROOT / 'data' / 'captures' / 'scurve_control_logs'
YAML_PATH = _ROOT / 'config' / 'main_planner.yaml'

# Soft corridor shared by lane-following trackers (pp/stanley/hybrid).
_PATH_COMMON: dict[str, Any] = {
    'cruise_throttle': 0.26,
    'curve_throttle': 0.16,
    'steering_rate_limit_per_sec': 12.0,
    'error_speed_cte_full_m': 0.18,
    'error_speed_steer_full': 0.50,
    'error_speed_min_scale': 0.55,
    'mask_corridor_mode': 'soft',
    'mask_corridor_half_width_m': 0.40,
    'mask_require_color_path': False,
}

VARIANTS: dict[str, dict[str, Any]] = {
    # --- mask_p (current SSOT family) ---
    'mask_sim_v2': {
        **zb._MASK_COMMON,
        'cruise_throttle': 0.26,
        'curve_throttle': 0.16,
        'mask_steer_law': 'sim_v2',
        'mask_steer_k': 2.0,
        'mask_steer_alpha': 0.40,
        'mask_center_mode': 'area',
        'mask_erode_px': 0,
        'mask_corridor_mode': 'off',
        'mask_far_blend': 0.0,
        'steering_rate_limit_per_sec': 12.0,
    },
    'mask_atan_row': zb.VARIANTS['atan_row'],
    'mask_atan_ridge': zb.VARIANTS['atan_ridge'],
    # --- Pure Pursuit ---
    'pp_base': {
        **_PATH_COMMON,
        'normal_tracker': 'pp',
        'lookahead_m': 0.70,
        'curve_lookahead_m': 0.45,
        'cte_gain': 0.10,
        'heading_gain': 0.30,
        'heading_preview_m': 0.45,
    },
    'pp_tight': {
        **_PATH_COMMON,
        'normal_tracker': 'pp',
        'lookahead_m': 0.55,
        'curve_lookahead_m': 0.35,
        'cte_gain': 0.14,
        'heading_gain': 0.40,
        'heading_preview_m': 0.35,
        'cruise_throttle': 0.24,
        'curve_throttle': 0.14,
    },
    'pp_soft': {
        **_PATH_COMMON,
        'normal_tracker': 'pp',
        'lookahead_m': 0.85,
        'curve_lookahead_m': 0.55,
        'cte_gain': 0.08,
        'heading_gain': 0.22,
        'heading_preview_m': 0.55,
    },
    # --- Stanley-lite ---
    'stanley_base': {
        **_PATH_COMMON,
        'normal_tracker': 'stanley',
        'stanley_k_cte': 1.20,
        'stanley_k_yaw': 1.0,
        'stanley_v_soft': 0.18,
        'stanley_curvature_ff_gain': 0.35,
        'stanley_steer_alpha': 0.35,
    },
    'stanley_agg': {
        **_PATH_COMMON,
        'normal_tracker': 'stanley',
        'stanley_k_cte': 1.60,
        'stanley_k_yaw': 1.15,
        'stanley_v_soft': 0.14,
        'stanley_curvature_ff_gain': 0.50,
        'stanley_steer_alpha': 0.40,
        'cruise_throttle': 0.24,
        'curve_throttle': 0.14,
    },
    'stanley_soft': {
        **_PATH_COMMON,
        'normal_tracker': 'stanley',
        'stanley_k_cte': 0.90,
        'stanley_k_yaw': 0.85,
        'stanley_v_soft': 0.22,
        'stanley_curvature_ff_gain': 0.25,
        'stanley_steer_alpha': 0.28,
    },
    # --- hybrid (PP straight / mask curves) ---
    'hybrid_base': {
        **_PATH_COMMON,
        'normal_tracker': 'hybrid',
        'lookahead_m': 0.65,
        'curve_lookahead_m': 0.40,
        'cte_gain': 0.12,
        'heading_gain': 0.32,
        'mask_steer_law': 'lateral_atan',
        'mask_steer_k': 1.80,
        'mask_steer_alpha': 0.35,
        'mask_center_mode': 'row_mid',
        'mask_erode_px': 2,
    },
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def family_of(name: str) -> str:
    if name.startswith('mask_'):
        return 'mask_p'
    if name.startswith('pp_'):
        return 'pp'
    if name.startswith('stanley_'):
        return 'stanley'
    if name.startswith('hybrid_'):
        return 'hybrid'
    return 'other'


def write_analysis(
    out_dir: Path,
    results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
) -> Path:
    crit = (results[0].get('criteria') if results else None) or {
        'cte_rms_m_max': 0.06,
        'cte_abs_max_m': 0.12,
        'heading_err_rms_rad_max': 0.25,
    }
    lines = [
        '# OUT S-curve control compare (`eval_s_curve`)',
        '',
        f'- criteria: CTE RMS≤{crit.get("cte_rms_m_max")} m, '
        f'|CTE|max≤{crit.get("cte_abs_max_m")} m, '
        f'heading RMS≤{crit.get("heading_err_rms_rad_max")} rad',
        f'- n_variants: {len(results)}',
        '',
        '## Ranking',
        '',
        '| rank | variant | family | pass | cte_rms | |cte|_max | h_rms | t_s | prog |',
        '|---:|---|---|:---:|---:|---:|---:|---:|---:|',
    ]
    # Per-run ranking (single zone → sort results directly)
    ordered = sorted(
        results,
        key=lambda r: (
            0 if r.get('pass') else 1,
            float(r.get('cte_rms_m') or 99.0),
            float(r.get('cte_abs_max_m') or 99.0),
            float(r.get('heading_err_rms_rad') or 99.0),
            float(r.get('transit_s') or 99.0),
        ),
    )
    for i, r in enumerate(ordered, 1):
        flag = 'PASS' if r.get('pass') else 'FAIL'
        lines.append(
            f'| {i} | `{r["variant"]}` | {family_of(str(r["variant"]))} | {flag} | '
            f'{r.get("cte_rms_m")} | {r.get("cte_abs_max_m")} | '
            f'{r.get("heading_err_rms_rad")} | {r.get("transit_s")} | '
            f'{r.get("max_progress_m")}/{r.get("length_m")} |'
        )

    winner = ordered[0] if ordered else None
    lines.extend(['', '## Verdict', ''])
    if winner:
        lines.append(
            f'- **best run:** `{winner["variant"]}` ({family_of(winner["variant"])}) '
            f'— pass={winner.get("pass")} cte_rms={winner.get("cte_rms_m")} '
            f't={winner.get("transit_s")}s'
        )
        # Best per family
        by_f: dict[str, dict[str, Any]] = {}
        for r in ordered:
            f = family_of(str(r['variant']))
            if f not in by_f:
                by_f[f] = r
        lines.append('- **best per family:**')
        for f, r in by_f.items():
            lines.append(
                f'  - `{f}`: `{r["variant"]}` cte_rms={r.get("cte_rms_m")} '
                f'pass={r.get("pass")}'
            )
        lines.extend(
            [
                '',
                '### Why (S-curve)',
                '- S-curve needs **fast CTE response + curvature foresight** without '
                'lagging COM (mask soft-corridor) or cutting apex (over-long PP lookahead).',
                '- Stanley: explicit CTE/yaw + curvature FF — usually strongest on '
                'sign-changing curvature.',
                '- PP: good continuity; short lookahead (`pp_tight`) for S waves.',
                '- mask_p: robust when paint is clear; can lag or bias on S inner walls.',
                '',
            ]
        )
    path = out_dir / 'ANALYSIS.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def apply_winner_to_yaml(winner: dict[str, Any]) -> None:
    """Write recommended OUT S-curve tracker block into main_planner.yaml comments+keys."""
    text = YAML_PATH.read_text(encoding='utf-8')
    fam = family_of(str(winner['variant']))
    cfg = winner.get('config') or VARIANTS[str(winner['variant'])]
    stamp = _stamp()

    # Update tracker.normal
    import re

    text2, n = re.subn(
        r'(tracker:\n  normal:\s*)\S+',
        rf'\1{cfg.get("normal_tracker", fam)}',
        text,
        count=1,
    )
    if n == 0:
        # still try simple form
        text2, n = re.subn(
            r'(normal:\s*)(mask_p|pp|stanley|hybrid)',
            rf'\1{cfg.get("normal_tracker", fam)}',
            text,
            count=1,
        )
    text = text2

    # Update stanley / pp gains if applicable
    if fam == 'stanley':
        for key, yaml_key in [
            ('stanley_k_cte', 'k_cte'),
            ('stanley_k_yaw', 'k_yaw'),
            ('stanley_v_soft', 'v_soft'),
            ('stanley_curvature_ff_gain', 'curvature_ff_gain'),
            ('stanley_steer_alpha', 'steer_alpha'),
        ]:
            if key in cfg:
                text, _ = re.subn(
                    rf'(stanley:\n(?:.*\n)*?  {yaml_key}:\s*)[^\n]+',
                    rf'\g<1>{cfg[key]}',
                    text,
                    count=1,
                )
    if fam == 'pp':
        for key, yaml_key in [
            ('lookahead_m', 'lookahead_m'),
            ('curve_lookahead_m', 'curve_lookahead_m'),
            ('cte_gain', 'cte_gain'),
            ('heading_gain', 'heading_gain'),
        ]:
            if key in cfg:
                text, _ = re.subn(
                    rf'(pure_pursuit:\n(?:.*\n)*?  {yaml_key}:\s*)[^\n]+',
                    rf'\g<1>{cfg[key]}',
                    text,
                    count=1,
                )

    banner = (
        f'# OUT S-curve compare winner ({stamp}): {winner["variant"]} '
        f'cte_rms={winner.get("cte_rms_m")} pass={winner.get("pass")}\n'
    )
    if 'OUT S-curve compare winner' not in text:
        text = banner + text
    else:
        text = re.sub(
            r'# OUT S-curve compare winner[^\n]*\n',
            banner,
            text,
            count=1,
        )

    YAML_PATH.write_text(text, encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--variants',
        default=','.join(VARIANTS.keys()),
        help='comma variant ids',
    )
    ap.add_argument('--route', default='out', choices=('out', 'in'))
    ap.add_argument('--settle', type=float, default=1.2)
    ap.add_argument('--timeout-scale', type=float, default=3.2)
    ap.add_argument('--finish-frac', type=float, default=0.92)
    ap.add_argument(
        '--camera-topic',
        default='/camera/image/compressed',
    )
    ap.add_argument('--viz', default='off', choices=('off', 'control', 'debug'))
    ap.add_argument(
        '--apply',
        action='store_true',
        help='write winning tracker into config/main_planner.yaml',
    )
    ap.add_argument(
        '--retest-winner',
        action='store_true',
        help='after ranking, re-run the winner once more for confirmation',
    )
    args = ap.parse_args()

    names = [v.strip() for v in args.variants.split(',') if v.strip()]
    for n in names:
        if n not in VARIANTS:
            print(f'unknown variant: {n}', file=sys.stderr)
            print('known:', ', '.join(VARIANTS), file=sys.stderr)
            return 2

    # Inject into mask_zone_ab_bench so build_planner/run_zone see them.
    zb.VARIANTS.update(VARIANTS)

    ref = zb._find_reference()
    zones = [z for z in zb.load_zones(ref) if z.get('id') == 'eval_s_curve']
    if not zones:
        print('eval_s_curve not found in', ref, file=sys.stderr)
        return 2
    zone = zones[0]
    length_m = float(zone.get('length_m') or 9.51)
    timeout = max(18.0, args.timeout_scale * length_m / 0.22)

    out_dir = zb._ensure(OUT_ROOT / _stamp())
    print(f'ref={ref}')
    print(f'out={out_dir}')
    print(f'zone={zone["id"]} length={length_m:.2f}m timeout={timeout:.1f}s')
    print(f'variants={names}')

    results: list[dict[str, Any]] = []
    for name in names:
        print(f'\n=== {name} ({family_of(name)}) ===', flush=True)
        try:
            summary = zb.run_zone(
                zone=zone,
                variant=name,
                route=args.route,
                out_dir=out_dir,
                camera_topic=args.camera_topic,
                settle_sec=args.settle,
                timeout_sec=timeout,
                finish_frac=args.finish_frac,
                viz=args.viz,
            )
        except Exception as exc:  # noqa: BLE001
            summary = {
                'variant': name,
                'zone_id': 'eval_s_curve',
                'pass': False,
                'error': str(exc),
                'cte_rms_m': 99.0,
                'cte_abs_max_m': 99.0,
                'heading_err_rms_rad': 99.0,
                'transit_s': 99.0,
                'max_progress_m': 0.0,
                'length_m': length_m,
                'config': VARIANTS[name],
            }
            print(f'ERROR: {exc}', flush=True)
        results.append(summary)
        flag = 'PASS' if summary.get('pass') else 'FAIL'
        print(
            f'  {flag} cte_rms={summary.get("cte_rms_m")} '
            f'max={summary.get("cte_abs_max_m")} '
            f'h_rms={summary.get("heading_err_rms_rad")} '
            f't={summary.get("transit_s")} '
            f'prog={summary.get("max_progress_m")}/{summary.get("length_m")}',
            flush=True,
        )
        time.sleep(0.4)

    ranked = zb.rank_variants(results)
    analysis = write_analysis(out_dir, results, ranked)
    (out_dir / 'results.json').write_text(
        json.dumps({'results': results, 'ranked': ranked}, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    zb.write_index(out_dir, ranked, results)

    ordered = sorted(
        results,
        key=lambda r: (
            0 if r.get('pass') else 1,
            float(r.get('cte_rms_m') or 99.0),
            float(r.get('cte_abs_max_m') or 99.0),
        ),
    )
    winner = ordered[0]
    print(f'\nWinner: {winner["variant"]} pass={winner.get("pass")} '
          f'cte_rms={winner.get("cte_rms_m")}')
    print(f'ANALYSIS: {analysis}')

    if args.retest_winner:
        print('\n=== retest winner ===', flush=True)
        confirm = zb.run_zone(
            zone=zone,
            variant=str(winner['variant']),
            route=args.route,
            out_dir=out_dir / 'retest',
            camera_topic=args.camera_topic,
            settle_sec=args.settle,
            timeout_sec=timeout,
            finish_frac=args.finish_frac,
            viz=args.viz,
        )
        (out_dir / 'retest' / 'confirm.json').write_text(
            json.dumps(confirm, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        print(
            f'  retest pass={confirm.get("pass")} cte_rms={confirm.get("cte_rms_m")}',
            flush=True,
        )
        winner = confirm

    if args.apply:
        apply_winner_to_yaml(winner)
        print(f'Applied to {YAML_PATH}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
