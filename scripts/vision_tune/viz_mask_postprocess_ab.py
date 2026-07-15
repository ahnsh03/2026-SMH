#!/usr/bin/env python3
"""A/B mask post-process techniques — raw HSV often beats weak rail fill.

Failure analysis (current pipeline)
-----------------------------------
1. Lane paint is sparse (ln~0–3%). Per-row min/max of paint ≠ lane walls.
2. ``road_in`` then fails valid_ratio and falls back to road-only.
3. Blob row-mid on a wide/bleeding road → zigzag centerline (worst part).
4. Morph open/close barely changes a already-solid road → looks "worse than raw"
   because the *centerline overlay* is garbage, not because morph ruined road.

Techniques compared (drivable / center)
---------------------------------------
A  raw_road          — black|red as-is
B  morph             — current open/close/hole (baseline)
C  near_blob         — morph + CC: largest component touching ego near-band
D  ego_clip          — C clipped to kinematic ego band (track_width ± reach)
E  paint_walls       — dilate course paint as walls, flood from bumper
F  dt_strip          — distance-transform ridge ± half track_width on near_blob
G  row_width_prior   — per-row keep mid±half_w around smoothed mid of C

Centers (overlaid on best-looking driveables):
  mid_raw / mid_smooth / dt_ridge / poly2

Example (in container)::

  PYTHONPATH=src/inference:scripts/vision_tune \\
    python3 scripts/vision_tune/viz_mask_postprocess_ab.py --from-bag out --index 25
  PYTHONPATH=src/inference:scripts/vision_tune \\
    python3 scripts/vision_tune/viz_mask_postprocess_ab.py --from-bag out --all --stride 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parents[1]
_INF = _ROOT / 'src' / 'inference'
for p in (_SCRIPT, _INF):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hsv import default_config_path  # noqa: E402
from out_drivable import prefer_yellow_for_course  # noqa: E402

OUT_DIR = _ROOT / 'data' / 'captures' / 'mask_pp_ab'
FROM_BAG = {
    'in': _ROOT / 'data' / 'captures' / 'from_bag' / 'in',
    'out': _ROOT / 'data' / 'captures' / 'from_bag' / 'out',
}
PANEL = 240
GAP = 3


def _import_blob():
    from inference.modules.perception.blob.masks import extract_bev_masks, get_ipm_params
    from inference.modules.perception.blob.corridor import denoise_road_mask
    from inference.modules.perception.blob.morph_blob import select_best_blob
    from inference.modules.perception.blob.rail_corridor import (
        build_kinematic_ego_band,
        resolve_course_lane_mask,
        vehicle_geom_for_platform,
    )
    from inference.modules.perception.blob.centerline import (
        blob_row_mids,
        mids_to_vehicle_points,
        smooth_mids_1d,
    )

    return (
        extract_bev_masks,
        get_ipm_params,
        denoise_road_mask,
        select_best_blob,
        build_kinematic_ego_band,
        resolve_course_lane_mask,
        vehicle_geom_for_platform,
        blob_row_mids,
        mids_to_vehicle_points,
        smooth_mids_1d,
    )


def _odd(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 else k + 1


def _fit(img: np.ndarray, w: int = PANEL, h: int = PANEL) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None or img.size == 0:
        return out
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = r
    return out


def _title(img: np.ndarray, text: str, color=(0, 255, 180)) -> None:
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _hud(img: np.ndarray, lines: list[str]) -> None:
    for i, t in enumerate(lines):
        cv2.putText(
            img, t, (6, 36 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 255), 1, cv2.LINE_AA
        )


def _cov(m: np.ndarray) -> float:
    return 0.0 if m.size == 0 else 100.0 * float(np.count_nonzero(m)) / float(m.size)


def _overlay(bev: np.ndarray, mask: np.ndarray, color, alpha=0.5) -> np.ndarray:
    base = bev.copy() if bev.ndim == 3 else cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    if mask is None or mask.size == 0:
        return base
    m = mask > 0
    tint = np.zeros_like(base)
    tint[:] = color
    base[m] = (base[m].astype(np.float32) * (1 - alpha) + tint[m].astype(np.float32) * alpha).astype(
        np.uint8
    )
    return base


def _draw_mids(bev: np.ndarray, mids: np.ndarray, color=(0, 255, 255)) -> None:
    h, w = bev.shape[:2]
    pts = []
    for v, u in enumerate(mids):
        if not np.isfinite(u):
            continue
        x = int(np.clip(round(float(u)), 0, w - 1))
        pts.append((x, v))
    for i in range(1, len(pts)):
        cv2.line(bev, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)


def _polyfit_mids(mids: np.ndarray, deg: int = 2) -> np.ndarray:
    out = mids.astype(np.float64).copy()
    rows = np.flatnonzero(np.isfinite(out))
    if rows.size < deg + 2:
        return mids.astype(np.float32)
    coef = np.polyfit(rows.astype(np.float64), out[rows], deg)
    pred = np.polyval(coef, np.arange(out.size, dtype=np.float64))
    pred[~np.isfinite(out)] = np.nan
    # keep only where original had support nearby
    return pred.astype(np.float32)


def _dt_ridge_mids(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    h, w = binary.shape
    mids = np.full(h, np.nan, dtype=np.float32)
    if not np.any(binary):
        return mids
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    for v in range(h):
        row = dist[v]
        if float(row.max()) <= 0:
            continue
        mids[v] = float(np.argmax(row))
    return mids


def tech_near_blob(road: np.ndarray, select_best_blob, track_w: float, mpp: float) -> np.ndarray:
    return select_best_blob(road, track_width_m=track_w, meters_per_pixel=mpp)[0]


def tech_ego_clip(
    road: np.ndarray,
    *,
    select_best_blob,
    build_kinematic_ego_band,
    vehicle_geom_for_platform,
    track_w: float,
    mpp: float,
    x_max: float,
) -> np.ndarray:
    blob = tech_near_blob(road, select_best_blob, track_w, mpp)
    if blob.size == 0:
        return blob
    geom = vehicle_geom_for_platform('sim')
    from dataclasses import replace

    geom = replace(geom, track_width_m=float(track_w))
    band = build_kinematic_ego_band(
        blob.shape[:2], meters_per_pixel=mpp, x_max_m=x_max, geom=geom
    )
    return cv2.bitwise_and(blob, band)


def tech_paint_walls(
    road: np.ndarray,
    lane: np.ndarray,
    *,
    dilate_m: float,
    mpp: float,
) -> np.ndarray:
    """Treat dilated lane paint as walls; flood-fill ego from bumper center."""
    h, w = road.shape[:2]
    if road.size == 0:
        return road.copy()
    k = _odd(max(3, int(round(dilate_m / mpp))))
    walls = np.zeros((h, w), dtype=np.uint8)
    if lane is not None and lane.size and lane.shape[:2] == (h, w):
        walls = cv2.dilate(
            (lane > 0).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
    seedable = ((road > 0) & (walls == 0)).astype(np.uint8) * 255
    seed_y = h - 2
    seed_x = w // 2
    # walk upward for a seed on road
    found = False
    for dy in range(0, min(40, h)):
        y = h - 1 - dy
        if seedable[y, seed_x] > 0:
            seed_y, found = y, True
            break
        for dx in range(1, 25):
            for sx in (seed_x - dx, seed_x + dx):
                if 0 <= sx < w and seedable[y, sx] > 0:
                    seed_y, seed_x, found = y, sx, True
                    break
            if found:
                break
        if found:
            break
    if not found:
        return seedable
    flood = seedable.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (int(seed_x), int(seed_y)), 128)
    return np.where(flood == 128, np.uint8(255), np.uint8(0))


def tech_dt_strip(
    road: np.ndarray,
    *,
    select_best_blob,
    track_w: float,
    mpp: float,
) -> np.ndarray:
    blob = tech_near_blob(road, select_best_blob, track_w, mpp)
    if not np.any(blob):
        return blob
    mids = _dt_ridge_mids(blob)
    half = 0.5 * float(track_w) / float(mpp)
    h, w = blob.shape
    out = np.zeros_like(blob)
    for v in range(h):
        if not np.isfinite(mids[v]):
            continue
        u0 = int(np.clip(np.floor(mids[v] - half), 0, w - 1))
        u1 = int(np.clip(np.ceil(mids[v] + half), 0, w - 1))
        # only keep road pixels inside strip
        row = blob[v, u0 : u1 + 1]
        out[v, u0 : u1 + 1] = row
    return out


def tech_row_width_prior(
    road: np.ndarray,
    *,
    select_best_blob,
    track_w: float,
    mpp: float,
    smooth_mids_1d,
) -> np.ndarray:
    blob = tech_near_blob(road, select_best_blob, track_w, mpp)
    if not np.any(blob):
        return blob
    from inference.modules.perception.blob.centerline import blob_row_mids

    mids = smooth_mids_1d(blob_row_mids(blob), win=21)
    half = 0.5 * float(track_w) / float(mpp)
    h, w = blob.shape
    out = np.zeros_like(blob)
    ego = (w - 1) * 0.5
    for v in range(h):
        mid = float(mids[v]) if np.isfinite(mids[v]) else ego
        u0 = int(np.clip(np.floor(mid - half), 0, w - 1))
        u1 = int(np.clip(np.ceil(mid + half), 0, w - 1))
        out[v, u0 : u1 + 1] = blob[v, u0 : u1 + 1]
    return out


def score_mask(mask: np.ndarray, track_w_px: float) -> dict[str, float]:
    """Heuristic quality: coverage, width consistency, mid smoothness."""
    if mask is None or mask.size == 0 or not np.any(mask):
        return {'cov': 0.0, 'width_std': 99.0, 'mid_jerk': 99.0, 'score': -1e9}
    h, w = mask.shape
    widths = []
    mids = []
    for v in range(h):
        cols = np.flatnonzero(mask[v] > 0)
        if cols.size < 2:
            continue
        widths.append(float(cols[-1] - cols[0] + 1))
        mids.append(0.5 * (float(cols[0]) + float(cols[-1])))
    if len(widths) < 8:
        return {'cov': _cov(mask), 'width_std': 99.0, 'mid_jerk': 99.0, 'score': -1e6}
    widths_a = np.asarray(widths, dtype=np.float64)
    mids_a = np.asarray(mids, dtype=np.float64)
    width_std = float(np.std(widths_a) / max(track_w_px, 1.0))
    mid_jerk = float(np.mean(np.abs(np.diff(mids_a)))) / max(w, 1)
    # prefer width near track prior, low std, low mid jerk, decent coverage
    width_bias = float(np.abs(np.median(widths_a) - track_w_px) / max(track_w_px, 1.0))
    cov = _cov(mask)
    score = cov - 40.0 * width_std - 80.0 * mid_jerk - 25.0 * width_bias
    return {
        'cov': cov,
        'width_std': width_std,
        'mid_jerk': mid_jerk,
        'width_bias': width_bias,
        'score': score,
    }


def build_ab(
    frame: np.ndarray,
    *,
    course: str,
    label: str,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    (
        extract_bev_masks,
        get_ipm_params,
        denoise_road_mask,
        select_best_blob,
        build_kinematic_ego_band,
        resolve_course_lane_mask,
        vehicle_geom_for_platform,
        blob_row_mids,
        mids_to_vehicle_points,
        smooth_mids_1d,
    ) = _import_blob()

    masks = extract_bev_masks(frame)
    ipm = get_ipm_params()
    mpp = float(ipm.meters_per_pixel)
    x_max = float(ipm.x_max_m)
    track_w = float(ipm.track_width_m)
    track_px = track_w / mpp
    prefer = prefer_yellow_for_course(course)
    lane, used_y = resolve_course_lane_mask(
        masks['white'], masks['yellow'], prefer_yellow=prefer
    )
    road_raw = masks['road_raw']
    morph = denoise_road_mask(road_raw)
    bev = masks['bev']

    variants: dict[str, np.ndarray] = {
        'A_raw': (road_raw > 0).astype(np.uint8) * 255,
        'B_morph': morph,
        'C_near': tech_near_blob(morph, select_best_blob, track_w, mpp),
        'D_ego': tech_ego_clip(
            morph,
            select_best_blob=select_best_blob,
            build_kinematic_ego_band=build_kinematic_ego_band,
            vehicle_geom_for_platform=vehicle_geom_for_platform,
            track_w=track_w,
            mpp=mpp,
            x_max=x_max,
        ),
        'E_walls': tech_paint_walls(morph, lane, dilate_m=0.04, mpp=mpp),
        'F_dtstrip': tech_dt_strip(morph, select_best_blob=select_best_blob, track_w=track_w, mpp=mpp),
        'G_wprior': tech_row_width_prior(
            morph,
            select_best_blob=select_best_blob,
            track_w=track_w,
            mpp=mpp,
            smooth_mids_1d=smooth_mids_1d,
        ),
    }

    # Current broken rails → between fill (for reference)
    from inference.modules.perception.blob.corridor import extract_drivable_blob

    cur, between, st, *_ = extract_drivable_blob(
        road_raw,
        masks['white'],
        masks['yellow'],
        prefer_yellow=prefer,
        track_width_m=track_w,
        meters_per_pixel=mpp,
        x_max_m=x_max,
    )
    variants['H_current'] = cur

    scores = {k: score_mask(v, track_px) for k, v in variants.items()}
    best_name = max(scores, key=lambda k: scores[k]['score'])

    # Centerline overlays on best + current
    def mid_panel(mask: np.ndarray, title: str) -> np.ndarray:
        vis = _overlay(bev, mask, (0, 200, 0), 0.45)
        m_raw = blob_row_mids(mask)
        m_sm = smooth_mids_1d(m_raw, win=25)
        m_dt = smooth_mids_1d(_dt_ridge_mids(mask), win=15)
        m_p2 = _polyfit_mids(m_sm, deg=2)
        _draw_mids(vis, m_raw, (80, 80, 255))  # red-ish = raw mid (bad)
        _draw_mids(vis, m_sm, (0, 255, 255))  # yellow = smooth
        _draw_mids(vis, m_dt, (255, 180, 0))  # cyan-ish = DT ridge
        _draw_mids(vis, m_p2, (255, 255, 255))  # white = poly2
        panel = _fit(vis)
        _title(panel, title)
        return panel

    # Row 0: origin + lane paint + raw road
    p0 = _fit(frame)
    _title(p0, '0 origin')
    p1 = _overlay(bev, road_raw, (90, 90, 90), 0.45)
    if lane.size:
        p1 = _overlay(p1, lane, (0, 255, 255) if used_y else (255, 255, 255), 0.55)
    p1 = _fit(p1)
    _title(p1, f'1 HSV road+{"Y" if used_y else "W"}')
    _hud(p1, [f'rd={_cov(road_raw):.0f}% ln={_cov(lane) if lane.size else 0:.0f}%'])

    panels = [p0, p1]
    order = [
        'A_raw',
        'B_morph',
        'C_near',
        'D_ego',
        'E_walls',
        'F_dtstrip',
        'G_wprior',
        'H_current',
    ]
    for name in order:
        mask = variants[name]
        sc = scores[name]
        color = (0, 255, 180) if name != best_name else (0, 255, 0)
        vis = _overlay(bev, mask, (0, 200, 0), 0.5)
        # draw DT ridge mid only (cleaner)
        _draw_mids(vis, smooth_mids_1d(_dt_ridge_mids(mask), win=11), (0, 255, 255))
        panel = _fit(vis)
        mark = '*' if name == best_name else ''
        _title(panel, f'{name}{mark}', color)
        _hud(
            panel,
            [
                f'cov={sc["cov"]:.1f}',
                f'score={sc["score"]:.1f}',
                f'jerk={sc["mid_jerk"]:.3f}',
            ],
        )
        panels.append(panel)

    # Extra row: centerline A/B on best vs current
    p_best = mid_panel(variants[best_name], f'CL on {best_name}')
    p_cur = mid_panel(variants['H_current'], 'CL on H_current')
    p_leg = np.full((PANEL, PANEL, 3), 28, dtype=np.uint8)
    _title(p_leg, 'legend')
    _hud(
        p_leg,
        [
            'red=row_mid raw',
            'yel=smooth mid',
            'blu=DT ridge',
            'wht=poly2',
            f'best={best_name}',
            f'road_in={int(st.road_in_mode)}',
            f'rail_v={st.rail_valid_ratio:.2f}',
        ],
    )
    panels.extend([p_best, p_cur, p_leg])

    # Layout: 2 + 8 + 3 = 13 →  wrap 5 cols
    cols = 5
    rows_needed = (len(panels) + cols - 1) // cols
    while len(panels) < rows_needed * cols:
        panels.append(np.zeros((PANEL, PANEL, 3), dtype=np.uint8))
    gap = np.full((PANEL, GAP, 3), 30, dtype=np.uint8)
    row_imgs = []
    for r in range(rows_needed):
        chunk = panels[r * cols : (r + 1) * cols]
        row = chunk[0]
        for p in chunk[1:]:
            row = np.hstack([row, gap, p])
        row_imgs.append(row)
    hgap = np.full((GAP, row_imgs[0].shape[1], 3), 30, dtype=np.uint8)
    mosaic = row_imgs[0]
    for r in row_imgs[1:]:
        mosaic = np.vstack([mosaic, hgap, r])

    footer = np.full((36, mosaic.shape[1], 3), 20, dtype=np.uint8)
    msg = (
        f'{label} course={course} used_y={int(used_y)}  '
        f'FAIL rail/road_in → zigzag mid  |  BEST={best_name} '
        f'score={scores[best_name]["score"]:.1f}  '
        f'(prefer low mid_jerk + width≈{track_w:.2f}m)'
    )
    cv2.putText(footer, msg[:160], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)
    mosaic = np.vstack([mosaic, footer])
    return mosaic, scores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-bag', choices=('in', 'out'), default='out')
    ap.add_argument('--course', choices=('in', 'out'), default=None)
    ap.add_argument('--index', type=int, default=1)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--stride', type=int, default=5)
    args = ap.parse_args(argv)

    course = args.course or args.from_bag
    folder = FROM_BAG[args.from_bag]
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg'})
    if not paths:
        raise SystemExit(f'No images in {folder}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = (
        list(range(0, len(paths), max(1, args.stride)))
        if args.all
        else [max(0, min(len(paths) - 1, args.index - 1))]
    )

    rank: dict[str, int] = {}
    for i in indices:
        frame = cv2.imread(str(paths[i]))
        if frame is None:
            continue
        mosaic, scores = build_ab(
            frame,
            course=course,
            label=f'[{i + 1}/{len(paths)}] {paths[i].name}',
        )
        best = max(scores, key=lambda k: scores[k]['score'])
        rank[best] = rank.get(best, 0) + 1
        out = OUT_DIR / f'{course}_{i + 1:04d}.png'
        cv2.imwrite(str(out), mosaic)
        sc = scores[best]
        print(
            f'Wrote {out.name} best={best} score={sc["score"]:.1f} '
            f'jerk={sc["mid_jerk"]:.3f} cov={sc["cov"]:.1f}  '
            f'H_cur score={scores["H_current"]["score"]:.1f} jerk={scores["H_current"]["mid_jerk"]:.3f}',
            flush=True,
        )

    print('Best-count:', dict(sorted(rank.items(), key=lambda x: -x[1])), flush=True)
    print(f'Dir: {OUT_DIR}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
