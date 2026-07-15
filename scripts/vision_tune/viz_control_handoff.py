#!/usr/bin/env python3
"""Visualize mask post-process → control handoff (mask_p + PP contract).

Shows how HSV masks are post-processed into what MainPlanner actually uses:

  Perception stages
    1 HSV raw: road(black|red), course lane (white OR yellow — never both)
    2 road denoise (morph open/close + speck drop)
    3 lane corridor fill (rails + ego band)  →  optional road_in clip
    4 final ``drivable_area``  (mask_p input)
    5 centerline (parallel-rail or blob mid)  (PP / hard-corridor anchor)

  Control overlay on panel 4–5
    - largest-blob COM (sim_v2, center_mode=area defaults)
    - near-band ROI used by mask_p
    - hard-corridor band around color_path (what OUT near-fork enables)

Examples (inside 2026-smh-sim)::

  python3 scripts/vision_tune/viz_control_handoff.py --from-bag out --course out
  python3 scripts/vision_tune/viz_control_handoff.py --from-bag in --course in --index 5
  python3 scripts/vision_tune/viz_control_handoff.py --from-bag out --all --stride 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_INFERENCE_SRC = _REPO_ROOT / 'src' / 'inference'
for p in (_SCRIPT_DIR, _INFERENCE_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hsv import default_config_path, load_hsv_ranges, make_mask, overlay_mask  # noqa: E402
from metric_ipm import load_metric_ipm, warp_metric_ipm  # noqa: E402
from out_drivable import prefer_yellow_for_course, road_mask_from_hsv  # noqa: E402

PANEL = 280
GAP = 4
OUT_DIR = _REPO_ROOT / 'data' / 'captures' / 'control_handoff'
FROM_BAG = {
    'in': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'in',
    'out': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'out',
}


def _import_ld():
    try:
        from inference.modules import lane_detection as ld
    except ModuleNotFoundError:
        from inference.inference.modules import lane_detection as ld  # type: ignore
    if hasattr(ld, 'set_perception_backend'):
        ld.set_perception_backend('blob')
    return ld


def _import_corridor():
    try:
        from inference.modules.perception.blob.corridor import (
            denoise_road_mask,
            extract_drivable_blob,
        )
        from inference.modules.perception.blob.masks import extract_bev_masks, get_ipm_params
        from inference.modules.perception.blob.rail_corridor import (
            resolve_course_lane_mask,
        )
    except ModuleNotFoundError:
        from inference.inference.modules.perception.blob.corridor import (  # type: ignore
            denoise_road_mask,
            extract_drivable_blob,
        )
        from inference.inference.modules.perception.blob.masks import (  # type: ignore
            extract_bev_masks,
            get_ipm_params,
        )
        from inference.inference.modules.perception.blob.rail_corridor import (  # type: ignore
            resolve_course_lane_mask,
        )
    return (
        denoise_road_mask,
        extract_drivable_blob,
        extract_bev_masks,
        get_ipm_params,
        resolve_course_lane_mask,
    )


def _fit(img: np.ndarray, w: int = PANEL, h: int = PANEL) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None or img.size == 0:
        return out
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _title(img: np.ndarray, text: str, color=(40, 220, 40)) -> None:
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _hud(img: np.ndarray, lines: list[str], y0: int = 42) -> None:
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (8, y0 + i * 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _cov(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(mask)) / float(mask.size)


def _draw_xy(
    bev_bgr: np.ndarray,
    pts: np.ndarray,
    *,
    x_max_m: float,
    mpp: float,
    color: tuple[int, int, int],
    radius: int = 2,
) -> None:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] < 2:
        return
    h, w = bev_bgr.shape[:2]
    for x_m, y_m in pts:
        if not (np.isfinite(x_m) and np.isfinite(y_m)):
            continue
        v = int(round((x_max_m - float(x_m)) / mpp))
        u = int(round((w - 1) * 0.5 - float(y_m) / mpp))
        if 0 <= v < h and 0 <= u < w:
            cv2.circle(bev_bgr, (u, v), radius, color, -1, lineType=cv2.LINE_AA)


def _largest_blob(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary * 255
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == best, np.uint8(255), np.uint8(0))


def _area_com(mask: np.ndarray, y_lo: int, y_hi: int) -> tuple[float, float] | None:
    roi = mask[y_lo:y_hi, :]
    moments = cv2.moments(roi, binaryImage=True)
    area = float(moments.get('m00', 0.0))
    if area < 100.0:
        return None
    cx = float(moments['m10'] / area)
    cy = float(moments['m01'] / area) + float(y_lo)
    return cx, cy


def _hard_corridor(
    path_xy: np.ndarray,
    h: int,
    w: int,
    *,
    x_max_m: float,
    mpp: float,
    half_width_m: float,
) -> np.ndarray:
    """Approximate planner hard corridor (dilated path skeleton)."""
    band = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(path_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return band
    half_px = max(1, int(round(half_width_m / mpp)))
    for x_m, y_m in pts:
        if not (np.isfinite(x_m) and np.isfinite(y_m)):
            continue
        v = int(round((x_max_m - float(x_m)) / mpp))
        u = int(round((w - 1) * 0.5 - float(y_m) / mpp))
        if 0 <= v < h and 0 <= u < w:
            cv2.circle(band, (u, v), half_px, 255, -1)
    return band


def build_handoff_mosaic(
    frame: np.ndarray,
    *,
    course: str,
    config_path: Path,
    label: str,
    corridor_mode: str = 'off',
) -> tuple[np.ndarray, dict[str, float | str | bool]]:
    (
        denoise_road_mask,
        extract_drivable_blob,
        extract_bev_masks,
        get_ipm_params,
        resolve_course_lane_mask,
    ) = _import_corridor()
    ld = _import_ld()

    prefer = prefer_yellow_for_course(course)
    masks = extract_bev_masks(frame)
    ipm = get_ipm_params()
    mpp = float(ipm.meters_per_pixel)
    x_max = float(ipm.x_max_m)
    track_w = float(ipm.track_width_m)

    road_raw = masks['road_raw']
    road_clean = denoise_road_mask(road_raw)
    lane_mask, used_yellow = resolve_course_lane_mask(
        masks['white'], masks['yellow'], prefer_yellow=prefer
    )
    blob, between, corr, left, right, used_y = extract_drivable_blob(
        road_raw,
        masks['white'],
        masks['yellow'],
        prefer_yellow=prefer,
        track_width_m=track_w,
        meters_per_pixel=mpp,
        x_max_m=x_max,
    )

    ld.reset_tracking_state()
    dets, _dbg = ld.detect_with_debug(frame, prefer_yellow=prefer, enable_fork=False)
    cl = np.asarray(
        dets.yellow_centerline if (prefer and used_y) else dets.white_centerline,
        dtype=np.float32,
    )
    if cl.size == 0:
        cl = np.asarray(dets.white_centerline, dtype=np.float32)
        if cl.size == 0:
            cl = np.asarray(dets.yellow_centerline, dtype=np.float32)

    work = _largest_blob(blob)
    h, w = work.shape[:2]
    near_ratio = 0.85
    y_lo = int(round(h * (1.0 - near_ratio)))
    com = _area_com(work, y_lo, h)

    # Control consumer view
    ctrl = (
        cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        if work.ndim == 2
        else work.copy()
    )
    # Near-band region (mask_p ROI)
    overlay_band = ctrl.copy()
    cv2.rectangle(overlay_band, (0, y_lo), (w - 1, h - 1), (0, 90, 0), -1)
    ctrl = cv2.addWeighted(overlay_band, 0.25, ctrl, 0.75, 0)
    cv2.line(ctrl, (0, y_lo), (w - 1, y_lo), (0, 200, 0), 1)

    corridor = np.zeros((h, w), dtype=np.uint8)
    if corridor_mode == 'hard' and cl.ndim == 2 and cl.shape[0] >= 2:
        corridor = _hard_corridor(
            cl, h, w, x_max_m=x_max, mpp=mpp, half_width_m=0.42
        )
        work_hard = cv2.bitwise_and(work, corridor)
        ctrl = overlay_mask(ctrl, corridor, color=(0, 140, 255), alpha=0.28)
        work_for_com = work_hard
        com = _area_com(work_for_com, y_lo, h) or com
    else:
        work_for_com = work

    _draw_xy(ctrl, cl, x_max_m=x_max, mpp=mpp, color=(0, 255, 255), radius=2)
    if com is not None:
        cx, cy = com
        cv2.drawMarker(
            ctrl,
            (int(round(cx)), int(round(cy))),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        # Image center (zero lateral error)
        img_c = (w - 1) * 0.5
        err_norm = (cx - img_c) / max(w, 1)
        cv2.arrowedLine(
            ctrl,
            (int(img_c), int(cy)),
            (int(cx), int(cy)),
            (0, 0, 255),
            2,
            tipLength=0.3,
        )
    else:
        err_norm = float('nan')

    # Panel 1: origin crop
    ranges = load_hsv_ranges(config_path)
    bev = masks['bev'] if masks['bev'].size else warp_metric_ipm(frame, load_metric_ipm(config_path))
    p_origin = _fit(frame)
    _title(p_origin, '0 ORIGIN')

    # Panel 2: HSV layers
    p_hsv = bev.copy() if bev.ndim == 3 else cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    p_hsv = overlay_mask(p_hsv, road_raw, color=(80, 80, 80), alpha=0.45)
    p_hsv = overlay_mask(p_hsv, lane_mask if lane_mask.size else np.zeros_like(road_raw), color=((0, 255, 255) if used_yellow else (255, 255, 255)), alpha=0.55)
    p_hsv = _fit(p_hsv)
    _title(p_hsv, f'1 HSV road+{"Y" if used_yellow else "W"}')
    _hud(p_hsv, [f'rd={_cov(road_raw):.0f}% ln={_cov(lane_mask) if lane_mask.size else 0:.0f}%'])

    # Panel 3: road denoise
    p_road = overlay_mask(
        bev.copy() if bev.ndim == 3 else cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR),
        road_clean,
        color=(90, 90, 90),
        alpha=0.55,
    )
    p_road = _fit(p_road)
    _title(p_road, '2 road denoise')
    _hud(p_road, [f'cov={_cov(road_clean):.1f}%'])

    # Panel 4: lane corridor / between
    p_corr = bev.copy() if bev.ndim == 3 else cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    p_corr = overlay_mask(p_corr, between, color=(0, 180, 0), alpha=0.45)
    if left.size and right.size:
        for v in range(min(left.size, between.shape[0])):
            if np.isfinite(left[v]):
                cv2.circle(p_corr, (int(left[v]), v), 1, (255, 80, 80), -1)
            if np.isfinite(right[v]):
                cv2.circle(p_corr, (int(right[v]), v), 1, (80, 80, 255), -1)
    p_corr = _fit(p_corr)
    _title(p_corr, '3 lane corridor')
    _hud(
        p_corr,
        [
            f'valid={corr.rail_valid_ratio:.2f}',
            f'road_in={int(corr.road_in_mode)}',
        ],
    )

    # Panel 5: final drivable (control input #1)
    p_drv = bev.copy() if bev.ndim == 3 else cv2.cvtColor(bev, cv2.COLOR_GRAY2BGR)
    p_drv = overlay_mask(p_drv, blob, color=(0, 255, 180), alpha=0.50)
    _draw_xy(p_drv, cl, x_max_m=x_max, mpp=mpp, color=(0, 255, 255), radius=2)
    p_drv = _fit(p_drv)
    _title(p_drv, '4 drivable_area → mask_p', (0, 255, 180))
    _hud(
        p_drv,
        [
            f'cov={_cov(blob):.1f}%  cl={cl.shape[0] if cl.ndim == 2 else 0}',
            f'mode={"road_in" if corr.road_in_mode else "road"}',
        ],
    )

    # Panel 6: control consumer (COM / corridor)
    p_ctrl = _fit(ctrl)
    _title(p_ctrl, f'5 control view [{corridor_mode}]', (0, 0, 255))
    steer_approx = float(np.clip(-err_norm * np.pi * 2.0, -1.0, 1.0)) if np.isfinite(err_norm) else float('nan')
    _hud(
        p_ctrl,
        [
            f'COM err={err_norm:+.3f}' if np.isfinite(err_norm) else 'COM=lost',
            f'steer~={steer_approx:+.2f}' if np.isfinite(steer_approx) else 'steer=—',
            '→ /perception/lane',
        ],
    )

    gap = np.full((PANEL, GAP, 3), 36, dtype=np.uint8)
    row1 = np.hstack([p_origin, gap, p_hsv, gap, p_road])
    row2 = np.hstack([p_corr, gap, p_drv, gap, p_ctrl])
    bridge = np.full((GAP, row1.shape[1], 3), 36, dtype=np.uint8)
    mosaic = np.vstack([row1, bridge, row2])

    # Footer contract strip
    footer = np.full((48, mosaic.shape[1], 3), 24, dtype=np.uint8)
    contract = (
        f'{label}  course={course} prefer_y={int(prefer)} used_y={int(used_y)}  '
        f'PASS: drivable_area(BEV mono8) + {"yellow" if used_y else "white"}_centerline(base_link m)  '
        f'+ mpp={mpp:.4f} x_max={x_max:.2f}  |  NORMAL→mask_p(COM)  FORK/CIRCLE→PP(path)'
    )
    cv2.putText(
        footer,
        contract[:140],
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        contract[140:] if len(contract) > 140 else '',
        (8, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    mosaic = np.vstack([mosaic, footer])

    stats: dict[str, float | str | bool] = {
        'road_pct': _cov(road_clean),
        'lane_pct': _cov(lane_mask) if lane_mask.size else 0.0,
        'corridor_pct': _cov(between),
        'drivable_pct': _cov(blob),
        'road_in': bool(corr.road_in_mode),
        'used_yellow': bool(used_y),
        'rail_valid': float(corr.rail_valid_ratio),
        'centerline_pts': float(cl.shape[0]) if cl.ndim == 2 else 0.0,
        'com_err': float(err_norm) if np.isfinite(err_norm) else -999.0,
        'steer_approx': float(steer_approx) if np.isfinite(steer_approx) else -999.0,
        'corridor_mode': corridor_mode,
    }
    return mosaic, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-bag', choices=('in', 'out'), default='out')
    parser.add_argument('--course', choices=('in', 'out'), default=None)
    parser.add_argument('--config', type=Path, default=default_config_path())
    parser.add_argument('--index', type=int, default=1, help='1-based frame index')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument(
        '--corridor',
        choices=('off', 'hard'),
        default='off',
        help='Simulate mask_p corridor_mode overlay (hard = OUT near-fork)',
    )
    args = parser.parse_args(argv)

    course = args.course or args.from_bag
    folder = FROM_BAG[args.from_bag]
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    if not paths:
        raise SystemExit(f'No images in {folder}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.all:
        indices = list(range(0, len(paths), max(1, args.stride)))
    else:
        indices = [max(0, min(len(paths) - 1, args.index - 1))]

    for i in indices:
        frame = cv2.imread(str(paths[i]))
        if frame is None:
            continue
        mosaic, stats = build_handoff_mosaic(
            frame,
            course=course,
            config_path=args.config,
            label=f'[{i + 1}/{len(paths)}] {paths[i].name}',
            corridor_mode=args.corridor,
        )
        out = OUT_DIR / f'{course}_{i + 1:04d}_{args.corridor}.png'
        cv2.imwrite(str(out), mosaic)
        print(
            f'Wrote {out}  road_in={stats["road_in"]} cl={int(stats["centerline_pts"])} '
            f'err={stats["com_err"]:+.3f} steer~={stats["steer_approx"]:+.2f}',
            flush=True,
        )
    print(f'Dir: {OUT_DIR}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
