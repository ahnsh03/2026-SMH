#!/usr/bin/env python3
"""OUT/IN drivable preview — blob corridor (default perception backend).

Pipeline: HSV → Metric IPM → morph → best road blob → row-mid centerline.
No polyfit rails (avoids leftover "11" when fit fails).

  OUT → prefer_yellow=False → white lane cue
  IN  → prefer_yellow=True  → yellow (+ soft white) lane cue
  Road = black_road | red_road

Compose modes:

  road     — black|red raw
  between  — selected drivable blob (road_clean)
  road_in  — road ∩ blob
  union    — blob ∪ road

Examples (inside 2026-smh-sim):

  python3 scripts/vision_tune/preview_out_drivable.py
  python3 scripts/vision_tune/preview_out_drivable.py --from-bag out --course out
  python3 scripts/vision_tune/preview_out_drivable.py --from-bag out --start 11 --fork
  python3 scripts/vision_tune/preview_out_drivable.py --from-bag in --course in
  python3 scripts/vision_tune/preview_out_drivable.py --compare --from-bag out --course out

Keys:
  1–4   compose mode
  f     toggle enable_fork (default OFF; OUT captures 11–12)
  n/p   next / prev frame
  SPACE auto-advance
  s     snapshot under data/captures/out_drivable_preview/
  q/ESC quit
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_INFERENCE_SRC = _REPO_ROOT / 'src' / 'inference'
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_SRC))

from hsv import (  # noqa: E402
    default_config_path,
    load_hsv_ranges,
    make_mask,
    overlay_mask,
)
from metric_ipm import (  # noqa: E402
    draw_crop_overlay,
    draw_metric_guides,
    load_metric_ipm,
    warp_metric_ipm,
)
from out_drivable import (  # noqa: E402
    COMPOSE_HELP,
    COMPOSE_MODES,
    compose_drivable,
    prefer_yellow_for_course,
    road_mask_from_hsv,
)

try:
    from inference.modules.perception.blob.corridor import (  # noqa: E402
        denoise_road_mask,
    )
except ImportError:
    from inference.inference.modules.perception.blob.corridor import (  # noqa: E402
        denoise_road_mask,
    )
from window_layout import (  # noqa: E402
    diagnose_window_placement,
    map_and_place_window,
    place_window,
)

# Keep title free of regex metachars — xdotool --name treats it as a regex.
WIN_MAIN = 'out_drivable'
PANEL_W = 420
PANEL_H = 360
GAP = 4

FROM_BAG_DIRS = {
    'in': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'in',
    'out': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'out',
    'in_cam': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'in_cam',
    'out_cam': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'out_cam',
}


def _import_lane_detection():
    try:
        from inference.modules import lane_detection as ld  # type: ignore
    except ModuleNotFoundError:
        try:
            from inference.inference.modules import lane_detection as ld  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                'Cannot import lane_detection. Inside 2026-smh-sim:\n'
                '  source /opt/ros/humble/setup.bash && source install/setup.bash\n'
                f'Original: {exc}'
            ) from exc
    # Preview always exercises blob backend (yaml default; force for A/B safety).
    if hasattr(ld, 'set_perception_backend'):
        ld.set_perception_backend('blob')
    return ld


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f'No such folder: {folder}')
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()
    )


def _fit_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is None or frame.size == 0:
        return out
    h, w = frame.shape[:2]
    if h < 1 or w < 1:
        return out
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _label_panel(img: np.ndarray, text: str) -> None:
    cv2.putText(
        img,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 220, 40),
        1,
        cv2.LINE_AA,
    )


def _coverage(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(mask)) / float(mask.size)


def _draw_fork_pairs(
    bev_bgr: np.ndarray,
    fork_lane_pairs: tuple | list,
    *,
    rank_filter: int | None = None,
) -> None:
    for pair in fork_lane_pairs or ():
        rank = int(getattr(pair, 'lateral_rank', -1))
        if rank_filter is not None and rank != int(rank_filter):
            continue
        if rank == 0:
            co, ci, cc = (0, 220, 0), (0, 160, 80), (0, 255, 128)
        else:
            co, ci, cc = (220, 0, 220), (160, 0, 180), (255, 0, 255)
        for u_attr, color in (
            ('outer_u', co),
            ('inner_u', ci),
            ('center_u', cc),
        ):
            col = np.asarray(getattr(pair, u_attr), dtype=np.float32)
            h = bev_bgr.shape[0]
            for y in range(min(h, col.size)):
                if not np.isfinite(col[y]):
                    continue
                x = int(np.clip(round(float(col[y])), 0, bev_bgr.shape[1] - 1))
                cv2.circle(bev_bgr, (x, y), 1, color, -1, lineType=cv2.LINE_AA)


def _draw_centerline_xy(
    bev_bgr: np.ndarray,
    points_xy: np.ndarray,
    *,
    ipm,
    color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    """base_link Nx2 [x forward, y left] → BEV dots."""

    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] < 2:
        return
    mpp = float(ipm.meters_per_pixel)
    x_max = float(ipm.x_max_m)
    w = bev_bgr.shape[1]
    h = bev_bgr.shape[0]
    for x_m, y_m in pts:
        if not (np.isfinite(x_m) and np.isfinite(y_m)):
            continue
        v = int(round((x_max - float(x_m)) / mpp))
        u = int(round((w - 1) * 0.5 - float(y_m) / mpp))
        if 0 <= v < h and 0 <= u < w:
            cv2.circle(bev_bgr, (u, v), 2, color, -1, lineType=cv2.LINE_AA)


def compare_road_modes(
    frames: list[np.ndarray],
    labels: list[str],
    *,
    ld: object,
    config_path: Path,
    course: str,
) -> None:
    """Headless A/B: denoised road vs runtime road_in (blob detect drivable)."""

    rows: list[dict[str, float | str]] = []
    for idx, (frame, label) in enumerate(zip(frames, labels)):
        ld.reset_tracking_state()  # type: ignore[attr-defined]
        dets, debug = ld.detect_with_debug(  # type: ignore[attr-defined]
            frame,
            prefer_yellow=prefer_yellow_for_course(course),
            enable_fork=False,
        )
        road = getattr(debug, 'road_raw', None)
        if road is None or not getattr(road, 'size', 0):
            ranges = load_hsv_ranges(config_path)
            ipm = load_metric_ipm(config_path)
            bev = warp_metric_ipm(frame, ipm)
            road = road_mask_from_hsv(
                bev, ranges, prefer_yellow=prefer_yellow_for_course(course)
            )
        road_only = denoise_road_mask(np.asarray(road, dtype=np.uint8))
        road_in = np.asarray(getattr(dets, 'drivable_area', road_only), dtype=np.uint8)
        road_in_comp = compose_drivable(road_only, road_in, 'road_in')
        cl = np.asarray(
            getattr(dets, 'yellow_centerline', ())
            if prefer_yellow_for_course(course)
            else getattr(dets, 'white_centerline', ()),
            dtype=np.float32,
        )
        rows.append(
            {
                'frame': label,
                'road_pct': _coverage(road_only),
                'road_in_pct': _coverage(road_in_comp),
                'delta_pct': _coverage(road_in_comp) - _coverage(road_only),
                'centerline_pts': float(cl.shape[0]) if cl.ndim == 2 else 0.0,
            }
        )

    print(f'=== road vs road_in  course={course}  n={len(rows)} ===', flush=True)
    for r in rows:
        print(
            f'{r["frame"]}: road={r["road_pct"]:.1f}%  road_in={r["road_in_pct"]:.1f}%  '
            f'delta={r["delta_pct"]:+.1f}%  cl={int(r["centerline_pts"])}',
            flush=True,
        )
    if rows:
        avg_delta = sum(float(r['delta_pct']) for r in rows) / len(rows)
        print(f'avg delta (road_in - road): {avg_delta:+.2f}%', flush=True)


def build_views(
    frame: np.ndarray,
    *,
    ld: object,
    config_path: Path,
    mode: str,
    course: str,
    label: str,
    enable_fork: bool = False,
    fork_rank: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    ranges = load_hsv_ranges(config_path)
    ipm = load_metric_ipm(config_path)
    origin = draw_crop_overlay(frame, ipm)
    bev = warp_metric_ipm(frame, ipm)

    ld.reset_tracking_state()  # type: ignore[attr-defined]
    use_fork = bool(enable_fork)
    dets, debug = ld.detect_with_debug(  # type: ignore[attr-defined]
        frame,
        prefer_yellow=prefer_yellow_for_course(course),
        enable_fork=use_fork,
    )

    if getattr(debug, 'bev', None) is not None and getattr(debug.bev, 'size', 0) > 0:
        bev_fit = debug.bev
        bev_vis = (
            cv2.cvtColor(bev_fit, cv2.COLOR_GRAY2BGR)
            if bev_fit.ndim == 2
            else bev_fit.copy()
        )
    else:
        bev_vis = bev.copy()
        bev_fit = bev

    if getattr(debug, 'road_raw', None) is not None and getattr(debug.road_raw, 'size', 0) > 0:
        road = debug.road_raw
    else:
        road = road_mask_from_hsv(
            bev_fit, ranges, prefer_yellow=prefer_yellow_for_course(course)
        )

    blob = getattr(debug, 'road_clean', None)
    if blob is None or getattr(blob, 'size', 0) == 0:
        blob = getattr(dets, 'drivable_area', None)
    if blob is None or getattr(blob, 'size', 0) == 0:
        blob = np.zeros(road.shape[:2], dtype=np.uint8)
    else:
        blob = np.asarray(blob, dtype=np.uint8)

    driv = compose_drivable(road, blob, mode)
    used_fork = bool(getattr(debug, 'fork_active', False)) and use_fork
    rail_label = 'yellow' if prefer_yellow_for_course(course) else 'white'

    if prefer_yellow_for_course(course):
        rail_mask = make_mask(bev_fit, ranges['yellow'])
        rail_color = (0, 255, 255)
        centerline = np.asarray(getattr(dets, 'yellow_centerline', ()), dtype=np.float32)
        if centerline.size == 0:
            centerline = np.asarray(getattr(dets, 'white_centerline', ()), dtype=np.float32)
    else:
        rail_mask = make_mask(bev_fit, ranges['white'])
        rail_color = (255, 255, 255)
        centerline = np.asarray(getattr(dets, 'white_centerline', ()), dtype=np.float32)

    overlay = draw_metric_guides(bev_vis, ipm)
    overlay = overlay_mask(overlay, road, color=(90, 90, 90), alpha=0.30)
    overlay = overlay_mask(overlay, blob, color=(0, 180, 0), alpha=0.35)
    overlay = overlay_mask(overlay, rail_mask, color=rail_color, alpha=0.40)
    overlay = overlay_mask(overlay, driv, color=(0, 255, 255), alpha=0.30)
    _draw_centerline_xy(overlay, centerline, ipm=ipm, color=(0, 255, 255))
    if used_fork:
        _draw_fork_pairs(
            overlay,
            getattr(debug, 'fork_lane_pairs', ()),
            rank_filter=fork_rank,
        )

    result = cv2.cvtColor(driv, cv2.COLOR_GRAY2BGR)
    result = overlay_mask(result, rail_mask, color=(255, 0, 255), alpha=0.45)
    _draw_centerline_xy(result, centerline, ipm=ipm, color=(0, 255, 255))
    if used_fork:
        _draw_fork_pairs(
            result,
            getattr(debug, 'fork_lane_pairs', ()),
            rank_filter=fork_rank,
        )

    fork_on = bool(getattr(debug, 'fork_active', False))
    n_pairs = len(tuple(getattr(debug, 'fork_lane_pairs', ()) or ()))
    stats = {
        'road': _coverage(road),
        'rail_hsv': _coverage(rail_mask),
        'between': _coverage(blob),
        'driv': _coverage(driv),
        'centerline_pts': float(centerline.shape[0]) if centerline.ndim == 2 else 0.0,
        'fork_active': 1.0 if fork_on else 0.0,
        'fork_pairs': float(n_pairs),
        'used_fork': 1.0 if used_fork else 0.0,
    }
    fork_tag = (
        f'fork={n_pairs}/{getattr(debug, "fork_split_source", "")}'
        if fork_on
        else 'fork=off'
    )
    hud = (
        f'{label}  course={course}/{rail_label}  mode={mode}  blob  '
        f'{fork_tag}  Rd={stats["road"]:.0f}% Blob={stats["between"]:.0f}% '
        f'Dr={stats["driv"]:.0f}%  cl={int(stats["centerline_pts"])}'
    )
    for img in (origin, overlay, result):
        cv2.putText(
            img,
            hud,
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    left = _fit_panel(origin, PANEL_W, PANEL_H)
    mid = _fit_panel(overlay, PANEL_W, PANEL_H)
    right = _fit_panel(result, PANEL_W, PANEL_H)
    _label_panel(left, 'ORIGIN')
    mid_title = 'BEV+fork' if used_fork else 'BEV road|blob|mid'
    _label_panel(mid, mid_title)
    _label_panel(right, f'RESULT [{mode}]')
    gap = np.full((PANEL_H, GAP, 3), 40, dtype=np.uint8)
    mosaic = np.hstack([left, gap, mid, gap, right])
    return mosaic, stats


def _resolve_folder(from_bag: str | None, folder: Path | None) -> Path:
    if folder is not None:
        return folder.expanduser().resolve()
    key = (from_bag or 'out').strip().lower()
    if key not in FROM_BAG_DIRS:
        raise SystemExit(
            f'Unknown --from-bag {from_bag!r}; use {"|".join(FROM_BAG_DIRS)}'
        )    path = FROM_BAG_DIRS[key]
    if not path.is_dir():
        raise SystemExit(f'Missing captures: {path}')
    return path.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=None)
    parser.add_argument(
        '--from-bag',
        choices=tuple(FROM_BAG_DIRS.keys()),
        default='out',
        help='Shortcut to data/captures/from_bag/<alias> (default: out)',
    )
    parser.add_argument(
        '--course',
        choices=('in', 'out'),
        default=None,
        help='Rail color: out=white, in=yellow (default: from --from-bag)',
    )
    parser.add_argument('--config', type=Path, default=default_config_path())
    parser.add_argument(
        '--mode',
        choices=COMPOSE_MODES,
        default='between',
        help='Initial compose mode (default: between = follow blob corridor)',
    )
    parser.add_argument(
        '--fork',
        action='store_true',
        help='Enable out_fork pair extraction (OUT captures ~11–12 only)',
    )
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='1-based start frame index (out fork ~11–12)',
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Headless: print road vs road_in stats for all frames and exit',
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Process start frame once, print stats, exit (smoke / headless)',
    )
    args = parser.parse_args(argv)

    course = args.course or args.from_bag or 'out'
    if str(course).endswith('_cam') or str(course).endswith('_course'):
        course = 'in' if str(course).startswith('in') else 'out'
    folder = _resolve_folder(args.from_bag, args.folder)
    paths = _list_images(folder)
    if not paths:
        raise SystemExit(f'No images in {folder}')

    frames: list[np.ndarray] = []
    labels: list[str] = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is not None:
            frames.append(img)
            labels.append(path.name)
    if not frames:
        raise SystemExit(f'Failed to load images from {folder}')

    ld = _import_lane_detection()

    if args.compare:
        compare_road_modes(
            frames,
            labels,
            ld=ld,
            config_path=args.config,
            course=course,
        )
        return 0

    mode_idx = COMPOSE_MODES.index(args.mode)
    enable_fork = bool(args.fork)
    fork_rank: int | None = None
    idx = max(0, min(len(frames) - 1, int(args.start) - 1))
    auto = False
    window_snapped = False
    snap_dir = _REPO_ROOT / 'data' / 'captures' / 'out_drivable_preview'
    mosaic_w = 3 * PANEL_W + 2 * GAP

    if args.once:
        mosaic, stats = build_views(
            frames[idx],
            ld=ld,
            config_path=args.config,
            mode=COMPOSE_MODES[mode_idx],
            course=course,
            label=f'[{idx + 1}/{len(frames)}] {labels[idx]}',
            enable_fork=enable_fork,
            fork_rank=fork_rank,
        )
        print(
            f'ONCE [{idx + 1}] {labels[idx]} course={course} fork={enable_fork} '
            f'stats={stats}',
            flush=True,
        )
        out = snap_dir / f'smoke_{course}_{idx + 1:04d}.png'
        snap_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), mosaic)
        print(f'Wrote {out}', flush=True)
        return 0

    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, mosaic_w, PANEL_H + 40)
    # Show splash immediately — first detect can take a moment; otherwise WSLg
    # looks like "nothing opened".
    splash = np.full((PANEL_H + 40, mosaic_w, 3), 32, dtype=np.uint8)
    cv2.putText(
        splash,
        'loading blob preview...',
        (24, PANEL_H // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # WSLg often maps OpenCV windows onto the secondary monitor; move is a no-op.
    map_and_place_window(WIN_MAIN, splash, x=48, y=48)
    snap_dir.mkdir(parents=True, exist_ok=True)
    live_path = snap_dir / 'live.png'
    cv2.imwrite(str(live_path), splash)
    print(
        f'Folder: {folder} ({len(frames)} images) start=[{idx + 1}]\n'
        f'Course: {course} (rails={"yellow" if prefer_yellow_for_course(course) else "white"})\n'
        f'Backend: blob  enable_fork={enable_fork}  Config: {args.config}\n'
        f'Keys: 1–4 mode  f=fork  0=branch L/R/both  n/p  SPACE  s  q\n'
        f'Live: {live_path}  (IDE에서 이 파일 열어두면 GUI 안 보여도 확인 가능)\n'
        f'WSLg: {diagnose_window_placement(WIN_MAIN)}\n'
        f'      OUT 11–12 → --fork',
        flush=True,
    )

    try:
        while True:
            mode = COMPOSE_MODES[mode_idx]
            mosaic, stats = build_views(
                frames[idx],
                ld=ld,
                config_path=args.config,
                mode=mode,
                course=course,
                label=f'[{idx + 1}/{len(frames)}] {labels[idx]}',
                enable_fork=enable_fork,
                fork_rank=fork_rank,
            )
            cv2.imshow(WIN_MAIN, mosaic)
            # Always mirror to disk — WSLg may put the GTK window on another monitor.
            cv2.imwrite(str(live_path), mosaic)
            if not window_snapped:
                map_and_place_window(WIN_MAIN, x=48, y=48, pumps=2)
                print(f'WSLg: {diagnose_window_placement(WIN_MAIN)}', flush=True)
                window_snapped = True
            delay = 400 if auto else 30
            key = cv2.waitKey(delay) & 0xFF

            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                auto = not auto
                print(f'auto={"ON" if auto else "OFF"}', flush=True)
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                mode_idx = key - ord('1')
                print(
                    f'mode → {COMPOSE_MODES[mode_idx]} '
                    f'({COMPOSE_HELP[COMPOSE_MODES[mode_idx]]})',
                    flush=True,
                )
            elif key == ord('f'):
                enable_fork = not enable_fork
                print(f'enable_fork → {enable_fork}', flush=True)
            elif key == ord('0'):
                if fork_rank is None:
                    fork_rank = 0
                elif fork_rank == 0:
                    fork_rank = 1
                else:
                    fork_rank = None
                print(
                    f'fork_rank → {"both" if fork_rank is None else fork_rank}',
                    flush=True,
                )
            elif key == ord('9'):
                fork_rank = 1 if fork_rank != 1 else None
                print(
                    f'fork_rank → {"both" if fork_rank is None else fork_rank}',
                    flush=True,
                )
            elif key == ord('n'):
                idx = (idx + 1) % len(frames)
                print(f'[{idx + 1}/{len(frames)}] {labels[idx]}', flush=True)
            elif key == ord('p'):
                idx = (idx - 1) % len(frames)
                print(f'[{idx + 1}/{len(frames)}] {labels[idx]}', flush=True)
            elif key == ord('s'):
                snap_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                out = snap_dir / f'{stamp}_{idx + 1:04d}_{mode}.png'
                cv2.imwrite(str(out), mosaic)
                print(f'Saved {out}', flush=True)
            elif auto:
                idx = (idx + 1) % len(frames)
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
