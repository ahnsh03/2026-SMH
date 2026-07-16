#!/usr/bin/env python3
"""Visualize OpenCV traffic-light detection on course bags (camera frames).

Same idea as BEV HSV mosaics: show raw frame + red/green masks + accepted
blobs, so you can check false positives (road paint, signs, glare).

Examples (inside 2026-smh-sim after sourcing ROS):

  # interactive bag scrubber
  python3 scripts/vision_tune/viz_traffic_light.py --from-bag out_cam
  python3 scripts/vision_tune/viz_traffic_light.py --from-bag in --hits-only

  # board thresholds (tighter red H) while using main bags/tooling
  PYTHONPATH=../2026-SMH-board/src/inference:$PYTHONPATH \\
    python3 scripts/vision_tune/viz_traffic_light.py --from-bag out_cam --tune

  # headless: dump mosaics + summary CSV for every hit
  python3 scripts/vision_tune/viz_traffic_light.py --from-bag out \\
      --export-dir data/captures/traffic_light_viz/out --no-gui --stride 2

Keys (GUI):
  SPACE       play / pause  (starts paused)
  ← / → , .   step frame
  [ / ]       slower / faster
  h           jump to next RED/GREEN hit
  d           dump current mosaic PNG
  r / Home    restart
  q / ESC     quit
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parents[1]
_INFERENCE = _ROOT / 'src' / 'inference'
for p in (_SCRIPT, _INFERENCE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from window_layout import place_window  # noqa: E402

from inference.modules.trafficsign.color_detector import (  # noqa: E402
    SignalInspect,
    _GREEN_RANGE,
    _MAX_ASPECT_RATIO,
    _MIN_CIRCULARITY,
    _MIN_GREEN_PIXELS,
    _MIN_RED_PIXELS,
    _RED_RANGES,
    inspect_signal,
)
from inference.types import TrafficSignal  # noqa: E402

PANEL_W = 480
PANEL_H = 270
GAP = 6
WIN = 'traffic_light_viz'
CTRL = 'traffic_light_ctrl'
CAMERA_TOPIC = '/camera/image/compressed'
OUT_DEFAULT = _ROOT / 'data' / 'captures' / 'traffic_light_viz'


def _fit(frame: np.ndarray, width: int, height: int, *, nearest: bool = False) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is None or frame.size == 0:
        return out
    img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h < 1 or w < 1:
        return out
    scale = min(width / w, height / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _label(panel: np.ndarray, text: str, color: tuple[int, int, int] = (0, 255, 255)) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 26), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _mask_panel(mask: np.ndarray, bgr: tuple[int, int, int]) -> np.ndarray:
    """Binary mask as solid color on black (no camera underlay)."""
    if mask is None or mask.size == 0:
        return np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    out = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    out[mask > 0] = bgr
    return out


def _signal_bgr(signal: TrafficSignal) -> tuple[int, int, int]:
    if signal == TrafficSignal.RED:
        return (0, 0, 255)
    if signal == TrafficSignal.GREEN:
        return (0, 220, 0)
    return (180, 180, 180)


def _annotate(frame: np.ndarray, report: SignalInspect, frame_idx: int) -> np.ndarray:
    out = frame.copy()
    for blob in report.blobs:
        x, y, w, h = blob.bbox
        if blob.color == 'red':
            color = (0, 0, 255) if blob.shape_ok else (0, 128, 255)
        else:
            color = (0, 255, 0) if blob.shape_ok else (0, 180, 180)
        thickness = 3 if blob.is_largest and blob.shape_ok else 1
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        tag = (
            f'{blob.color[0].upper()} a={blob.area:.0f} '
            f'c={blob.circularity:.2f} ar={blob.aspect_ratio:.1f}'
        )
        if blob.is_largest:
            tag = '* ' + tag
        cv2.putText(
            out,
            tag,
            (x, max(14, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    sig_color = _signal_bgr(report.signal)
    lines = [
        f'#{frame_idx}  signal={report.signal.value}',
        f'red_px={report.red_pixels} ok={int(report.red_ok)}  '
        f'green_px={report.green_pixels} ok={int(report.green_ok)}',
        f'min_px={_MIN_RED_PIXELS}/{_MIN_GREEN_PIXELS}  '
        f'circ>={_MIN_CIRCULARITY}  ar<={_MAX_ASPECT_RATIO}',
    ]
    y = 28
    for line in lines:
        cv2.putText(
            out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sig_color, 1, cv2.LINE_AA
        )
        y += 22
    return out


def _mosaic(frame: np.ndarray, report: SignalInspect, frame_idx: int) -> np.ndarray:
    cam = _fit(frame, PANEL_W, PANEL_H)
    _label(cam, f'camera #{frame_idx}')

    red = _fit(
        _mask_panel(report.red_mask, (0, 0, 255)), PANEL_W, PANEL_H, nearest=True
    )
    _label(red, f'red mask  px={report.red_pixels}  ok={int(report.red_ok)}', (0, 0, 255))

    green = _fit(
        _mask_panel(report.green_mask, (0, 255, 0)), PANEL_W, PANEL_H, nearest=True
    )
    _label(
        green,
        f'green mask  px={report.green_pixels}  ok={int(report.green_ok)}',
        (0, 220, 0),
    )

    ann = _fit(_annotate(frame, report, frame_idx), PANEL_W, PANEL_H)
    _label(ann, f'decision={report.signal.value}', _signal_bgr(report.signal))

    top = np.hstack([cam, np.full((PANEL_H, GAP, 3), 40, np.uint8), red])
    bot = np.hstack([green, np.full((PANEL_H, GAP, 3), 40, np.uint8), ann])
    return np.vstack([top, np.full((GAP, top.shape[1], 3), 40, np.uint8), bot])


def _default_trackbar_values() -> dict[str, int]:
    r0_lo, r0_hi = _RED_RANGES[0]
    r1_lo, r1_hi = _RED_RANGES[1]
    g_lo, g_hi = _GREEN_RANGE
    return {
        'r0_h_lo': int(r0_lo[0]),
        'r0_h_hi': int(r0_hi[0]),
        'r1_h_lo': int(r1_lo[0]),
        'r1_h_hi': int(r1_hi[0]),
        'r_s_lo': int(r0_lo[1]),
        'r_v_lo': int(r0_lo[2]),
        'g_h_lo': int(g_lo[0]),
        'g_h_hi': int(g_hi[0]),
        'g_s_lo': int(g_lo[1]),
        'g_v_lo': int(g_lo[2]),
        'min_px': int(_MIN_RED_PIXELS),
    }


def _ranges_from_trackbars() -> tuple[
    tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
    tuple[tuple[int, int, int], tuple[int, int, int]],
    int,
]:
    r0_h_lo = cv2.getTrackbarPos('r0_h_lo', CTRL)
    r0_h_hi = max(r0_h_lo, cv2.getTrackbarPos('r0_h_hi', CTRL))
    r1_h_lo = cv2.getTrackbarPos('r1_h_lo', CTRL)
    r1_h_hi = max(r1_h_lo, cv2.getTrackbarPos('r1_h_hi', CTRL))
    r_s = cv2.getTrackbarPos('r_s_lo', CTRL)
    r_v = cv2.getTrackbarPos('r_v_lo', CTRL)
    g_h_lo = cv2.getTrackbarPos('g_h_lo', CTRL)
    g_h_hi = max(g_h_lo, cv2.getTrackbarPos('g_h_hi', CTRL))
    g_s = cv2.getTrackbarPos('g_s_lo', CTRL)
    g_v = cv2.getTrackbarPos('g_v_lo', CTRL)
    min_px = max(1, cv2.getTrackbarPos('min_px', CTRL))
    red = (
        ((r0_h_lo, r_s, r_v), (r0_h_hi, 255, 255)),
        ((r1_h_lo, r_s, r_v), (r1_h_hi, 255, 255)),
    )
    green = ((g_h_lo, g_s, g_v), (g_h_hi, 255, 255))
    return red, green, min_px


def _setup_trackbars() -> None:
    cv2.namedWindow(CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL, 420, 360)
    place_window(CTRL, 48, 48 + PANEL_H * 2 + 80)
    vals = _default_trackbar_values()
    specs = [
        ('r0_h_lo', 179),
        ('r0_h_hi', 179),
        ('r1_h_lo', 179),
        ('r1_h_hi', 179),
        ('r_s_lo', 255),
        ('r_v_lo', 255),
        ('g_h_lo', 179),
        ('g_h_hi', 179),
        ('g_s_lo', 255),
        ('g_v_lo', 255),
        ('min_px', 500),
    ]
    for name, vmax in specs:
        cv2.createTrackbar(name, CTRL, int(vals[name]), int(vmax), lambda _v: None)


def _inspect_frame(
    frame: np.ndarray,
    *,
    tune: bool,
) -> SignalInspect:
    if not tune:
        return inspect_signal(frame)
    red, green, min_px = _ranges_from_trackbars()
    return inspect_signal(
        frame,
        red_ranges=red,
        green_range=green,
        min_red_pixels=min_px,
        min_green_pixels=min_px,
    )


def _load_source(args: argparse.Namespace):
    if args.folder is not None:
        folder = args.folder.expanduser()
        if not folder.is_absolute():
            folder = (_ROOT / folder).resolve()
        paths = sorted(
            p
            for p in folder.iterdir()
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        )
        if not paths:
            raise SystemExit(f'No images in {folder}')

        def decode(i: int) -> np.ndarray | None:
            return cv2.imread(str(paths[i]), cv2.IMREAD_COLOR)

        return f'folder:{folder.name}', len(paths), decode, paths

    from capture_from_bag import decode_jpeg, load_camera_jpegs, resolve_bag

    bag_dir = resolve_bag(args.from_bag, args.bag)
    print(f'Loading bag {bag_dir} …', flush=True)
    jpegs, _stamps = load_camera_jpegs(bag_dir, args.topic)

    def decode(i: int) -> np.ndarray | None:
        return decode_jpeg(jpegs[i])

    return bag_dir.name, len(jpegs), decode, None


def _print_thresholds() -> None:
    print('OpenCV traffic-light thresholds (loaded package):')
    print(f'  RED   {_RED_RANGES}')
    print(f'  GREEN {_GREEN_RANGE}')
    print(
        f'  min_px={_MIN_RED_PIXELS}/{_MIN_GREEN_PIXELS}  '
        f'circ>={_MIN_CIRCULARITY}  ar<={_MAX_ASPECT_RATIO}'
    )


def run(args: argparse.Namespace) -> int:
    _print_thresholds()
    label, n_total, decode, paths = _load_source(args)
    stride = max(1, int(args.stride))
    indices = list(range(0, n_total, stride))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]
    if not indices:
        raise SystemExit('No frames')

    export_dir: Path | None = None
    if args.export_dir is not None:
        export_dir = args.export_dir.expanduser()
        if not export_dir.is_absolute():
            export_dir = (_ROOT / export_dir).resolve()
        export_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    hit_rows: list[dict] = []
    hit_indices: list[int] = []

    # Pre-scan for --hits-only / summary (decode once if exporting headless).
    if args.hits_only or args.no_gui:
        filtered: list[int] = []
        for k, i in enumerate(indices):
            frame = decode(i)
            if frame is None:
                continue
            report = inspect_signal(frame)
            counts[report.signal.value] += 1
            if report.signal != TrafficSignal.UNKNOWN:
                filtered.append(i)
                hit_indices.append(i)
                hit_rows.append(
                    {
                        'frame': i + 1,
                        'signal': report.signal.value,
                        'red_px': report.red_pixels,
                        'green_px': report.green_pixels,
                        'n_blobs': len(report.blobs),
                    }
                )
                if export_dir is not None:
                    mosaic = _mosaic(frame, report, i + 1)
                    out = export_dir / f'{i + 1:05d}_{report.signal.value}.png'
                    cv2.imwrite(str(out), mosaic)
            if (k + 1) % 100 == 0:
                print(f'  scanned {k + 1}/{len(indices)}', flush=True)
        if args.hits_only:
            indices = filtered
            print(f'hits-only: {len(indices)} detection frames', flush=True)
        _print_summary(label, counts, hit_indices)
        if export_dir is not None and hit_rows:
            csv_path = export_dir / 'hits.csv'
            with csv_path.open('w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(hit_rows[0].keys()))
                w.writeheader()
                w.writerows(hit_rows)
            print(f'wrote {csv_path}  mosaics→{export_dir}')
        if args.no_gui:
            return 0
        # GUI continues over filtered (or all) indices; reset live counts.
        counts = Counter()
        hit_rows = []
        hit_indices = []

    if args.tune:
        _setup_trackbars()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, PANEL_W * 2 + GAP, PANEL_H * 2 + GAP)
    place_window(WIN, 48, 48)

    paused = True
    rate = max(0.05, float(args.rate))
    pos = 0
    print(
        f'source={label}  frames={len(indices)}  stride={stride}  '
        f'rate={rate}  tune={args.tune}',
        flush=True,
    )
    print('SPACE play/pause · ←→ step · h next-hit · d dump · q quit', flush=True)

    last_t = 0.0
    while 0 <= pos < len(indices):
        i = indices[pos]
        frame = decode(i)
        if frame is None:
            pos += 1
            continue
        report = _inspect_frame(frame, tune=args.tune)
        counts[report.signal.value] += 1
        if report.signal != TrafficSignal.UNKNOWN:
            hit_indices.append(i)

        mosaic = _mosaic(frame, report, i + 1)
        cv2.imshow(WIN, mosaic)

        now = time.monotonic()
        delay_ms = 1 if paused else max(1, int(1000.0 / (15.0 * rate)))
        key = cv2.waitKey(delay_ms) & 0xFF

        if key in (ord('q'), 27):
            break
        if key == ord(' '):
            paused = not paused
        elif key in (81, ord(',')):  # left
            pos = max(0, pos - 1)
            paused = True
            continue
        elif key in (83, ord('.')):  # right
            pos = min(len(indices) - 1, pos + 1)
            paused = True
            continue
        elif key == ord('['):
            rate = max(0.05, rate * 0.7)
            print(f'rate={rate:.2f}', flush=True)
        elif key == ord(']'):
            rate = min(8.0, rate * 1.4)
            print(f'rate={rate:.2f}', flush=True)
        elif key == ord('h'):
            found = None
            for j in range(pos + 1, len(indices)):
                fr = decode(indices[j])
                if fr is None:
                    continue
                if inspect_signal(fr).signal != TrafficSignal.UNKNOWN:
                    found = j
                    break
            if found is None:
                print('no more hits ahead', flush=True)
            else:
                pos = found
                paused = True
                continue
        elif key == ord('d'):
            dump_dir = export_dir or (OUT_DEFAULT / label)
            dump_dir.mkdir(parents=True, exist_ok=True)
            out = dump_dir / f'{i + 1:05d}_{report.signal.value}.png'
            cv2.imwrite(str(out), mosaic)
            print(f'dumped {out}', flush=True)
        elif key in (ord('r'),):
            pos = 0
            paused = True
            continue

        if not paused:
            if now - last_t >= 1.0 / (15.0 * rate):
                pos += 1
                last_t = now
        # when paused, waitKey already blocked briefly

    cv2.destroyAllWindows()
    _print_summary(label, counts, hit_indices)
    return 0


def _print_summary(label: str, counts: Counter[str], hit_indices: list[int]) -> None:
    total = sum(counts.values())
    print(f'--- summary {label} ---')
    print(
        f'total={total}  '
        f'RED={counts.get(TrafficSignal.RED.value, 0)}  '
        f'GREEN={counts.get(TrafficSignal.GREEN.value, 0)}  '
        f'UNKNOWN={counts.get(TrafficSignal.UNKNOWN.value, 0)}'
    )
    if hit_indices:
        uniq = sorted({i + 1 for i in hit_indices})
        preview = uniq[:40]
        more = '' if len(uniq) <= 40 else f' … (+{len(uniq) - 40})'
        print(f'hit frames (1-based, unique={len(uniq)}): {preview}{more}')
    else:
        print('hit frames: (none)')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        '--from-bag',
        choices=('in', 'out', 'out_cam', 'in_cam', 'sign_right', 'sign_left'),
        help='course bag alias under bags/',
    )
    src.add_argument('--folder', type=Path, help='folder of camera PNG/JPG frames')
    ap.add_argument('--bag', type=Path, default=None, help='explicit rosbag2 directory')
    ap.add_argument('--topic', default=CAMERA_TOPIC)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--rate', type=float, default=1.0, help='playback rate multiplier')
    ap.add_argument(
        '--hits-only',
        action='store_true',
        help='skip UNKNOWN frames (scan first, then scrub hits)',
    )
    ap.add_argument(
        '--tune',
        action='store_true',
        help='HSV / min_px trackbars (overrides package defaults live)',
    )
    ap.add_argument(
        '--export-dir',
        type=Path,
        default=None,
        help='write mosaics (+ hits.csv with --no-gui/--hits-only)',
    )
    ap.add_argument(
        '--no-gui',
        action='store_true',
        help='headless scan + optional export; print summary and exit',
    )
    args = ap.parse_args()
    if args.no_gui and args.tune:
        raise SystemExit('--tune needs a GUI; drop --no-gui')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
