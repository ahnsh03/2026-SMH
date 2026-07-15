#!/usr/bin/env python3
"""Play bag / BEV video with the **same** ``6_ego_blob`` pipeline as photo mosaics.

Why bag looked wrong before
---------------------------
``play_bag_drivable`` previously used runtime ``blob`` perception
(``extract_bev_masks`` = HSV on **camera** then warp + morph 3/5 +
``select_best_blob``). Photo mosaics use ``viz_raw_hsv_masks.extract_five``:
HSV on **BEV**, course paint|road (IN: Y if present else W; never W∧Y),
morph open=5/close=17, then bottom ego CC.

Correct workflow
----------------
1. Export Metric IPM BEV video from bag (warp once, save).
2. Play that BEV with ``extract_five_from_bev`` → left BEV | right white ego.

Examples (inside 2026-smh-sim after sourcing ROS):

  # export BEV mp4 from bag
  python3 scripts/vision_tune/play_bag_drivable.py in --export-bev \\
      data/captures/bev_videos/in.mp4

  # play saved BEV with 6_ego_blob (recommended)
  python3 scripts/vision_tune/play_bag_drivable.py --from-bev \\
      data/captures/bev_videos/in.mp4

  # live bag → BEV once → same 6_ego_blob (no save)
  python3 scripts/vision_tune/play_bag_drivable.py out --rate 0.4

Keys: SPACE play/pause · ,/. step · [] rate · o overlay · r restart · q quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
for p in (_SCRIPT_DIR,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hsv import default_config_path  # noqa: E402
from metric_ipm import load_metric_ipm, warp_metric_ipm  # noqa: E402
from viz_raw_hsv_masks import extract_five_from_bev  # noqa: E402
from window_layout import place_window  # noqa: E402

PANEL_W = 480
PANEL_H = 360
GAP = 6
WIN_PREFIX = 'bag_drivable_ego6'
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')


def _chown_to_host(path: Path) -> None:
    if os.geteuid() != 0:
        return
    uid = int(os.environ.get('HOST_UID') or os.environ.get('SUDO_UID') or 1000)
    gid = int(os.environ.get('HOST_GID') or os.environ.get('SUDO_GID') or uid)
    try:
        os.chown(path, uid, gid)
    except OSError:
        pass


def _fit(
    frame: np.ndarray, width: int, height: int, *, nearest: bool = False
) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is None or frame.size == 0:
        return out
    img = frame
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
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


def _label(panel: np.ndarray, text: str) -> None:
    cv2.putText(
        panel,
        text,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _coverage(mask: np.ndarray) -> float:
    if mask is None or mask.size == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(mask)) / float(mask.size)


def ego6_from_bev(
    bev: np.ndarray,
    config_path: Path,
    *,
    open_k: int,
    close_k: int,
    course: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Same as photo mosaic panel 6 — returns (bev, white ego_blob, cov%)."""

    five = extract_five_from_bev(
        bev,
        config_path,
        open_k=open_k,
        close_k=close_k,
        course=course,
    )
    blob = five['ego_blob']
    return five['bev'], blob, _coverage(blob)


def build_right_panel(
    bev: np.ndarray,
    white_mask: np.ndarray,
    *,
    overlay: bool,
) -> np.ndarray:
    if overlay:
        base = (
            bev.copy()
            if bev is not None and bev.size
            else np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        )
        dim = (base.astype(np.float32) * 0.35).astype(np.uint8)
        on = white_mask > 0
        dim[on] = (255, 255, 255)
        panel = _fit(dim, PANEL_W, PANEL_H, nearest=True)
        _label(panel, f'BEV+6_ego white  {_coverage(white_mask):.1f}%')
        return panel

    panel = _fit(white_mask, PANEL_W, PANEL_H, nearest=True)
    _label(panel, f'6_ego_blob (white)  {_coverage(white_mask):.1f}%')
    return panel


def compose_view(
    bev: np.ndarray,
    white_mask: np.ndarray,
    *,
    overlay: bool,
    index: int,
    n: int,
    t_rel: float,
    rate: float,
    paused: bool,
    course: str,
) -> np.ndarray:
    left = _fit(bev, PANEL_W, PANEL_H)
    _label(left, 'BEV')
    right = build_right_panel(bev, white_mask, overlay=overlay)
    gap = np.full((PANEL_H, GAP, 3), 28, dtype=np.uint8)
    row = np.hstack([left, gap, right])

    status = 'PAUSE' if paused else f'x{rate:.2f}'
    footer = np.full((44, row.shape[1], 3), 16, dtype=np.uint8)
    mode = 'overlay' if overlay else 'white-mask'
    lines = [
        f'{course}  [{index + 1}/{n}]  t={t_rel:.2f}s  {status}  '
        f'pipe=extract_five/6_ego  right={mode}',
        'SPACE=play/pause  ,/.=step  []=rate  o=overlay  r=restart  q=quit',
    ]
    y = 18
    for text in lines:
        cv2.putText(
            footer,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 18
    return np.vstack([row, footer])


def _open_window(title: str, width: int, height: int) -> str:
    win = f'{title} [{_stamp()[-10:]}]'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, width, height)
    place_window(win, 48, 48)
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        'loading…',
        (24, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(win, blank)
    cv2.waitKey(1)
    place_window(win, 48, 48)
    return win


def _load_bag_camera(course: str | None, bag: Path | None, topic: str):
    from capture_from_bag import (  # local import — needs sourced ROS
        decode_jpeg,
        load_camera_jpegs,
        resolve_bag,
    )

    bag_dir = resolve_bag(course, bag)
    label = course or bag_dir.name
    print(f'Loading bag {bag_dir} …', flush=True)
    jpegs, stamps = load_camera_jpegs(bag_dir, topic)
    return label, jpegs, stamps, decode_jpeg


def export_bev_video(
    jpegs: list[bytes],
    stamps: list[float],
    decode_jpeg,
    out_path: Path,
    config_path: Path,
    *,
    stride: int = 1,
    fps: float | None = None,
) -> Path:
    """Warp each camera JPEG → BEV once and write an mp4 (no ego mask)."""

    out_path = out_path.expanduser()
    if not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ipm = load_metric_ipm(config_path)
    stride = max(1, int(stride))
    indices = list(range(0, len(jpegs), stride))
    if not indices:
        raise SystemExit('No frames to export')

    # Probe size from first good frame.
    bev0 = None
    for i in indices:
        fr = decode_jpeg(jpegs[i])
        if fr is None:
            continue
        bev0 = warp_metric_ipm(fr, ipm)
        break
    if bev0 is None:
        raise SystemExit('Could not decode any frame for BEV export')

    h, w = bev0.shape[:2]
    if fps is None or fps <= 0:
        if len(stamps) >= 2:
            dts = [
                stamps[min(i + stride, len(stamps) - 1)] - stamps[i]
                for i in indices[:-1]
            ]
            med = float(np.median([d for d in dts if d > 1e-4])) if dts else 1 / 15
            fps = max(1.0, min(30.0, 1.0 / med))
        else:
            fps = 15.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise SystemExit(f'Failed to open VideoWriter: {out_path}')

    print(
        f'Exporting BEV → {out_path}  frames={len(indices)}  '
        f'stride={stride}  fps={fps:.2f}  size={w}x{h}',
        flush=True,
    )
    written = 0
    for k, i in enumerate(indices):
        fr = decode_jpeg(jpegs[i])
        if fr is None:
            continue
        bev = warp_metric_ipm(fr, ipm)
        writer.write(bev)
        written += 1
        if (k + 1) % 50 == 0 or k + 1 == len(indices):
            print(f'  {k + 1}/{len(indices)}', flush=True)
    writer.release()
    _chown_to_host(out_path)
    print(f'Done: {written} BEV frames → {out_path}', flush=True)
    return out_path


def _list_bev_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_bev_source(
    path: Path,
):
    """Return (label, stamps, reader) where ``reader(i) -> bgr|None``."""

    path = path.expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()

    if path.is_dir():
        paths = _list_bev_images(path)
        if not paths:
            raise SystemExit(f'No images in {path}')
        stamps = [i / 15.0 for i in range(len(paths))]
        cache: dict[int, np.ndarray] = {}

        def reader(i: int) -> np.ndarray | None:
            hit = cache.get(i)
            if hit is not None:
                return hit
            img = cv2.imread(str(paths[i]))
            if img is None:
                return None
            if len(cache) > 64:
                cache.clear()
            cache[i] = img
            return img

        return path.name, stamps, reader

    if path.is_file() and path.suffix.lower() in {'.mp4', '.avi', '.mkv', '.mov'}:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f'Cannot open video: {path}')
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        if n <= 0:
            # Fallback: scan
            frames: list[np.ndarray] = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
            cap.release()
            if not frames:
                raise SystemExit(f'Empty video: {path}')
            stamps = [i / max(fps, 1.0) for i in range(len(frames))]

            def reader_mem(i: int) -> np.ndarray | None:
                return frames[i]

            return path.stem, stamps, reader_mem

        stamps = [i / max(fps, 1.0) for i in range(n)]
        cache: dict[int, np.ndarray] = {}

        def reader_vid(i: int) -> np.ndarray | None:
            hit = cache.get(i)
            if hit is not None:
                return hit
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok or fr is None:
                return None
            if len(cache) > 64:
                cache.clear()
            cache[i] = fr
            return fr

        # Keep cap alive via closure attribute
        reader_vid._cap = cap  # type: ignore[attr-defined]
        return path.stem, stamps, reader_vid

    raise SystemExit(f'--from-bev expects a video file or image folder: {path}')


def run_player(
    *,
    course_label: str,
    n: int,
    stamps: list[float],
    get_bev,  # (i) -> BEV bgr (already warped)
    rate: float,
    start_index: int,
    overlay: bool,
    config_path: Path,
    open_k: int,
    close_k: int,
    course: str | None = None,
) -> int:
    idx = max(0, min(start_index, n - 1))
    paused = True
    rate = max(0.05, float(rate))
    overlay = bool(overlay)
    paint_course = course or course_label
    mask_cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    view_w = PANEL_W * 2 + GAP
    view_h = PANEL_H + 44
    win = _open_window(f'{WIN_PREFIX}_{course_label}', view_w, view_h)
    print(
        f'Loaded {n} BEV frames ({course_label})  paint_course={paint_course}\n'
        f'Window "{win}" — BEV | 6_ego_blob white  '
        f'(extract_five_from_bev open={open_k} close={close_k})\n'
        f'IN: yellow|road if yellow else white|road · OUT: white|road\n'
        f'Starts PAUSED. SPACE=play  o=overlay  q=quit',
        flush=True,
    )
    if not os.environ.get('DISPLAY'):
        print('WARNING: DISPLAY is empty — OpenCV window will not appear.', flush=True)

    def masks_at(i: int, bev: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        hit = mask_cache.get(i)
        if hit is not None:
            return hit
        b, white, cov = ego6_from_bev(
            bev,
            config_path,
            open_k=open_k,
            close_k=close_k,
            course=paint_course,
        )
        if len(mask_cache) > 48:
            mask_cache.clear()
        mask_cache[i] = (b, white, cov)
        return b, white, cov

    last_advance = time.perf_counter()
    try:
        while True:
            bev = get_bev(idx)
            if bev is None:
                idx = min(n - 1, idx + 1)
                continue
            bev, white, _ = masks_at(idx, bev)
            view = compose_view(
                bev,
                white,
                overlay=overlay,
                index=idx,
                n=n,
                t_rel=stamps[idx] if idx < len(stamps) else 0.0,
                rate=rate,
                paused=paused,
                course=course_label,
            )
            cv2.imshow(win, view)
            key = cv2.waitKeyEx(20)
            key8 = key & 0xFF

            if key8 in (ord('q'), 27) or key in (ord('q'), 27):
                break
            if key8 == ord(' '):
                paused = not paused
                last_advance = time.perf_counter()
            elif key8 == ord('o'):
                overlay = not overlay
                print(f'right={"overlay" if overlay else "white-mask"}', flush=True)
            elif key in (65361, 2424832, 81) or key8 == ord(','):
                idx = max(0, idx - 1)
                paused = True
            elif key in (65363, 2555904, 83) or key8 == ord('.'):
                idx = min(n - 1, idx + 1)
                paused = True
            elif key8 == ord('['):
                rate = max(0.05, rate / 1.25)
                print(f'rate={rate:.2f}', flush=True)
            elif key8 == ord(']'):
                rate = min(8.0, rate * 1.25)
                print(f'rate={rate:.2f}', flush=True)
            elif key8 == ord('r') or key in (65360, 2359296):
                idx = 0
                paused = True
            elif key in (65367, 2293760):
                idx = n - 1
                paused = True

            if not paused and n > 1:
                now = time.perf_counter()
                if idx + 1 < n and idx + 1 < len(stamps):
                    dt = max(1e-3, stamps[idx + 1] - stamps[idx])
                elif idx > 0 and idx < len(stamps):
                    dt = max(1e-3, stamps[idx] - stamps[idx - 1])
                else:
                    dt = 1 / 15
                if now - last_advance >= dt / rate:
                    if idx + 1 < n:
                        idx += 1
                    else:
                        paused = True
                    last_advance = now
    except KeyboardInterrupt:
        print('\ninterrupted', flush=True)
    finally:
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        'course',
        nargs='?',
        choices=('in', 'out'),
        help='bag course alias (ignored with --from-bev)',
    )
    ap.add_argument('--bag', type=Path, default=None)
    ap.add_argument('--topic', default='/camera/image/compressed')
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument('--rate', type=float, default=0.5)
    ap.add_argument('--start', type=float, default=0.0, help='seconds (bag) or frame idx if --from-bev')
    ap.add_argument('--overlay', action='store_true')
    ap.add_argument('--open-k', type=int, default=5)
    ap.add_argument('--close-k', type=int, default=17)
    ap.add_argument(
        '--export-bev',
        type=Path,
        default=None,
        help='write BEV mp4 from bag then exit (or play if --play-after-export)',
    )
    ap.add_argument(
        '--play-after-export',
        action='store_true',
        help='after --export-bev, play the saved file with 6_ego_blob',
    )
    ap.add_argument(
        '--export-stride',
        type=int,
        default=1,
        help='keep every Nth bag frame when exporting BEV',
    )
    ap.add_argument(
        '--export-fps',
        type=float,
        default=0.0,
        help='forced fps for BEV mp4 (0 = from bag timestamps)',
    )
    ap.add_argument(
        '--from-bev',
        type=Path,
        default=None,
        help='play from saved BEV mp4 or image folder (no second IPM warp)',
    )
    ap.add_argument(
        '--paint-course',
        choices=('in', 'out'),
        default=None,
        help='paint rule for ego fill: in=Y|road if yellow else W|road; '
        'out=W|road. Default: positional course or inferred from --from-bev name',
    )
    args = ap.parse_args(argv)
    config_path = args.config.expanduser().resolve()

    def _resolve_paint_course(fallback: str | None) -> str:
        if args.paint_course:
            return args.paint_course
        if args.course:
            return args.course
        name = (fallback or '').lower()
        if 'out' in name:
            return 'out'
        return 'in'

    # --- Recommended path: play already-exported BEV ---
    if args.from_bev is not None:
        label, stamps, reader = load_bev_source(args.from_bev)
        n = len(stamps)
        start_index = 0
        if args.start > 0:
            # treat as seconds if float looks like time, else frame
            if args.start >= 1.0 and abs(args.start - int(args.start)) < 1e-6:
                start_index = min(n - 1, max(0, int(args.start) - 1))
            else:
                for i, t in enumerate(stamps):
                    if t >= args.start:
                        start_index = i
                        break
        return run_player(
            course_label=label,
            n=n,
            stamps=stamps,
            get_bev=reader,
            rate=args.rate,
            start_index=start_index,
            overlay=args.overlay,
            config_path=config_path,
            open_k=args.open_k,
            close_k=args.close_k,
            course=_resolve_paint_course(label),
        )

    # --- Bag path ---
    label, jpegs, stamps, decode_jpeg = _load_bag_camera(
        args.course, args.bag, args.topic
    )
    ipm = load_metric_ipm(config_path)
    decode_cache: dict[int, np.ndarray] = {}

    def camera_at(i: int):
        hit = decode_cache.get(i)
        if hit is not None:
            return hit
        img = decode_jpeg(jpegs[i])
        if img is None:
            return None
        if len(decode_cache) > 32:
            decode_cache.clear()
        decode_cache[i] = img
        return img

    if args.export_bev is not None:
        out = export_bev_video(
            jpegs,
            stamps,
            decode_jpeg,
            args.export_bev,
            config_path,
            stride=args.export_stride,
            fps=args.export_fps or None,
        )
        if not args.play_after_export:
            print(
                f'Next: python3 scripts/vision_tune/play_bag_drivable.py '
                f'--from-bev {out}',
                flush=True,
            )
            return 0
        # fall through to play exported file
        return main(
            [
                '--from-bev',
                str(out),
                '--rate',
                str(args.rate),
                '--config',
                str(config_path),
                '--open-k',
                str(args.open_k),
                '--close-k',
                str(args.close_k),
            ]
            + (['--overlay'] if args.overlay else [])
        )

    # Live: camera → BEV once per frame → extract_five_from_bev (photo SSOT)
    n = len(jpegs)
    start_index = 0
    if args.start > 0:
        for i, t in enumerate(stamps):
            if t >= args.start:
                start_index = i
                break
        else:
            start_index = n - 1

    bev_cache: dict[int, np.ndarray] = {}

    def get_bev(i: int):
        hit = bev_cache.get(i)
        if hit is not None:
            return hit
        cam = camera_at(i)
        if cam is None:
            return None
        bev = warp_metric_ipm(cam, ipm)
        if len(bev_cache) > 32:
            bev_cache.clear()
        bev_cache[i] = bev
        return bev

    return run_player(
        course_label=label,
        n=n,
        stamps=stamps,
        get_bev=get_bev,
        rate=args.rate,
        start_index=start_index,
        overlay=args.overlay,
        config_path=config_path,
        open_k=args.open_k,
        close_k=args.close_k,
        course=_resolve_paint_course(label),
    )


if __name__ == '__main__':
    raise SystemExit(main())
