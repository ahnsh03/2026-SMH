#!/usr/bin/env python3
"""Play IN/OUT course rosbag and hotkey-capture frames (Gazebo-free).

Joystick bags live under ``bags/`` (copied into the workspace for the sim
container). Default course aliases:

  in  → bags/in_course   (← monorepo data/bag_20260711_150234)
  out → bags/out_course  (← monorepo data/bag_20260711_144948)

Examples (inside 2026-smh-sim after sourcing ROS):

  python3 scripts/vision_tune/capture_from_bag.py in
  python3 scripts/vision_tune/capture_from_bag.py out --rate 0.4
  python3 scripts/vision_tune/capture_from_bag.py --bag bags/in_course \\
      --out data/captures/from_bag/in

Keys (focus the player window):
  c           save current frame as PNG
  SPACE       pause / resume  (starts paused)
  ← / →       previous / next frame (also , / .)
  [ / ]       slower / faster playback
  Home / r    jump to start
  End         jump to last frame
  q / ESC     quit
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

try:
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from sensor_msgs.msg import CompressedImage
except ModuleNotFoundError as exc:
    raise SystemExit(
        'rosbag2_py / rclpy not found. Run inside 2026-smh-sim (or the board) after:\n'
        '  source /opt/ros/humble/setup.bash\n'
        '  source install/setup.bash\n'
        f'Original error: {exc}'
    ) from exc

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from window_layout import place_window  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_W = 640
PREVIEW_H = 360
CAMERA_TOPIC = '/camera/image/compressed'

COURSE_BAGS: dict[str, Path] = {
    'in': _REPO_ROOT / 'bags' / 'in_course',
    'out': _REPO_ROOT / 'bags' / 'out_course',
}

# Optional monorepo mount (docker-compose: ../data → /host_data)
HOST_DATA_ALIASES: dict[str, Path] = {
    'in': Path('/host_data/bag_20260711_150234'),
    'out': Path('/host_data/bag_20260711_144948'),
}


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')


def _host_ids() -> tuple[int, int] | None:
    env_uid = os.environ.get('HOST_UID') or os.environ.get('SUDO_UID')
    env_gid = os.environ.get('HOST_GID') or os.environ.get('SUDO_GID')
    if env_uid is not None:
        return int(env_uid), int(env_gid if env_gid is not None else env_uid)
    return None


def _chown_to_host(path: Path, fallback_dir: Path | None = None) -> None:
    if os.geteuid() != 0:
        return
    ids = _host_ids()
    if ids is None and fallback_dir is not None:
        probe = fallback_dir if fallback_dir.exists() else fallback_dir.parent
        try:
            st = probe.stat()
            if st.st_uid != 0:
                ids = (st.st_uid, st.st_gid)
        except OSError:
            ids = None
    if ids is None:
        ids = (1000, 1000)
    try:
        os.chown(path, ids[0], ids[1])
    except OSError:
        pass


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _chown_to_host(out_dir, out_dir.parent)
    return out_dir


def _is_bag_dir(path: Path) -> bool:
    return path.is_dir() and (path / 'metadata.yaml').is_file()


def resolve_bag(course: str | None, bag: Path | None) -> Path:
    if bag is not None:
        path = bag.expanduser()
        if not path.is_absolute():
            path = (_REPO_ROOT / path).resolve()
        else:
            path = path.resolve()
        if not _is_bag_dir(path):
            raise SystemExit(f'Not a rosbag2 directory (missing metadata.yaml): {path}')
        return path

    if course is None:
        raise SystemExit('Provide course (in|out) or --bag PATH')

    key = course.strip().lower()
    if key not in COURSE_BAGS:
        raise SystemExit(f'Unknown course {course!r}; use in / out or --bag')

    candidates = [COURSE_BAGS[key], HOST_DATA_ALIASES[key]]
    for cand in candidates:
        if _is_bag_dir(cand):
            return cand.resolve()

    raise SystemExit(
        'Bag not found. Expected one of:\n'
        + '\n'.join(f'  - {c}' for c in candidates)
        + '\nCopy from monorepo data/ or mount ../data at /host_data.'
    )


def load_camera_jpegs(
    bag_dir: Path,
    topic: str,
) -> tuple[list[bytes], list[float]]:
    """Load JPEG payloads + relative timestamps (decode on demand)."""
    storage = StorageOptions(uri=str(bag_dir), storage_id='sqlite3')
    converter = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )
    reader = SequentialReader()
    reader.open(storage, converter)

    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in topics:
        available = ', '.join(sorted(topics)) or '(none)'
        raise SystemExit(f'Topic {topic} not in bag. Available: {available}')

    jpegs: list[bytes] = []
    stamps: list[float] = []
    t0_ns: int | None = None

    while reader.has_next():
        name, data, t_ns = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, CompressedImage)
        payload = bytes(msg.data)
        if not payload:
            continue
        if t0_ns is None:
            t0_ns = int(t_ns)
        jpegs.append(payload)
        stamps.append((int(t_ns) - t0_ns) * 1e-9)

    if not jpegs:
        raise SystemExit(f'No frames on {topic} in {bag_dir}')
    return jpegs, stamps


def decode_jpeg(payload: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)


def save_frame(frame: np.ndarray, out_dir: Path, index: int, t_rel: float) -> Path:
    name = f'frame_{_stamp()}_{index:04d}.png'
    path = out_dir / name
    cv2.imwrite(str(path), frame)
    meta = path.with_suffix('.txt')
    h, w = frame.shape[:2]
    meta.write_text(
        f'width={w}\nheight={h}\nsource=bag\nbag_index={index}\n'
        f't_rel_s={t_rel:.6f}\n',
        encoding='utf-8',
    )
    _chown_to_host(path, out_dir)
    _chown_to_host(meta, out_dir)
    return path


def draw_hud(
    frame: np.ndarray,
    *,
    index: int,
    n: int,
    t_rel: float,
    rate: float,
    paused: bool,
    saved: int,
    course: str,
) -> np.ndarray:
    view = cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)
    status = 'PAUSE' if paused else f'x{rate:.2f}'
    lines = [
        f'{course}  [{index + 1}/{n}]  t={t_rel:.2f}s  {status}  saved={saved}',
        'c=save  SPACE=play/pause  ,/.=step  []=rate  r=restart  q=quit',
    ]
    y = 22
    for text in lines:
        cv2.putText(
            view,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return view


def _open_window(title: str) -> str:
    """Create a visible HighGUI window (WSLg-friendly)."""
    # Fresh name avoids a stuck previous WINDOW_NORMAL handle after Ctrl+C.
    win = f'{title} [{_stamp()[-10:]}]'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, PREVIEW_W, PREVIEW_H)
    place_window(win, 48, 48)
    # Force a first paint — some WSLg sessions otherwise show no window.
    blank = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        'loading preview…',
        (24, PREVIEW_H // 2),
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


def run_player(
    jpegs: list[bytes],
    stamps: list[float],
    out_dir: Path,
    *,
    course_label: str,
    rate: float,
    start_index: int,
) -> int:
    out_dir = _ensure_out_dir(out_dir)
    n = len(jpegs)
    idx = max(0, min(start_index, n - 1))
    # Start paused so a static frame is visible immediately (playback can hide
    # a briefly flashing window on some WSLg sessions).
    paused = True
    saved = 0
    rate = max(0.05, float(rate))
    decode_cache: dict[int, np.ndarray] = {}

    win = _open_window(f'bag_capture_{course_label}')
    print(
        f'Loaded {n} frames → {out_dir}\n'
        f'Window "{win}" (starts PAUSED). SPACE=play  c=save  q=quit',
        flush=True,
    )
    if not os.environ.get('DISPLAY'):
        print('WARNING: DISPLAY is empty — OpenCV window will not appear.', flush=True)

    def frame_at(i: int) -> np.ndarray | None:
        hit = decode_cache.get(i)
        if hit is not None:
            return hit
        img = decode_jpeg(jpegs[i])
        if img is None:
            return None
        # Keep a small LRU-ish cache so seeking stays snappy without huge RAM.
        if len(decode_cache) > 64:
            decode_cache.clear()
        decode_cache[i] = img
        return img

    last_advance = time.perf_counter()
    try:
        while True:
            frame = frame_at(idx)
            if frame is None:
                idx = min(n - 1, idx + 1)
                continue
            t_rel = stamps[idx]
            view = draw_hud(
                frame,
                index=idx,
                n=n,
                t_rel=t_rel,
                rate=rate,
                paused=paused,
                saved=saved,
                course=course_label,
            )
            cv2.imshow(win, view)
            # waitKeyEx: arrow keys need the full code (not & 0xFF).
            key = cv2.waitKeyEx(20)
            key8 = key & 0xFF

            if key8 in (ord('q'), 27) or key in (ord('q'), 27):
                break
            if key8 == ord('c'):
                path = save_frame(frame, out_dir, idx, t_rel)
                saved += 1
                print(f'saved {path.name} ({saved} total)', flush=True)
            elif key8 == ord(' '):
                paused = not paused
                last_advance = time.perf_counter()
            elif key in (65361, 2424832, 81) or key8 == ord(','):  # ←
                idx = max(0, idx - 1)
                paused = True
            elif key in (65363, 2555904, 83) or key8 == ord('.'):  # →
                idx = min(n - 1, idx + 1)
                paused = True
            elif key8 == ord('['):
                rate = max(0.05, rate / 1.25)
                print(f'rate={rate:.2f}', flush=True)
            elif key8 == ord(']'):
                rate = min(8.0, rate * 1.25)
                print(f'rate={rate:.2f}', flush=True)
            elif key8 == ord('r') or key in (65360, 2359296):  # Home
                idx = 0
                paused = True
            elif key in (65367, 2293760):  # End
                idx = n - 1
                paused = True

            if not paused and n > 1:
                now = time.perf_counter()
                if idx + 1 < n:
                    dt = max(1e-3, stamps[idx + 1] - stamps[idx])
                else:
                    dt = max(1e-3, stamps[idx] - stamps[idx - 1]) if idx > 0 else 1 / 30
                need = dt / rate
                if now - last_advance >= need:
                    if idx + 1 < n:
                        idx += 1
                    else:
                        paused = True
                    last_advance = now
    except KeyboardInterrupt:
        print('\ninterrupted', flush=True)
    finally:
        print(f'saved_total={saved} dir={out_dir}', flush=True)
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Play IN/OUT course bag and capture frames with hotkeys.',
    )
    parser.add_argument(
        'course',
        nargs='?',
        choices=('in', 'out'),
        help='Course alias: in (=bags/in_course) or out (=bags/out_course)',
    )
    parser.add_argument(
        '--bag',
        type=Path,
        default=None,
        help='rosbag2 directory (overrides course alias)',
    )
    parser.add_argument(
        '--topic',
        default=CAMERA_TOPIC,
        help=f'camera topic (default: {CAMERA_TOPIC})',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='PNG output dir (default: data/captures/from_bag/<course>)',
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=0.5,
        help='playback speed vs bag time (default: 0.5)',
    )
    parser.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='start at bag-relative seconds (default: 0)',
    )
    args = parser.parse_args()

    bag_dir = resolve_bag(args.course, args.bag)
    course_label = args.course or bag_dir.name

    if args.out is not None:
        out_dir = args.out.expanduser()
        if not out_dir.is_absolute():
            out_dir = (_REPO_ROOT / out_dir).resolve()
    else:
        out_dir = (_REPO_ROOT / 'data' / 'captures' / 'from_bag' / course_label).resolve()

    print(f'Loading {bag_dir} …', flush=True)
    jpegs, stamps = load_camera_jpegs(bag_dir, args.topic)
    start_index = 0
    if args.start > 0:
        for i, t in enumerate(stamps):
            if t >= args.start:
                start_index = i
                break
        else:
            start_index = len(jpegs) - 1

    return run_player(
        jpegs,
        stamps,
        out_dir,
        course_label=course_label,
        rate=args.rate,
        start_index=start_index,
    )


if __name__ == '__main__':
    raise SystemExit(main())
