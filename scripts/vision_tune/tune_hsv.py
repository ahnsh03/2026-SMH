#!/usr/bin/env python3
"""Sim / real-car HSV mask tuner (shared YAML for Won Tae + runtime).

Tune **6** channels on Metric IPM BEV (or camera):
  white / yellow / black_road / red_road / black_cyan / black_cyan_2

Examples (inside 2026-smh-sim after sourcing ROS):

  # All previous bag captures (in+out+glare+black_tune+in_yellow) — keys 1–6
  python3 scripts/vision_tune/tune_hsv.py --from-bag all

  python3 scripts/vision_tune/tune_hsv.py --from-bag out_glare --channel black_cyan
  python3 scripts/vision_tune/tune_hsv.py --from-bag black_road --channel black_road
  python3 scripts/vision_tune/tune_hsv.py --from-bag in_yellow --channel yellow
  python3 scripts/vision_tune/tune_hsv.py --folder data/captures/from_bag/out

Keys:
  1–6   white / yellow / black_road / red_road / black_cyan / black_cyan_2
  b     reset active channel → origin/board field baseline
  d     reset active channel → Won Tae seed defaults
  s     save all channels to lane_vision.yaml
  n/p   next/prev image (folder mode)
  q/ESC quit

Click on the ORIGIN or BEV panel to expand range to sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hsv import (  # noqa: E402
    CHANNEL_NAMES,
    HsvRange,
    board_range,
    default_config_path,
    default_range,
    expand_range_with_sample,
    load_hsv_ranges,
    make_mask,
    overlay_mask,
    save_hsv_ranges,
)
from metric_ipm import (  # noqa: E402
    draw_crop_overlay,
    draw_metric_guides,
    load_metric_ipm,
    warp_metric_ipm,
)
from window_layout import place_window  # noqa: E402

# Single mosaic window (WSLg drops extra HighGUI windows).
WIN_MAIN = 'hsv_tune (origin | BEV | mask)'
WIN_CTRL = WIN_MAIN

PANEL_W = 420
PANEL_H = 360
GAP = 4

CHANNEL_COLORS = {
    'white': (255, 255, 255),
    'yellow': (0, 255, 255),
    'black_road': (80, 80, 80),
    'red_road': (0, 0, 255),
    'black_cyan': (255, 220, 0),  # cyan-ish in BGR preview
    'black_cyan_2': (255, 180, 80),
}

FROM_BAG_DIRS = {
    'in': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'in',
    'out': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'out',
    # OUT LED billboard floor-wash frames only (7 captures, 2026-07-15).
    'out_glare': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'out_glare',
    # Black V / curve off-track samples (IN+OUT frames for black_road tune).
    'black_road': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'black_road_tune_in_out',
    # IN BEV yellow-dense band (~870–955).
    'in_yellow': _REPO_ROOT / 'data' / 'captures' / 'from_bag' / 'in_bev_tune_870_955',
}

# Suggested capture sets per channel (keys into FROM_BAG_DIRS or merge tags).
CHANNEL_CAPTURE_HINT = {
    'white': 'out / in  (흰 테이프)',
    'yellow': 'in / in_yellow',
    'black_road': 'black_road / in / out',
    'red_road': 'out',
    'black_cyan': 'out_glare',
    'black_cyan_2': 'in / in_yellow',
}

# --from-bag all : merge these folders (existing only).
ALL_BAG_KEYS = ('in', 'out', 'out_glare', 'black_road', 'in_yellow')


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f'No such folder: {folder}')
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()
    )


def _decode_compressed(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _fit_panel(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is None or frame.size == 0:
        return out
    h, w = frame.shape[:2]
    if h < 1 or w < 1:
        return out
    scale = min(width / w, height / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_NEAREST)
    x0 = (width - nw) // 2
    y0 = (height - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _label_panel(panel: np.ndarray, title: str) -> None:
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _panel_from_mosaic_x(mx: int) -> str | None:
    if mx < PANEL_W:
        return 'origin'
    if mx < PANEL_W + GAP + PANEL_W:
        return None
    if mx < 2 * PANEL_W + GAP:
        return 'bev'
    return None


def _mosaic_to_frame_xy(
    mx: int,
    my: int,
    which: str,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int] | None:
    if which == 'origin':
        panel_x = mx
    elif which == 'bev':
        panel_x = mx - (PANEL_W + GAP)
    else:
        return None
    if not (0 <= panel_x < PANEL_W and 28 <= my < PANEL_H):
        return None
    # Undo letterbox in _fit_panel.
    scale = min(PANEL_W / frame_w, PANEL_H / frame_h)
    nw = max(1, int(round(frame_w * scale)))
    nh = max(1, int(round(frame_h * scale)))
    x0 = (PANEL_W - nw) // 2
    y0 = (PANEL_H - nh) // 2
    px = panel_x - x0
    py = my - y0
    if not (0 <= px < nw and 0 <= py < nh):
        return None
    fx = int(round(px * frame_w / nw))
    fy = int(round(py * frame_h / nh))
    if not (0 <= fx < frame_w and 0 <= fy < frame_h):
        return None
    return fx, fy


class HsvTuneState:
    def __init__(self, ranges: dict[str, HsvRange]):
        self.ranges = {k: v.clamp() for k, v in ranges.items()}
        self.channel_idx = 0
        self._suppress_trackbar = False
        self._ui_ready = False
        self.last_origin: np.ndarray | None = None
        self.last_bev: np.ndarray | None = None
        self.last_sample: str = ''
        self.frame: np.ndarray | None = None
        self.config_path = default_config_path()

    @property
    def channel(self) -> str:
        return CHANNEL_NAMES[self.channel_idx]

    def active(self) -> HsvRange:
        return self.ranges[self.channel]

    def set_channel(self, idx: int) -> None:
        self.channel_idx = int(np.clip(idx, 0, len(CHANNEL_NAMES) - 1))
        self._push_trackbars()

    def sync_from_trackbars(self) -> HsvRange:
        if self._suppress_trackbar:
            return self.active()
        ch_idx = cv2.getTrackbarPos('channel', WIN_CTRL)
        if ch_idx != self.channel_idx:
            self.channel_idx = int(np.clip(ch_idx, 0, len(CHANNEL_NAMES) - 1))
            self._push_trackbars()
            return self.active()
        rng = HsvRange(
            h_min=cv2.getTrackbarPos('h_min', WIN_CTRL),
            h_max=cv2.getTrackbarPos('h_max', WIN_CTRL),
            s_min=cv2.getTrackbarPos('s_min', WIN_CTRL),
            s_max=cv2.getTrackbarPos('s_max', WIN_CTRL),
            v_min=cv2.getTrackbarPos('v_min', WIN_CTRL),
            v_max=cv2.getTrackbarPos('v_max', WIN_CTRL),
        ).clamp()
        self.ranges[self.channel] = rng
        return rng

    def _push_trackbars(self) -> None:
        self._suppress_trackbar = True
        p = self.active()
        cv2.setTrackbarPos('channel', WIN_CTRL, self.channel_idx)
        cv2.setTrackbarPos('h_min', WIN_CTRL, p.h_min)
        cv2.setTrackbarPos('h_max', WIN_CTRL, p.h_max)
        cv2.setTrackbarPos('s_min', WIN_CTRL, p.s_min)
        cv2.setTrackbarPos('s_max', WIN_CTRL, p.s_max)
        cv2.setTrackbarPos('v_min', WIN_CTRL, p.v_min)
        cv2.setTrackbarPos('v_max', WIN_CTRL, p.v_max)
        self._suppress_trackbar = False

    def on_trackbar(self, _value: int = 0) -> None:
        if not self._ui_ready or self.frame is None:
            return
        _show_frame(self.frame, self, self.config_path)

    def reset_active_default(self) -> None:
        self.ranges[self.channel] = default_range(self.channel)
        self._push_trackbars()

    def reset_active_board(self) -> None:
        self.ranges[self.channel] = board_range(self.channel)
        self._push_trackbars()

    def apply_click(self, which: str, mx: int, my: int) -> None:
        frame = self.last_bev if which == 'bev' else self.last_origin
        if frame is None:
            return
        mapped = _mosaic_to_frame_xy(mx, my, which, frame.shape[1], frame.shape[0])
        if mapped is None:
            return
        fx, fy = mapped
        hsv = cv2.cvtColor(frame[fy : fy + 1, fx : fx + 1], cv2.COLOR_BGR2HSV)[0, 0]
        before = self.active()
        after = expand_range_with_sample(before, hsv)
        self.ranges[self.channel] = after
        self._push_trackbars()
        self.last_sample = (
            f'{self.channel} click HSV=({int(hsv[0])},{int(hsv[1])},{int(hsv[2])})'
        )


def _on_mouse(state: HsvTuneState):
    def _cb(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        which = _panel_from_mosaic_x(x)
        if which is None:
            return
        state.apply_click(which, x, y)
        if state.frame is not None:
            _show_frame(state.frame, state, state.config_path)

    return _cb


def _init_ui(state: HsvTuneState) -> None:
    state._ui_ready = False
    mosaic_w = 3 * PANEL_W + 2 * GAP
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_MAIN, mosaic_w, PANEL_H + 80)
    place_window(WIN_MAIN, 48, 48)

    blank = np.zeros((PANEL_H, mosaic_w, 3), dtype=np.uint8)
    cv2.putText(
        blank,
        'origin | BEV | mask — loading…',
        (24, PANEL_H // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(WIN_MAIN, blank)
    cv2.waitKey(1)
    place_window(WIN_MAIN, 48, 48)

    cb = state.on_trackbar
    cv2.createTrackbar('channel', WIN_CTRL, state.channel_idx, len(CHANNEL_NAMES) - 1, cb)
    for key, vmax in (
        ('h_min', 179),
        ('h_max', 179),
        ('s_min', 255),
        ('s_max', 255),
        ('v_min', 255),
        ('v_max', 255),
    ):
        cv2.createTrackbar(key, WIN_CTRL, 0, vmax, cb)
    state._push_trackbars()
    cv2.setMouseCallback(WIN_MAIN, _on_mouse(state))
    state._ui_ready = True


def _compose_views(
    frame: np.ndarray,
    state: HsvTuneState,
    config_path: Path,
) -> tuple[np.ndarray, HsvRange, float]:
    rng = state.sync_from_trackbars()
    ipm = load_metric_ipm(config_path)
    origin = draw_crop_overlay(frame, ipm)
    bev = warp_metric_ipm(frame, ipm)
    state.last_origin = origin
    state.last_bev = bev

    mask = make_mask(bev, rng)
    color = CHANNEL_COLORS.get(state.channel, (0, 255, 0))
    bev_ov = overlay_mask(draw_metric_guides(bev, ipm), mask, color=color)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cov = 100.0 * float(np.count_nonzero(mask)) / float(mask.size)
    label = (
        f'{state.channel}  H[{rng.h_min},{rng.h_max}] '
        f'S[{rng.s_min},{rng.s_max}] V[{rng.v_min},{rng.v_max}]  cov={cov:.1f}%'
    )
    for img in (origin, bev_ov, mask_bgr):
        cv2.putText(
            img,
            label,
            (8, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if state.last_sample:
        cv2.putText(
            bev_ov,
            state.last_sample,
            (8, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    left = _fit_panel(origin, PANEL_W, PANEL_H)
    mid = _fit_panel(bev_ov, PANEL_W, PANEL_H)
    right = _fit_panel(mask_bgr, PANEL_W, PANEL_H)
    _label_panel(left, 'ORIGIN')
    _label_panel(mid, 'BEV + mask')
    _label_panel(right, 'MASK')
    gap = np.full((PANEL_H, GAP, 3), 40, dtype=np.uint8)
    mosaic = np.hstack([left, gap, mid, gap, right])
    return mosaic, rng, cov


def _show_frame(frame: np.ndarray, state: HsvTuneState, config_path: Path) -> HsvRange:
    state.frame = frame
    mosaic, rng, _cov = _compose_views(frame, state, config_path)
    cv2.imshow(WIN_MAIN, mosaic)
    return rng


def _handle_key(key: int, state: HsvTuneState, config_path: Path) -> str | None:
    """Return 'quit' | 'next' | 'prev' | None."""
    if key in (ord('q'), 27):
        return 'quit'
    if key == ord('s'):
        state.sync_from_trackbars()
        path = save_hsv_ranges(state.ranges, config_path)
        print(f'Saved HSV → {path}')
        return None
    if key == ord('d'):
        state.reset_active_default()
        print(f'Reset {state.channel} → Won Tae seed default')
        return None
    if key == ord('b'):
        state.reset_active_board()
        print(f'Reset {state.channel} → origin/board field baseline')
        return None
    if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')):
        state.set_channel(key - ord('1'))
        print(f'Channel → {state.channel}')
        return None
    if key == ord('n'):
        return 'next'
    if key == ord('p'):
        return 'prev'
    return None


def _list_images_many(folders: list[Path]) -> list[tuple[Path, str]]:
    """Return (path, label) unique by filename, stable order."""

    seen: set[str] = set()
    out: list[tuple[Path, str]] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in _list_images(folder):
            key = path.name
            if key in seen:
                continue
            seen.add(key)
            out.append((path, f'{folder.name}/{path.name}'))
    return out


def run_folder(
    folder: Path | list[Path],
    state: HsvTuneState,
    config_path: Path,
    *,
    label: str | None = None,
) -> int:
    folders = [folder] if isinstance(folder, Path) else list(folder)
    entries = _list_images_many(folders)
    if not entries:
        raise SystemExit(f'No images in {folders}')
    frames = []
    labels = []
    for path, lab in entries:
        img = cv2.imread(str(path))
        if img is not None:
            frames.append(img)
            labels.append(lab)
    if not frames:
        raise SystemExit(f'Failed to load images from {folders}')
    state.config_path = config_path
    _init_ui(state)
    idx = 0
    src = label or (str(folders[0]) if len(folders) == 1 else f'{len(folders)} folders')
    hints = '  '.join(f'{i + 1}:{n}' for i, n in enumerate(CHANNEL_NAMES))
    print(
        f'Folder: {src} ({len(frames)} images)\n'
        f'Channels: {hints}\n'
        f'Hints: '
        + ' | '.join(f'{k}→{v}' for k, v in CHANNEL_CAPTURE_HINT.items())
        + '\n'
        f'Keys: 1-6 channel  b=board  d=seed  s=save  n/p  q=quit  (click ORIGIN/BEV)',
        flush=True,
    )
    try:
        while True:
            _show_frame(frames[idx], state, config_path)
            key = cv2.waitKey(16) & 0xFF
            action = _handle_key(key, state, config_path)
            if action == 'quit':
                break
            if action == 'next' and len(frames) > 1:
                idx = (idx + 1) % len(frames)
                print(f'[{idx + 1}/{len(frames)}] {labels[idx]}', flush=True)
            if action == 'prev' and len(frames) > 1:
                idx = (idx - 1) % len(frames)
                print(f'[{idx + 1}/{len(frames)}] {labels[idx]}', flush=True)
    except KeyboardInterrupt:
        print('\ninterrupted', flush=True)
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


def run_topic(topic: str, state: HsvTuneState, config_path: Path) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'rclpy not found. Inside 2026-smh-sim run:\n'
            '  source /opt/ros/humble/setup.bash\n'
            f'Original error: {exc}'
        ) from exc

    image_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    class HsvTuneNode(Node):
        def __init__(self):
            super().__init__('hsv_tune')
            self.frame: np.ndarray | None = None
            self.create_subscription(
                CompressedImage, topic, self._on_compressed, image_qos
            )
            self.get_logger().info(f'Live HSV tune on {topic}')

        def _on_compressed(self, msg: CompressedImage) -> None:
            frame = _decode_compressed(bytes(msg.data))
            if frame is not None:
                self.frame = frame

    state.config_path = config_path
    _init_ui(state)
    rclpy.init()
    node = HsvTuneNode()
    print('Live mode. Keys: 1-5  b=board  d=seed  s=save  q=quit  (click ORIGIN/BEV)')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.frame is not None:
                _show_frame(node.frame, state, config_path)
            key = cv2.waitKey(1) & 0xFF
            if _handle_key(key, state, config_path) == 'quit':
                break
    except KeyboardInterrupt:
        print('\ninterrupted', flush=True)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    return 0


def _resolve_folders(args: argparse.Namespace) -> tuple[list[Path] | None, str | None]:
    """Return (folders, label) or (None, None) for live topic mode."""

    if args.from_bag is not None:
        key = args.from_bag.strip().lower()
        if key == 'both':
            folders = [FROM_BAG_DIRS['in'], FROM_BAG_DIRS['out']]
            present = [p for p in folders if p.is_dir()]
            if not present:
                raise SystemExit(f'No bag captures under {FROM_BAG_DIRS["in"].parent}')
            return present, 'from_bag/in+out'
        if key == 'all':
            folders = [FROM_BAG_DIRS[k] for k in ALL_BAG_KEYS]
            present = [p for p in folders if p.is_dir()]
            if not present:
                raise SystemExit(
                    f'No bag captures under {FROM_BAG_DIRS["in"].parent} '
                    '(run capture_from_bag.py first)'
                )
            return present, 'from_bag/all(in+out+glare+black+in_yellow)'
        if key not in FROM_BAG_DIRS:
            raise SystemExit(
                f'Unknown --from-bag {args.from_bag!r}; '
                f'use {", ".join(list(FROM_BAG_DIRS) + ["both", "all"])}'
            )
        folder = FROM_BAG_DIRS[key]
        if not folder.is_dir():
            raise SystemExit(f'Missing captures: {folder} (run capture_from_bag.py first)')
        return [folder], f'from_bag/{key}'
    if args.folder is not None:
        return [args.folder.expanduser().resolve()], str(args.folder)
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Tune lane/road HSV masks')
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--folder', type=Path, default=None)
    parser.add_argument(
        '--from-bag',
        choices=(*FROM_BAG_DIRS.keys(), 'both', 'all'),
        default=None,
        help=(
            'Capture set under data/captures/from_bag/. '
            'all = in+out+out_glare+black_road+in_yellow (6 HSV channels). '
            'Keys 1–6: white yellow black_road red_road black_cyan black_cyan_2.'
        ),
    )
    parser.add_argument('--config', type=Path, default=default_config_path())
    parser.add_argument(
        '--channel',
        choices=list(CHANNEL_NAMES),
        default='white',
        help='Initial channel (1–6)',
    )
    args = parser.parse_args(argv)

    ranges = load_hsv_ranges(args.config)
    state = HsvTuneState(ranges)
    state.channel_idx = CHANNEL_NAMES.index(args.channel)

    folders, label = _resolve_folders(args)
    if folders is not None:
        return run_folder(folders, state, args.config, label=label)
    return run_topic(args.topic, state, args.config)


if __name__ == '__main__':
    raise SystemExit(main())
