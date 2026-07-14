#!/usr/bin/env python3
"""Sim / real-car HSV mask tuner (shared YAML for Won Tae + runtime).

Tune white / yellow / black_road / red_road ranges on Metric IPM BEV (or
camera view). Click a pixel to expand the active range around that sample.
Saves ``config/lane_vision.yaml`` → ``hsv:``.

Precise competition values may still be refined by Won Tae; this tool is the
shared field/sim interface.

Examples (inside 2026-smh-sim after source /opt/ros/humble/setup.bash):

  python3 scripts/vision_tune/tune_hsv.py
  python3 scripts/vision_tune/tune_hsv.py --folder data/captures/sim
  python3 scripts/vision_tune/tune_hsv.py --topic /camera/image/compressed

Keys:
  1–4   select channel (white/yellow/black/red)
  d     reset active channel to Won Tae seed defaults
  s     save all channels to lane_vision.yaml
  n/p   next/prev image (folder mode)
  q/ESC quit

Click on ``hsv_tune_bev`` or ``hsv_tune_origin`` to expand range to sample.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from hsv import (  # noqa: E402
    CHANNEL_NAMES,
    HsvRange,
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

WIN_ORIGIN = 'hsv_tune_origin'
WIN_BEV = 'hsv_tune_bev'
WIN_MASK = 'hsv_tune_mask'
WIN_CTRL = 'hsv_tune_controls'
PREVIEW_W = 640
PREVIEW_H = 360

CHANNEL_COLORS = {
    'white': (255, 255, 255),
    'yellow': (0, 255, 255),
    'black_road': (80, 80, 80),
    'red_road': (0, 0, 255),
}


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


def _scale_to_preview(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)


def _preview_to_frame_xy(
    px: int,
    py: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int] | None:
    if frame_w <= 0 or frame_h <= 0:
        return None
    x = int(round(px * frame_w / PREVIEW_W))
    y = int(round(py * frame_h / PREVIEW_H))
    if not (0 <= x < frame_w and 0 <= y < frame_h):
        return None
    return x, y


class HsvTuneState:
    def __init__(self, ranges: dict[str, HsvRange]):
        self.ranges = {k: v.clamp() for k, v in ranges.items()}
        self.channel_idx = 0
        self._suppress_trackbar = False
        self.last_origin: np.ndarray | None = None
        self.last_bev: np.ndarray | None = None
        self.last_sample: str = ''

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

    def reset_active_default(self) -> None:
        self.ranges[self.channel] = default_range(self.channel)
        self._push_trackbars()

    def apply_click(self, which: str, x: int, y: int) -> None:
        frame = self.last_bev if which == 'bev' else self.last_origin
        if frame is None:
            return
        mapped = _preview_to_frame_xy(x, y, frame.shape[1], frame.shape[0])
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


def _on_mouse_factory(state: HsvTuneState, which: str):
    def _cb(event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state.apply_click(which, x, y)

    return _cb


def _init_ui(state: HsvTuneState) -> None:
    for name in (WIN_ORIGIN, WIN_BEV, WIN_MASK):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, PREVIEW_W, PREVIEW_H)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 320)
    cv2.createTrackbar('channel', WIN_CTRL, state.channel_idx, 3, lambda *_: None)
    for key, vmax in (
        ('h_min', 179),
        ('h_max', 179),
        ('s_min', 255),
        ('s_max', 255),
        ('v_min', 255),
        ('v_max', 255),
    ):
        cv2.createTrackbar(key, WIN_CTRL, 0, vmax, lambda *_: None)
    state._push_trackbars()
    cv2.setMouseCallback(WIN_ORIGIN, _on_mouse_factory(state, 'origin'))
    cv2.setMouseCallback(WIN_BEV, _on_mouse_factory(state, 'bev'))


def _show_frame(frame: np.ndarray, state: HsvTuneState, config_path: Path) -> None:
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
        f'S[{rng.s_min},{rng.s_max}] V[{rng.v_min},{rng.v_max}]  '
        f'cov={cov:.1f}%'
    )
    for img in (origin, bev_ov, mask_bgr):
        cv2.putText(
            img,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if state.last_sample:
        cv2.putText(
            bev_ov,
            state.last_sample,
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow(WIN_ORIGIN, _scale_to_preview(origin))
    cv2.imshow(WIN_BEV, _scale_to_preview(bev_ov))
    cv2.imshow(WIN_MASK, _scale_to_preview(mask_bgr))


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
    if key in (ord('1'), ord('2'), ord('3'), ord('4')):
        state.set_channel(key - ord('1'))
        print(f'Channel → {state.channel}')
        return None
    if key == ord('n'):
        return 'next'
    if key == ord('p'):
        return 'prev'
    return None


def run_folder(folder: Path, state: HsvTuneState, config_path: Path) -> int:
    paths = _list_images(folder)
    if not paths:
        raise SystemExit(f'No images in {folder}')
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit(f'Failed to load images from {folder}')
    _init_ui(state)
    idx = 0
    print('Keys: 1-4 channel  d=default  s=save  n/p  q=quit  (click to sample)')
    while True:
        _show_frame(frames[idx], state, config_path)
        key = cv2.waitKey(30) & 0xFF
        action = _handle_key(key, state, config_path)
        if action == 'quit':
            break
        if action == 'next' and len(frames) > 1:
            idx = (idx + 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {paths[idx].name}')
        if action == 'prev' and len(frames) > 1:
            idx = (idx - 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {paths[idx].name}')
    cv2.destroyAllWindows()
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
        # Newest frame only — a deeper queue shows stale frames (see tune_metric_ipm).
        depth=1,
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

    _init_ui(state)
    rclpy.init()
    node = HsvTuneNode()
    print('Keys: 1-4 channel  d=default  s=save  q=quit  (click to sample)')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.frame is not None:
                _show_frame(node.frame, state, config_path)
            key = cv2.waitKey(1) & 0xFF
            if _handle_key(key, state, config_path) == 'quit':
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Tune lane/road HSV masks')
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--folder', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=default_config_path())
    parser.add_argument(
        '--channel',
        choices=list(CHANNEL_NAMES),
        default='white',
        help='Initial channel',
    )
    args = parser.parse_args(argv)

    ranges = load_hsv_ranges(args.config)
    state = HsvTuneState(ranges)
    state.channel_idx = CHANNEL_NAMES.index(args.channel)
    if args.folder is not None:
        return run_folder(args.folder, state, args.config)
    return run_topic(args.topic, state, args.config)


if __name__ == '__main__':
    raise SystemExit(main())
