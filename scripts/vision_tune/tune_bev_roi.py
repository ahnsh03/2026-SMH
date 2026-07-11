#!/usr/bin/env python3
"""Phase 0: tune extended-trapezoid BEV ROI with trackbars.

Shows three windows: camera/original, ROI overlay, BEV warp.
Supports a still image, an image folder, or a ROS2 compressed camera topic.

Examples (inside 2026-smh-sim or host with DISPLAY + ROS2):

  python3 scripts/vision_tune/tune_bev_roi.py --image path/to.png
  python3 scripts/vision_tune/tune_bev_roi.py --topic /camera/image/compressed
  python3 scripts/vision_tune/tune_bev_roi.py --folder ../../data/captures/sim

Keys:
  s  save YAML (config/lane_vision.yaml)
  n  next image (folder mode)
  p  previous image (folder mode)
  q / ESC  quit

See docs/lane-drive-strategy.md §4.
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

from bev_roi import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    BevRoiParams,
    draw_roi_overlay,
    load_bev_roi,
    save_bev_roi,
    warp_bev,
)

WIN_ORIGIN = 'bev_tune_origin'
WIN_ROI = 'bev_tune_roi'
WIN_BEV = 'bev_tune_bev'
WIN_CTRL = 'bev_tune_controls'


def _list_images(folder: Path) -> list[Path]:
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()
    )


def _decode_compressed(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class TrackbarState:
    def __init__(self, params: BevRoiParams):
        self.params = params.clamp()

    def sync_from_trackbars(self) -> BevRoiParams:
        crop_pct = cv2.getTrackbarPos('crop_top_%', WIN_CTRL)
        bottom_pct = cv2.getTrackbarPos('bottom_half_%', WIN_CTRL)
        bev_w = cv2.getTrackbarPos('bev_w', WIN_CTRL)
        bev_h = cv2.getTrackbarPos('bev_h', WIN_CTRL)
        self.params = BevRoiParams(
            crop_top_ratio=crop_pct / 100.0,
            bottom_half_width_ratio=max(bottom_pct, 50) / 100.0,
            bev_width=max(bev_w, 64),
            bev_height=max(bev_h, 64),
        ).clamp()
        return self.params


def _init_ui(state: TrackbarState) -> None:
    cv2.namedWindow(WIN_ORIGIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_ROI, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_BEV, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 160)
    p = state.params
    cv2.createTrackbar('crop_top_%', WIN_CTRL, int(round(p.crop_top_ratio * 100)), 50, lambda _v: None)
    cv2.createTrackbar(
        'bottom_half_%',
        WIN_CTRL,
        int(round(p.bottom_half_width_ratio * 100)),
        300,
        lambda _v: None,
    )
    cv2.createTrackbar('bev_w', WIN_CTRL, p.bev_width, 640, lambda _v: None)
    cv2.createTrackbar('bev_h', WIN_CTRL, p.bev_height, 640, lambda _v: None)


def _show_frame(frame: np.ndarray, state: TrackbarState) -> BevRoiParams:
    params = state.sync_from_trackbars()
    roi = draw_roi_overlay(frame, params)
    bev = warp_bev(frame, params)
    cv2.imshow(WIN_ORIGIN, frame)
    cv2.imshow(WIN_ROI, roi)
    cv2.imshow(WIN_BEV, bev)
    return params


def run_image_sources(
    frames: list[np.ndarray],
    labels: list[str],
    state: TrackbarState,
    config_path: Path,
) -> int:
    if not frames:
        print('No images to show.', file=sys.stderr)
        return 1
    _init_ui(state)
    idx = 0
    print('Keys: s=save  n/p=next/prev  q=quit')
    while True:
        frame = frames[idx]
        params = _show_frame(frame, state)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            path = save_bev_roi(params, config_path)
            print(f'Saved BEV ROI → {path}')
        if key == ord('n') and len(frames) > 1:
            idx = (idx + 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {labels[idx]}')
        if key == ord('p') and len(frames) > 1:
            idx = (idx - 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {labels[idx]}')
    cv2.destroyAllWindows()
    return 0


def run_topic(topic: str, state: TrackbarState, config_path: Path) -> int:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage

    class BevTuneNode(Node):
        def __init__(self):
            super().__init__('bev_roi_tune')
            self.frame: np.ndarray | None = None
            self.create_subscription(CompressedImage, topic, self._on_image, 10)
            self.get_logger().info(f'Tuning BEV ROI on {topic}')

        def _on_image(self, msg: CompressedImage) -> None:
            frame = _decode_compressed(bytes(msg.data))
            if frame is not None:
                self.frame = frame

    rclpy.init()
    node = BevTuneNode()
    _init_ui(state)
    print('Keys: s=save  q=quit')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.frame is not None:
                params = _show_frame(node.frame, state)
            else:
                params = state.sync_from_trackbars()
                blank = np.zeros((180, 320, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    'waiting for image...',
                    (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WIN_ORIGIN, blank)
                cv2.imshow(WIN_ROI, blank)
                cv2.imshow(WIN_BEV, blank)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                path = save_bev_roi(params, config_path)
                print(f'Saved BEV ROI → {path}')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', type=Path, help='single image file')
    src.add_argument('--folder', type=Path, help='folder of images')
    src.add_argument(
        '--topic',
        type=str,
        help='ROS2 CompressedImage topic (e.g. /camera/image/compressed)',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f'YAML path (default: {DEFAULT_CONFIG_PATH})',
    )
    args = parser.parse_args()

    state = TrackbarState(load_bev_roi(args.config))

    if args.topic:
        return run_topic(args.topic, state, args.config)

    if args.image:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            print(f'Failed to read {args.image}', file=sys.stderr)
            return 1
        return run_image_sources([frame], [str(args.image)], state, args.config)

    paths = _list_images(args.folder)
    if not paths:
        print(f'No images in {args.folder}', file=sys.stderr)
        return 1
    frames = []
    labels = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
            labels.append(str(path))
    return run_image_sources(frames, labels, state, args.config)


if __name__ == '__main__':
    raise SystemExit(main())
