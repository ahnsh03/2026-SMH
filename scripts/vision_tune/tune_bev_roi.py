#!/usr/bin/env python3
"""Phase 0: tune extended-trapezoid BEV ROI with live camera + trackbars.

Default source is the live ROS2 camera topic (real-time). Still images /
folders remain available for offline tuning.

Windows:
  bev_tune_origin / bev_tune_roi — initial size 640×360 (same as D-Racer Camera)
  bev_tune_bev — sized to bev_w×bev_h trackbars (not free-resized by hand)

Examples (inside 2026-smh-sim, after source /opt/ros/humble/setup.bash):

  python3 scripts/vision_tune/tune_bev_roi.py
  python3 scripts/vision_tune/tune_bev_roi.py --topic /camera/image/compressed
  python3 scripts/vision_tune/tune_bev_roi.py --image path/to.png

Keys: s=save YAML  q/ESC=quit  (folder: n/p next/prev)

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
    draw_bev_guides,
    draw_roi_overlay,
    load_bev_roi,
    save_bev_roi,
    warp_bev,
)

WIN_ORIGIN = 'bev_tune_origin'
WIN_ROI = 'bev_tune_roi'
WIN_BEV = 'bev_tune_bev'
WIN_CTRL = 'bev_tune_controls'

# Match sim_bringup D-Racer Camera preview (320×180 shown at 2×).
PREVIEW_W = 640
PREVIEW_H = 360


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f'No such folder: {folder}\n'
            'Capture with hotkey first:\n'
            '  source /opt/ros/humble/setup.bash\n'
            '  python3 scripts/vision_tune/capture_camera.py --out data/captures/sim'
        )
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()
    )


def _decode_compressed(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _scale_to_preview(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (PREVIEW_W, PREVIEW_H), interpolation=cv2.INTER_NEAREST)


class TrackbarState:
    def __init__(self, params: BevRoiParams):
        self.params = params.clamp()
        self._last_bev_size: tuple[int, int] | None = None

    def sync_from_trackbars(self) -> BevRoiParams:
        crop_pct = cv2.getTrackbarPos('crop_top_%', WIN_CTRL)
        bottom_pct = cv2.getTrackbarPos('bottom_half_%', WIN_CTRL)
        bev_w = cv2.getTrackbarPos('bev_w', WIN_CTRL)
        bev_h = cv2.getTrackbarPos('bev_h', WIN_CTRL)
        guide_half = cv2.getTrackbarPos('guide_half_px', WIN_CTRL)
        self.params = BevRoiParams(
            crop_top_ratio=crop_pct / 100.0,
            bottom_half_width_ratio=max(bottom_pct, 50) / 100.0,
            bev_width=max(bev_w, 64),
            bev_height=max(bev_h, 64),
            guide_half_width_px=max(guide_half, 5),
            track_width_m=0.35,
        ).clamp()
        return self.params


def _init_ui(state: TrackbarState) -> None:
    # Camera-like windows: free-ish NORMAL at D-Racer preview size.
    for name in (WIN_ORIGIN, WIN_ROI):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, PREVIEW_W, PREVIEW_H)

    # BEV follows parameter size; AUTOSIZE so the client area matches the image.
    cv2.namedWindow(WIN_BEV, cv2.WINDOW_AUTOSIZE)

    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 480, 200)

    p = state.params
    cv2.createTrackbar(
        'crop_top_%', WIN_CTRL, int(round(p.crop_top_ratio * 100)), 50, lambda _v: None
    )
    # Large expansion so small crop_top can still flatten the ground plane.
    cv2.createTrackbar(
        'bottom_half_%',
        WIN_CTRL,
        int(round(p.bottom_half_width_ratio * 100)),
        1500,
        lambda _v: None,
    )
    cv2.createTrackbar('bev_w', WIN_CTRL, p.bev_width, 640, lambda _v: None)
    cv2.createTrackbar('bev_h', WIN_CTRL, p.bev_height, 640, lambda _v: None)
    cv2.createTrackbar(
        'guide_half_px', WIN_CTRL, p.guide_half_width_px, 200, lambda _v: None
    )
    state._last_bev_size = (p.bev_width, p.bev_height)


def _show_frame(frame: np.ndarray, state: TrackbarState) -> BevRoiParams:
    params = state.sync_from_trackbars()
    roi = draw_roi_overlay(frame, params)
    bev = draw_bev_guides(warp_bev(frame, params), params)

    cv2.imshow(WIN_ORIGIN, _scale_to_preview(frame))
    cv2.imshow(WIN_ROI, _scale_to_preview(roi))
    cv2.imshow(WIN_BEV, bev)
    state._last_bev_size = (params.bev_width, params.bev_height)
    return params


def _spin_drain(node, rounds: int = 8) -> None:
    import rclpy

    for _ in range(rounds):
        rclpy.spin_once(node, timeout_sec=0.0)


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
    print('Keys: s=save  n/p=next/prev  q=quit  (offline stills — use default topic for live)')
    while True:
        params = _show_frame(frames[idx], state)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            print(f'Saved BEV ROI → {save_bev_roi(params, config_path)}')
        if key == ord('n') and len(frames) > 1:
            idx = (idx + 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {labels[idx]}')
        if key == ord('p') and len(frames) > 1:
            idx = (idx - 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {labels[idx]}')
    cv2.destroyAllWindows()
    return 0


def run_topic(topic: str, state: TrackbarState, config_path: Path) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage, Image
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

    class BevTuneNode(Node):
        def __init__(self):
            super().__init__('bev_roi_tune')
            self.frame: np.ndarray | None = None
            self.frame_count = 0
            if topic.endswith('/compressed') or 'compressed' in topic:
                self.create_subscription(
                    CompressedImage, topic, self._on_compressed, image_qos
                )
            else:
                self.create_subscription(Image, topic, self._on_raw, image_qos)
            self.get_logger().info(
                f'Live BEV ROI tune on {topic} '
                f'(origin/roi windows {PREVIEW_W}x{PREVIEW_H})'
            )

        def _on_compressed(self, msg: CompressedImage) -> None:
            frame = _decode_compressed(bytes(msg.data))
            if frame is not None:
                self.frame = frame
                self.frame_count += 1

        def _on_raw(self, msg: Image) -> None:
            # Prefer numpy path without cv_bridge dependency for raw.
            if msg.encoding not in ('bgr8', 'rgb8', 'mono8'):
                self.get_logger().warning(f'unsupported encoding {msg.encoding}')
                return
            h, w = msg.height, msg.width
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'mono8':
                frame = buf.reshape((h, w))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif msg.encoding == 'rgb8':
                frame = buf.reshape((h, w, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame = buf.reshape((h, w, 3)).copy()
            self.frame = frame
            self.frame_count += 1

    rclpy.init()
    node = BevTuneNode()
    _init_ui(state)
    print('Live mode. Keys: s=save  q=quit')
    print('Make sure sim-bringup is running so the camera topic publishes.')
    last_log = 0
    try:
        while rclpy.ok():
            _spin_drain(node)
            if node.frame is not None:
                params = _show_frame(node.frame, state)
                if node.frame_count - last_log >= 30:
                    h, w = node.frame.shape[:2]
                    print(
                        f'live frames={node.frame_count} src={w}x{h} '
                        f'bev={params.bev_width}x{params.bev_height}',
                        flush=True,
                    )
                    last_log = node.frame_count
            else:
                params = state.sync_from_trackbars()
                blank = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    f'waiting for {topic} ...',
                    (24, PREVIEW_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WIN_ORIGIN, blank)
                cv2.imshow(WIN_ROI, blank)
                cv2.imshow(
                    WIN_BEV,
                    np.zeros((params.bev_height, params.bev_width, 3), dtype=np.uint8),
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                print(f'Saved BEV ROI → {save_bev_roi(params, config_path)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument('--image', type=Path, help='single still image (offline)')
    src.add_argument('--folder', type=Path, help='folder of stills (offline)')
    src.add_argument(
        '--topic',
        type=str,
        default=None,
        help='ROS2 image topic (default: /camera/image/compressed)',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f'YAML path (default: {DEFAULT_CONFIG_PATH})',
    )
    args = parser.parse_args()

    state = TrackbarState(load_bev_roi(args.config))

    if args.image:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            print(f'Failed to read {args.image}', file=sys.stderr)
            return 1
        return run_image_sources([frame], [str(args.image)], state, args.config)

    if args.folder:
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

    topic = args.topic or '/camera/image/compressed'
    return run_topic(topic, state, args.config)


if __name__ == '__main__':
    raise SystemExit(main())
