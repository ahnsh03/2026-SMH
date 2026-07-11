#!/usr/bin/env python3
"""Live / offline viewer for metric IPM (team BEV SSOT).

Preferred entry: scripts/vision_tune/tune_bev.py
(also: tune_metric_ipm.py, tune_bev_roi.py without --trapezoid)

Default coverage (locked 2026-07-12):
  crop_top ≈ 39%  →  x_max ≈ 1.5 m
  image bottom    →  x_min ≈ 0.22 m
  ±y_half         →  locked ±0.77 m (y_half_cm=77)

Examples (inside 2026-smh-sim):

  python3 scripts/vision_tune/tune_bev.py
  python3 scripts/vision_tune/tune_bev.py --compare
  python3 scripts/vision_tune/tune_metric_ipm.py --topic /camera/image/compressed
  python3 scripts/vision_tune/tune_bev.py --folder data/captures/sim

Keys: s=save YAML  f=snap y_half to full image width  q/ESC=quit  (folder: n/p)
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

from metric_ipm import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MetricIpmParams,
    draw_crop_overlay,
    draw_metric_guides,
    load_metric_ipm,
    save_metric_ipm,
    warp_metric_ipm,
)

WIN_ORIGIN = 'ipm_tune_origin'
WIN_BEV = 'ipm_tune_bev'
WIN_TRAP = 'ipm_tune_trapezoid'
WIN_CTRL = 'ipm_tune_controls'

PREVIEW_W = 640
PREVIEW_H = 360


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


class TrackbarState:
    def __init__(self, params: MetricIpmParams):
        self.params = params.clamp()

    def sync_from_trackbars(self) -> MetricIpmParams:
        crop_pct = cv2.getTrackbarPos('crop_top_%', WIN_CTRL)
        x_min_cm = cv2.getTrackbarPos('x_min_cm', WIN_CTRL)
        x_max_cm = cv2.getTrackbarPos('x_max_cm', WIN_CTRL)
        y_half_cm = cv2.getTrackbarPos('y_half_cm', WIN_CTRL)
        mpp_mm = cv2.getTrackbarPos('mpp_mm', WIN_CTRL)
        pitch_ddeg = cv2.getTrackbarPos('pitch_x10', WIN_CTRL)
        height_cm = cv2.getTrackbarPos('height_cm', WIN_CTRL)
        self.params = MetricIpmParams(
            hfov_deg=self.params.hfov_deg,
            camera_height_m=max(height_cm, 5) / 100.0,
            pitch_down_deg=max(pitch_ddeg, 0) / 10.0,
            x_min_m=max(x_min_cm, 5) / 100.0,
            x_max_m=max(x_max_cm, 30) / 100.0,
            y_half_width_m=max(y_half_cm, 15) / 100.0,
            meters_per_pixel=max(mpp_mm, 1) / 1000.0,
            crop_top_ratio=crop_pct / 100.0,
            track_width_m=0.35,
        ).clamp()
        return self.params


def _init_ui(state: TrackbarState, compare: bool) -> None:
    cv2.namedWindow(WIN_ORIGIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_ORIGIN, PREVIEW_W, PREVIEW_H)
    cv2.namedWindow(WIN_BEV, cv2.WINDOW_AUTOSIZE)
    if compare:
        cv2.namedWindow(WIN_TRAP, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 520, 260)

    p = state.params
    cv2.createTrackbar(
        'crop_top_%', WIN_CTRL, int(round(p.crop_top_ratio * 100)), 60, lambda _v: None
    )
    cv2.createTrackbar(
        'x_min_cm', WIN_CTRL, int(round(p.x_min_m * 100)), 100, lambda _v: None
    )
    cv2.createTrackbar(
        'x_max_cm', WIN_CTRL, int(round(p.x_max_m * 100)), 300, lambda _v: None
    )
    cv2.createTrackbar(
        'y_half_cm', WIN_CTRL, int(round(p.y_half_width_m * 100)), 200, lambda _v: None
    )
    cv2.createTrackbar(
        'mpp_mm',
        WIN_CTRL,
        int(round(p.meters_per_pixel * 1000)),
        20,
        lambda _v: None,
    )
    cv2.createTrackbar(
        'pitch_x10',
        WIN_CTRL,
        int(round(p.pitch_down_deg * 10)),
        300,
        lambda _v: None,
    )
    cv2.createTrackbar(
        'height_cm',
        WIN_CTRL,
        int(round(p.camera_height_m * 100)),
        40,
        lambda _v: None,
    )


def _snap_full_width(state: TrackbarState, frame_shape: tuple[int, ...]) -> MetricIpmParams:
    h, w = frame_shape[:2]
    state.params = state.params.with_full_image_width(w, h)
    cv2.setTrackbarPos(
        'y_half_cm', WIN_CTRL, int(round(state.params.y_half_width_m * 100))
    )
    print(
        f'y_half → full image width ±{state.params.y_half_width_m:.3f} m '
        f'(bev {state.params.bev_width}x{state.params.bev_height})',
        flush=True,
    )
    return state.params


def _trapezoid_bev(frame: np.ndarray) -> np.ndarray | None:
    try:
        from bev_roi import draw_bev_guides, load_bev_roi, warp_bev
    except ImportError:
        return None
    params = load_bev_roi()
    return draw_bev_guides(warp_bev(frame, params), params)


def _show_frame(
    frame: np.ndarray,
    state: TrackbarState,
    compare: bool,
) -> MetricIpmParams:
    params = state.sync_from_trackbars()
    origin = draw_crop_overlay(frame, params)
    bev = draw_metric_guides(warp_metric_ipm(frame, params), params)
    cv2.imshow(WIN_ORIGIN, _scale_to_preview(origin))
    cv2.imshow(WIN_BEV, bev)
    if compare:
        trap = _trapezoid_bev(frame)
        if trap is not None:
            cv2.imshow(WIN_TRAP, trap)
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
    compare: bool,
) -> int:
    if not frames:
        print('No images to show.', file=sys.stderr)
        return 1
    _init_ui(state, compare)
    idx = 0
    print('Keys: s=save  f=full-width  n/p=next/prev  q=quit')
    while True:
        params = _show_frame(frames[idx], state, compare)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            print(f'Saved metric IPM → {save_metric_ipm(params, config_path)}')
        if key == ord('f'):
            params = _snap_full_width(state, frames[idx].shape)
        if key == ord('n') and len(frames) > 1:
            idx = (idx + 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {labels[idx]}')
        if key == ord('p') and len(frames) > 1:
            idx = (idx - 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {labels[idx]}')
    cv2.destroyAllWindows()
    return 0


def run_topic(
    topic: str,
    state: TrackbarState,
    config_path: Path,
    compare: bool,
) -> int:
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

    class IpmTuneNode(Node):
        def __init__(self):
            super().__init__('metric_ipm_tune')
            self.frame: np.ndarray | None = None
            self.frame_count = 0
            if topic.endswith('/compressed') or 'compressed' in topic:
                self.create_subscription(
                    CompressedImage, topic, self._on_compressed, image_qos
                )
            else:
                self.create_subscription(Image, topic, self._on_raw, image_qos)
            self.get_logger().info(f'Live metric IPM tune on {topic}')

        def _on_compressed(self, msg: CompressedImage) -> None:
            frame = _decode_compressed(bytes(msg.data))
            if frame is not None:
                self.frame = frame
                self.frame_count += 1

        def _on_raw(self, msg: Image) -> None:
            if msg.encoding not in ('bgr8', 'rgb8', 'mono8'):
                self.get_logger().warning(f'unsupported encoding {msg.encoding}')
                return
            h, w = msg.height, msg.width
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'mono8':
                frame = cv2.cvtColor(buf.reshape((h, w)), cv2.COLOR_GRAY2BGR)
            elif msg.encoding == 'rgb8':
                frame = cv2.cvtColor(buf.reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
            else:
                frame = buf.reshape((h, w, 3)).copy()
            self.frame = frame
            self.frame_count += 1

    rclpy.init()
    node = IpmTuneNode()
    _init_ui(state, compare)
    print('Live mode. Keys: s=save  f=full-width  q=quit')
    last_log = 0
    params = state.params
    try:
        while rclpy.ok():
            _spin_drain(node)
            if node.frame is not None:
                params = _show_frame(node.frame, state, compare)
                if node.frame_count - last_log >= 30:
                    print(
                        f'live frames={node.frame_count} '
                        f'bev={params.bev_width}x{params.bev_height} '
                        f'x=[{params.x_min_m:.2f},{params.x_max_m:.2f}] '
                        f'|y|<={params.y_half_width_m:.2f}',
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
                cv2.imshow(
                    WIN_BEV,
                    np.zeros((params.bev_height, params.bev_width, 3), dtype=np.uint8),
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                print(f'Saved metric IPM → {save_metric_ipm(params, config_path)}')
            if key == ord('f') and node.frame is not None:
                params = _snap_full_width(state, node.frame.shape)
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
    parser.add_argument(
        '--compare',
        action='store_true',
        help='also show trapezoid BEV from lane_vision.yaml bev_roi',
    )
    args = parser.parse_args()

    state = TrackbarState(load_metric_ipm(args.config))

    if args.image:
        frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if frame is None:
            print(f'Failed to read {args.image}', file=sys.stderr)
            return 1
        return run_image_sources(
            [frame], [str(args.image)], state, args.config, args.compare
        )

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
        return run_image_sources(frames, labels, state, args.config, args.compare)

    topic = args.topic or '/camera/image/compressed'
    return run_topic(topic, state, args.config, args.compare)


if __name__ == '__main__':
    raise SystemExit(main())
