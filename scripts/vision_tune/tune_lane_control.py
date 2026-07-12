#!/usr/bin/env python3
"""Sim / real-car control-gain tuner for white-lane planner.

Tune P / EMA / rate-limit / look-ahead / cruise overlay. Saves
``config/lane_control.yaml``. HSV masks are tuned separately with
``tune_hsv.py``; this tool visualizes stub white detections + planner output.

Examples (inside 2026-smh-sim after source /opt/ros/humble/setup.bash):

  python3 scripts/vision_tune/tune_lane_control.py
  python3 scripts/vision_tune/tune_lane_control.py --folder data/captures/sim
  python3 scripts/vision_tune/tune_lane_control.py --topic /camera/image/compressed

Keys: s=save YAML  r=reset planner state  n/p=next/prev (folder)  q/ESC=quit
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

from metric_ipm import (  # noqa: E402
    draw_crop_overlay,
    draw_metric_guides,
    load_metric_ipm,
    warp_metric_ipm,
)
from inference.modules.lane_detection import detect_markings  # noqa: E402
from inference.modules.lane_planner import (  # noqa: E402
    LaneControlParams,
    LanePlanner,
    default_control_config_path,
    load_control_params,
    plan,
    save_control_params,
)

WIN_ORIGIN = 'lane_ctrl_origin'
WIN_BEV = 'lane_ctrl_bev'
WIN_CTRL = 'lane_ctrl_controls'
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
    def __init__(self, params: LaneControlParams, cruise: float):
        self.params = params.clamp()
        self.cruise = cruise

    def sync_from_trackbars(self) -> LaneControlParams:
        la_cm = cv2.getTrackbarPos('lookahead_cm', WIN_CTRL)
        kp_x10 = cv2.getTrackbarPos('kp_x10', WIN_CTRL)
        ema_pct = cv2.getTrackbarPos('ema_%', WIN_CTRL)
        rate_x100 = cv2.getTrackbarPos('rate_x100', WIN_CTRL)
        cruise_pct = cv2.getTrackbarPos('cruise_%', WIN_CTRL)
        slow_pct = cv2.getTrackbarPos('slow_scale_%', WIN_CTRL)
        self.cruise = max(cruise_pct, 1) / 100.0
        self.params = LaneControlParams(
            lookahead_x_m=max(la_cm, 30) / 100.0,
            kp=max(kp_x10, 1) / 10.0,
            ema_alpha=max(ema_pct, 5) / 100.0,
            steer_rate_limit=max(rate_x100, 1) / 100.0,
            max_steer=self.params.max_steer,
            track_half_width_m=self.params.track_half_width_m,
            steer_slowdown_thresh=self.params.steer_slowdown_thresh,
            steer_slowdown_scale=max(slow_pct, 20) / 100.0,
            min_confidence=self.params.min_confidence,
            hold_decay=self.params.hold_decay,
        ).clamp()
        return self.params


def _init_ui(state: TrackbarState) -> None:
    cv2.namedWindow(WIN_ORIGIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_ORIGIN, PREVIEW_W, PREVIEW_H)
    cv2.namedWindow(WIN_BEV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_BEV, PREVIEW_W, PREVIEW_H)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 420, 280)
    p = state.params
    cv2.createTrackbar('lookahead_cm', WIN_CTRL, int(round(p.lookahead_x_m * 100)), 150, lambda *_: None)
    cv2.createTrackbar('kp_x10', WIN_CTRL, int(round(p.kp * 10)), 80, lambda *_: None)
    cv2.createTrackbar('ema_%', WIN_CTRL, int(round(p.ema_alpha * 100)), 100, lambda *_: None)
    cv2.createTrackbar('rate_x100', WIN_CTRL, int(round(p.steer_rate_limit * 100)), 100, lambda *_: None)
    cv2.createTrackbar('cruise_%', WIN_CTRL, int(round(state.cruise * 100)), 100, lambda *_: None)
    cv2.createTrackbar(
        'slow_scale_%',
        WIN_CTRL,
        int(round(p.steer_slowdown_scale * 100)),
        100,
        lambda *_: None,
    )


def _draw_polyline_bev(
    bev: np.ndarray,
    xy: np.ndarray,
    params,
    color: tuple[int, int, int],
) -> None:
    if xy.shape[0] < 2:
        return
    # Inverse of bev_uv_to_xy
    p = params.clamp()
    u = ((p.y_half_width_m - xy[:, 1]) / p.meters_per_pixel).astype(np.int32)
    v = ((p.x_max_m - xy[:, 0]) / p.meters_per_pixel).astype(np.int32)
    pts = np.stack([u, v], axis=1).reshape(-1, 1, 2)
    cv2.polylines(bev, [pts], False, color, 2, cv2.LINE_AA)


def _show_frame(frame: np.ndarray, state: TrackbarState, planner: LanePlanner) -> None:
    params = state.sync_from_trackbars()
    planner.set_params(params)

    ipm = load_metric_ipm(_REPO_ROOT / 'config' / 'lane_vision.yaml')
    bev = warp_metric_ipm(frame, ipm)
    bev_vis = draw_metric_guides(bev, ipm)

    dets = detect_markings(frame)
    result = plan(dets, planner)

    left = dets.white_left()
    right = dets.white_right()
    if left is not None:
        _draw_polyline_bev(bev_vis, left.xy(), ipm, (0, 0, 255))
    if right is not None:
        _draw_polyline_bev(bev_vis, right.xy(), ipm, (255, 0, 0))

    # Look-ahead center marker
    y_c = None
    from inference.modules.lane_planner import centerline_y_at_lookahead

    y_c, _ = centerline_y_at_lookahead(
        dets, params.lookahead_x_m, params.track_half_width_m
    )
    if y_c is not None:
        u = int(round((ipm.y_half_width_m - y_c) / ipm.meters_per_pixel))
        v = int(round((ipm.x_max_m - params.lookahead_x_m) / ipm.meters_per_pixel))
        cv2.circle(bev_vis, (u, v), 6, (0, 255, 0), -1)

    origin = draw_crop_overlay(frame, ipm)
    h, w = origin.shape[:2]
    bar_w = int(abs(result.steering_offset) * (w // 2))
    mid = w // 2
    color = (0, 165, 255) if result.steering_offset >= 0 else (255, 128, 0)
    if result.steering_offset >= 0:
        cv2.rectangle(origin, (mid, h - 24), (mid + bar_w, h - 8), color, -1)
    else:
        cv2.rectangle(origin, (mid - bar_w, h - 24), (mid, h - 8), color, -1)
    throttle = state.cruise * result.throttle_scale if result.confidence > 0.1 else 0.0
    cv2.putText(
        origin,
        f'steer={result.steering_offset:+.2f}  conf={result.confidence:.2f}  '
        f'throttle~={throttle:.2f}',
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bev_vis,
        f'kp={params.kp:.1f} ema={params.ema_alpha:.2f} '
        f'rate={params.steer_rate_limit:.2f} la={params.lookahead_x_m:.2f}m',
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(WIN_ORIGIN, _scale_to_preview(origin))
    cv2.imshow(WIN_BEV, _scale_to_preview(bev_vis))


def run_folder(folder: Path, state: TrackbarState, config_path: Path) -> int:
    paths = _list_images(folder)
    if not paths:
        raise SystemExit(f'No images in {folder}')
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit(f'Failed to load images from {folder}')
    planner = LanePlanner(state.params)
    _init_ui(state)
    idx = 0
    print('Keys: s=save  r=reset  n/p=next/prev  q=quit')
    while True:
        _show_frame(frames[idx], state, planner)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            state.sync_from_trackbars()
            print(f'Saved control → {save_control_params(state.params, config_path)}')
        if key == ord('r'):
            planner.reset()
            print('Planner state reset')
        if key == ord('n') and len(frames) > 1:
            idx = (idx + 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {paths[idx].name}')
        if key == ord('p') and len(frames) > 1:
            idx = (idx - 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {paths[idx].name}')
    cv2.destroyAllWindows()
    return 0


def run_topic(topic: str, state: TrackbarState, config_path: Path) -> int:
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

    class CtrlTuneNode(Node):
        def __init__(self):
            super().__init__('lane_control_tune')
            self.frame: np.ndarray | None = None
            self.create_subscription(
                CompressedImage, topic, self._on_compressed, image_qos
            )
            self.get_logger().info(f'Live lane-control tune on {topic}')

        def _on_compressed(self, msg: CompressedImage) -> None:
            frame = _decode_compressed(bytes(msg.data))
            if frame is not None:
                self.frame = frame

    planner = LanePlanner(state.params)
    _init_ui(state)
    rclpy.init()
    node = CtrlTuneNode()
    print('Keys: s=save  r=reset  q=quit')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.frame is not None:
                _show_frame(node.frame, state, planner)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                state.sync_from_trackbars()
                print(f'Saved control → {save_control_params(state.params, config_path)}')
            if key == ord('r'):
                planner.reset()
                print('Planner state reset')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Tune white-lane planner gains')
    parser.add_argument(
        '--topic',
        default='/camera/image/compressed',
        help='Compressed image topic (live mode)',
    )
    parser.add_argument(
        '--folder',
        type=Path,
        default=None,
        help='Offline image folder (skips ROS)',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=default_control_config_path(),
        help='lane_control.yaml path',
    )
    parser.add_argument(
        '--cruise',
        type=float,
        default=0.35,
        help='Display cruise throttle (not written to yaml)',
    )
    args = parser.parse_args(argv)

    params = load_control_params(args.config)
    state = TrackbarState(params, cruise=args.cruise)
    if args.folder is not None:
        return run_folder(args.folder, state, args.config)
    return run_topic(args.topic, state, args.config)


if __name__ == '__main__':
    raise SystemExit(main())
