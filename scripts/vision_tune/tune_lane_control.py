#!/usr/bin/env python3
"""Live control-gain tuner for single-color lane planner (sim / real).

Trackbars adjust planner gains in real time. With ``--drive``, also publishes
``/control`` so you can tune while the car moves (do **not** run
``inference_node`` at the same time).

Smoothing chain (verified in ``lane_planner.step``):
  raw = -Kp * y_error  →  EMA(α)  →  rate-limit(|Δ|/step)  →  steering

Early curve cut → lower lookahead_cm.
S-curve exit / understeer → raise rate_x100 and max_steer_%, lower cruise_%.
Speed → cruise_% trackbar (live in --drive).

Examples (inside 2026-smh-sim):

  source /opt/ros/humble/setup.bash
  source /workspace/install/setup.bash   # if built

  python3 scripts/vision_tune/tune_lane_control.py --drive
  python3 scripts/vision_tune/tune_lane_control.py --drive --color yellow

Keys: s=save YAML  r=reset planner  space=pause throttle(drive)  q/ESC=quit(+stop)
"""

from __future__ import annotations

import argparse
import sys
import time
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
from inference.modules import lane_detection  # noqa: E402
from inference.modules.lane_detection import detect_markings  # noqa: E402
from inference.modules.lane_planner import (  # noqa: E402
    LaneControlParams,
    LanePlanner,
    centerline_y_at_lookahead,
    default_control_config_path,
    load_control_params,
    plan,
    save_control_params,
)
from inference.types import LaneMarking  # noqa: E402

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
        self.paused = False

    def sync_from_trackbars(self) -> LaneControlParams:
        cruise_pct = cv2.getTrackbarPos('cruise_%', WIN_CTRL)
        la_cm = cv2.getTrackbarPos('lookahead_cm', WIN_CTRL)
        kp_x10 = cv2.getTrackbarPos('kp_x10', WIN_CTRL)
        ema_pct = cv2.getTrackbarPos('ema_%', WIN_CTRL)
        rate_x100 = cv2.getTrackbarPos('rate_x100', WIN_CTRL)
        max_pct = cv2.getTrackbarPos('max_steer_%', WIN_CTRL)
        slow_pct = cv2.getTrackbarPos('slow_scale_%', WIN_CTRL)
        half_cm = cv2.getTrackbarPos('half_w_cm', WIN_CTRL)
        hold_pct = cv2.getTrackbarPos('hold_%', WIN_CTRL)
        color_idx = cv2.getTrackbarPos('color_0w1y', WIN_CTRL)
        follow = 'yellow' if color_idx >= 1 else 'white'
        self.cruise = max(cruise_pct, 0) / 100.0
        self.params = LaneControlParams(
            lookahead_x_m=max(la_cm, 30) / 100.0,
            kp=max(kp_x10, 1) / 10.0,
            ema_alpha=max(ema_pct, 5) / 100.0,
            steer_rate_limit=max(rate_x100, 1) / 100.0,
            max_steer=max(max_pct, 10) / 100.0,
            track_half_width_m=max(half_cm, 5) / 100.0,
            steer_slowdown_thresh=self.params.steer_slowdown_thresh,
            steer_slowdown_scale=max(slow_pct, 20) / 100.0,
            min_confidence=self.params.min_confidence,
            hold_decay=max(hold_pct, 50) / 100.0,
            follow_color=follow,
        ).clamp()
        return self.params


def _init_ui(state: TrackbarState) -> None:
    cv2.namedWindow(WIN_ORIGIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_ORIGIN, PREVIEW_W, PREVIEW_H)
    cv2.namedWindow(WIN_BEV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_BEV, PREVIEW_W, PREVIEW_H)
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, 520, 420)
    p = state.params
    # Speed first — used immediately in --drive.
    cv2.createTrackbar('cruise_%', WIN_CTRL, int(round(state.cruise * 100)), 100, lambda *_: None)
    cv2.createTrackbar(
        'lookahead_cm', WIN_CTRL, int(round(p.lookahead_x_m * 100)), 150, lambda *_: None
    )
    cv2.createTrackbar('kp_x10', WIN_CTRL, int(round(p.kp * 10)), 80, lambda *_: None)
    cv2.createTrackbar('ema_%', WIN_CTRL, int(round(p.ema_alpha * 100)), 100, lambda *_: None)
    # Allow up to 0.50 / frame so continuous curves can build steer quickly.
    cv2.createTrackbar(
        'rate_x100', WIN_CTRL, int(round(p.steer_rate_limit * 100)), 50, lambda *_: None
    )
    cv2.createTrackbar(
        'max_steer_%', WIN_CTRL, int(round(p.max_steer * 100)), 100, lambda *_: None
    )
    cv2.createTrackbar(
        'slow_scale_%',
        WIN_CTRL,
        int(round(p.steer_slowdown_scale * 100)),
        100,
        lambda *_: None,
    )
    cv2.createTrackbar(
        'half_w_cm',
        WIN_CTRL,
        int(round(p.track_half_width_m * 100)),
        40,
        lambda *_: None,
    )
    cv2.createTrackbar(
        'hold_%', WIN_CTRL, int(round(p.hold_decay * 100)), 99, lambda *_: None
    )
    cv2.createTrackbar(
        'color_0w1y',
        WIN_CTRL,
        1 if p.follow_color == 'yellow' else 0,
        1,
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
    p = params.clamp()
    u = ((p.y_half_width_m - xy[:, 1]) / p.meters_per_pixel).astype(np.int32)
    v = ((p.x_max_m - xy[:, 0]) / p.meters_per_pixel).astype(np.int32)
    pts = np.stack([u, v], axis=1).reshape(-1, 1, 2)
    cv2.polylines(bev, [pts], False, color, 2, cv2.LINE_AA)


def _show_frame(
    frame: np.ndarray,
    state: TrackbarState,
    planner: LanePlanner,
) -> tuple[float, float]:
    """Update UI; return (steering, throttle) for optional /control publish."""
    params = state.sync_from_trackbars()
    planner.set_params(params)
    lane_detection.set_follow_color(params.follow_color)

    ipm = load_metric_ipm(_REPO_ROOT / 'config' / 'lane_vision.yaml')
    bev = warp_metric_ipm(frame, ipm)
    bev_vis = draw_metric_guides(bev, ipm)

    dets = detect_markings(frame, follow_color=params.follow_color)
    result = plan(dets, planner)
    dbg = planner.last_debug

    color_id = (
        LaneMarking.COLOR_YELLOW
        if params.follow_color == 'yellow'
        else LaneMarking.COLOR_WHITE
    )
    left, right = dets.pair_for_color(color_id)
    if left is not None:
        _draw_polyline_bev(bev_vis, left.xy(), ipm, (0, 0, 255))
    if right is not None:
        _draw_polyline_bev(bev_vis, right.xy(), ipm, (255, 0, 0))

    y_c, _ = centerline_y_at_lookahead(
        dets,
        params.lookahead_x_m,
        params.track_half_width_m,
        follow_color=params.follow_color,
    )
    if y_c is not None:
        u = int(round((ipm.y_half_width_m - y_c) / ipm.meters_per_pixel))
        v = int(round((ipm.x_max_m - params.lookahead_x_m) / ipm.meters_per_pixel))
        cv2.circle(bev_vis, (u, v), 6, (0, 255, 0), -1)
        cv2.line(bev_vis, (0, v), (bev_vis.shape[1] - 1, v), (0, 255, 0), 1)

    origin = draw_crop_overlay(frame, ipm)
    h, w = origin.shape[:2]
    mid = w // 2

    def _bar(val: float, y0: int, y1: int, col: tuple[int, int, int]) -> None:
        bar_w = int(abs(val) * (w // 2))
        if val >= 0:
            cv2.rectangle(origin, (mid, y0), (mid + bar_w, y1), col, -1)
        else:
            cv2.rectangle(origin, (mid - bar_w, y0), (mid, y1), col, -1)

    # raw (cyan) / ema (magenta) / out (orange) — proves smoothing chain
    _bar(float(dbg.get('raw', 0.0)), h - 54, h - 42, (255, 255, 0))
    _bar(float(dbg.get('ema', 0.0)), h - 40, h - 28, (255, 0, 255))
    _bar(result.steering_offset, h - 26, h - 10, (0, 165, 255))

    throttle = 0.0
    if not state.paused and result.confidence > 0.1:
        throttle = state.cruise * float(np.clip(result.throttle_scale, 0.0, 1.0))
    mode = 'DRIVE' if state.cruise > 0 else 'VIEW'
    if state.paused:
        mode = 'PAUSED'
    cv2.putText(
        origin,
        f'[{mode}] {params.follow_color}  steer={result.steering_offset:+.2f}  '
        f'conf={result.confidence:.2f}  thr={throttle:.2f}',
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        origin,
        f'smooth raw={dbg.get("raw", 0):+.2f} → ema={dbg.get("ema", 0):+.2f} '
        f'→ out={result.steering_offset:+.2f}  y_c={dbg.get("y_c", float("nan")):+.3f}m',
        (8, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bev_vis,
        f'kp={params.kp:.1f} ema={params.ema_alpha:.2f} '
        f'rate={params.steer_rate_limit:.2f} max={params.max_steer:.2f} '
        f'la={params.lookahead_x_m:.2f}m cruise={state.cruise:.2f}',
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bev_vis,
        'S-curve exit? ↑rate_x100 + ↑max_steer_% ; early cut? ↓lookahead',
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow(WIN_ORIGIN, _scale_to_preview(origin))
    cv2.imshow(WIN_BEV, _scale_to_preview(bev_vis))
    return float(result.steering_offset), float(throttle)


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
        if key == ord('p') and len(frames) > 1:
            idx = (idx - 1) % len(frames)
    cv2.destroyAllWindows()
    return 0


def run_topic(
    topic: str,
    state: TrackbarState,
    config_path: Path,
    *,
    drive: bool,
    control_topic: str,
) -> int:
    try:
        import rclpy
        from control_msgs.msg import Control
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
            self.control_pub = None
            if drive:
                self.control_pub = self.create_publisher(Control, control_topic, 10)
            self.get_logger().info(
                f'Live lane-control tune on {topic}'
                + (f'  DRIVE→{control_topic}' if drive else '  (view only)')
            )

        def _on_compressed(self, msg: CompressedImage) -> None:
            frame = _decode_compressed(bytes(msg.data))
            if frame is not None:
                self.frame = frame

        def publish_cmd(self, steering: float, throttle: float) -> None:
            if self.control_pub is None:
                return
            msg = Control()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.steering = float(np.clip(steering, -1.0, 1.0))
            msg.throttle = float(np.clip(throttle, -1.0, 1.0))
            self.control_pub.publish(msg)

        def publish_stop(self, n: int = 8) -> None:
            for _ in range(n):
                self.publish_cmd(0.0, 0.0)
                time.sleep(0.02)

    planner = LanePlanner(state.params)
    _init_ui(state)
    rclpy.init()
    node = CtrlTuneNode()
    if drive:
        print(
            'DRIVE mode: publishes /control. Do NOT run inference_node.\n'
            'Keys: s=save  r=reset  space=pause throttle  q=quit(+stop)'
        )
    else:
        print('VIEW mode (no /control). Keys: s=save  r=reset  q=quit')
        print('For live drive tuning: add --drive')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.frame is not None:
                steer, thr = _show_frame(node.frame, state, planner)
                if drive:
                    node.publish_cmd(steer, thr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                state.sync_from_trackbars()
                print(f'Saved control → {save_control_params(state.params, config_path)}')
            if key == ord('r'):
                planner.reset()
                print('Planner state reset')
            if key == ord(' ') and drive:
                state.paused = not state.paused
                if state.paused:
                    node.publish_cmd(0.0, 0.0)
                print('PAUSED' if state.paused else 'RESUMED')
    finally:
        if drive:
            try:
                node.publish_stop()
            except Exception:  # noqa: BLE001
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Tune lane planner gains (optional live /control drive)'
    )
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
        default=0.30,
        help='Cruise throttle (trackbar + --drive publish)',
    )
    parser.add_argument(
        '--drive',
        action='store_true',
        help='Publish /control while tuning (stop inference_node first)',
    )
    parser.add_argument(
        '--control-topic',
        default='/control',
        help='Control topic when --drive',
    )
    parser.add_argument(
        '--color',
        choices=('white', 'yellow'),
        default=None,
        help='Initial follow color (else yaml follow_color)',
    )
    args = parser.parse_args(argv)

    params = load_control_params(args.config)
    if args.color is not None:
        params.follow_color = args.color
        params = params.clamp()
    state = TrackbarState(params, cruise=args.cruise)
    lane_detection.set_follow_color(params.follow_color)

    if args.folder is not None:
        return run_folder(args.folder, state, args.config)
    return run_topic(
        args.topic,
        state,
        args.config,
        drive=args.drive,
        control_topic=args.control_topic,
    )


if __name__ == '__main__':
    raise SystemExit(main())
