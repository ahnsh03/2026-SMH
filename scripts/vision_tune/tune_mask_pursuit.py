#!/usr/bin/env python3
"""Live ``hybrid`` / ``mask_p`` / ``pp`` gain tuner (MainPlanner) with OpenCV trackbars.

Sim bringup ON, **sim-auto / inference_node OFF** — this script owns ``/control``
when ``--drive`` is set. Watch ``Lane drive`` + ``mask_p controls``.

``trk_0p1m2h``: 0=pp, 1=mask_p, 2=hybrid (SSOT gated PP+mask).

Keys
----
  s     save current knobs → ``config/main_planner.yaml``
  r     reload YAML + reset trackbars
  space pause / resume (drive only; sends stop while paused)
  t     teleport to ``inout_fork`` (S-curve start)
  g     teleport to ``start``
  w     reposition windows
  q/ESC quit (+ stop)

Knob cheat-sheet (lag / centering)
----------------------------------
  far_blend↑     — see curve earlier (too high → cut inside)
  near_band↓     — later turn-in / less foresight
  alpha↑ / rate↑ — snappier steer follow
  path_corr=1    — white CTE+heading keep lane center
  cte / head↑    — stronger recenter (needs path_corr)
  steer_k↑       — stronger COM→steer gain

Example (inside 2026-smh-sim)::

  source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash
  cd /workspace
  PYTHONUNBUFFERED=1 python3 scripts/vision_tune/tune_mask_pursuit.py --drive
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_INFERENCE_SRC = _REPO_ROOT / 'src' / 'inference'
_DRIVE = _REPO_ROOT / 'scripts' / 'drive_test'
for p in (_SCRIPT_DIR, _INFERENCE_SRC, _DRIVE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from window_layout import place_windows  # noqa: E402
from viz_util import apply_lane_viz  # noqa: E402
from inference.modules import lane_detection as ld  # noqa: E402
from inference.pipeline import (  # noqa: E402
    MainPlanner,
    PlannerConfig,
    default_planner_config_path,
    load_planner_config,
)
from inference.types import DrivingState  # noqa: E402

WIN_CTRL = 'mask_p controls'
WIN_HUD = 'mask_p hud'
CTRL_W, CTRL_H = 560, 720
HUD_W, HUD_H = 440, 320


def _teleport(spawn: str) -> None:
    script = _REPO_ROOT / 'scripts' / 'teleport_spawn_pose.py'
    subprocess.check_call([sys.executable, str(script), spawn], cwd=str(_REPO_ROOT))


def _place_ui() -> None:
    place_windows(
        (WIN_CTRL, WIN_HUD),
        widths=(CTRL_W, HUD_W),
        heights=(CTRL_H, HUD_H),
    )


class MaskTuneState:
    def __init__(self, cfg: PlannerConfig, config_path: Path):
        self.cfg = cfg
        self.config_path = config_path
        self.paused = False
        self._baseline = cfg

    def apply_to_planner(self, planner: MainPlanner) -> None:
        planner.config = self.cfg
        # Keep NORMAL so mask_p track is exercised (not stuck in fork FSM).
        planner.state = DrivingState.NORMAL

    def sync_from_trackbars(self) -> PlannerConfig:
        steer_k = cv2.getTrackbarPos('steer_k_x100', WIN_CTRL) / 100.0
        alpha = max(cv2.getTrackbarPos('alpha_%', WIN_CTRL), 1) / 100.0
        near = max(cv2.getTrackbarPos('near_%', WIN_CTRL), 10) / 100.0
        far = max(cv2.getTrackbarPos('far_%', WIN_CTRL), near * 100) / 100.0
        far = float(np.clip(far, near, 1.0))
        far_blend = cv2.getTrackbarPos('far_blend_%', WIN_CTRL) / 100.0
        path_corr = cv2.getTrackbarPos('path_corr', WIN_CTRL) >= 1
        tracker_idx = cv2.getTrackbarPos('trk_0p1m2h', WIN_CTRL)
        tracker = ('pp', 'mask_p', 'hybrid')[int(np.clip(tracker_idx, 0, 2))]
        rate = max(cv2.getTrackbarPos('rate_x10', WIN_CTRL), 1) / 10.0
        max_cmd = max(cv2.getTrackbarPos('max_cmd_%', WIN_CTRL), 10) / 100.0
        cruise = cv2.getTrackbarPos('cruise_%', WIN_CTRL) / 100.0
        curve = cv2.getTrackbarPos('curve_%', WIN_CTRL) / 100.0
        cte = cv2.getTrackbarPos('cte_x100', WIN_CTRL) / 100.0
        head = cv2.getTrackbarPos('head_x100', WIN_CTRL) / 100.0
        half_cm = max(cv2.getTrackbarPos('corr_half_cm', WIN_CTRL), 5)
        e_db = cv2.getTrackbarPos('e_dead_x100', WIN_CTRL) / 100.0
        e_lo = cv2.getTrackbarPos('e_lo_x100', WIN_CTRL) / 100.0
        e_hi = max(cv2.getTrackbarPos('e_hi_x100', WIN_CTRL), int(e_lo * 100) + 1) / 100.0
        k_lo = cv2.getTrackbarPos('k_lo_x100', WIN_CTRL) / 100.0
        k_hi = max(cv2.getTrackbarPos('k_hi_x100', WIN_CTRL), int(k_lo * 100) + 1) / 100.0
        self.cfg = replace(
            self.cfg,
            normal_tracker=tracker,
            mask_steer_k=max(0.05, steer_k),
            mask_steer_alpha=float(np.clip(alpha, 0.01, 1.0)),
            mask_near_band_ratio=float(np.clip(near, 0.1, 1.0)),
            mask_far_band_ratio=float(np.clip(far, near, 1.0)),
            mask_far_blend=float(np.clip(far_blend, 0.0, 0.8)),
            mask_use_path_correction=bool(path_corr),
            steering_rate_limit_per_sec=float(rate),
            max_steering_command=float(np.clip(max_cmd, 0.1, 1.0)),
            cruise_throttle=float(np.clip(cruise, 0.0, 1.0)),
            curve_throttle=float(np.clip(curve, 0.0, 1.0)),
            cte_gain=max(0.0, cte),
            heading_gain=max(0.0, head),
            mask_corridor_half_width_m=half_cm / 100.0,
            mask_error_deadband=max(0.0, e_db),
            mask_blend_error_lo=max(0.0, e_lo),
            mask_blend_error_hi=max(e_lo + 0.01, e_hi),
            mask_blend_curvature_lo=max(0.0, k_lo),
            mask_blend_curvature_hi=max(k_lo + 0.01, k_hi),
        )
        return self.cfg

    def push_trackbars(self) -> None:
        c = self.cfg
        trk = {'pp': 0, 'mask_p': 1, 'hybrid': 2}.get(
            str(c.normal_tracker or 'hybrid').lower(), 2
        )
        cv2.setTrackbarPos('trk_0p1m2h', WIN_CTRL, trk)
        cv2.setTrackbarPos('steer_k_x100', WIN_CTRL, int(round(c.mask_steer_k * 100)))
        cv2.setTrackbarPos('alpha_%', WIN_CTRL, int(round(c.mask_steer_alpha * 100)))
        cv2.setTrackbarPos('near_%', WIN_CTRL, int(round(c.mask_near_band_ratio * 100)))
        cv2.setTrackbarPos('far_%', WIN_CTRL, int(round(c.mask_far_band_ratio * 100)))
        cv2.setTrackbarPos('far_blend_%', WIN_CTRL, int(round(c.mask_far_blend * 100)))
        cv2.setTrackbarPos('path_corr', WIN_CTRL, 1 if c.mask_use_path_correction else 0)
        cv2.setTrackbarPos(
            'rate_x10', WIN_CTRL, int(round(c.steering_rate_limit_per_sec * 10))
        )
        cv2.setTrackbarPos(
            'max_cmd_%', WIN_CTRL, int(round(c.max_steering_command * 100))
        )
        cv2.setTrackbarPos('cruise_%', WIN_CTRL, int(round(c.cruise_throttle * 100)))
        cv2.setTrackbarPos('curve_%', WIN_CTRL, int(round(c.curve_throttle * 100)))
        cv2.setTrackbarPos('cte_x100', WIN_CTRL, int(round(c.cte_gain * 100)))
        cv2.setTrackbarPos('head_x100', WIN_CTRL, int(round(c.heading_gain * 100)))
        cv2.setTrackbarPos(
            'corr_half_cm',
            WIN_CTRL,
            int(round(c.mask_corridor_half_width_m * 100)),
        )
        cv2.setTrackbarPos(
            'e_dead_x100', WIN_CTRL, int(round(c.mask_error_deadband * 100))
        )
        cv2.setTrackbarPos(
            'e_lo_x100', WIN_CTRL, int(round(c.mask_blend_error_lo * 100))
        )
        cv2.setTrackbarPos(
            'e_hi_x100', WIN_CTRL, int(round(c.mask_blend_error_hi * 100))
        )
        cv2.setTrackbarPos(
            'k_lo_x100', WIN_CTRL, int(round(c.mask_blend_curvature_lo * 100))
        )
        cv2.setTrackbarPos(
            'k_hi_x100', WIN_CTRL, int(round(c.mask_blend_curvature_hi * 100))
        )

    def reload(self) -> None:
        self.cfg = load_planner_config(self.config_path, route_mode='out')
        self._baseline = self.cfg
        self.push_trackbars()

    def save_yaml(self) -> Path:
        """Patch scalar keys in-place so YAML comments survive."""
        path = self.config_path
        text = path.read_text(encoding='utf-8') if path.is_file() else ''
        c = self.cfg
        replacements: list[tuple[str, str]] = [
            ('normal', str(c.normal_tracker)),
            ('steer_k', f'{c.mask_steer_k:.3f}'),
            ('steer_alpha', f'{c.mask_steer_alpha:.3f}'),
            ('near_band_ratio', f'{c.mask_near_band_ratio:.3f}'),
            ('far_band_ratio', f'{c.mask_far_band_ratio:.3f}'),
            ('far_blend', f'{c.mask_far_blend:.3f}'),
            (
                'use_path_correction',
                'true' if c.mask_use_path_correction else 'false',
            ),
            ('corridor_half_width_m', f'{c.mask_corridor_half_width_m:.3f}'),
            (
                'steering_rate_limit_per_sec',
                f'{c.steering_rate_limit_per_sec:.2f}',
            ),
            ('max_steering_command', f'{c.max_steering_command:.3f}'),
            ('cte_gain', f'{c.cte_gain:.3f}'),
            ('heading_gain', f'{c.heading_gain:.3f}'),
            ('cruise_throttle', f'{c.cruise_throttle:.3f}'),
            ('curve_throttle', f'{c.curve_throttle:.3f}'),
            ('error_deadband', f'{c.mask_error_deadband:.3f}'),
            ('blend_error_lo', f'{c.mask_blend_error_lo:.3f}'),
            ('blend_error_hi', f'{c.mask_blend_error_hi:.3f}'),
            ('blend_curvature_lo', f'{c.mask_blend_curvature_lo:.3f}'),
            ('blend_curvature_hi', f'{c.mask_blend_curvature_hi:.3f}'),
            ('curve_lookahead_m', f'{c.curve_lookahead_m:.3f}'),
        ]

        def _set_key(src: str, key: str, value: str) -> str:
            import re

            pat = re.compile(
                rf'^([ \t]*{re.escape(key)}\s*:\s*)([^#\n]+)(.*)$',
                re.MULTILINE,
            )
            m = pat.search(src)
            if not m:
                # Insert under mask_pursuit if missing (far_* etc.).
                insert_pat = re.compile(
                    r'(^mask_pursuit:\s*\n)', re.MULTILINE
                )
                if key in (
                    'far_band_ratio',
                    'far_blend',
                    'use_path_correction',
                ) and insert_pat.search(src):
                    return insert_pat.sub(
                        rf'\1  {key}: {value}\n', src, count=1
                    )
                print(f'[save] skip missing key: {key}', flush=True)
                return src
            return pat.sub(rf'\g<1>{value}\g<3>', src, count=1)

        for key, value in replacements:
            text = _set_key(text, key, value)
        path.write_text(text, encoding='utf-8')
        print(f'[save] patched {path}', flush=True)
        return path


def _init_ui(state: MaskTuneState) -> None:
    cv2.namedWindow(WIN_CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CTRL, CTRL_W, CTRL_H)
    cv2.namedWindow(WIN_HUD, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_HUD, HUD_W, HUD_H)
    _place_ui()
    # Trackbar ranges for hybrid / mask_p / pp live tuning.
    cv2.createTrackbar('trk_0p1m2h', WIN_CTRL, 2, 2, lambda *_: None)
    cv2.createTrackbar('steer_k_x100', WIN_CTRL, 155, 300, lambda *_: None)
    cv2.createTrackbar('alpha_%', WIN_CTRL, 28, 100, lambda *_: None)
    cv2.createTrackbar('near_%', WIN_CTRL, 55, 100, lambda *_: None)
    cv2.createTrackbar('far_%', WIN_CTRL, 90, 100, lambda *_: None)
    cv2.createTrackbar('far_blend_%', WIN_CTRL, 28, 80, lambda *_: None)
    cv2.createTrackbar('path_corr', WIN_CTRL, 0, 1, lambda *_: None)
    cv2.createTrackbar('rate_x10', WIN_CTRL, 90, 200, lambda *_: None)
    cv2.createTrackbar('max_cmd_%', WIN_CTRL, 95, 100, lambda *_: None)
    cv2.createTrackbar('cruise_%', WIN_CTRL, 34, 80, lambda *_: None)
    cv2.createTrackbar('curve_%', WIN_CTRL, 22, 60, lambda *_: None)
    cv2.createTrackbar('cte_x100', WIN_CTRL, 10, 40, lambda *_: None)
    cv2.createTrackbar('head_x100', WIN_CTRL, 30, 60, lambda *_: None)
    cv2.createTrackbar('corr_half_cm', WIN_CTRL, 38, 60, lambda *_: None)
    cv2.createTrackbar('e_dead_x100', WIN_CTRL, 4, 20, lambda *_: None)
    cv2.createTrackbar('e_lo_x100', WIN_CTRL, 8, 40, lambda *_: None)
    cv2.createTrackbar('e_hi_x100', WIN_CTRL, 35, 60, lambda *_: None)
    cv2.createTrackbar('k_lo_x100', WIN_CTRL, 40, 150, lambda *_: None)
    cv2.createTrackbar('k_hi_x100', WIN_CTRL, 120, 200, lambda *_: None)
    state.push_trackbars()


def _draw_hud(output, cfg: PlannerConfig, paused: bool) -> None:
    canvas = np.zeros((HUD_H, HUD_W, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)
    cmd = output.command
    dbg = output.debug or {}
    lines = [
        f"{'PAUSED' if paused else 'DRIVE'}  trk={cfg.normal_tracker}",
        f"src={output.path_source.value}  steer={cmd.steering:+.3f} thr={cmd.throttle:.2f}",
        f"hybrid_w={dbg.get('hybrid_w')} mode={dbg.get('hybrid_mode')}",
        f"k={cfg.mask_steer_k:.2f} a={cfg.mask_steer_alpha:.2f}",
        f"near={cfg.mask_near_band_ratio:.2f} far_blend_max={cfg.mask_far_blend:.2f}",
        f"far_eff={dbg.get('hybrid_far_blend')} path_corr={int(cfg.mask_use_path_correction)}",
        f"rate={cfg.steering_rate_limit_per_sec:.1f} max={cfg.max_steering_command:.2f}",
        f"e_db={cfg.mask_error_deadband:.2f} e=[{cfg.mask_blend_error_lo:.2f},{cfg.mask_blend_error_hi:.2f}]",
        f"CTE={dbg.get('cross_track_error_m')} err={dbg.get('mask_error_near')}",
        's=save r=reload t=inout g=start space=pause',
    ]
    y = 22
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        y += 22
    cv2.imshow(WIN_HUD, canvas)


def run(*, topic: str, drive: bool, config_path: Path, route_mode: str) -> int:
    import rclpy
    from control_msgs.msg import Control
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    cfg = load_planner_config(config_path, route_mode=route_mode)
    state = MaskTuneState(cfg, config_path)
    planner = MainPlanner(cfg)
    apply_lane_viz('control')
    ld._apply_detect_tune_from_yaml()
    _init_ui(state)

    rclpy.init()
    node = Node('tune_mask_pursuit')
    bridge = CvBridge()
    latest: dict[str, Any] = {'frame': None}

    def _cb_raw(msg: Image) -> None:
        latest['frame'] = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _cb_compressed(msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        latest['frame'] = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if topic.endswith('compressed'):
        node.create_subscription(CompressedImage, topic, _cb_compressed, 10)
    else:
        node.create_subscription(Image, topic, _cb_raw, 10)
    control_pub = node.create_publisher(Control, '/control', 10) if drive else None

    def publish(steer: float, thr: float) -> None:
        if control_pub is None:
            return
        msg = Control()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.steering = float(steer)
        msg.throttle = float(thr)
        control_pub.publish(msg)

    print('=== tune_mask_pursuit ===', flush=True)
    print(f'config={config_path} drive={drive} topic={topic}', flush=True)
    print('Keys: s save | r reload | t inout_fork | g start | space pause | q quit', flush=True)
    if drive:
        print('WATCH: OpenCV ``Lane drive`` + ``mask_p controls`` / ``mask_p hud``', flush=True)
        print('Do NOT run sim-auto / inference_node while driving.', flush=True)

    last = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            frame = latest['frame']
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('w'):
                _place_ui()
            if key == ord('s'):
                state.sync_from_trackbars()
                state.save_yaml()
            if key == ord('r'):
                state.reload()
                planner = MainPlanner(state.cfg)
                print('[reload] yaml + planner reset', flush=True)
            if key == ord('t'):
                try:
                    _teleport('inout_fork')
                    print('[teleport] inout_fork', flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f'[teleport] fail: {exc}', flush=True)
            if key == ord('g'):
                try:
                    _teleport('start')
                    print('[teleport] start', flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f'[teleport] fail: {exc}', flush=True)
            if key == ord(' ') and drive:
                state.paused = not state.paused
                if state.paused:
                    publish(0.0, 0.0)
                print('PAUSED' if state.paused else 'RESUMED', flush=True)

            if frame is None:
                continue

            cfg_now = state.sync_from_trackbars()
            state.apply_to_planner(planner)
            now = time.time()
            _ = max(0.02, now - last)
            last = now
            output = planner.step(frame, now_sec=now)
            _draw_hud(output, cfg_now, state.paused)

            if drive and not state.paused:
                publish(output.command.steering, output.command.throttle)
            elif drive and state.paused:
                publish(0.0, 0.0)
    finally:
        if drive:
            for _ in range(8):
                publish(0.0, 0.0)
                rclpy.spin_once(node, timeout_sec=0.02)
                time.sleep(0.02)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument(
        '--config',
        type=Path,
        default=default_planner_config_path(),
        help='main_planner.yaml path',
    )
    parser.add_argument('--route-mode', choices=('out', 'in'), default='out')
    parser.add_argument(
        '--drive',
        action='store_true',
        help='Publish /control (stop inference_node / sim-auto first)',
    )
    parser.add_argument(
        '--view-only',
        action='store_true',
        help='Alias: do not publish (default without --drive)',
    )
    args = parser.parse_args(argv)
    drive = bool(args.drive) and not bool(args.view_only)
    return run(
        topic=args.topic,
        drive=drive,
        config_path=args.config,
        route_mode=args.route_mode,
    )


if __name__ == '__main__':
    raise SystemExit(main())
