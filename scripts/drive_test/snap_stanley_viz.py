#!/usr/bin/env python3
"""Grab a few Live stanley/circle overlays from the sim camera for review."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'src' / 'inference'))

from inference.pipeline import MainPlanner, load_planner_config  # noqa: E402
from inference.modules import lane_detection as ld  # noqa: E402


def _draw_path(canvas: np.ndarray, path_xy: np.ndarray, color: tuple[int, int, int]) -> None:
    if path_xy is None or len(path_xy) < 2:
        return
    mpp = float(ld.METERS_PER_PIXEL)
    h, w = canvas.shape[:2]
    # Metric IPM: x forward from bottom, y left from image center (lane_detection SSOT).
    pts = []
    for x, y in path_xy:
        u = int(round(w * 0.5 - float(y) / mpp))
        v = int(round(h - 1 - float(x) / mpp))
        if 0 <= u < w and 0 <= v < h:
            pts.append((u, v))
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
    if pts:
        cv2.circle(canvas, pts[0], 4, (0, 255, 255), -1)


class Snapper(Node):
    def __init__(self, out_dir: Path, count: int = 6, interval_sec: float = 1.2):
        super().__init__('snap_stanley_viz')
        self.bridge = CvBridge()
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.count = count
        self.interval = interval_sec
        self.saved = 0
        self.last_t = 0.0
        self.frame = None
        cfg = load_planner_config(
            '/workspace/config/main_planner.yaml', route_mode='in'
        )
        # Capture NORMAL stanley without false circle/fork switches.
        from dataclasses import replace

        cfg = replace(
            cfg,
            roundabout_entry_on_yellow=False,
            circle_ignore_fork_for_control=True,
        )
        self.planner = MainPlanner(cfg)
        self.planner._forkish_for_mask = lambda _lane: False  # type: ignore[method-assign]
        self.create_subscription(Image, '/camera/image_raw', self._on_img, 10)
        self.create_timer(0.05, self._tick)

    def _on_img(self, msg: Image) -> None:
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _tick(self) -> None:
        if self.frame is None or self.saved >= self.count:
            if self.saved >= self.count:
                raise SystemExit(0)
            return
        now = time.time()
        if now - self.last_t < self.interval:
            return
        self.last_t = now
        frame = self.frame.copy()
        out = self.planner.step(frame, now_sec=now)
        lane, dbg = ld.detect_with_debug(
            frame,
            prefer_yellow=True,
            enable_fork=True,
        )
        if dbg.fork_active or len(getattr(dbg, 'road_branches', ()) or ()) >= 2:
            canvas = ld.make_drive_preview(
                dbg.bev,
                dbg.road_clean,
                white_left=dbg.white_left,
                white_right=dbg.white_right,
                yellow_left=dbg.yellow_left,
                yellow_right=dbg.yellow_right,
                prefer_yellow=True,
                fork_active=bool(dbg.fork_active),
                fork_lane_pairs=dbg.fork_lane_pairs,
                road_branches=dbg.road_branches,
                road_cells=dbg.road_cells,
                fork_split_source=str(getattr(dbg, 'fork_split_source', '') or ''),
                ego_road_color=getattr(dbg, 'ego_road_color', None),
            )
        else:
            mode = 'yellow' if out.path_source.name.startswith('YELLOW') else 'white'
            canvas = ld.render_mode_preview(mode, dbg)

        path = getattr(lane, 'yellow_centerline', None)
        if path is None or len(path) < 2:
            path = getattr(lane, 'white_centerline', None)
        _draw_path(canvas, path, (0, 220, 255))

        dbg_p = out.debug or {}
        lines = [
            f'decision={out.decision}',
            f'state={out.state.name} path={out.path_source.name}',
            f'tracker={dbg_p.get("normal_tracker")} circle={dbg_p.get("circle_tracker")}',
            f'steer={out.command.steering:+.3f} thr={out.command.throttle:+.3f}',
            f'psi={dbg_p.get("stanley_psi")} cte={dbg_p.get("stanley_cte_f")}',
        ]
        y = 18
        for text in lines:
            cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
            y += 18

        path_out = self.out_dir / f'snap_{self.saved:02d}_{out.decision}.png'
        cv2.imwrite(str(path_out), canvas)
        print(f'[snap] {path_out.name} {lines[0]} {lines[1]}', flush=True)
        self.saved += 1


def main() -> None:
    out = _ROOT / 'data' / 'captures' / 'stanley_smooth_viz'
    rclpy.init()
    node = Snapper(out, count=6, interval_sec=1.0)
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f'[snap] done → {out}', flush=True)


if __name__ == '__main__':
    main()
