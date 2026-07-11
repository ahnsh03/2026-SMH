#!/usr/bin/env python3
"""Phase 0: capture frames from /camera/image/compressed to disk.

Saves PNG (and optional JPEG) under a captures directory for later
BEV/HSV tuning on sim or the real D3-G board.

Examples:

  # Gazebo bringup already running
  python3 scripts/vision_tune/capture_camera.py --out data/captures/sim

  # Board
  python3 scripts/vision_tune/capture_camera.py --out ~/captures/board --every 5

Keys (when --preview):
  c  save one frame now
  q  quit

Without --preview, frames are saved automatically every --every messages
until Ctrl-C (or --count is reached).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')


class CaptureNode(Node):
    def __init__(
        self,
        topic: str,
        out_dir: Path,
        every: int,
        count: int | None,
        preview: bool,
    ):
        super().__init__('camera_capture')
        self.out_dir = out_dir
        self.every = max(1, every)
        self.count_limit = count
        self.preview = preview
        self.frame_idx = 0
        self.saved = 0
        self.latest: np.ndarray | None = None
        self.done = False
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.create_subscription(CompressedImage, topic, self._on_image, 10)
        self.get_logger().info(
            f'capture → {out_dir} topic={topic} every={self.every} preview={preview}'
        )

    def _on_image(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.latest = frame
        self.frame_idx += 1
        if self.preview:
            return
        if self.frame_idx % self.every != 0:
            return
        self._save(frame)
        if self.count_limit is not None and self.saved >= self.count_limit:
            self.done = True

    def _save(self, frame: np.ndarray) -> Path:
        name = f'frame_{_stamp()}_{self.saved:04d}.png'
        path = self.out_dir / name
        cv2.imwrite(str(path), frame)
        self.saved += 1
        meta = path.with_suffix('.txt')
        h, w = frame.shape[:2]
        meta.write_text(f'width={w}\nheight={h}\nsource=compressed\n', encoding='utf-8')
        self.get_logger().info(f'saved {path.name} ({w}x{h}) total={self.saved}')
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='output directory (e.g. data/captures/sim)',
    )
    parser.add_argument(
        '--every',
        type=int,
        default=10,
        help='save every Nth message when not in preview mode',
    )
    parser.add_argument('--count', type=int, default=None, help='stop after N saves')
    parser.add_argument(
        '--preview',
        action='store_true',
        help='show window; press c to capture, q to quit',
    )
    args = parser.parse_args()

    rclpy.init()
    node = CaptureNode(
        args.topic,
        args.out.expanduser().resolve(),
        args.every,
        args.count,
        args.preview,
    )
    try:
        if args.preview:
            cv2.namedWindow('capture_preview', cv2.WINDOW_NORMAL)
            print('Keys: c=capture  q=quit')
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.latest is not None:
                    cv2.imshow('capture_preview', node.latest)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('c') and node.latest is not None:
                    node._save(node.latest)
                    if args.count is not None and node.saved >= args.count:
                        break
            cv2.destroyAllWindows()
        else:
            while rclpy.ok() and not node.done:
                rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print(f'saved_total={node.saved} dir={node.out_dir}')
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
