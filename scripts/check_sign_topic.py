#!/usr/bin/env python3
"""Print left/right sign predictions off the live camera topic.

Runs alongside camera_node and monitor_node, so you can aim the sign using the
web dashboard while watching the prediction here. Ctrl-C to stop.

    ros2 run camera camera_node &
    python3 scripts/check_sign_topic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

_PKG_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'inference'
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.direction_sign import detect_signs  # noqa: E402


class SignProbe(Node):
    def __init__(self, topic: str, conf: float, save: Path | None):
        super().__init__('sign_probe')
        self.conf = conf
        self.save = save
        self.previous = ''
        self.create_subscription(CompressedImage, topic, self.on_image, 10)
        self.get_logger().info(f'listening on {topic} — hold a sign up to the camera')

    def on_image(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        detections = [d for d in detect_signs(frame) if d.score >= self.conf]
        if detections:
            top = detections[0]
            label = top.turn.value.upper()
            summary = f'{label} {top.score:.2f}'
        else:
            label = 'none'
            summary = 'no sign'

        # Live status line; keep only the transitions in the scrollback.
        if label != self.previous:
            self.previous = label
            print(f'\r{summary:<30}')
        print(f'\r{summary:<30}', end='', flush=True)

        if self.save and detections:
            cv2.imwrite(str(self.save), frame)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence floor')
    parser.add_argument('--save', type=Path, help='write frames that contain a sign here')
    args = parser.parse_args()

    rclpy.init()
    node = SignProbe(args.topic, args.conf, args.save)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
