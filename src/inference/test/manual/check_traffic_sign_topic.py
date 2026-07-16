#!/usr/bin/env python3
"""Print red/green traffic-light verdicts off the live camera topic.

Runs alongside camera_node and monitor_node, so you can aim the light using the
web dashboard while watching the verdict here. Ctrl-C to stop.

    ros2 run camera camera_node &
    python3 test/manual/check_traffic_sign_topic.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.traffic_sign import detect  # noqa: E402
from inference.types import TrafficSignal  # noqa: E402


class LightProbe(Node):
    def __init__(self, topic: str, save: Path | None, every: int):
        super().__init__('traffic_light_probe')
        self.save = save
        self.every = every
        self.counts: Counter[TrafficSignal] = Counter()
        self.frame_idx = 0
        self.create_subscription(CompressedImage, topic, self.on_image, 10)
        self.get_logger().info(f'listening on {topic} — hold a traffic light up to the camera')

    def on_image(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        signal = detect(frame).signal
        self.counts[signal] += 1
        self.frame_idx += 1

        if self.frame_idx % self.every == 0:
            print(signal.value, flush=True)
        if self.save and signal is not TrafficSignal.UNKNOWN:
            cv2.imwrite(str(self.save), frame)

    def report(self) -> None:
        total = sum(self.counts.values())
        print(
            f'\nframes={total} '
            f'red={self.counts[TrafficSignal.RED]} '
            f'green={self.counts[TrafficSignal.GREEN]} '
            f'unknown={self.counts[TrafficSignal.UNKNOWN]}'
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--every', type=int, default=1, help='print every Nth frame')
    parser.add_argument('--save', type=Path, help='write frames where a light was detected')
    args = parser.parse_args()

    rclpy.init()
    node = LightProbe(args.topic, args.save, max(args.every, 1))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        node.report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
