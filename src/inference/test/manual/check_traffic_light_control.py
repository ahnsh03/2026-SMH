#!/usr/bin/env python3
"""Publish a Control command driven by the traffic light, without touching pipeline.py.

Red (or no light) -> throttle 0.0, green -> --throttle. Publishes to /debug/control
by default so control_node ignores it and the car stays put. Point it at /control
only when you actually want the car to move.

    ros2 run camera camera_node &
    python3 test/manual/check_traffic_light_control.py
    ros2 topic echo /debug/control

Throttle is normalized -1.0..1.0 (d3racer clips to that range), not a percent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from control_msgs.msg import Control
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.traffic_sign import detect  # noqa: E402
from inference.types import TrafficSignal  # noqa: E402

_MAX_THROTTLE = 1.0


class LightControl(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__('traffic_light_control_probe')
        self.green_throttle = args.throttle
        self.steering = args.steering
        self.previous: TrafficSignal | None = None

        self.publisher = self.create_publisher(Control, args.control_topic, 10)
        self.create_subscription(CompressedImage, args.image_topic, self.on_image, 10)
        self.get_logger().info(
            f'{args.image_topic} -> {args.control_topic} | '
            f'green throttle={self.green_throttle}'
        )

    def on_image(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        signal = detect(frame).signal
        # Anything that is not a confirmed green keeps the car stopped.
        throttle = self.green_throttle if signal is TrafficSignal.GREEN else 0.0

        command = Control()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'base_link'
        command.steering = float(self.steering)
        command.throttle = float(throttle)
        self.publisher.publish(command)

        if signal is not self.previous:
            self.previous = signal
            print(f'{signal.value:<8} -> throttle={throttle:.2f}', flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--image-topic', default='/camera/image/compressed')
    parser.add_argument(
        '--control-topic',
        default='/debug/control',
        help='use /control only when the car should actually move',
    )
    parser.add_argument(
        '--throttle',
        type=float,
        default=0.10,
        help='throttle on green, normalized -1.0..1.0 (default: 0.10)',
    )
    parser.add_argument('--steering', type=float, default=0.0)
    args = parser.parse_args()

    if abs(args.throttle) > _MAX_THROTTLE:
        parser.error(
            f'--throttle {args.throttle} is outside -1.0..1.0; '
            'd3racer clips it, so 10 would mean full speed'
        )

    rclpy.init()
    node = LightControl(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
