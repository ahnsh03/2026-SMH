#!/usr/bin/env python3
"""Display a ROS 2 CompressedImage topic with OpenCV imshow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage

_PKG_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'inference'
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


class CompressedImageViewer(Node):
    def __init__(
        self,
        topic: str,
        window_name: str,
        window_width: int,
        window_height: int,
        detect_signs: bool,
        detect_lights: bool,
        confidence: float,
    ) -> None:
        super().__init__('compressed_image_viewer')
        self.window_name = window_name
        self.stopped = False
        self.detect_signs = detect_signs
        self.detect_lights = detect_lights
        self.confidence = confidence
        self._detect_signs = None
        self._detect_signal = None

        if self.detect_signs:
            from inference.modules.direction_sign import detect_signs as detector

            self._detect_signs = detector

        if self.detect_lights:
            from inference.modules.trafficsign import detect_signal

            self._detect_signal = detect_signal

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(CompressedImage, topic, self._on_image, qos)
        cv2.namedWindow(
            self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
        )
        cv2.resizeWindow(self.window_name, window_width, window_height)
        self.get_logger().info(f'displaying {topic}; press q or Esc to quit')

    def _on_image(self, message: CompressedImage) -> None:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('failed to decode compressed image')
            return

        if self._detect_signs is not None:
            detections = [
                detection
                for detection in self._detect_signs(frame)
                if detection.score >= self.confidence
            ]
            for detection in detections:
                x1, y1 = int(detection.x1), int(detection.y1)
                x2, y2 = int(detection.x2), int(detection.y2)
                color = (
                    (0, 165, 255)
                    if detection.turn.value == 'left'
                    else (0, 255, 0)
                )
                label = f'{detection.turn.value.upper()} {detection.score:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 7, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if self._detect_signal is not None:
            signal = self._detect_signal(frame)
            signal_name = signal.value.upper()
            signal_color = {
                'RED': (0, 0, 255),
                'GREEN': (0, 255, 0),
            }.get(signal_name, (160, 160, 160))
            cv2.rectangle(frame, (8, 8), (218, 48), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f'TRAFFIC LIGHT: {signal_name}',
                (15, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                signal_color,
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            self.stopped = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--window', default='camera bag')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument(
        '--detect-signs',
        action='store_true',
        help='run the ONNX left/right sign detector and draw its detections',
    )
    parser.add_argument(
        '--detect-lights',
        action='store_true',
        help='run the OpenCV red/green traffic-light detector',
    )
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error('--width and --height must be greater than zero')
    if not 0.0 <= args.conf <= 1.0:
        parser.error('--conf must be between 0 and 1')

    rclpy.init()
    node = CompressedImageViewer(
        args.topic,
        args.window,
        args.width,
        args.height,
        args.detect_signs,
        args.detect_lights,
        args.conf,
    )
    try:
        while rclpy.ok() and not node.stopped:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
