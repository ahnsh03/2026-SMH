#!/usr/bin/env python3
"""Print left/right direction-sign verdicts off the live camera topic.

Uses the trained model (weights/sign_best.onnx via direction_sign.detect_turn);
falls back to the blue-circle/white-arrow rule if ONNX Runtime is unavailable.
Runs alongside camera_node so you can aim a sign at the camera while watching
the verdict here. Ctrl-C to stop.

    ros2 run camera camera_node &
    python3 test/check_direction_sign_topic.py
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

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.direction_sign.detector import (  # noqa: E402
    detect_signs,
    detect_turn,
)
from inference.types import TurnSign  # noqa: E402


class SignProbe(Node):
    def __init__(self, topic: str, save: Path | None, every: int):
        super().__init__('direction_sign_probe')
        self.save = save
        self.every = every
        self.counts: Counter[TurnSign] = Counter()
        self.frame_idx = 0
        self.create_subscription(CompressedImage, topic, self.on_image, 10)
        self.get_logger().info(
            f'listening on {topic} — hold a left/right sign up to the camera'
        )

    def on_image(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        # detect_signs surfaces the model's score/box; detect_turn adds the
        # rule-based fallback so the verdict matches what the pipeline sees.
        detections = []
        try:
            detections = detect_signs(frame)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if self.frame_idx == 0:
                self.get_logger().warning(f'ONNX unavailable, using rule fallback: {exc}')

        turn = detect_turn(frame)
        self.counts[turn] += 1
        self.frame_idx += 1

        if self.frame_idx % self.every == 0:
            if detections:
                best = detections[0]
                print(
                    f'{turn.value:7s} score={best.score:.2f} '
                    f'box=({best.x1:.0f},{best.y1:.0f},{best.x2:.0f},{best.y2:.0f})',
                    flush=True,
                )
            else:
                print(f'{turn.value:7s} (no model box)', flush=True)
        if self.save and turn is not TurnSign.UNKNOWN:
            cv2.imwrite(str(self.save), frame)

    def report(self) -> None:
        total = sum(self.counts.values())
        print(
            f'\nframes={total} '
            + ' '.join(f'{sign.value}={self.counts[sign]}' for sign in TurnSign),
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='camera/image/compressed')
    parser.add_argument(
        '--every', type=int, default=5, help='print verdict every N frames'
    )
    parser.add_argument(
        '--save', type=Path, default=None, help='write last non-UNKNOWN frame here'
    )
    args = parser.parse_args()

    rclpy.init()
    node = SignProbe(args.topic, args.save, args.every)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
