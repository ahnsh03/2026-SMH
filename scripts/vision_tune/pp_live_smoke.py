#!/usr/bin/env python3
"""One-shot live PP smoke: one camera frame → detect → Pure Pursuit."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / 'src' / 'inference'), str(_ROOT / 'scripts' / 'vision_tune')]

from inference.lane_adapters import detections_from_module  # noqa: E402
from inference.modules import lane_detection  # noqa: E402
from inference.modules.lane_planner import LanePlanner, load_control_params  # noqa: E402


class Once(Node):
    def __init__(self) -> None:
        super().__init__(
            'pp_smoke',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True),
            ],
        )
        self.frame: np.ndarray | None = None
        self.create_subscription(
            CompressedImage, '/camera/image/compressed', self._cb, 10
        )

    def _cb(self, msg: CompressedImage) -> None:
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        self.frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main() -> int:
    rclpy.init()
    node = Once()
    t0 = time.time()
    while time.time() - t0 < 12.0 and node.frame is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.frame is None:
        print('SMOKE_FAIL no camera frame')
        node.destroy_node()
        rclpy.shutdown()
        return 1
    dets = detections_from_module(lane_detection.detect(node.frame))
    planner = LanePlanner(load_control_params())
    result = planner.step(dets)
    print(
        'live_pp_smoke',
        'shape', node.frame.shape,
        'lanes', len(dets.lanes),
        'steer', round(result.steering_offset, 3),
        'conf', round(result.confidence, 3),
        'y_c', planner.last_debug.get('y_c'),
        'alpha', round(float(planner.last_debug.get('alpha', 0.0)), 3),
    )
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
