"""
Autonomous driving inference node.

Subscribes to camera images, runs perception/planning pipeline, publishes /control.

Module integration points (assign per docs/roles.md):
  - lane_detection    : 장원태
  - traffic_sign      : 장원정
  - aruco_detection   : 안승현, 박성준
  - roundabout        : 양서준
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from control_msgs.msg import Control
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
import yaml


def get_default_vehicle_config_path() -> str:
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/SEA-Me-Hackathon/src/config/vehicle_config.yaml'


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('default_throttle', 0.0)
        self.declare_parameter('steer_trim', 0.0)

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        image_topic = str(self.get_parameter('image_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        self.default_throttle = float(self.get_parameter('default_throttle').value)
        self.steer_trim = float(self.load_steer_trim())

        if publish_hz <= 0.0:
            raise ValueError('publish_hz must be greater than 0')

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.latest_frame: np.ndarray | None = None
        self.steering = self.steer_trim
        self.throttle = self.default_throttle

        self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            image_qos,
        )
        self.control_pub = self.create_publisher(Control, control_topic, 10)
        self.create_timer(1.0 / publish_hz, self.publish_control)

        self.get_logger().info(
            f'inference_node started: image_topic={image_topic}, '
            f'control_topic={control_topic}, steer_trim={self.steer_trim}'
        )

    def image_callback(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('Failed to decode camera frame')
            return

        self.latest_frame = frame
        self.steering, self.throttle = self.run_pipeline(frame)

    def run_pipeline(self, frame: np.ndarray) -> tuple[float, float]:
        """
        Main perception + planning pipeline.

        TODO: integrate team modules here.
        Returns (steering, throttle) in range [-1.0, 1.0].
        """
        _ = frame
        return self.steer_trim, self.default_throttle

    def publish_control(self):
        msg = Control()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.steering = float(self.steering)
        msg.throttle = float(self.throttle)
        self.control_pub.publish(msg)

    def load_steer_trim(self) -> float:
        param_trim = float(self.get_parameter('steer_trim').value)
        if param_trim != 0.0:
            return param_trim

        if not os.path.exists(self.vehicle_config_file):
            return 0.0

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except OSError as exc:
            self.get_logger().warning(f'Failed to read {self.vehicle_config_file}: {exc}')
            return 0.0

        return float(config.get('STEER_TRIM', 0.0))


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down inference_node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
