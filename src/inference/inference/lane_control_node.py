"""Temporary lane control node — 담당: 안승현.

Subscribes to /perception/lane, runs lane_planner (Pure Pursuit + EMA), publishes /control.
Keeps perception (inference_node) and control separated on ROS topics.

Sim and real share the same topics (/perception/lane → /control).
Differences are launch parameters only (use_sim_time, steer trim, cruise).
"""

from __future__ import annotations

import os
from pathlib import Path

import rclpy
import yaml
from control_msgs.msg import Control
from lane_msgs.msg import LaneDetections as LaneDetectionsMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time

from inference.lane_adapters import detections_from_msg
from inference.modules.lane_planner import (
    LanePlanner,
    default_control_config_path,
    load_control_params,
)


def get_default_vehicle_config_path() -> str:
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/2026-SMH/src/config/vehicle_config.yaml'


class LaneControlNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_control_node')

        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter(
            'control_config_file',
            str(default_control_config_path()),
        )
        self.declare_parameter('default_throttle', 0.0)
        self.declare_parameter('cruise_throttle', 0.35)
        # Real car: leave override=false → load STEER_TRIM from vehicle_config.
        # Sim: launch sets steer_trim_override=true, steer_trim=0.0 (no hardware bias).
        self.declare_parameter('steer_trim_override', False)
        self.declare_parameter('steer_trim', 0.0)
        self.declare_parameter('publish_hz', 10.0)
        # Stop if perception stops publishing (camera hang / node crash).
        self.declare_parameter('lane_timeout_sec', 0.5)
        self.declare_parameter('lane_follow_color', '')

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        lane_topic = str(self.get_parameter('lane_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        control_config = os.path.expanduser(
            str(self.get_parameter('control_config_file').value)
        )
        self.default_throttle = float(self.get_parameter('default_throttle').value)
        self.cruise_throttle = float(self.get_parameter('cruise_throttle').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        if publish_hz <= 0.0:
            raise ValueError('publish_hz must be greater than 0')

        self.lane_timeout_sec = float(self.get_parameter('lane_timeout_sec').value)
        if self.lane_timeout_sec <= 0.0:
            raise ValueError('lane_timeout_sec must be greater than 0')

        if bool(self.get_parameter('steer_trim_override').value):
            self.steer_trim = float(self.get_parameter('steer_trim').value)
        else:
            self.steer_trim = self._load_steer_trim()

        params = load_control_params(Path(control_config))
        follow_override = str(self.get_parameter('lane_follow_color').value).strip()
        if follow_override:
            from dataclasses import replace

            params = replace(params, follow_color=follow_override).clamp()
        self.planner = LanePlanner(params)
        self._latest_command: Control | None = None
        self._last_lane_time: Time | None = None

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            LaneDetectionsMsg,
            lane_topic,
            self.on_lane,
            qos,
        )
        self.control_pub = self.create_publisher(Control, control_topic, qos)
        self.create_timer(1.0 / publish_hz, self.publish_control)

        self.get_logger().info(
            f'lane_control_node started: lane_topic={lane_topic}, '
            f'control_topic={control_topic}, cruise={self.cruise_throttle}, '
            f'steer_trim={self.steer_trim}, timeout={self.lane_timeout_sec}s, '
            f'config={control_config}, follow_color={self.planner.params.follow_color}'
        )

    def _load_steer_trim(self) -> float:
        if not os.path.exists(self.vehicle_config_file):
            return 0.0
        try:
            with open(self.vehicle_config_file, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except OSError as exc:
            self.get_logger().warning(
                f'Failed to read {self.vehicle_config_file}: {exc}'
            )
            return 0.0
        return float(config.get('STEER_TRIM', 0.0))

    def _stopped_command(self) -> Control:
        command = Control()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'base_link'
        command.steering = float(self.steer_trim)
        command.throttle = float(self.default_throttle)
        return command

    def on_lane(self, msg: LaneDetectionsMsg) -> None:
        self._last_lane_time = self.get_clock().now()
        detections = detections_from_msg(msg)
        result = self.planner.step(detections)

        steering = float(self.steer_trim + result.steering_offset)
        steering = max(-1.0, min(1.0, steering))
        if result.confidence > 0.1:
            scale = float(max(0.0, min(1.0, result.throttle_scale)))
            throttle = self.cruise_throttle * scale
        else:
            throttle = self.default_throttle

        command = Control()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = 'base_link'
        command.steering = steering
        command.throttle = float(throttle)
        self._latest_command = command

    def publish_control(self) -> None:
        if self._latest_command is None or self._last_lane_time is None:
            self.control_pub.publish(self._stopped_command())
            return

        age_sec = (
            self.get_clock().now() - self._last_lane_time
        ).nanoseconds / 1e9
        if age_sec > self.lane_timeout_sec:
            self.control_pub.publish(self._stopped_command())
            return

        self._latest_command.header.stamp = self.get_clock().now().to_msg()
        self.control_pub.publish(self._latest_command)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down lane_control_node')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
