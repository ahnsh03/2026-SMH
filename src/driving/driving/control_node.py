"""제어 노드 (driving / lane_control).

  구독 : /perception/lane   (lane_msgs/LaneDetections, 인지 계층)
  발행 : /control           (control_msgs/Control, D-Racer control_node 가 서보로 구동)

LaneDetections 를 받아 LanePlanner(Pure Pursuit)로 조향/스로틀을 만들고,
고정 주기(기본 10Hz)로 Control 을 발행한다. 인지가 timeout 이상 끊기면
안전정지(throttle 0) 명령을 낸다.
"""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from control_msgs.msg import Control
from lane_msgs.msg import LaneDetections

from .planner.lane_planner import LanePlanner


class ControlNode(Node):
    def __init__(self):
        super().__init__('lane_control_node')

        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('command_hz', 10.0)
        self.declare_parameter('lane_timeout', 0.5)      # s, 인지 소실 안전정지
        self.declare_parameter('steer_trim', 0.0)        # 실차 조향 오프셋 (sim=0)

        lane_topic = self.get_parameter('lane_topic').value
        control_topic = self.get_parameter('control_topic').value
        self.command_hz = float(self.get_parameter('command_hz').value)
        self.lane_timeout = float(self.get_parameter('lane_timeout').value)
        self.steer_trim = float(self.get_parameter('steer_trim').value)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.planner = LanePlanner()
        self._last_cmd_steer = 0.0
        self._last_cmd_throttle = 0.0
        self._last_lane_sec = None

        self.sub = self.create_subscription(
            LaneDetections, lane_topic, self.on_lane, qos)
        self.pub = self.create_publisher(Control, control_topic, qos)
        self.timer = self.create_timer(1.0 / self.command_hz, self.on_timer)

        self.get_logger().info(
            f'lane_control_node up: {lane_topic} -> {control_topic} '
            f'@ {self.command_hz:.0f}Hz')

    def on_lane(self, msg: LaneDetections):
        cmd = self.planner.plan(msg)
        self._last_cmd_steer = cmd.steering
        self._last_cmd_throttle = cmd.throttle
        self._last_lane_sec = self._now()

    def on_timer(self):
        out = Control()
        out.header.stamp = self.get_clock().now().to_msg()

        if self._timed_out():
            # 인지 소실 → 안전정지 (조향 유지, 스로틀 0)
            out.steering = float(self._last_cmd_steer + self.steer_trim)
            out.throttle = 0.0
        else:
            out.steering = float(self._last_cmd_steer + self.steer_trim)
            out.throttle = float(self._last_cmd_throttle)

        self.pub.publish(out)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _timed_out(self) -> bool:
        if self._last_lane_sec is None:
            return True
        return (self._now() - self._last_lane_sec) > self.lane_timeout


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
