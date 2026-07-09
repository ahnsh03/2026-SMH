"""Bridge D-Racer /control to Gazebo /cmd_vel using a bicycle model."""

from __future__ import annotations

import math

import rclpy
from control_msgs.msg import Control
from geometry_msgs.msg import Twist
from rclpy.node import Node


class ControlBridge(Node):
  def __init__(self):
    super().__init__('sim_control_bridge')
    self.declare_parameter('control_topic', '/control')
    self.declare_parameter('cmd_vel_topic', '/cmd_vel')
    self.declare_parameter('max_linear_speed', 1.2)
    self.declare_parameter('max_steer_angle_rad', 0.45)
    self.declare_parameter('wheelbase_m', 0.20)

    control_topic = str(self.get_parameter('control_topic').value)
    cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
    self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
    self.max_steer_angle_rad = float(self.get_parameter('max_steer_angle_rad').value)
    self.wheelbase_m = float(self.get_parameter('wheelbase_m').value)

    self.throttle = 0.0
    self.steering = 0.0

    self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
    self.create_subscription(Control, control_topic, self._on_control, 10)
    self.create_timer(0.05, self._publish_cmd_vel)

    self.get_logger().info(
      f'Control bridge: {control_topic} -> {cmd_vel_topic} '
      f'(v_max={self.max_linear_speed}, steer_max={self.max_steer_angle_rad})'
    )

  def _on_control(self, msg: Control):
    self.throttle = float(msg.throttle)
    self.steering = float(msg.steering)

  def _publish_cmd_vel(self):
    linear = max(-1.0, min(1.0, self.throttle)) * self.max_linear_speed
    steer_angle = max(-1.0, min(1.0, self.steering)) * self.max_steer_angle_rad

    twist = Twist()
    twist.linear.x = linear
    if abs(linear) < 1e-4:
      twist.angular.z = steer_angle
    else:
      twist.angular.z = (linear / self.wheelbase_m) * math.tan(steer_angle)
    self.cmd_pub.publish(twist)


def main(args=None):
  rclpy.init(args=args)
  node = ControlBridge()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
