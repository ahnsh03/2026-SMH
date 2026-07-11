"""Bridge D-Racer /control to Gazebo /cmd_vel (Ackermann steer angle).

D-Racer convention (same as real hardware / inference):
  steering  -1.0 = left,  +1.0 = right
  throttle  -1.0 = reverse, +1.0 = forward

gazebo_ros_ackermann_drive expects:
  linear.x  = speed (m/s)
  angular.z = front wheel steer angle (rad), NOT yaw rate
  +angular.z = left (ROS REP-103 / +Z CCW)

See control_mapping.control_to_cmd_vel for the sign/angle conversion.
"""

from __future__ import annotations

import rclpy
from control_msgs.msg import Control
from geometry_msgs.msg import Twist
from joystick_msgs.msg import Joystick
from rclpy.node import Node

from dracer_sim.control_mapping import CMD_MODE_ACKERMANN, control_to_cmd_vel


class ControlBridge(Node):
  def __init__(self):
    super().__init__('sim_control_bridge')
    self.declare_parameter('control_topic', '/control')
    self.declare_parameter('cmd_vel_topic', '/cmd_vel')
    self.declare_parameter('joystick_topic', 'joystick')
    self.declare_parameter('max_linear_speed', 1.2)
    self.declare_parameter('max_steer_angle_rad', 0.5236)
    self.declare_parameter('cmd_mode', CMD_MODE_ACKERMANN)
    self.declare_parameter('wheelbase_m', 0.24)

    control_topic = str(self.get_parameter('control_topic').value)
    cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
    joystick_topic = str(self.get_parameter('joystick_topic').value)
    self.max_linear_speed = float(self.get_parameter('max_linear_speed').value)
    self.max_steer_angle_rad = float(self.get_parameter('max_steer_angle_rad').value)
    self.cmd_mode = str(self.get_parameter('cmd_mode').value)
    self.wheelbase_m = float(self.get_parameter('wheelbase_m').value)

    self.throttle = 0.0
    self.steering = 0.0
    self.e_stop_active = False

    self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
    self.create_subscription(Control, control_topic, self._on_control, 10)
    self.create_subscription(Joystick, joystick_topic, self._on_joystick, 10)
    self.create_timer(0.05, self._publish_cmd_vel)

    self.get_logger().info(
      f'Control bridge: {control_topic} -> {cmd_vel_topic} '
      f'(mode={self.cmd_mode}, v_max={self.max_linear_speed}, '
      f'steer_max={self.max_steer_angle_rad}, '
      f'steering: +right→Gazebo left-negated, e_stop via {joystick_topic})'
    )

  def _on_control(self, msg: Control):
    if self.e_stop_active:
      return
    self.throttle = float(msg.throttle)
    self.steering = float(msg.steering)

  def _on_joystick(self, msg: Joystick):
    """Match real control_node: latch E-Stop and force throttle to 0."""
    if not bool(msg.e_stop_en):
      return
    if self.e_stop_active:
      return
    self.e_stop_active = True
    self.throttle = 0.0
    self.get_logger().warning('E-STOP engaged. Ignoring incoming throttle commands.')

  def _publish_cmd_vel(self):
    throttle = 0.0 if self.e_stop_active else self.throttle
    linear, angular_z = control_to_cmd_vel(
      self.steering,
      throttle,
      max_linear_speed=self.max_linear_speed,
      max_steer_angle_rad=self.max_steer_angle_rad,
      cmd_mode=self.cmd_mode,
      wheelbase_m=self.wheelbase_m,
    )
    twist = Twist()
    twist.linear.x = linear
    twist.angular.z = angular_z
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
