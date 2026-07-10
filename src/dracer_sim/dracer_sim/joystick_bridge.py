"""Forward /joystick commands to /control for simulation (no I2C control_node)."""

from __future__ import annotations

import rclpy
from control_msgs.msg import Control
from joystick_msgs.msg import Joystick
from rclpy.node import Node


class JoystickBridge(Node):
  def __init__(self):
    super().__init__('sim_joystick_bridge')
    self.declare_parameter('joystick_topic', 'joystick')
    self.declare_parameter('control_topic', '/control')

    joystick_topic = str(self.get_parameter('joystick_topic').value)
    control_topic = str(self.get_parameter('control_topic').value)

    self.pub = self.create_publisher(Control, control_topic, 10)
    self.create_subscription(Joystick, joystick_topic, self._on_joystick, 10)
    self.get_logger().info(f'Joystick bridge: {joystick_topic} -> {control_topic}')

  def _on_joystick(self, msg: Joystick):
    if bool(msg.e_stop_en):
      stop = Control()
      stop.throttle = 0.0
      stop.steering = float(msg.control_msg.steering)
      self.pub.publish(stop)
      return

    out = Control()
    out.steering = float(msg.control_msg.steering)
    out.throttle = float(msg.control_msg.throttle)
    self.pub.publish(out)


def main(args=None):
  rclpy.init(args=args)
  node = JoystickBridge()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
