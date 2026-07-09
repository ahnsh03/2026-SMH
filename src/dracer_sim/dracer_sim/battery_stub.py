"""Publish a constant battery level for monitor in simulation (no I2C)."""

from __future__ import annotations

import rclpy
from battery_msgs.msg import Battery
from rclpy.node import Node


class BatteryStub(Node):
  def __init__(self):
    super().__init__('sim_battery_stub')
    self.declare_parameter('publish_topic', '/battery_status')
    self.declare_parameter('battery_percent', 80.0)
    self.declare_parameter('publish_hz', 10.0)

    publish_topic = str(self.get_parameter('publish_topic').value)
    self.battery_percent = float(self.get_parameter('battery_percent').value)
    publish_hz = float(self.get_parameter('publish_hz').value)

    self.pub = self.create_publisher(Battery, publish_topic, 10)
    self.create_timer(1.0 / publish_hz, self._publish)
    self.get_logger().info(f'Battery stub on {publish_topic} = {self.battery_percent}%')

  def _publish(self):
    msg = Battery()
    msg.battery_status = self.battery_percent
    self.pub.publish(msg)


def main(args=None):
  rclpy.init(args=args)
  node = BatteryStub()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
