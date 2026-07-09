"""Publish latched /robot_description for RViz2 RobotModel (Topic source)."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class RobotDescriptionPublisher(Node):
  def __init__(self):
    super().__init__('robot_description_publisher')
    self.declare_parameter('robot_description', '')
    description = str(self.get_parameter('robot_description').value)
    if not description.strip():
      self.get_logger().error('robot_description parameter is empty')
      return

    qos = QoSProfile(
      history=HistoryPolicy.KEEP_LAST,
      depth=1,
      reliability=ReliabilityPolicy.RELIABLE,
      durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    self._publisher = self.create_publisher(String, '/robot_description', qos)
    self._message = String()
    self._message.data = description
    self._publisher.publish(self._message)
    self.create_timer(1.0, self._republish)
    self.get_logger().info('Publishing /robot_description for RViz')

  def _republish(self):
    self._publisher.publish(self._message)


def main(args=None):
  rclpy.init(args=args)
  node = RobotDescriptionPublisher()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
