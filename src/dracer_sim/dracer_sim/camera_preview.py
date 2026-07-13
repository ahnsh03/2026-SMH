"""OpenCV camera preview for simulation (16:9 window, configurable initial size)."""

from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class CameraPreview(Node):
  def __init__(self):
    super().__init__('sim_camera_preview')
    self.declare_parameter('image_topic', '/camera/image_raw')
    self.declare_parameter('window_name', 'D-Racer Camera')
    self.declare_parameter('window_width', 640)
    self.declare_parameter('window_height', 360)
    self.declare_parameter('window_x', 40)
    self.declare_parameter('window_y', 40)

    topic = str(self.get_parameter('image_topic').value)
    self.window_name = str(self.get_parameter('window_name').value)
    width = int(self.get_parameter('window_width').value)
    height = int(self.get_parameter('window_height').value)
    window_x = int(self.get_parameter('window_x').value)
    window_y = int(self.get_parameter('window_y').value)

    if width <= 0 or height <= 0:
      raise ValueError('window_width and window_height must be positive')

    self.bridge = CvBridge()
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(self.window_name, width, height)
    try:
      cv2.moveWindow(self.window_name, window_x, window_y)
    except cv2.error:
      pass

    image_qos = QoSProfile(
      history=HistoryPolicy.KEEP_LAST,
      depth=10,
      reliability=ReliabilityPolicy.RELIABLE,
      durability=DurabilityPolicy.VOLATILE,
    )
    self.create_subscription(Image, topic, self._on_image, image_qos)
    self.get_logger().info(
      f'Camera preview: {topic} -> {self.window_name} ({width}x{height})'
    )

  def _on_image(self, msg: Image):
    try:
      frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as exc:
      self.get_logger().warning(f'cv_bridge failed: {exc}')
      return
    cv2.imshow(self.window_name, frame)
    cv2.waitKey(1)

  def destroy_node(self):
    cv2.destroyAllWindows()
    super().destroy_node()


def main(args=None):
  rclpy.init(args=args)
  node = CameraPreview()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
