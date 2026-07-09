"""Republish Gazebo camera as D-Racer /camera/image/compressed (320x180 JPEG)."""

from __future__ import annotations

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


class CameraRepublish(Node):
  def __init__(self):
    super().__init__('sim_camera_republish')
    self.declare_parameter('input_topic', '/gazebo/camera/image_raw')
    self.declare_parameter('output_topic', '/camera/image/compressed')
    self.declare_parameter('raw_output_topic', '/camera/image_raw')
    self.declare_parameter('image_width', 320)
    self.declare_parameter('image_height', 180)
    self.declare_parameter('jpeg_quality', 90)
    self.declare_parameter('frame_id', 'camera')

    input_topic = str(self.get_parameter('input_topic').value)
    output_topic = str(self.get_parameter('output_topic').value)
    self.raw_output_topic = str(self.get_parameter('raw_output_topic').value).strip()
    self.image_width = int(self.get_parameter('image_width').value)
    self.image_height = int(self.get_parameter('image_height').value)
    self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)
    self.frame_id = str(self.get_parameter('frame_id').value)

    self.bridge = CvBridge()
    image_qos = QoSProfile(
      history=HistoryPolicy.KEEP_LAST,
      depth=10,
      reliability=ReliabilityPolicy.RELIABLE,
      durability=DurabilityPolicy.VOLATILE,
    )
    self.pub = self.create_publisher(CompressedImage, output_topic, image_qos)
    self.raw_pub = None
    if self.raw_output_topic:
      self.raw_pub = self.create_publisher(Image, self.raw_output_topic, image_qos)
    self.create_subscription(Image, input_topic, self._on_image, qos_profile_sensor_data)

    raw_info = f', raw={self.raw_output_topic}' if self.raw_pub else ''
    self.get_logger().info(
      f'Camera republish: {input_topic} -> {output_topic}{raw_info} '
      f'({self.image_width}x{self.image_height} JPEG)'
    )

  def _on_image(self, msg: Image):
    try:
      frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as exc:
      self.get_logger().warning(f'cv_bridge failed: {exc}')
      return

    resized = cv2.resize(
      frame,
      (self.image_width, self.image_height),
      interpolation=cv2.INTER_AREA,
    )

    if self.raw_pub is not None:
      raw = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
      raw.header = msg.header
      raw.header.frame_id = self.frame_id
      self.raw_pub.publish(raw)

    ok, encoded = cv2.imencode(
      '.jpg',
      resized,
      [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
    )
    if not ok:
      self.get_logger().warning('JPEG encode failed')
      return

    out = CompressedImage()
    out.header = msg.header
    out.header.frame_id = self.frame_id
    out.format = 'jpeg'
    out.data = encoded.tobytes()
    self.pub.publish(out)


def main(args=None):
  rclpy.init(args=args)
  node = CameraRepublish()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
