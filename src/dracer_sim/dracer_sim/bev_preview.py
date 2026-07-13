"""Metric IPM BEV preview for simulation (locked lane_vision.yaml SSOT).

Shows bird's-eye view beside the D-Racer camera window on sim-bringup.
Does not launch Gazebo or run lane detection — remaps camera frames only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def _locate_vision_tune() -> Path:
  for parent in Path(__file__).resolve().parents:
    candidate = parent / 'scripts' / 'vision_tune' / 'metric_ipm.py'
    if candidate.is_file():
      return candidate.parent
  raise ImportError(
    'scripts/vision_tune/metric_ipm.py not found '
    '(need repo root with config/lane_vision.yaml)'
  )


_VISION_TUNE = _locate_vision_tune()
if str(_VISION_TUNE) not in sys.path:
  sys.path.insert(0, str(_VISION_TUNE))

from metric_ipm import (  # noqa: E402
  DEFAULT_CONFIG_PATH,
  draw_metric_guides,
  load_metric_ipm,
  warp_metric_ipm,
)


class BevPreview(Node):
  def __init__(self) -> None:
    super().__init__('sim_bev_preview')
    self.declare_parameter('image_topic', '/camera/image_raw')
    self.declare_parameter('window_name', 'Metric IPM BEV')
    self.declare_parameter('window_scale', 2.0)
    self.declare_parameter('window_x', 700)
    self.declare_parameter('window_y', 40)
    self.declare_parameter('config_path', str(DEFAULT_CONFIG_PATH))
    self.declare_parameter('show_guides', True)

    topic = str(self.get_parameter('image_topic').value)
    self.window_name = str(self.get_parameter('window_name').value)
    self.window_scale = float(self.get_parameter('window_scale').value)
    window_x = int(self.get_parameter('window_x').value)
    window_y = int(self.get_parameter('window_y').value)
    config_raw = str(self.get_parameter('config_path').value).strip()
    config_path = Path(config_raw) if config_raw else DEFAULT_CONFIG_PATH
    self.show_guides = bool(self.get_parameter('show_guides').value)

    if self.window_scale <= 0:
      raise ValueError('window_scale must be positive')

    self.params = load_metric_ipm(config_path if config_path.is_file() else None)
    self.bridge = CvBridge()

    bev_w = int(round(self.params.bev_width * self.window_scale))
    bev_h = int(round(self.params.bev_height * self.window_scale))
    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(self.window_name, bev_w, bev_h)
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
    p = self.params.clamp()
    self.get_logger().info(
      f'BEV preview: {topic} -> {self.window_name} '
      f'(Metric IPM x={p.x_min_m:.2f}..{p.x_max_m:.2f}m '
      f'y=±{p.y_half_width_m:.2f}m mpp={p.meters_per_pixel} '
      f'from {config_path})'
    )

  def _on_image(self, msg: Image) -> None:
    try:
      frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as exc:
      self.get_logger().warning(f'cv_bridge failed: {exc}')
      return
    bev = warp_metric_ipm(frame, self.params)
    if self.show_guides:
      bev = draw_metric_guides(bev, self.params)
    if abs(self.window_scale - 1.0) > 1e-3:
      bev = cv2.resize(
        bev,
        None,
        fx=self.window_scale,
        fy=self.window_scale,
        interpolation=cv2.INTER_NEAREST,
      )
    cv2.imshow(self.window_name, bev)
    cv2.waitKey(1)

  def destroy_node(self) -> None:
    cv2.destroyWindow(self.window_name)
    super().destroy_node()


def main(args=None) -> None:
  rclpy.init(args=args)
  node = BevPreview()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
