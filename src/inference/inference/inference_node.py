"""인지 노드 (perception).

  구독 : /camera/image/compressed   (sensor_msgs/CompressedImage, D-Racer camera_node)
  발행 : /perception/lane           (lane_msgs/LaneDetections)

카메라 프레임을 받아 PerceptionPipeline 을 돌리고, 결과 LaneResult 를
LaneDetections 메시지로 변환해 제어(driving) 계층으로 넘긴다.
비전 알고리즘 자체는 perception.pipeline / perception.modules 에 있다.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import Point32
from sensor_msgs.msg import CompressedImage

from lane_msgs.msg import LaneDetections, LaneMarking, RoadBranch

from .pipeline import PerceptionPipeline
from .types import LaneResult


def _to_point32_list(points):
    out = []
    for x, y in points:
        p = Point32()
        p.x = float(x)
        p.y = float(y)
        p.z = 0.0
        out.append(p)
    return out


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('vehicle_config_file', '')
        # metric_ipm / hsv / lane_detect 블록을 담은 lane_vision.yaml 경로.
        # 빈 값이면 LaneDetector 가 설치 share / 소스 트리에서 자동 탐색.
        self.declare_parameter('vision_config_file', '')

        image_topic = self.get_parameter('image_topic').value
        lane_topic = self.get_parameter('lane_topic').value
        vision_config = self.get_parameter('vision_config_file').value or None

        # D-Racer camera_node 의 QoS(RELIABLE, depth 10)와 호환.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pipeline = PerceptionPipeline({'lane': vision_config})

        self.sub = self.create_subscription(
            CompressedImage, image_topic, self.on_image, qos)
        self.pub = self.create_publisher(LaneDetections, lane_topic, qos)

        self.get_logger().info(
            f'inference_node up: {image_topic} -> {lane_topic}')

    def on_image(self, msg: CompressedImage):
        frame = self._decode(msg)
        if frame is None:
            return
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        lane_result, _sign_result = self.pipeline.process(frame, now_sec)
        self.pub.publish(self._to_msg(lane_result, msg.header))

    def _decode(self, msg: CompressedImage):
        try:
            import cv2
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'decode failed: {exc}')
            return None

    def _to_msg(self, r: LaneResult, header) -> LaneDetections:
        m = LaneDetections()
        m.header = header

        for lm in r.lanes:
            msg_lm = LaneMarking()
            msg_lm.color = lm.color
            msg_lm.side_hint = lm.side_hint
            msg_lm.confidence = lm.confidence
            msg_lm.length = lm.length
            msg_lm.heading = lm.heading
            msg_lm.curvature = lm.curvature
            msg_lm.points = _to_point32_list(lm.points)
            m.lanes.append(msg_lm)

        m.white_visible = r.white_visible
        m.yellow_visible = r.yellow_visible
        m.left_visible = r.left_visible
        m.right_visible = r.right_visible
        m.white_confidence = r.white_confidence
        m.yellow_confidence = r.yellow_confidence
        m.left_confidence = r.left_confidence
        m.right_confidence = r.right_confidence

        m.white_centerline = _to_point32_list(r.white_centerline)
        m.yellow_centerline = _to_point32_list(r.yellow_centerline)
        m.yellow_crossing_line = r.yellow_crossing_line

        m.fork_active = r.fork_active
        for br in r.branches:
            msg_br = RoadBranch()
            msg_br.branch_id = br.branch_id
            msg_br.confidence = br.confidence
            msg_br.width = br.width
            msg_br.centerline = _to_point32_list(br.centerline)
            m.branches.append(msg_br)

        m.meters_per_pixel = r.meters_per_pixel
        m.x_forward_max = r.x_forward_max
        return m


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
