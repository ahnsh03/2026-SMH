"""
Autonomous driving perception node.

Subscribes to camera images, runs lane/aruco perception, publishes:
  - /perception/lane  (lane_msgs/LaneDetections)
  - /debug/aruco

Control (/control) is owned by ``lane_control_node`` (temporary P/EMA) or a
future mission planner. See docs/lane-perception-topic.md.

ArUco 보드 확인:
  ros2 topic echo /debug/aruco
  # 또는 launch 로그에서 [aruco] 상태 변경만 출력
"""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point32
from lane_msgs.msg import LaneDetections as LaneDetectionsMsg
from lane_msgs.msg import LaneMarking as LaneMarkingMsg
from lane_msgs.msg import RoadBranch as RoadBranchMsg
from pathlib import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from inference.modules import aruco_detection, lane_detection
from inference.types import ArucoResult


def _to_point32_list(points: np.ndarray) -> list:
    """Nx2 또는 Nx3 base_link 배열을 geometry_msgs/Point32 리스트로 변환."""
    result = []
    for point in np.asarray(points, dtype=np.float32):
        z = float(point[2]) if point.shape[0] > 2 else 0.0
        result.append(Point32(x=float(point[0]), y=float(point[1]), z=z))
    return result


def get_default_vehicle_config_path() -> str:
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/2026-SMH/src/config/vehicle_config.yaml'


class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        self.declare_parameter('vehicle_config_file', get_default_vehicle_config_path())
        self.declare_parameter('image_topic', '/camera/image/compressed')
        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('aruco_debug_topic', '/debug/aruco')
        self.declare_parameter('aruco_debug_log', True)

        image_topic = str(self.get_parameter('image_topic').value)
        lane_topic = str(self.get_parameter('lane_topic').value)
        aruco_debug_topic = str(self.get_parameter('aruco_debug_topic').value)
        self.aruco_debug_log = bool(self.get_parameter('aruco_debug_log').value)

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        lane_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.latest_frame: np.ndarray | None = None
        self._last_aruco_log_key: tuple[bool, bool, int | None] | None = None

        self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            image_qos,
        )
        # 인지 전용: /perception/lane 만 발행. /control 은 lane_control_node.
        self.lane_pub = self.create_publisher(LaneDetectionsMsg, lane_topic, lane_qos)
        self.aruco_debug_pub = self.create_publisher(String, aruco_debug_topic, 10)

        self.get_logger().info(
            f'inference_node (perception-only) started: '
            f'image_topic={image_topic}, lane_topic={lane_topic}, '
            f'aruco_debug_topic={aruco_debug_topic}'
        )

    def image_callback(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('Failed to decode camera frame')
            return

        self.latest_frame = frame
        self.run_pipeline(frame)

    def run_pipeline(self, frame: np.ndarray) -> None:
        """인지 모듈만 직접 호출·발행한다.

        판제(roundabout.plan 등)는 실행하지 않는다. run_perception은
        판제(roundabout)까지 묶으므로 우회하고, 이 노드가 발행하는 인지
        (lane, aruco)만 계산한다.
        """
        lane = lane_detection.detect(frame)
        aruco = aruco_detection.detect(frame)
        self.publish_aruco_debug(aruco)
        self.publish_lane_detections(lane)

    def publish_lane_detections(self, lane) -> None:
        """인지 LaneDetections(dataclass)를 단일 토픽 msg로 변환·발행한다."""
        msg = LaneDetectionsMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        for marking in lane.lanes:
            lane_marking = LaneMarkingMsg()
            lane_marking.id = int(marking.id)
            lane_marking.color = int(marking.color)
            lane_marking.side_hint = int(marking.side_hint)
            lane_marking.confidence = float(marking.confidence)
            lane_marking.length = float(marking.length)
            lane_marking.heading = float(marking.heading)
            lane_marking.curvature = float(marking.curvature)
            lane_marking.points = _to_point32_list(marking.points)
            msg.lanes.append(lane_marking)

        msg.white_visible = bool(lane.white_visible)
        msg.yellow_visible = bool(lane.yellow_visible)
        msg.left_visible = bool(lane.left_visible)
        msg.right_visible = bool(lane.right_visible)
        msg.white_confidence = float(lane.white_confidence)
        msg.yellow_confidence = float(lane.yellow_confidence)
        msg.left_confidence = float(lane.left_confidence)
        msg.right_confidence = float(lane.right_confidence)

        msg.white_centerline = _to_point32_list(lane.white_centerline)
        msg.yellow_centerline = _to_point32_list(lane.yellow_centerline)
        msg.yellow_crossing_line = bool(lane.yellow_crossing_line)

        msg.fork_active = bool(lane.fork_active)
        for branch in lane.branches:
            road_branch = RoadBranchMsg()
            road_branch.branch_id = int(branch.lateral_rank)
            road_branch.confidence = float(branch.confidence)
            road_branch.width = float(branch.width)
            road_branch.centerline = _to_point32_list(branch.points)
            msg.branches.append(road_branch)

        grid = np.ascontiguousarray(lane.drivable_area, dtype=np.uint8)
        drivable = Image()
        drivable.header = msg.header
        drivable.height = int(grid.shape[0]) if grid.ndim == 2 else 0
        drivable.width = int(grid.shape[1]) if grid.ndim == 2 else 0
        drivable.encoding = 'mono8'
        drivable.is_bigendian = 0
        drivable.step = drivable.width
        drivable.data = grid.tobytes()
        msg.drivable_area = drivable

        msg.meters_per_pixel = float(lane_detection.METERS_PER_PIXEL)
        msg.x_forward_max = float(lane_detection.X_MAX_M)

        self.lane_pub.publish(msg)

    def publish_aruco_debug(self, aruco: ArucoResult) -> None:
        """매 프레임 /debug/aruco 발행 + 상태 변경 시에만 로그."""
        line = (
            f'detected={int(aruco.detected)} '
            f'should_stop={int(aruco.should_stop)} '
            f'marker_id={aruco.marker_id}'
        )
        msg = String()
        msg.data = line
        self.aruco_debug_pub.publish(msg)

        key = (aruco.detected, aruco.should_stop, aruco.marker_id)
        if self.aruco_debug_log and key != self._last_aruco_log_key:
            self.get_logger().info(f'[aruco] {line}')
            self._last_aruco_log_key = key


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down inference_node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
