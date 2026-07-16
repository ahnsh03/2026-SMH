"""통합 주행 노드 — 인지+제어 in-process (인지→제어 토픽 미사용).

  구독 : /camera/image/compressed  (sensor_msgs/CompressedImage, D-Racer camera_node)
  발행 : /control                  (control_msgs/Control, D-Racer control_node)

프레임마다 한 프로세스 안에서:
  frame → LaneDetector.detect() → LaneResult → MissionController.plan() → Control

★ 인지 결과를 ROS 토픽(LaneDetections)으로 주고받지 않고 **함수 호출(import)** 로
  넘긴다 → 직렬화/전송 지연 제거. (LaneDetections 발행이 필요하면 inference_node
  를 별도로 쓴다.)

course_mode 파라미터로 In/Out 코스 모드를 고른다.
게인은 ROS 파라미터로 오버라이드 가능:
  ros2 run driving lane_drive_node --ros-args -p course_mode:=in \
       -p base_lookahead_m:=0.9 -p cruise_throttle:=0.25
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult

from control_msgs.msg import Control
from sensor_msgs.msg import CompressedImage

from inference.modules.lane_detection import LaneDetector
from .planner.mission import MissionController

# ROS 파라미터로 노출할 제어/미션 게인 (이름: 기본값). None → 코드 기본값 사용.
GAIN_DEFAULTS = {
    "course_mode": "out",
    # LaneController
    "wheelbase_m": 0.175,
    "max_steer_rad": 0.3054,
    "camera_to_axle_m": 0.20,
    "base_lookahead_m": 0.85,
    "curve_lookahead_m": 0.45,
    "curvature_full_scale": 0.60,
    "curvature_ff_gain": 0.5,
    "cruise_throttle": 0.22,
    "curve_throttle": 0.13,
    "steer_ema": 0.5,
    "steer_rate_limit": 0.25,
    "steer_sign": -1.0,        # 이 D-Racer: 오른쪽=−, 왼쪽=+
    "steer_trim": 0.10,
    # MissionController (In 코스)
    "entry_min_yellow_pts": 5,
    "entry_confirm_frames": 3,
    "enter_commit_deg": 45.0,
    "roundabout_exit_deg": 300.0,
    "roundabout_speed_mps": 0.33,
    "roundabout_lap_time_s": 0.0,
    "roundabout_min_time_s": 2.0,
    "roundabout_max_time_s": 20.0,
    "exit_confirm_frames": 4,
}


class LaneDriveNode(Node):
    def __init__(self):
        super().__init__("lane_drive_node")

        self.declare_parameter("image_topic", "/camera/image/compressed")
        self.declare_parameter("control_topic", "/control")
        self.declare_parameter("command_hz", 20.0)
        self.declare_parameter("lane_timeout", 0.5)       # s, 인지 소실 안전정지
        self.declare_parameter("vision_config_file", "")  # lane_vision.yaml (빈값=자동탐색)
        for name, default in GAIN_DEFAULTS.items():
            self.declare_parameter(name, default)

        image_topic = self.get_parameter("image_topic").value
        control_topic = self.get_parameter("control_topic").value
        self.command_hz = float(self.get_parameter("command_hz").value)
        self.lane_timeout = float(self.get_parameter("lane_timeout").value)
        vision_config = self.get_parameter("vision_config_file").value or None
        gains = {k: self.get_parameter(k).value for k in GAIN_DEFAULTS}

        self.detector = LaneDetector(vision_config)
        self.mission = MissionController(gains)

        # 주행 중 라이브 튜닝: ros2 param set /lane_drive_node steer_trim 0.15
        self.add_on_set_parameters_callback(self._on_param_update)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=10,
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE,
        )
        self.sub = self.create_subscription(CompressedImage, image_topic, self.on_image, qos)
        self.pub = self.create_publisher(Control, control_topic, qos)
        self.timer = self.create_timer(1.0 / self.command_hz, self.on_timer)

        self._last_steer = 0.0
        self._last_throttle = 0.0
        self._last_img_sec = None

        self.get_logger().info(
            f"lane_drive_node up [{gains['course_mode']}]: {image_topic} -> {control_topic} "
            f"(인지→제어 in-process, no lane topic)")

    # -------------------------------------------------------- 라이브 파라미터
    def _on_param_update(self, params):
        """ros2 param set 으로 게인/트림을 주행 중 즉시 반영."""
        applied = []
        for p in params:
            if self.mission.update(p.name, p.value):
                applied.append(f"{p.name}={p.value}")
        if applied:
            self.get_logger().info("param update: " + ", ".join(applied))
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------ image
    def on_image(self, msg: CompressedImage):
        frame = self._decode(msg)
        if frame is None:
            return
        now = self._now()
        dt = 0.05 if self._last_img_sec is None else max(1e-3, now - self._last_img_sec)
        self._last_img_sec = now

        lane = self.detector.detect(frame)               # in-process 인지
        cmd, dstate = self.mission.plan(lane, dt)        # in-process 제어
        self._last_steer = float(cmd.steering)
        self._last_throttle = float(cmd.throttle)

    def _decode(self, msg: CompressedImage):
        try:
            import cv2
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"decode failed: {exc}")
            return None

    # ------------------------------------------------------------------ timer
    def on_timer(self):
        out = Control()
        out.header.stamp = self.get_clock().now().to_msg()
        if self._timed_out():
            out.steering = float(self._last_steer)   # 조향 유지, 스로틀 0 (안전정지)
            out.throttle = 0.0
        else:
            out.steering = float(self._last_steer)
            out.throttle = float(self._last_throttle)
        self.pub.publish(out)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _timed_out(self) -> bool:
        if self._last_img_sec is None:
            return True
        return (self._now() - self._last_img_sec) > self.lane_timeout


def main(args=None):
    rclpy.init(args=args)
    node = LaneDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
