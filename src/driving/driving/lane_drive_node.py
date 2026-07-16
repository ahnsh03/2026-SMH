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

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from rcl_interfaces.msg import SetParametersResult

from control_msgs.msg import Control
from sensor_msgs.msg import CompressedImage

from inference.modules.lane_detection import LaneDetector
from inference.modules.aruco_stop import ArucoStopDetector
from inference.modules.sign_light import SignLightDetector, SIGN_LEFT, SIGN_RIGHT
from .planner.mission import MissionController

# 표지판(좌/우) + 신호등(빨강/초록) YOLO. 빨강=정지, 표지=감속+해당차선 추종.
SIGNLIGHT_DEFAULTS = {
    "enable_sign_light": True,        # YOLO 표지/신호 기능 on/off
    "sign_light_model_path": "",      # 빈값=자동탐색(team-new/weights → 2026-SMH/weights)
    "sign_light_conf": 0.35,          # YOLO 신뢰도 임계
    "sign_light_threads": 3,          # onnxruntime 스레드(4코어 중 3, 172ms; 1개는 제어용)
    "sign_slow_factor": 0.5,          # 표지 인식 중 스로틀 배율(감속)
    "sign_lane_offset_m": 0.15,       # 표지 방향으로 centerline 측면 바이어스(우+/좌−)
    "light_enter_seconds": 0.2,       # 빨강/초록 이 시간 지속돼야 상태 변경
    "sign_hold_seconds": 1.0,         # 표지 사라져도 이 시간 유지(감속/차선)
}

# ArUco 정지 파라미터 (마커 인식 → 정지). 대회 정지마커 DICT_6X6_50 ID 3.
ARUCO_DEFAULTS = {
    "enable_aruco_stop": True,        # 아루코 정지 기능 on/off
    "aruco_enter_stop_seconds": 0.15,  # 이 시간 연속 검출되면 정지 진입
    "aruco_exit_stop_seconds": 1.5,    # 이 시간 연속 미검출되면 재출발
    "aruco_min_marker_px": 0.0,        # 마커 변 길이(px) 하한(거리 게이팅). 0=거리무관
}

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
        for name, default in ARUCO_DEFAULTS.items():
            self.declare_parameter(name, default)
        for name, default in SIGNLIGHT_DEFAULTS.items():
            self.declare_parameter(name, default)

        image_topic = self.get_parameter("image_topic").value
        control_topic = self.get_parameter("control_topic").value
        self.command_hz = float(self.get_parameter("command_hz").value)
        self.lane_timeout = float(self.get_parameter("lane_timeout").value)
        vision_config = self.get_parameter("vision_config_file").value or None
        gains = {k: self.get_parameter(k).value for k in GAIN_DEFAULTS}

        self.detector = LaneDetector(vision_config)
        self.mission = MissionController(gains)

        # ArUco 정지: 마커 인식 시 throttle=0 (미션 명령 오버라이드)
        self.enable_aruco_stop = bool(self.get_parameter("enable_aruco_stop").value)
        self.aruco = ArucoStopDetector({
            "enter_stop_seconds": self.get_parameter("aruco_enter_stop_seconds").value,
            "exit_stop_seconds": self.get_parameter("aruco_exit_stop_seconds").value,
            "min_marker_px": self.get_parameter("aruco_min_marker_px").value,
        })
        self._aruco_stopped = False

        # 표지판/신호등 YOLO — aarch64 CPU 에서 느려(~0.4s) 별도 스레드에서 실행.
        self.enable_sign_light = bool(self.get_parameter("enable_sign_light").value)
        self.sign_slow_factor = float(self.get_parameter("sign_slow_factor").value)
        self.sign_lane_offset_m = float(self.get_parameter("sign_lane_offset_m").value)
        self.sign_light = SignLightDetector({
            "sign_light_model_path": self.get_parameter("sign_light_model_path").value or None,
            "sign_light_conf": self.get_parameter("sign_light_conf").value,
            "sign_light_threads": self.get_parameter("sign_light_threads").value,
            "light_enter_seconds": self.get_parameter("light_enter_seconds").value,
            "sign_hold_seconds": self.get_parameter("sign_hold_seconds").value,
        })
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._yolo_stop = threading.Event()
        self._last_light_stop = False
        self._last_sign_dir = 0
        if self.enable_sign_light and self.sign_light.available:
            self._yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
            self._yolo_thread.start()
            self.get_logger().info("sign/light YOLO thread up (async)")
        else:
            self._yolo_thread = None
            if self.enable_sign_light:
                self.get_logger().warn(
                    "sign/light YOLO 비활성: onnxruntime/모델 없음 → 표지·신호 무시")

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
            elif p.name == "enable_aruco_stop":
                self.enable_aruco_stop = bool(p.value)
                applied.append(f"{p.name}={p.value}")
            elif p.name == "aruco_min_marker_px":
                self.aruco.min_marker_px = float(p.value)
                applied.append(f"{p.name}={p.value}")
            elif p.name == "aruco_enter_stop_seconds":
                self.aruco.enter_seconds = float(p.value)
                applied.append(f"{p.name}={p.value}")
            elif p.name == "aruco_exit_stop_seconds":
                self.aruco.exit_seconds = float(p.value)
                applied.append(f"{p.name}={p.value}")
            elif p.name == "enable_sign_light":
                self.enable_sign_light = bool(p.value)
                applied.append(f"{p.name}={p.value}")
            elif p.name == "sign_slow_factor":
                self.sign_slow_factor = float(p.value)
                applied.append(f"{p.name}={p.value}")
            elif p.name == "sign_lane_offset_m":
                self.sign_lane_offset_m = float(p.value)
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

        # YOLO 스레드가 처리할 최신 프레임 공유 (제어는 안 기다림)
        if self.enable_sign_light and self._yolo_thread is not None:
            with self._frame_lock:
                self._latest_frame = frame

        lane = self.detector.detect(frame)               # in-process 인지

        # 표지판 방향 → 해당 차선 쪽으로 centerline 측면 바이어스 (우 표지=우측 추종)
        sl = self.sign_light.state() if self.enable_sign_light else None
        if sl is not None and sl.sign_dir != 0 and self.sign_lane_offset_m > 0:
            off = self.sign_lane_offset_m * (1.0 if sl.sign_dir == SIGN_RIGHT else -1.0)
            lane.white_centerline = [(x, y + off) for x, y in lane.white_centerline]

        cmd, dstate = self.mission.plan(lane, dt)        # in-process 제어
        throttle = float(cmd.throttle)

        # 표지 인식 중 → 감속
        if sl is not None and sl.sign_dir != 0:
            throttle *= self.sign_slow_factor
        self._log_sign_light(sl)

        # ArUco 정지 마커: 인식되면 throttle=0 (미션보다 우선). 조향은 유지.
        if self.enable_aruco_stop:
            should_stop, marker_id = self.aruco.stop(frame)
            if should_stop != self._aruco_stopped:
                self._aruco_stopped = should_stop
                self.get_logger().info(
                    f"ArUco {'STOP (marker %s)' % marker_id if should_stop else 'resume'}")
            if should_stop:
                throttle = 0.0

        # 신호등 빨강 → 정지 (초록/미검출이면 주행). 최우선.
        if sl is not None and sl.stop_for_light:
            throttle = 0.0

        self._last_steer = float(cmd.steering)
        self._last_throttle = throttle

    # -------------------------------------------------------- YOLO 백그라운드
    def _yolo_loop(self):
        """별도 스레드: 최신 프레임에 YOLO 실행 → sign_light 상태 갱신.

        한 프레임 추론이 실패해도(예외) 스레드는 죽지 않고 계속 → 주행 지속.
        """
        import time as _t
        while not self._yolo_stop.is_set() and rclpy.ok():
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                _t.sleep(0.02)
                continue
            try:
                self.sign_light.infer(frame)
            except Exception as exc:  # noqa: BLE001 — 인식 실패가 주행을 막지 않게
                self.get_logger().warn(f"sign/light infer skipped: {exc}")
                _t.sleep(0.05)
            _t.sleep(0.005)   # 다른 코어에 양보

    def _log_sign_light(self, sl):
        if sl is None:
            return
        if sl.stop_for_light != self._last_light_stop:
            self._last_light_stop = sl.stop_for_light
            self.get_logger().info(
                f"traffic light: {'RED → STOP' if sl.stop_for_light else 'GREEN/none → go'}")
        if sl.sign_dir != self._last_sign_dir:
            self._last_sign_dir = sl.sign_dir
            name = {0: "none", SIGN_LEFT: "LEFT", SIGN_RIGHT: "RIGHT"}[sl.sign_dir]
            self.get_logger().info(f"direction sign: {name} (감속+차선추종)")

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

    def destroy_node(self):
        self._yolo_stop.set()          # YOLO 스레드 종료 신호
        if self._yolo_thread is not None:
            self._yolo_thread.join(timeout=1.0)
        super().destroy_node()


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
