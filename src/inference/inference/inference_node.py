"""
Autonomous driving inference node.

Subscribes to camera images, runs perception/planning pipeline, publishes /control.

Integration is handled in pipeline.py — assignees edit modules/ only.
See docs/collaboration.md for branch and PR rules.

ArUco 보드 확인:
  ros2 topic echo /debug/aruco
  # 또는 launch 로그에서 [aruco] 상태 변경만 출력
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point32
from control_msgs.msg import Control
from lane_msgs.msg import LaneDetections as LaneDetectionsMsg
from lane_msgs.msg import LaneMarking as LaneMarkingMsg
from lane_msgs.msg import RoadBranch as RoadBranchMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from inference import pipeline
from inference.pipeline import MainPlanner, load_planner_config
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
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('lane_topic', '/perception/lane')
        self.declare_parameter('aruco_debug_topic', '/debug/aruco')
        self.declare_parameter('planner_debug_topic', '/debug/planner')
        self.declare_parameter(
            'planner_config_file', str(pipeline.default_planner_config_path())
        )
        self.declare_parameter('route_mode', '')
        self.declare_parameter(
            'forced_turn',
            '',
        )  # left|right|'' — sim test override (IN: left=exit, right=stay)
        # Mid-track board tests: skip WAIT_GREEN / red stop (ArUco still on).
        self.declare_parameter('traffic_pass', False)
        self.declare_parameter('aruco_debug_log', True)
        self.declare_parameter('publish_hz', 10.0)
        self.declare_parameter('steer_trim', 0.0)
        self.declare_parameter('use_vehicle_steer_trim', True)
        # Monitor BEV overlays (lane paint + drivable road) at this rate.
        self.declare_parameter('bev_debug_hz', 5.0)
        self.declare_parameter(
            'bev_lane_topic', '/debug/bev/lane/compressed'
        )
        self.declare_parameter(
            'bev_road_topic', '/debug/bev/road/compressed'
        )
        self.declare_parameter('publish_bev_debug', True)

        self.vehicle_config_file = os.path.expanduser(
            str(self.get_parameter('vehicle_config_file').value)
        )
        image_topic = str(self.get_parameter('image_topic').value)
        control_topic = str(self.get_parameter('control_topic').value)
        lane_topic = str(self.get_parameter('lane_topic').value)
        aruco_debug_topic = str(self.get_parameter('aruco_debug_topic').value)
        planner_debug_topic = str(self.get_parameter('planner_debug_topic').value)
        planner_config_file = str(self.get_parameter('planner_config_file').value)
        route_mode = str(self.get_parameter('route_mode').value).strip() or None
        forced_turn_raw = str(self.get_parameter('forced_turn').value).strip().lower()
        traffic_pass_raw = self.get_parameter('traffic_pass').value
        if isinstance(traffic_pass_raw, str):
            traffic_pass = traffic_pass_raw.strip().lower() in (
                '1',
                'true',
                'yes',
                'on',
            )
        else:
            traffic_pass = bool(traffic_pass_raw)
        self.aruco_debug_log = bool(self.get_parameter('aruco_debug_log').value)
        publish_hz = float(self.get_parameter('publish_hz').value)
        self.steer_trim = float(self.load_steer_trim())
        self.publish_bev_debug = bool(self.get_parameter('publish_bev_debug').value)
        self.bev_debug_hz = max(0.5, float(self.get_parameter('bev_debug_hz').value))
        bev_lane_topic = str(self.get_parameter('bev_lane_topic').value)
        bev_road_topic = str(self.get_parameter('bev_road_topic').value)
        self._last_bev_debug_sec: float | None = None

        if publish_hz <= 0.0:
            raise ValueError('publish_hz must be greater than 0')

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            # Control must consume the newest camera frame. Queuing old frames
            # creates apparent steering lag when perception is slower than FPS.
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        planner_config = load_planner_config(
            planner_config_file,
            route_mode=route_mode,
            traffic_pass=True if traffic_pass else None,
        )
        self.latest_frame: np.ndarray | None = None
        self.latest_command = pipeline.ControlCommand(
            steering=self.steer_trim,
            throttle=planner_config.default_throttle,
        )
        self._last_frame_time_sec: float | None = None
        self._last_aruco_log_key: tuple[bool, bool, int | None] | None = None
        self._last_planner_log_key: tuple | None = None
        self._last_planner_debug_publish_sec: float | None = None
        self.planner = MainPlanner(planner_config, steer_trim=self.steer_trim)
        if forced_turn_raw in ('left', 'right'):
            from inference.types import TurnSign

            forced = (
                TurnSign.LEFT if forced_turn_raw == 'left' else TurnSign.RIGHT
            )
            self.planner.apply_forced_turn(forced)
            self.get_logger().info(
                f'forced_turn={forced.value} '
                f'(IN: left=roundabout exit, right=stay circulating)'
            )
        elif forced_turn_raw not in ('', 'none', 'auto'):
            self.get_logger().warning(
                f'Ignoring unknown forced_turn={forced_turn_raw!r}; '
                f'use left|right|empty'
            )

        self.create_subscription(
            CompressedImage,
            image_topic,
            self.image_callback,
            image_qos,
        )
        self.lane_pub = self.create_publisher(LaneDetectionsMsg, lane_topic, 10)
        self.aruco_debug_pub = self.create_publisher(String, aruco_debug_topic, 10)
        self.planner_debug_pub = self.create_publisher(String, planner_debug_topic, 10)
        self.control_pub = self.create_publisher(Control, control_topic, 10)
        # Monitor subscribes with default RELIABLE (depth=10). BEST_EFFORT
        # publishers never match → empty Grayscale/Blur panels.
        # Match camera_node (RELIABLE) so /debug/bev/* shows on the web UI.
        jpeg_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.bev_lane_pub = self.create_publisher(
            CompressedImage, bev_lane_topic, jpeg_qos
        )
        self.bev_road_pub = self.create_publisher(
            CompressedImage, bev_road_topic, jpeg_qos
        )
        self.create_timer(1.0 / publish_hz, self.publish_control)

        self.get_logger().info(
            f'inference_node started: '
            f'image_topic={image_topic}, lane_topic={lane_topic}, '
            f'control_topic={control_topic}, route={planner_config.route_mode.value}, '
            f'forced_turn={forced_turn_raw or "-"}, '
            f'traffic_pass={traffic_pass}, '
            f'require_green={planner_config.require_green_to_start}, '
            f'stop_on_red={planner_config.stop_on_red}, '
            f'bev_debug={self.publish_bev_debug} '
            f'({bev_lane_topic}, {bev_road_topic}), '
            f'planner_config={planner_config_file}'
        )
        if traffic_pass:
            self.get_logger().warn(
                '*** TRAFFIC_PASS active: skip WAIT_GREEN / red stop '
                '(ArUco still stops) ***'
            )
        elif planner_config.require_green_to_start:
            self.get_logger().warn(
                '*** WAIT_GREEN armed (throttle=0 until green). '
                'Mid-track test: traffic_pass:=true ***'
            )
        if forced_turn_raw in ('left', 'right'):
            # One more loud line so experimental runs are easy to confirm.
            self.get_logger().warn(
                f'*** FORCED_TURN={forced_turn_raw.upper()} active '
                f'(IN: LEFT=exit / RIGHT=stay circle) ***'
            )

    def publish_stop(self, *, bursts: int = 5) -> None:
        """Publish neutral /control so Gazebo does not keep the last throttle."""
        stop = pipeline.ControlCommand(steering=0.0, throttle=0.0)
        self.latest_command = stop
        self.planner.neutralize_steering()
        for _ in range(max(1, int(bursts))):
            self._publish_control_command(stop)

    def destroy_node(self) -> None:
        try:
            self.publish_stop()
            self.get_logger().info('Published stop /control on shutdown')
        except Exception as exc:  # noqa: BLE001 — best-effort stop
            try:
                self.get_logger().warning(f'stop publish failed: {exc}')
            except Exception:  # noqa: BLE001
                pass
        super().destroy_node()

    def image_callback(self, msg: CompressedImage):
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning('Failed to decode camera frame')
            return

        self.latest_frame = frame
        self.run_pipeline(frame)

    def run_pipeline(self, frame: np.ndarray) -> None:
        """Run synchronized perception/planning and publish debug outputs."""
        now_sec = self.get_clock().now().nanoseconds / 1_000_000_000.0
        output = self.planner.step(frame, now_sec=now_sec)
        self.latest_command = output.command
        self._last_frame_time_sec = now_sec
        # Do not wait for the lower-rate heartbeat timer: a valid corner path
        # may exist for only one perception frame.
        self._publish_control_command(self.latest_command)
        self.publish_aruco_debug(output.aruco)
        self.publish_lane_detections(output.lane)
        self.publish_bev_debug_frames(output)
        self.publish_planner_debug(output)

    @staticmethod
    def _overlay_mask_bgr(
        bev: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
        *,
        alpha: float = 0.50,
    ) -> np.ndarray:
        out = bev.copy()
        if mask is None or getattr(mask, 'size', 0) == 0:
            return out
        if mask.shape[:2] != out.shape[:2]:
            mask = cv2.resize(
                mask, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        selected = mask > 0
        if not np.any(selected):
            return out
        tint = np.zeros_like(out)
        tint[:] = color
        out[selected] = (
            (1.0 - alpha) * out[selected].astype(np.float32)
            + alpha * tint[selected].astype(np.float32)
        ).astype(np.uint8)
        return out

    def _publish_jpeg(self, pub, bgr: np.ndarray, *, frame_id: str) -> None:
        ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        pub.publish(msg)

    def publish_bev_debug_frames(self, output) -> None:
        """BEV + course lane paint, BEV + drivable road — for monitor panels."""
        if not self.publish_bev_debug:
            return
        now_sec = self.get_clock().now().nanoseconds / 1_000_000_000.0
        period = 1.0 / self.bev_debug_hz
        if (
            self._last_bev_debug_sec is not None
            and now_sec - self._last_bev_debug_sec < period
        ):
            return
        dbg = getattr(output, 'lane_debug', None)
        if dbg is None:
            return
        bev = np.asarray(getattr(dbg, 'bev', None))
        if bev.ndim != 3 or bev.size == 0:
            return

        prefer_yellow = bool(output.debug.get('prefer_yellow', False))
        if str(output.debug.get('route', '')).lower() == 'out':
            prefer_yellow = False
        try:
            from inference.modules.perception.blob.rail_corridor import (
                resolve_course_lane_mask,
            )

            lane_mask, used_y = resolve_course_lane_mask(
                getattr(dbg, 'white_bev', None),
                getattr(dbg, 'yellow_bev', None),
                prefer_yellow=prefer_yellow,
            )
        except Exception:  # noqa: BLE001
            lane_mask = getattr(dbg, 'white_bev', None)
            used_y = False
        lane_color = (0, 255, 255) if used_y else (255, 255, 255)  # BGR
        lane_ov = self._overlay_mask_bgr(bev, lane_mask, lane_color)
        cv2.putText(
            lane_ov,
            f'lane ({"Y" if used_y else "W"})',
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )

        road = getattr(dbg, 'road_clean', None)
        if road is None or getattr(road, 'size', 0) == 0:
            road = getattr(output.lane, 'drivable_area', None)
        road_ov = self._overlay_mask_bgr(bev, road, (0, 220, 0))
        cv2.putText(
            road_ov,
            'road / drivable',
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        self._publish_jpeg(self.bev_lane_pub, lane_ov, frame_id='bev_lane')
        self._publish_jpeg(self.bev_road_pub, road_ov, frame_id='bev_road')
        self._last_bev_debug_sec = now_sec

    def publish_planner_debug(self, output) -> None:
        """Publish sign/fork decisions immediately and periodic snapshots."""
        debug = output.debug
        key = (
            output.state.value,
            output.path_source.value,
            output.decision,
            debug['turn_sign'],
            debug['desired_turn'],
            debug['sign_candidate'],
            debug['sign_candidate_frames'],
            debug['fork_locked_turn'],
            debug['fork_active'],
            debug['branch_count'],
            debug['selected_branch_rank'],
            debug['branch_selection_reason'],
        )
        config = self.planner.config
        state_changed = (
            self._last_planner_log_key is None
            or key[0] != self._last_planner_log_key[0]
        )
        decision_changed = (
            self._last_planner_log_key is None
            or key[1:3] != self._last_planner_log_key[1:3]
        )
        sign_changed = (
            self._last_planner_log_key is None
            or key[3:8] != self._last_planner_log_key[3:8]
        )
        fork_changed = (
            self._last_planner_log_key is None
            or key[8:] != self._last_planner_log_key[8:]
        )
        now_sec = self.get_clock().now().nanoseconds / 1_000_000_000.0
        publish_period = 1.0 / config.debug_publish_hz
        periodic_due = (
            self._last_planner_debug_publish_sec is None
            or now_sec < self._last_planner_debug_publish_sec
            or now_sec - self._last_planner_debug_publish_sec >= publish_period
        )
        if not (
            state_changed
            or decision_changed
            or sign_changed
            or fork_changed
            or periodic_due
        ):
            self._last_planner_log_key = key
            return

        msg = String()
        selected_rank = debug['selected_branch_rank']
        selected_rank_text = '-' if selected_rank is None else str(selected_rank)
        msg.data = (
            f"sign_seen={debug['turn_sign']} "
            f"candidate={debug['sign_candidate']}/{debug['sign_candidate_frames']} "
            f"latched={debug['desired_turn']} locked={debug['fork_locked_turn']} "
            f"forced={debug.get('forced_turn', 'unknown')} | "
            f"state={debug['state']} fork={int(debug['fork_active'])}/"
            f"{debug['branch_count']} fork_on={int(bool(debug.get('fork_perception', True)))} "
            f"event={int(debug['branch_event'])} "
            f"events={debug['branch_events']} | "
            f"choice={debug['branch_selection_reason']} rank={selected_rank_text} "
            f"path={debug['path_source']} decision={debug['decision']} | "
            f"steer={debug['steering']:+.3f} throttle={debug['throttle']:+.3f}"
        )
        if str(debug.get('forced_turn', 'unknown')) in ('left', 'right'):
            # 카메라 표지는 무시 중임이 로그에서 바로 보이게.
            msg.data = (
                f"sign_ignored(forced={debug['forced_turn']}) "
                f"latched={debug['desired_turn']} locked={debug['fork_locked_turn']} | "
                f"state={debug['state']} fork={int(debug['fork_active'])}/"
                f"{debug['branch_count']} fork_on={int(bool(debug.get('fork_perception', True)))} "
                f"event={int(debug['branch_event'])} "
                f"events={debug['branch_events']} | "
                f"choice={debug['branch_selection_reason']} rank={selected_rank_text} "
                f"path={debug['path_source']} decision={debug['decision']} | "
                f"steer={debug['steering']:+.3f} throttle={debug['throttle']:+.3f}"
            )
        self.planner_debug_pub.publish(msg)
        self._last_planner_debug_publish_sec = now_sec

        if (config.log_state_changes and state_changed) or (
            config.log_decision_changes and decision_changed
        ) or sign_changed or fork_changed:
            self.get_logger().info(f'[sign] {msg.data}')
        self._last_planner_log_key = key

    def _publish_control_command(self, command: pipeline.ControlCommand) -> None:
        """Publish one already validated planner command immediately."""
        msg = Control()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.steering = float(command.steering)
        msg.throttle = float(command.throttle)
        self.control_pub.publish(msg)

    def publish_control(self) -> None:
        """Publish a command heartbeat and force neutral on stale camera."""
        now_sec = self.get_clock().now().nanoseconds / 1_000_000_000.0
        stale = (
            self._last_frame_time_sec is None
            or now_sec - self._last_frame_time_sec > self.planner.config.command_watchdog_sec
        )
        if stale:
            self.planner.neutralize_steering()
            command = pipeline.ControlCommand(steering=0.0, throttle=0.0)
        else:
            command = self.latest_command
        self._publish_control_command(command)

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

        from inference.modules import lane_detection

        msg.meters_per_pixel = float(
            getattr(lane, 'meters_per_pixel', 0.0) or lane_detection.METERS_PER_PIXEL
        )
        msg.x_forward_max = float(
            getattr(lane, 'x_forward_max', 0.0) or lane_detection.X_MAX_M
        )

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

    def load_steer_trim(self) -> float:
        param_trim = float(self.get_parameter('steer_trim').value)
        use_vehicle_trim = bool(
            self.get_parameter('use_vehicle_steer_trim').value
        )
        if not use_vehicle_trim:
            return param_trim
        if param_trim != 0.0:
            return param_trim

        if not os.path.exists(self.vehicle_config_file):
            return 0.0

        try:
            with open(self.vehicle_config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        except OSError as exc:
            self.get_logger().warning(f'Failed to read {self.vehicle_config_file}: {exc}')
            return 0.0

        return float(config.get('STEER_TRIM', 0.0))


def main(args=None):
    rclpy.init(args=args)
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down inference_node')
    finally:
        try:
            node.publish_stop()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
