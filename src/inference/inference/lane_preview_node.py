"""Real-time lane/fork perception preview for sim bringup.

Runs the same ``lane_detection.detect_with_debug`` path as the planner so you
can tell whether a miss is **perception** (overlay wrong) vs **control lag**
(overlay OK but robot slow to follow).

Typical use (Gazebo already up)::

  ros2 run inference lane_preview_node
  # or via sim_auto_stack.launch.py use_lane_view:=true
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
try:
    from cv_bridge import CvBridge
except ImportError:  # 보드엔 cv_bridge가 없을 수 있다. compressed 경로는 불필요.
    CvBridge = None
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

try:
    from control_msgs.msg import Control
except ImportError:  # pragma: no cover
    Control = None

from inference.modules import lane_detection as ld
from inference.modules.active_lane import (
    focus_name_for_rank,
    parse_fork_perception_from_planner_debug,
    parse_selected_rank_from_planner_debug,
)


def _panel(img: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) <= 1e-3:
        return img
    return cv2.resize(
        img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )


class LanePreviewNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_preview_node')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('compressed_topic', '/camera/image/compressed')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('planner_debug_topic', '/debug/planner')
        self.declare_parameter('control_topic', '/control')
        self.declare_parameter('window_name', 'Lane / Fork Perception')
        self.declare_parameter('window_scale', 1.5)
        self.declare_parameter('window_x', 40)
        self.declare_parameter('window_y', 420)
        self.declare_parameter('focus', 'all')  # all|left|right
        self.declare_parameter('max_hz', 12.0)
        self.declare_parameter('route_mode', '')  # in|out → color contract
        self.declare_parameter('prefer_yellow', -1)  # -1=from route, 0/1 override
        # tune 모드 웹 패널 매핑 (모니터 opencv 패널 기본 토픽과 일치):
        #   edge      = 경계 검출 결과 프리뷰
        #   grayscale = 원본 흰 HSV 마스크 (dbg.white_bev)
        #   blur      = 점선 연결본 (검출에 실제 들어가는 마스크)
        self.declare_parameter('web_topic', '/opencv/image/edge')
        self.declare_parameter('web_topic_hsv', '/opencv/image/grayscale')
        self.declare_parameter('web_topic_connected', '/opencv/image/blur')

        self.window_name = str(self.get_parameter('window_name').value)
        self.window_scale = float(self.get_parameter('window_scale').value)
        self.focus = str(self.get_parameter('focus').value).strip().lower() or 'all'
        self.max_hz = max(1.0, float(self.get_parameter('max_hz').value))
        window_x = int(self.get_parameter('window_x').value)
        window_y = int(self.get_parameter('window_y').value)
        route = str(self.get_parameter('route_mode').value).strip().lower()
        pref_param = int(self.get_parameter('prefer_yellow').value)
        if pref_param in (0, 1):
            self.prefer_yellow = bool(pref_param)
        elif route == 'in':
            self.prefer_yellow = True
        elif route == 'out':
            self.prefer_yellow = False
        else:
            self.prefer_yellow = False  # sim Out-default safe

        # tune 모드: cv2 창(imshow) 대신 흰 차선 프리뷰를 CompressedImage로
        # 웹(모니터 debug_image 패널)에 발행한다. ld.VISUALIZE_MODE는 import 시
        # LANE_VISUALIZE 환경변수로 이미 결정돼 있으므로 여기서 읽어 둔다.
        self.tune_mode = ld.VISUALIZE_MODE == ld.VISUALIZE_TUNE

        # Do not open lane_detection's own debug windows; this node owns one view.
        ld.VISUALIZE = False
        ld.VISUALIZE_MODE = ld.VISUALIZE_OFF

        self.bridge = CvBridge() if CvBridge is not None else None
        self._last_process = 0.0
        self._planner_line = ''
        self._active_rank: int | None = None
        self._enable_fork: bool = True
        self._manual_focus = False
        self._steer = 0.0
        self._throttle = 0.0
        self._have_control = False

        if self.tune_mode:
            # 헤드리스 보드: 창을 열지 않고 웹 토픽으로만 발행한다.
            self.web_pub = self.create_publisher(
                CompressedImage, str(self.get_parameter('web_topic').value), 2
            )
            self.web_pub_hsv = self.create_publisher(
                CompressedImage, str(self.get_parameter('web_topic_hsv').value), 2
            )
            self.web_pub_connected = self.create_publisher(
                CompressedImage,
                str(self.get_parameter('web_topic_connected').value),
                2,
            )
        else:
            self.web_pub = None
            self.web_pub_hsv = None
            self.web_pub_connected = None
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            try:
                cv2.moveWindow(self.window_name, window_x, window_y)
            except cv2.error:
                pass

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        use_compressed = bool(self.get_parameter('use_compressed').value)
        if use_compressed:
            topic = str(self.get_parameter('compressed_topic').value)
            self.create_subscription(
                CompressedImage, topic, self._on_compressed, qos
            )
        else:
            topic = str(self.get_parameter('image_topic').value)
            self.create_subscription(Image, topic, self._on_image, reliable)

        self.create_subscription(
            String,
            str(self.get_parameter('planner_debug_topic').value),
            self._on_planner_debug,
            10,
        )
        if Control is not None:
            self.create_subscription(
                Control,
                str(self.get_parameter('control_topic').value),
                self._on_control,
                10,
            )

        if self.tune_mode:
            self.get_logger().info(
                f'Lane preview [tune]: {topic} → '
                f'{self.get_parameter("web_topic").value} '
                f'(white boundary → 웹 모니터 패널, cv2 창 없음)'
            )
        else:
            self.get_logger().info(
                f'Lane preview: {topic} focus={self.focus} '
                f'prefer_yellow={self.prefer_yellow} route={route or "-"} '
                f'(keys: 0=all 1=left 2=right a=auto-from-planner q=quit)'
            )

    def _on_planner_debug(self, msg: String) -> None:
        self._planner_line = str(msg.data or '')[:180]
        rank = parse_selected_rank_from_planner_debug(self._planner_line)
        self._active_rank = rank
        fork_on = parse_fork_perception_from_planner_debug(self._planner_line)
        if fork_on is not None:
            self._enable_fork = fork_on
        if not self._manual_focus:
            self.focus = focus_name_for_rank(rank)

    def _on_control(self, msg) -> None:
        self._steer = float(getattr(msg, 'steering', 0.0))
        self._throttle = float(getattr(msg, 'throttle', 0.0))
        self._have_control = True

    def _on_compressed(self, msg: CompressedImage) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            self._process(frame)

    def _on_image(self, msg: Image) -> None:
        if self.bridge is None:
            self.get_logger().warning(
                'cv_bridge 없음 — raw Image 경로를 쓸 수 없습니다. '
                'use_compressed:=true 로 실행하거나 ros-humble-cv-bridge 설치.'
            )
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'cv_bridge failed: {exc}')
            return
        self._process(frame)

    def _process(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_process < 1.0 / self.max_hz:
            # tune 모드엔 창이 없으므로 GUI 이벤트 펌프(waitKey)를 건너뛴다.
            if not self.tune_mode:
                cv2.waitKey(1)
            return
        self._last_process = now

        # 키보드 포커스 전환은 창이 있을 때만. tune(웹)은 planner 자동 포커스만.
        if not self.tune_mode:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('0'):
                self.focus = 'all'
                self._manual_focus = True
                self._active_rank = None
            elif key == ord('1'):
                self.focus = 'left'
                self._manual_focus = True
                self._active_rank = 0
            elif key == ord('2'):
                self.focus = 'right'
                self._manual_focus = True
                self._active_rank = 1
            elif key == ord('a'):
                # Resume auto focus from planner lock.
                self._manual_focus = False
                self.focus = focus_name_for_rank(self._active_rank)
            elif key in (ord('q'), 27):
                self.get_logger().info('quit key — destroy window')
                cv2.destroyWindow(self.window_name)
                return

        try:
            lane, dbg = ld.detect_with_debug(
                frame,
                active_branch_rank=self._active_rank,
                prefer_yellow=self.prefer_yellow,
                enable_fork=self._enable_fork,
            )
        except Exception as exc:
            self.get_logger().warning(f'detect failed: {exc}')
            return

        # tune 모드는 흰 차선 생성 결과만 본다(요청 범위). 웹 패널 1개 = 흰 경계.
        if self.tune_mode:
            canvas = ld.render_mode_preview('white', dbg)
        # One canvas: white/yellow course + road; fork rails only when armed+active.
        elif self._enable_fork and (
            dbg.fork_lane_pairs
            or bool(dbg.fork_active)
            or len(dbg.road_branches) >= 2
        ):
            canvas = ld.make_drive_preview(
                dbg.bev,
                dbg.road_clean,
                white_left=dbg.white_left,
                white_right=dbg.white_right,
                yellow_left=dbg.yellow_left,
                yellow_right=dbg.yellow_right,
                prefer_yellow=self.prefer_yellow,
                fork_active=bool(dbg.fork_active),
                fork_lane_pairs=dbg.fork_lane_pairs,
                road_branches=dbg.road_branches,
                road_cells=dbg.road_cells,
                fork_split_source=str(getattr(dbg, 'fork_split_source', '') or ''),
                ego_road_color=getattr(dbg, 'ego_road_color', None),
            )
        elif self.prefer_yellow:
            canvas = ld.render_mode_preview('yellow', dbg)
        else:
            canvas = ld.render_mode_preview('white', dbg)

        canvas = _panel(canvas, self.window_scale)
        n_br = len(getattr(lane, 'branches', ()) or ())
        lines = [
            f'focus={self.focus}  policy={getattr(dbg, "lane_policy", "?")}  '
            f'active={getattr(dbg, "active_branch_rank", None)}  '
            f'fork={int(bool(lane.fork_active))}  branches={n_br}  '
            f'fork_on={int(self._enable_fork)}  '
            f'src={getattr(dbg, "fork_split_source", "?")}',
            f'white_c={float(lane.white_confidence):.2f}  '
            f'yellow_c={float(lane.yellow_confidence):.2f}',
        ]
        if self._have_control:
            lines.append(
                f'control steer={self._steer:+.3f} throttle={self._throttle:+.3f}'
            )
        if self._planner_line:
            lines.append(self._planner_line[:120])

        y = 18
        for text in lines:
            cv2.putText(
                canvas,
                text,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                (8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )
            y += 18

        # HUD hint: if overlay follows paint but car cuts, suspect control lag.
        hint = 'overlay=paint? OK → check steer lag | overlay off paint → perception'
        cv2.putText(
            canvas,
            hint,
            (8, canvas.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (80, 220, 80),
            1,
            cv2.LINE_AA,
        )
        if self.tune_mode:
            self._publish_web(self.web_pub, canvas)
            # 진단용: 원본 흰 HSV → 점선 연결본을 각 패널로. HSV가 선을 제대로
            # 잡는지 vs 좌우 분류가 틀리는지를 갈라서 볼 수 있다.
            self._publish_web(
                self.web_pub_hsv, self._mask_to_bgr(getattr(dbg, 'white_bev', None))
            )
            self._publish_web(
                self.web_pub_connected,
                self._mask_to_bgr(getattr(dbg, 'white_dash_connected_bev', None)),
            )
        else:
            cv2.imshow(self.window_name, canvas)

    @staticmethod
    def _mask_to_bgr(mask) -> np.ndarray | None:
        """단일채널 마스크(0/1 또는 0/255)를 볼 수 있는 BGR로 변환."""

        if mask is None or getattr(mask, 'size', 0) == 0:
            return None
        m = np.asarray(mask)
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        if m.max(initial=0) <= 1:  # 0/1 이진 마스크면 보이도록 스케일
            m = m * 255
        if m.ndim == 2:
            m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        return m

    def _publish_web(self, pub, image) -> None:
        """이미지를 JPEG CompressedImage로 웹 토픽에 발행한다(모니터 패널용)."""

        if pub is None or image is None:
            return
        ok, buf = cv2.imencode('.jpg', image)
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        pub.publish(msg)

    def destroy_node(self) -> None:
        if not self.tune_mode:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LanePreviewNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
