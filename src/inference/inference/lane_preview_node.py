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
from cv_bridge import CvBridge
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

        # Do not open lane_detection's own debug windows; this node owns one view.
        ld.VISUALIZE = False
        ld.VISUALIZE_MODE = ld.VISUALIZE_OFF

        self.bridge = CvBridge()
        self._last_process = 0.0
        self._planner_line = ''
        self._active_rank: int | None = None
        self._manual_focus = False
        self._steer = 0.0
        self._throttle = 0.0
        self._have_control = False

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

        self.get_logger().info(
            f'Lane preview: {topic} focus={self.focus} '
            f'prefer_yellow={self.prefer_yellow} route={route or "-"} '
            f'(keys: 0=all 1=left 2=right a=auto-from-planner q=quit)'
        )

    def _on_planner_debug(self, msg: String) -> None:
        self._planner_line = str(msg.data or '')[:180]
        rank = parse_selected_rank_from_planner_debug(self._planner_line)
        self._active_rank = rank
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
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'cv_bridge failed: {exc}')
            return
        self._process(frame)

    def _process(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_process < 1.0 / self.max_hz:
            cv2.waitKey(1)
            return
        self._last_process = now

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
            )
        except Exception as exc:
            self.get_logger().warning(f'detect failed: {exc}')
            return

        # Same overlays as vision_tune / fork harness — perception SSOT.
        if (
            dbg.fork_lane_pairs
            or dbg.fork_active
            or len(dbg.road_branches) >= 1
            or getattr(dbg, 'active_branch_rank', None) is not None
        ):
            canvas = ld.make_fork_focus_preview(dbg, focus=self.focus)
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
        cv2.imshow(self.window_name, canvas)

    def destroy_node(self) -> None:
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
