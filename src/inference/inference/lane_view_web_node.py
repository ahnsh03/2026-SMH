"""lane_view: lane_detection(detect_with_debug) 결과를 웹 모니터 패널로 시각화.

mask_check(hsv_mask_web_node)의 웹 발행 구조를 그대로 쓰되, 마스크를 이 노드의
자체 inRange가 아니라 **lane_detection.detect_with_debug 의 결과(dbg)**에서 받는다.
→ "패널에 보이는 것 = 실제 주행 인지가 쓰는 것" 이 되어 진단·튜닝이 정확하다.

빨강은 주행가능영역으로 포함한다(mask_check 의 road=black|red 방식 참고):
  drivable = road_clean | red_bev

  grayscale = WHITE  : HSV 마스크 | 흰 차선 경계 결과
  blur      = YELLOW : HSV 마스크 | 노란 차선 경계 결과
  edge      = 주행가능영역 (road_clean | red) 1개

  ros2 launch inference lane_view.launch.py
  # browser → http://<board-ip>:5000
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from inference.modules import lane_detection as ld


def _pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """두 BEV 뷰를 한 창에 가로로 이어붙인다(가운데 얇은 구분선)."""
    h = max(left.shape[0], right.shape[0])

    def _fit(img: np.ndarray) -> np.ndarray:
        if img.shape[0] != h:
            w = max(1, int(round(img.shape[1] * h / img.shape[0])))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        return img

    sep = np.full((h, 3, 3), 60, dtype=np.uint8)
    return cv2.hconcat([_fit(left), sep, _fit(right)])


def _label(bgr: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(
        bgr, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        bgr, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA
    )
    return bgr


def _masked_bev(bev: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """BEV 픽셀을 mask 있는 곳만 남기고 나머지는 검게."""
    out = np.zeros_like(bev)
    if mask is not None and mask.size == bev.shape[0] * bev.shape[1]:
        selected = mask > 0
        if np.any(selected):
            out[selected] = bev[selected]
    return out


class LaneViewWebNode(Node):
    def __init__(self) -> None:
        super().__init__('lane_view_web_node')
        self.declare_parameter('compressed_topic', '/camera/image/compressed')
        self.declare_parameter('max_hz', 1.5)
        self.declare_parameter('jpeg_quality', 55)
        self.declare_parameter('white_topic', '/opencv/image/grayscale')
        self.declare_parameter('yellow_topic', '/opencv/image/blur')
        self.declare_parameter('drivable_topic', '/opencv/image/edge')

        self.max_hz = max(0.5, float(self.get_parameter('max_hz').value))
        self.jpeg_quality = int(np.clip(self.get_parameter('jpeg_quality').value, 30, 90))

        # detect_with_debug 내부 cv2.imshow 경로를 끈다(헤드리스 보드).
        ld.VISUALIZE = False
        ld.VISUALIZE_MODE = ld.VISUALIZE_OFF

        sub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        topic = str(self.get_parameter('compressed_topic').value)
        self.create_subscription(CompressedImage, topic, self._on_image, sub_qos)

        pub_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub_white = self.create_publisher(
            CompressedImage, str(self.get_parameter('white_topic').value), pub_qos
        )
        self.pub_yellow = self.create_publisher(
            CompressedImage, str(self.get_parameter('yellow_topic').value), pub_qos
        )
        self.pub_drivable = self.create_publisher(
            CompressedImage, str(self.get_parameter('drivable_topic').value), pub_qos
        )

        self._last = 0.0
        self.get_logger().info(
            f'lane_view: {topic} → 모니터 패널 @ ≤{self.max_hz:.1f} Hz '
            f'(dbg 기반: WHITE/YELLOW=HSV|경계, EDGE=주행가능(road_clean|red))'
        )

    def _on_image(self, msg: CompressedImage) -> None:
        now = time.monotonic()
        if now - self._last < 1.0 / self.max_hz:
            return
        self._last = now

        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return

        try:
            _lane, dbg = ld.detect_with_debug(frame)
        except Exception as exc:
            self.get_logger().warning(f'detect failed: {exc}')
            return

        bev = dbg.bev
        if bev is None or bev.size == 0:
            return

        # 빨강을 주행가능영역으로 포함(mask_check 의 road=black|red 참고).
        # lane_detection 의 road_clean(검정+구멍메움+가로마킹)에 red_bev 를 OR.
        drivable = dbg.road_clean
        red = getattr(dbg, 'red_bev', None)
        if red is not None and red.shape[:2] == drivable.shape[:2]:
            drivable = cv2.bitwise_or(drivable, red)

        # 흰: HSV 마스크 | 흰 차선 경계 결과
        white_combo = _pair(
            _label(_masked_bev(bev, dbg.white_bev),
                   f'WHITE HSV  px={int(np.count_nonzero(dbg.white_bev))}'),
            _label(ld.make_boundary_preview(
                bev, drivable, dbg.white_left, dbg.white_right, 'WHITE'),
                'WHITE BOUNDARY'),
        )
        # 노랑: HSV 마스크 | 노란 차선 경계 결과
        yellow_combo = _pair(
            _label(_masked_bev(bev, dbg.yellow_bev),
                   f'YELLOW HSV  px={int(np.count_nonzero(dbg.yellow_bev))}'),
            _label(ld.make_boundary_preview(
                bev, drivable, dbg.yellow_left, dbg.yellow_right, 'YELLOW'),
                'YELLOW BOUNDARY'),
        )
        # 주행가능영역(빨강 포함): 한 창에 1개.
        drivable_v = _label(
            _masked_bev(bev, drivable),
            f'DRIVABLE(road|red)  px={int(np.count_nonzero(drivable))}',
        )

        self._publish(self.pub_white, white_combo)
        self._publish(self.pub_yellow, yellow_combo)
        self._publish(self.pub_drivable, drivable_v)

    def _publish(self, pub, image: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = buf.tobytes()
        pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneViewWebNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
