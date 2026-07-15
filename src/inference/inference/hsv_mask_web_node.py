"""Low-rate HSV mask + lane-boundary preview → monitor debug panels.

Publishes JPEG CompressedImage on the monitor's OpenCV debug topics so the
web UI can show, in one panel each:
  grayscale = WHITE  : HSV 마스크 | 흰 차선 경계 결과
  blur      = YELLOW : HSV 마스크 | 노란 차선 경계 결과
  edge      = 주행가능영역(black|red) 1개
경계는 lane_detection.detect_with_debug 로 만든다(= 실제 주행 인지와 동일 경로).
IPM+inRange만 쓰던 이전보다 무겁다(detect 1회 ~255ms). 기본 ~1.5 Hz 유지.

  ros2 launch inference mask_check.launch.py
  # browser → http://<board-ip>:5000
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage


def _vision_tune_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'scripts' / 'vision_tune'
        if (candidate / 'metric_ipm.py').is_file() and (candidate / 'hsv.py').is_file():
            return candidate
    raise ImportError('scripts/vision_tune/{metric_ipm,hsv}.py not found')


_VT = _vision_tune_dir()
if str(_VT) not in sys.path:
    sys.path.insert(0, str(_VT))

from hsv import load_hsv_ranges, make_mask  # noqa: E402
from metric_ipm import load_metric_ipm, warp_metric_ipm  # noqa: E402

from inference.modules import lane_detection as ld  # noqa: E402


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
    out = bgr
    cv2.putText(
        out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA
    )
    return out


def _masked_bev(bev: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep BEV pixels only where mask is set; everywhere else black."""
    out = np.zeros_like(bev)
    selected = mask > 0
    if np.any(selected):
        out[selected] = bev[selected]
    return out


def _red_mask(bev_bgr: np.ndarray, rng, wrap: int) -> np.ndarray:
    """Primary red inRange (+ low-H wrap, same contract as lane_detection)."""
    hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
    p = rng.clamp()
    mask = cv2.inRange(hsv, p.lower(), p.upper())
    if wrap > 0:
        lo = np.array([0, p.s_min, p.v_min], dtype=np.uint8)
        hi = np.array([wrap, p.s_max, p.v_max], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


class HsvMaskWebNode(Node):
    def __init__(self) -> None:
        super().__init__('hsv_mask_web_node')
        self.declare_parameter('compressed_topic', '/camera/image/compressed')
        self.declare_parameter('max_hz', 1.5)
        self.declare_parameter('jpeg_quality', 55)
        self.declare_parameter('white_topic', '/opencv/image/grayscale')
        self.declare_parameter('yellow_topic', '/opencv/image/blur')
        self.declare_parameter('mosaic_topic', '/opencv/image/edge')
        self.declare_parameter('vision_config', '')

        self.max_hz = max(0.5, float(self.get_parameter('max_hz').value))
        self.jpeg_quality = int(np.clip(self.get_parameter('jpeg_quality').value, 30, 90))
        cfg = str(self.get_parameter('vision_config').value).strip()
        self.cfg_path = Path(cfg).expanduser() if cfg else None

        self.ipm = load_metric_ipm(self.cfg_path)
        self.ranges = load_hsv_ranges(self.cfg_path)
        self.red_wrap = 15
        try:
            path = self.cfg_path or (_VT.parents[1] / 'config' / 'lane_vision.yaml')
            with path.open('r', encoding='utf-8') as stream:
                data = yaml.safe_load(stream) or {}
            wrap = (data.get('detect_tune') or {}).get('red_h_low_wrap')
            if wrap is not None:
                self.red_wrap = int(np.clip(int(wrap), 0, 30))
        except OSError:
            pass

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        topic = str(self.get_parameter('compressed_topic').value)
        self.create_subscription(CompressedImage, topic, self._on_image, qos)

        # depth=1 publishers — monitor only needs the latest JPEG
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
        self.pub_mosaic = self.create_publisher(
            CompressedImage, str(self.get_parameter('mosaic_topic').value), pub_qos
        )

        # detect_with_debug 내부 cv2.imshow 경로를 끈다(헤드리스 보드).
        ld.VISUALIZE = False
        ld.VISUALIZE_MODE = ld.VISUALIZE_OFF

        self._last = 0.0
        self.get_logger().info(
            f'HSV mask + boundary web: {topic} → monitor panels @ ≤{self.max_hz:.1f} Hz '
            f'(WHITE/YELLOW = HSV|경계, EDGE = 주행가능영역; red_wrap={self.red_wrap})'
        )

    def _on_image(self, msg: CompressedImage) -> None:
        now = time.monotonic()
        if now - self._last < 1.0 / self.max_hz:
            return
        self._last = now

        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return

        # 차선 경계는 실제 주행 인지와 동일 경로(detect_with_debug)로 만든다.
        try:
            _lane, dbg = ld.detect_with_debug(frame)
        except Exception as exc:
            self.get_logger().warning(f'detect failed: {exc}')
            return

        bev = dbg.bev
        if bev is None or bev.size == 0:
            return

        # HSV 마스크는 기존과 동일하게 이 노드가 inRange로 계산(주행가능영역 패널의
        # black|red 표시를 그대로 유지). 경계는 dbg에서 가져온다.
        white = make_mask(bev, self.ranges['white'], morph=True)
        yellow = make_mask(bev, self.ranges['yellow'], morph=True)
        black = make_mask(bev, self.ranges['black_road'], morph=True)
        red = _red_mask(bev, self.ranges['red_road'], self.red_wrap)
        road = cv2.bitwise_or(black, red)

        # 흰: HSV 마스크 | 흰 차선 경계 결과
        white_combo = _pair(
            _label(_masked_bev(bev, white), f'WHITE HSV  px={int(np.count_nonzero(white))}'),
            _label(
                ld.make_boundary_preview(
                    bev, dbg.road_clean, dbg.white_left, dbg.white_right, 'WHITE',
                ),
                'WHITE BOUNDARY',
            ),
        )
        # 노랑: HSV 마스크 | 노란 차선 경계 결과
        yellow_combo = _pair(
            _label(_masked_bev(bev, yellow), f'YELLOW HSV  px={int(np.count_nonzero(yellow))}'),
            _label(
                ld.make_boundary_preview(
                    bev, dbg.road_clean, dbg.yellow_left, dbg.yellow_right, 'YELLOW',
                ),
                'YELLOW BOUNDARY',
            ),
        )
        # 주행가능영역: 한 창에 1개 그대로.
        road_v = _label(
            _masked_bev(bev, road),
            f'DRIVABLE(black|red)  px={int(np.count_nonzero(road))}',
        )

        self._publish(self.pub_white, white_combo)
        self._publish(self.pub_yellow, yellow_combo)
        self._publish(self.pub_mosaic, road_v)

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
    node = HsvMaskWebNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
