"""Low-rate HSV mask preview → monitor debug panels (no full lane detection).

Publishes JPEG CompressedImage on the monitor's OpenCV debug topics so the
web UI can show white / yellow / road masks without doubling ``inference_node``
work. IPM warp + ``inRange`` only; default ~1.5 Hz.

  ros2 launch inference mask_check.launch.py
  # browser → http://<board-ip>:5000  (Grayscale=white, Blur=yellow, Edge=road+mosaic)
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


def _label(bgr: np.ndarray, text: str) -> np.ndarray:
    out = bgr
    cv2.putText(
        out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
    )
    cv2.putText(
        out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA
    )
    return out


def _mask_bgr(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask > 0] = color
    return canvas


def _overlay(bev: np.ndarray, mask: np.ndarray, color: tuple[int, int, int],
             alpha: float = 0.55) -> np.ndarray:
    """Tint masked pixels on the BEV so the lane/road context stays visible."""
    out = bev.copy()
    selected = mask > 0
    if not np.any(selected):
        return out
    tint = np.array(color, dtype=np.float32)
    out[selected] = (
        (1.0 - alpha) * out[selected].astype(np.float32) + alpha * tint
    ).astype(np.uint8)
    return out


def _binary_side(bev: np.ndarray, mask: np.ndarray,
                 color: tuple[int, int, int]) -> np.ndarray:
    """BEV | solid mask side-by-side for unambiguous mask check."""
    left = bev
    right = _mask_bgr(mask, color)
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    return np.hstack([left[:h, :w], right[:h, :w]])


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

        self._last = 0.0
        self.get_logger().info(
            f'HSV mask web: {topic} → monitor panels @ ≤{self.max_hz:.1f} Hz '
            f'(IPM+inRange only, no lane detect; red_wrap={self.red_wrap})'
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
            bev = warp_metric_ipm(frame, self.ipm)
        except Exception as exc:
            self.get_logger().warning(f'IPM failed: {exc}')
            return

        white = make_mask(bev, self.ranges['white'], morph=True)
        yellow = make_mask(bev, self.ranges['yellow'], morph=True)
        black = make_mask(bev, self.ranges['black_road'], morph=True)
        red = _red_mask(bev, self.ranges['red_road'], self.red_wrap)

        # Monitor UI titles stay Grayscale/Blur/Edge — bake HSV names into pixels.
        # Left=BEV, right=binary mask (clearly not OpenCV grayscale/blur/edge).
        white_v = _label(
            _binary_side(bev, white, (255, 255, 255)),
            f'WHITE mask  px={int(np.count_nonzero(white))}',
        )
        yellow_v = _label(
            _binary_side(bev, yellow, (0, 220, 255)),
            f'YELLOW mask  px={int(np.count_nonzero(yellow))}',
        )

        # Edge: all masks tinted on BEV (W=white Y=cyan Bk=gray R=red).
        edge_v = bev.copy()
        edge_v = _overlay(edge_v, black, (60, 60, 60), alpha=0.45)
        edge_v = _overlay(edge_v, red, (0, 0, 255), alpha=0.55)
        edge_v = _overlay(edge_v, yellow, (0, 220, 255), alpha=0.55)
        edge_v = _overlay(edge_v, white, (255, 255, 255), alpha=0.65)
        edge_v = _label(edge_v, 'ALL overlay  W=white Y=cyan Bk=gray R=red')
        if edge_v.shape[1] > 640:
            edge_v = cv2.resize(
                edge_v, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_AREA
            )
        if white_v.shape[1] > 640:
            white_v = cv2.resize(
                white_v, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA
            )
            yellow_v = cv2.resize(
                yellow_v, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA
            )

        self._publish(self.pub_white, white_v)
        self._publish(self.pub_yellow, yellow_v)
        self._publish(self.pub_mosaic, edge_v)

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
