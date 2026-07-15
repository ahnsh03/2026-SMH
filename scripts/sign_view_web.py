#!/usr/bin/env python3
"""Stream live direction-sign detection to a browser as MJPEG (headless board).

Subscribes to the camera's CompressedImage topic, runs the trained ONNX
direction-sign model (weights/sign_best.onnx via direction_sign.detect_signs;
falls back to the blue-circle/white-arrow rule if ONNX Runtime is missing),
draws the LEFT/RIGHT boxes, and serves the annotated frames over HTTP.

The board is headless (reached over SSH), so there is no cv2.imshow window —
open the printed URL in a browser on your laptop instead.

    ros2 run camera camera_node &
    python3 scripts/sign_view_web.py
    # browser → http://<board-ip>:8088
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage

# Make the `inference` package importable without sourcing the install overlay.
_PKG_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'inference'
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.direction_sign import detect_signs  # noqa: E402

# BGR, matches view_compressed_topic.py: LEFT=orange, RIGHT=green.
_TURN_COLOR = {'left': (0, 165, 255), 'right': (0, 255, 0)}


class _Latest:
    """The most recent JPEG frame, shared between the ROS node and HTTP server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._event = threading.Event()

    def set(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
        self._event.set()

    def wait_for_next(self, timeout: float) -> bytes | None:
        if not self._event.wait(timeout):
            return None
        self._event.clear()
        with self._lock:
            return self._jpeg


class SignViewNode(Node):
    def __init__(self, topic: str, latest: _Latest, max_hz: float,
                 conf: float, jpeg_quality: int) -> None:
        super().__init__('sign_view_web')
        self.latest = latest
        self.conf = conf
        self.jpeg_quality = jpeg_quality
        self.min_period = 1.0 / max(0.5, max_hz)
        self._last = 0.0
        self._frames = 0
        self._hits = 0
        self._fps = 0.0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(CompressedImage, topic, self._on_image, qos)
        self.get_logger().info(f'sign_view: {topic} → MJPEG @ ≤{max_hz:.1f} Hz')

    def _on_image(self, msg: CompressedImage) -> None:
        now = time.monotonic()
        if now - self._last < self.min_period:
            return
        dt = now - self._last
        self._last = now

        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return

        try:
            detections = [d for d in detect_signs(frame) if d.score >= self.conf]
        except Exception as exc:  # keep the stream alive on a bad frame/model call
            self.get_logger().warning(f'detect failed: {exc}')
            detections = []

        self._frames += 1
        if detections:
            self._hits += 1
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

        self._annotate(frame, detections)
        ok, buf = cv2.imencode(
            '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if ok:
            self.latest.set(buf.tobytes())

    def _annotate(self, frame: np.ndarray, detections) -> None:
        for det in detections:
            color = _TURN_COLOR.get(det.turn.value, (200, 200, 200))
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(
                frame, f'{det.turn.value.upper()} {det.score:.2f}',
                (p1[0], max(p1[1] - 7, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2, cv2.LINE_AA,
            )

        top = detections[0].turn.value.upper() if detections else 'none'
        status = f'sign={top}  det={len(detections)}  {self._fps:4.1f}fps  hits={self._hits}/{self._frames}'
        cv2.rectangle(frame, (6, 6), (6 + 9 * len(status), 34), (0, 0, 0), -1)
        cv2.putText(frame, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (240, 240, 240), 1, cv2.LINE_AA)


_PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>direction-sign view</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center}
img{max-width:100vw;max-height:100vh}</style></head>
<body><img src="/stream"></body></html>"""


def _make_handler(latest: _Latest):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):  # silence per-request logging
            pass

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(_PAGE)
                return
            if self.path != '/stream':
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header(
                'Content-Type', 'multipart/x-mixed-replace; boundary=frame'
            )
            self.end_headers()
            try:
                while True:
                    jpeg = latest.wait_for_next(timeout=5.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n')
                    self.wfile.write(
                        f'Content-Length: {len(jpeg)}\r\n\r\n'.encode()
                    )
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser tab closed

    return Handler


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        s.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--max-hz', type=float, default=8.0)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--jpeg-quality', type=int, default=70)
    args = parser.parse_args()

    latest = _Latest()
    server = ThreadingHTTPServer(('0.0.0.0', args.port), _make_handler(latest))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f'browser → http://{_lan_ip()}:{args.port}   (Ctrl-C to stop)', flush=True)

    rclpy.init()
    node = SignViewNode(
        args.topic, latest, args.max_hz, args.conf, args.jpeg_quality
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
