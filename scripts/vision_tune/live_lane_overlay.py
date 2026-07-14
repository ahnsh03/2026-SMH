#!/usr/bin/env python3
"""Live illumination-invariant lane / drivable-area overlay, viewed in a browser.

The board is headless (no DISPLAY), so cv2.imshow cannot open a window. Instead
this node subscribes to the live camera topic, runs the SAME detector as
`preview_lane_illuminv.py` (imported, not copied -- so what you see here is the
real code path), and serves a raw|overlay MJPEG stream over HTTP. Open the
printed URL in a laptop browser on the same network.

It only consumes the ROS topic, never opens /dev/video1 -- so it does not
conflict with camera_node, and it does not touch the monitor's config.

Run (on the car, after the camera is already publishing):

  # terminal 1 -- bring the camera up (exposure-lock patch applies here)
  ros2 run camera camera_node        # or your usual driving launch

  # terminal 2
  source /opt/ros/humble/setup.bash
  python3 scripts/vision_tune/live_lane_overlay.py
  #   -> open http://<board-ip>:8081 in a browser

Keys don't apply (no window); tune live with flags: --mpp --blue-thr --red-thr
--line-frac. Ctrl-C to stop.
"""

from __future__ import annotations

import argparse
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview_lane_illuminv import LaneIlluminInv  # noqa: E402  (same detector)


class LatestFrame:
    """Thread-safe hand-off of the newest JPEG from ROS thread to HTTP threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None
        self._event = threading.Condition(self._lock)

    def set(self, jpeg_bytes):
        with self._event:
            self._jpeg = jpeg_bytes
            self._event.notify_all()

    def wait_next(self, timeout=1.0):
        with self._event:
            self._event.wait(timeout)
            return self._jpeg

    def get(self):
        with self._lock:
            return self._jpeg


PAGE = b"""<!doctype html><meta charset=utf-8>
<title>live lane overlay</title>
<style>body{margin:0;background:#111;color:#ccc;font:14px sans-serif;text-align:center}
img{max-width:100vw;image-rendering:pixelated;margin-top:8px}
p{margin:6px}</style>
<p>illumination-invariant lane / drivable-area &mdash; raw | overlay
(red=lane, green=drivable, blue=off-track)</p>
<img src="/stream.mjpg">
"""


def make_handler(latest: LatestFrame):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):        # silence per-request logging
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        jpeg = latest.wait_next()
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return          # browser tab closed; drop this stream
                return
            self.send_error(404)

    return Handler


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


class OverlayRenderer:
    """Frame -> (raw|overlay) JPEG. No ROS dependency, so it is unit-testable."""

    def __init__(self, args):
        self.args = args
        self.det = None
        self.scale = args.scale
        self.jpeg_q = args.jpeg_quality
        # rolling FPS estimate
        self._t = cv2.getTickCount()
        self._fps = 0.0

    def _ensure_det(self, h, w):
        if self.det is None:
            self.det = LaneIlluminInv(
                img_w=w, img_h=h, meters_per_pixel=self.args.mpp,
                blue_thr=self.args.blue_thr, red_thr=self.args.red_thr,
                line_thr_frac=self.args.line_frac)
            print(f"detector: BEV {self.det.bev_w}x{self.det.bev_h} "
                  f"@ {self.args.mpp} m/px  tophat kernel={self.det.kernel_px}px")

    def render(self, img):
        h, w = img.shape[:2]
        self._ensure_det(h, w)
        d = self.det.detect(img)
        ov, cov = self.det.overlay(d)
        raw = cv2.resize(img, (self.det.bev_w, self.det.bev_h))
        view = np.hstack([raw, ov])
        if self.scale != 1:
            view = cv2.resize(view, None, fx=self.scale, fy=self.scale,
                              interpolation=cv2.INTER_NEAREST)
        # fps
        now = cv2.getTickCount()
        dt = (now - self._t) / cv2.getTickFrequency()
        self._t = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        cv2.putText(view, f"{self._fps:4.1f} fps   drivable {cov:3.0f}%",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        ok, buf = cv2.imencode(".jpg", view,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        return buf.tobytes() if ok else None


def build_parser():
    ap = argparse.ArgumentParser(
        description="Live lane/drivable overlay over MJPEG (headless-friendly).")
    ap.add_argument("--topic", default="/camera/image/compressed")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--mpp", type=float, default=0.006)
    ap.add_argument("--blue-thr", type=float, default=0.06)
    ap.add_argument("--red-thr", type=float, default=0.12)
    ap.add_argument("--line-frac", type=float, default=0.45)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--selftest", action="store_true",
                    help="feed one bag frame through the renderer and exit "
                         "(no ROS, no camera) -- takes --bag")
    ap.add_argument("--bag", help="bag dir for --selftest")
    return ap


def run_selftest(args):
    """Headless CI: prove render() produces a JPEG from a real frame."""
    from preview_lane_illuminv import load_compressed_frames
    r = OverlayRenderer(args)
    n = 0
    for img in load_compressed_frames(args.bag, args.topic):
        jpeg = r.render(img)
        assert jpeg and jpeg[:2] == b"\xff\xd8", "not a JPEG"
        n += 1
        if n >= 5:
            break
    print(f"selftest OK: rendered {n} frames, last JPEG {len(jpeg)} bytes")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        if not args.bag:
            build_parser().error("--selftest needs --bag")
        return run_selftest(args)

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                           ReliabilityPolicy)
    from sensor_msgs.msg import CompressedImage

    latest = LatestFrame()
    renderer = OverlayRenderer(args)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(latest))
    threading.Thread(target=server.serve_forever, daemon=True).start()

    import socket
    ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except OSError:
        pass
    print(f"MJPEG server up -> open  http://{ip}:{args.port}   "
          f"(binding {args.host}:{args.port})")
    print("waiting for frames on", args.topic, "...")

    rclpy.init()
    node = Node("live_lane_overlay")
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.VOLATILE)

    def on_image(msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        jpeg = renderer.render(img)
        if jpeg is not None:
            latest.set(jpeg)

    node.create_subscription(CompressedImage, args.topic, on_image, qos)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
