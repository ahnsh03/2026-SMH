#!/usr/bin/env python3
"""Phase 0: hotkey capture from /camera/image/compressed.

Sim already shows "D-Racer Camera". This tool only listens and saves when you
press a key — it does not auto-dump many frames.

Examples (2026-smh-sim, after source /opt/ros/humble/setup.bash):

  python3 scripts/vision_tune/capture_camera.py --out data/captures/sim

Keys (focus the capture window):
  c / SPACE  save current frame as PNG
  q / ESC    quit
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CompressedImage
except ModuleNotFoundError as exc:
    raise SystemExit(
        'rclpy/sensor_msgs not found. Run inside 2026-smh-sim (or the board) after:\n'
        '  source /opt/ros/humble/setup.bash\n'
        f'Original error: {exc}'
    ) from exc

# Same initial size as sim D-Racer Camera preview.
PREVIEW_W = 640
PREVIEW_H = 360
WIN_NAME = 'capture_hotkey (c=save)'


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')


def _host_ids() -> tuple[int, int] | None:
    """Prefer HOST_UID/GID; else inherit non-root owner of the output parent."""
    env_uid = os.environ.get('HOST_UID') or os.environ.get('SUDO_UID')
    env_gid = os.environ.get('HOST_GID') or os.environ.get('SUDO_GID')
    if env_uid is not None:
        return int(env_uid), int(env_gid if env_gid is not None else env_uid)
    return None


def _chown_to_host(path: Path, fallback_dir: Path | None = None) -> None:
    """Avoid root-owned captures that the host user cannot delete."""
    if os.geteuid() != 0:
        return
    ids = _host_ids()
    if ids is None and fallback_dir is not None:
        probe = fallback_dir if fallback_dir.exists() else fallback_dir.parent
        try:
            st = probe.stat()
            if st.st_uid != 0:
                ids = (st.st_uid, st.st_gid)
        except OSError:
            ids = None
    if ids is None:
        # Workspace mount is usually owned by the developer (uid 1000).
        ids = (1000, 1000)
    try:
        os.chown(path, ids[0], ids[1])
    except OSError:
        pass


def _ensure_out_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _chown_to_host(out_dir, out_dir.parent)
    return out_dir


class CaptureNode(Node):
    def __init__(self, topic: str, out_dir: Path):
        super().__init__('camera_capture')
        self.out_dir = _ensure_out_dir(out_dir)
        self.saved = 0
        self.latest: np.ndarray | None = None
        self.frame_count = 0
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            # Hotkey capture wants the newest frame, not a backlog. Revisit if
            # burst capture is ever added.
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(CompressedImage, topic, self._on_image, qos)
        self.get_logger().info(
            f'Hotkey capture on {topic} → {out_dir}  (c/SPACE=save, q=quit)'
        )

    def _on_image(self, msg: CompressedImage) -> None:
        frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        self.latest = frame
        self.frame_count += 1

    def save_latest(self) -> Path | None:
        if self.latest is None:
            self.get_logger().warning('No frame yet — is sim-bringup running?')
            return None
        name = f'frame_{_stamp()}_{self.saved:04d}.png'
        path = self.out_dir / name
        cv2.imwrite(str(path), self.latest)
        self.saved += 1
        h, w = self.latest.shape[:2]
        meta = path.with_suffix('.txt')
        meta.write_text(
            f'width={w}\nheight={h}\nsource=compressed\n',
            encoding='utf-8',
        )
        _chown_to_host(path, self.out_dir)
        _chown_to_host(meta, self.out_dir)
        self.get_logger().info(f'saved {path.name} ({w}x{h}) total={self.saved}')
        return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('data/captures/sim'),
        help='output directory (default: data/captures/sim)',
    )
    args = parser.parse_args()

    rclpy.init()
    node = CaptureNode(args.topic, args.out.expanduser().resolve())
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, PREVIEW_W, PREVIEW_H)
    print(f'Focus "{WIN_NAME}" → c/SPACE save, q quit', flush=True)

    try:
        while rclpy.ok():
            for _ in range(8):
                rclpy.spin_once(node, timeout_sec=0.0)
            if node.latest is not None:
                view = cv2.resize(
                    node.latest,
                    (PREVIEW_W, PREVIEW_H),
                    interpolation=cv2.INTER_NEAREST,
                )
                cv2.putText(
                    view,
                    f'c=save  saved={node.saved}  frames={node.frame_count}',
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WIN_NAME, view)
            else:
                blank = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    f'waiting for {args.topic} ...',
                    (24, PREVIEW_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WIN_NAME, blank)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key in (ord('c'), ord(' ')):
                node.save_latest()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print(f'saved_total={node.saved} dir={node.out_dir}')
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
