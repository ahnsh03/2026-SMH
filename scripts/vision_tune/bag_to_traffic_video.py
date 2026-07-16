#!/usr/bin/env python3
"""Convert a rosbag2 camera topic into an mp4 for the traffic-light tools.

Reads frames with the pure-Python ``rosbags`` library (no ROS environment
needed — unlike ``capture_from_bag.py``, which requires ``rclpy``/``rosbag2_py``
sourced from a ROS install). Output feeds directly into
``src/inference/test/tune_traffic_sign_video.py`` and
``src/inference/test/check_traffic_sign_video.py``, which only know ``--video``
(``cv2.VideoCapture``), not rosbag2.

Example:
    python3 scripts/vision_tune/bag_to_traffic_video.py \\
        --bag /mnt/c/bag_20260715_204143/bag_20260715_204143 \\
        --out data/captures/traffic_light_videos/bag_20260715_204143.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

_REPO_ROOT = Path(__file__).resolve().parents[2]
CAMERA_TOPIC = '/camera/image/compressed'


def iter_bag_timestamps(bag_dir: Path, topic: str):
    """Yield t_rel_seconds for each message on topic, without decoding payloads (cheap)."""
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = ', '.join(sorted({c.topic for c in reader.connections})) or '(none)'
            raise SystemExit(f'Topic {topic} not in bag. Available: {available}')

        t0_ns: int | None = None
        for _conn, timestamp, _rawdata in reader.messages(connections=conns):
            if t0_ns is None:
                t0_ns = timestamp
            yield (timestamp - t0_ns) * 1e-9


def iter_bag_frames(bag_dir: Path, topic: str):
    """Yield (t_rel_seconds, BGR frame) for a compressed-image topic in a rosbag2 dir."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = ', '.join(sorted({c.topic for c in reader.connections})) or '(none)'
            raise SystemExit(f'Topic {topic} not in bag. Available: {available}')

        t0_ns: int | None = None
        for _conn, timestamp, rawdata in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(rawdata, conns[0].msgtype)
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if t0_ns is None:
                t0_ns = timestamp
            yield (timestamp - t0_ns) * 1e-9, frame


def _resolve_bag(bag: Path) -> Path:
    path = bag.expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.is_dir() or not (path / 'metadata.yaml').is_file():
        raise SystemExit(f'Not a rosbag2 directory (missing metadata.yaml): {path}')
    return path


def _resolve_out(out: Path | None, bag_dir: Path) -> Path:
    if out is not None:
        path = out.expanduser()
        if not path.is_absolute():
            path = (_REPO_ROOT / path).resolve()
        return path
    return (_REPO_ROOT / 'data' / 'captures' / 'traffic_light_videos' / f'{bag_dir.name}.mp4').resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True, help='rosbag2 directory (has metadata.yaml)')
    parser.add_argument('--topic', default=CAMERA_TOPIC, help=f'camera topic (default: {CAMERA_TOPIC})')
    parser.add_argument('--out', type=Path, default=None, help='output mp4 (default: data/captures/traffic_light_videos/<bag_name>.mp4)')
    parser.add_argument('--start', type=float, default=0.0, help='start at bag-relative seconds (default: 0)')
    parser.add_argument('--end', type=float, default=None, help='stop at bag-relative seconds (default: end of bag)')
    args = parser.parse_args()

    bag_dir = _resolve_bag(args.bag)
    out_path = _resolve_out(args.out, bag_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'Reading {bag_dir} (topic={args.topic}) ...', flush=True)

    # Pass 1: timestamps only (no JPEG decode) to estimate fps within [start, end]
    # without buffering every decoded frame in memory.
    deltas: list[float] = []
    prev_t: float | None = None
    for t_rel in iter_bag_timestamps(bag_dir, args.topic):
        if t_rel < args.start:
            continue
        if args.end is not None and t_rel > args.end:
            break
        if prev_t is not None:
            dt = t_rel - prev_t
            if dt > 0:
                deltas.append(dt)
        prev_t = t_rel

    if prev_t is None:
        raise SystemExit(f'No frames on {args.topic} in [{args.start}, {args.end}] for {bag_dir}')

    fps = 1.0 / (sum(deltas) / len(deltas)) if deltas else 30.0
    fps = max(1.0, min(60.0, fps))

    # Pass 2: decode + write directly, streaming (only one frame in memory at a time).
    writer: cv2.VideoWriter | None = None
    written = 0
    width = height = 0
    for t_rel, frame in iter_bag_frames(bag_dir, args.topic):
        if t_rel < args.start:
            continue
        if args.end is not None and t_rel > args.end:
            break
        if writer is None:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            if not writer.isOpened():
                raise SystemExit(f'Could not open VideoWriter for {out_path}')
        writer.write(frame)
        written += 1

    if writer is not None:
        writer.release()

    print(f'frames={written} fps={fps:.2f} size={width}x{height} -> {out_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
