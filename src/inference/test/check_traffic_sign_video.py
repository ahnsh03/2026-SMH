"""Run traffic-light detection on a video or a generated red-light demo.

Examples:
    python test/check_traffic_sign_video.py --generate-demo --show --loop
    python test/check_traffic_sign_video.py --video path/to/red_light.mp4 --output annotated.mp4
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.traffic_sign import detect  # noqa: E402
from inference.types import TrafficSignal  # noqa: E402


def _make_demo_video(path: Path, width: int = 640, height: int = 360, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f'Could not create demo video: {path}')

    for frame_idx in range(fps * 5):
        frame = np.full((height, width, 3), (35, 35, 35), dtype=np.uint8)
        cv2.rectangle(frame, (255, 55), (385, 305), (18, 18, 18), -1)
        cv2.rectangle(frame, (255, 55), (385, 305), (80, 80, 80), 3)
        cv2.circle(frame, (320, 115), 38, (0, 0, 235), -1)
        cv2.circle(frame, (320, 115), 45, (20, 20, 20), 5)
        cv2.circle(frame, (320, 180), 38, (25, 25, 25), -1)
        cv2.circle(frame, (320, 245), 38, (25, 25, 25), -1)

        # Small motion/noise keeps this closer to a video stream than a still image.
        offset = int(8 * np.sin(frame_idx / 7.0))
        frame = np.roll(frame, offset, axis=1)
        noise = np.random.default_rng(frame_idx).integers(0, 8, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        writer.write(frame)

    writer.release()


def _draw_result(frame: np.ndarray, signal: TrafficSignal, frame_idx: int) -> np.ndarray:
    annotated = frame.copy()
    color = (0, 0, 255) if signal == TrafficSignal.RED else (0, 255, 0)
    if signal == TrafficSignal.UNKNOWN:
        color = (180, 180, 180)
    text = f'frame={frame_idx} signal={signal.value}'
    cv2.putText(annotated, text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return annotated


def _run_video(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f'Could not create output video: {output_path}')

    counts: Counter[TrafficSignal] = Counter()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            if args.loop and args.show:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        result = detect(frame)
        counts[result.signal] += 1
        annotated = _draw_result(frame, result.signal, frame_idx)

        if writer is not None:
            writer.write(annotated)
        if args.show:
            cv2.imshow('traffic sign detection', annotated)
            key = cv2.waitKey(args.delay_ms) & 0xFF
            if key == ord('q') or key == 27:
                break

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    total = sum(counts.values())
    red_ratio = counts[TrafficSignal.RED] / total if total else 0.0
    print(f'video={video_path}')
    print(f'frames={total}')
    print(f'red={counts[TrafficSignal.RED]} green={counts[TrafficSignal.GREEN]} unknown={counts[TrafficSignal.UNKNOWN]}')
    print(f'red_ratio={red_ratio:.3f}')

    if red_ratio < args.min_red_ratio:
        print(f'FAIL: red ratio below threshold {args.min_red_ratio:.3f}', file=sys.stderr)
        return 1
    print('PASS: red traffic light detected')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Check traffic-light detection on video frames.')
    parser.add_argument('--video', help='Input video path. If omitted with --generate-demo, a demo is created.')
    parser.add_argument('--generate-demo', action='store_true', help='Create and run a synthetic red-light video.')
    parser.add_argument('--show', action='store_true', help='Display annotated frames. Press q to quit.')
    parser.add_argument('--loop', action='store_true', help='Loop playback when used with --show.')
    parser.add_argument('--delay-ms', type=int, default=1, help='Display delay between frames.')
    parser.add_argument('--output', help='Optional annotated output mp4 path.')
    parser.add_argument('--max-frames', type=int, default=0, help='Optional frame limit.')
    parser.add_argument('--min-red-ratio', type=float, default=0.80, help='Required red detection ratio.')
    args = parser.parse_args()

    if args.generate_demo:
        demo_path = Path(__file__).with_name('artifacts') / 'red_light_demo.mp4'
        _make_demo_video(demo_path)
        if not args.video:
            args.video = str(demo_path)

    if not args.video:
        parser.error('provide --video path/to/file.mp4 or use --generate-demo')

    return _run_video(args)


if __name__ == '__main__':
    raise SystemExit(main())
