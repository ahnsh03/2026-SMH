#!/usr/bin/env python3
"""Run traffic-light detection on a live webcam, printing red/green per frame.

Headless by default (the board is normally reached over SSH), so the verdict
goes to the terminal instead of a window.

Examples:
    python3 test/manual/check_traffic_sign_webcam.py
    python3 test/manual/check_traffic_sign_webcam.py --save annotated.jpg
    python3 test/manual/check_traffic_sign_webcam.py --source red_light.png
    python3 test/manual/check_traffic_sign_webcam.py --show
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.traffic_sign import detect  # noqa: E402
from inference.types import TrafficSignal  # noqa: E402

_SIGNAL_COLOR = {
    TrafficSignal.RED: (0, 0, 235),
    TrafficSignal.GREEN: (0, 200, 0),
    TrafficSignal.UNKNOWN: (180, 180, 180),
}
_FPS_SMOOTHING = 0.9


def _annotate(frame, signal: TrafficSignal):
    annotated = frame.copy()
    cv2.putText(
        annotated,
        f'signal={signal.value}',
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        _SIGNAL_COLOR[signal],
        2,
    )
    return annotated


def _run_image(args: argparse.Namespace) -> int:
    frame = cv2.imread(str(args.source))
    if frame is None:
        print(f'cannot read image: {args.source}', file=sys.stderr)
        return 1

    signal = detect(frame).signal
    print(signal.value)
    if args.save:
        cv2.imwrite(str(args.save), _annotate(frame, signal))
        print(f'wrote {args.save}')
    return 0


def _run_camera(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f'cannot open /dev/video{args.device}', file=sys.stderr)
        return 1

    print(f'/dev/video{args.device} open — hold a traffic light up, Ctrl-C to stop')
    counts: Counter[TrafficSignal] = Counter()
    fps = 0.0
    previous = None

    try:
        while True:
            start = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                print('\nframe grab failed', file=sys.stderr)
                return 1

            signal = detect(frame).signal
            counts[signal] += 1

            # Keep the scrollback for transitions only; the rest is a status line.
            if signal is not previous:
                previous = signal
                print(f'\r{signal.value:<10}')
            print(f'\r{fps:5.1f} fps  {signal.value:<10}', end='', flush=True)

            if args.save:
                cv2.imwrite(str(args.save), _annotate(frame, signal))
            if args.show:
                cv2.imshow('traffic light', _annotate(frame, signal))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            elapsed = time.monotonic() - start
            if elapsed > 0:
                decay = _FPS_SMOOTHING if fps else 0.0
                fps = decay * fps + (1 - decay) * (1.0 / elapsed)
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    total = sum(counts.values())
    print(
        f'frames={total} '
        f'red={counts[TrafficSignal.RED]} '
        f'green={counts[TrafficSignal.GREEN]} '
        f'unknown={counts[TrafficSignal.UNKNOWN]}'
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Check traffic-light detection on a webcam.')
    parser.add_argument('--device', type=int, default=1, help='/dev/videoN index (default: 1)')
    parser.add_argument('--source', type=Path, help='run on an image file instead of the camera')
    parser.add_argument('--save', type=Path, help='write the annotated frame to this path')
    parser.add_argument('--show', action='store_true', help='display frames (needs DISPLAY)')
    args = parser.parse_args()

    return _run_image(args) if args.source else _run_camera(args)


if __name__ == '__main__':
    raise SystemExit(main())
