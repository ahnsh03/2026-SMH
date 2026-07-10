#!/usr/bin/env python3
"""Live left/right sign recognition from a webcam — manual check for weights/sign_best.onnx.

Prints the most confident detection per frame. Headless by default (no window),
since the board is normally reached over SSH.

    python3 scripts/check_sign_webcam.py                  # /dev/video1, terminal only
    python3 scripts/check_sign_webcam.py --save out.jpg   # also write annotated frames
    python3 scripts/check_sign_webcam.py --source pic.jpg # single image instead of camera
    python3 scripts/check_sign_webcam.py --show           # OpenCV window (needs DISPLAY)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_PKG_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'inference'
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inference.modules.direction_sign import Detection, detect_signs  # noqa: E402
from inference.types import TurnSign  # noqa: E402

_BOX_COLOR = {TurnSign.LEFT: (0, 165, 255), TurnSign.RIGHT: (0, 255, 0)}
_FPS_SMOOTHING = 0.9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', type=int, default=1, help='/dev/videoN index (default: 1)')
    parser.add_argument('--source', type=Path, help='run on an image file instead of the camera')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence floor (default: 0.25)')
    parser.add_argument('--save', type=Path, help='write the annotated frame to this path')
    parser.add_argument('--show', action='store_true', help='open an OpenCV window (needs DISPLAY)')
    return parser.parse_args()


def _annotate(frame, detections: list[Detection]):
    canvas = frame.copy()
    for detection in detections:
        color = _BOX_COLOR[detection.turn]
        corner1 = (int(detection.x1), int(detection.y1))
        corner2 = (int(detection.x2), int(detection.y2))
        cv2.rectangle(canvas, corner1, corner2, color, 2)
        cv2.putText(
            canvas,
            f'{detection.turn.value.upper()} {detection.score:.2f}',
            (corner1[0], max(corner1[1] - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return canvas


def _describe(detections: list[Detection]) -> str:
    if not detections:
        return 'no sign'
    top = detections[0]
    extra = f' (+{len(detections) - 1} more)' if len(detections) > 1 else ''
    return f'{top.turn.value.upper():<5} {top.score:.2f}{extra}'


def _run_once(args: argparse.Namespace) -> int:
    frame = cv2.imread(str(args.source))
    if frame is None:
        print(f'cannot read image: {args.source}', file=sys.stderr)
        return 1

    detections = [d for d in detect_signs(frame) if d.score >= args.conf]
    print(_describe(detections))
    if args.save:
        cv2.imwrite(str(args.save), _annotate(frame, detections))
        print(f'wrote {args.save}')
    return 0


def _run_camera(args: argparse.Namespace) -> int:
    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        print(f'cannot open /dev/video{args.device}', file=sys.stderr)
        return 1

    print(f'/dev/video{args.device} open — hold a sign up to the camera, Ctrl-C to stop')
    fps = 0.0
    previous = None
    try:
        while True:
            start = time.monotonic()
            ok, frame = capture.read()
            if not ok:
                print('\nframe grab failed', file=sys.stderr)
                return 1

            detections = [d for d in detect_signs(frame) if d.score >= args.conf]
            summary = _describe(detections)

            # Keep the scrollback for transitions only; the rest is a live status line.
            if summary.split()[0] != (previous or ''):
                previous = summary.split()[0]
                print(f'\r{summary:<30}')
            print(f'\r{fps:5.1f} fps  {summary:<30}', end='', flush=True)

            if args.save:
                cv2.imwrite(str(args.save), _annotate(frame, detections))
            if args.show:
                cv2.imshow('direction sign', _annotate(frame, detections))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            elapsed = time.monotonic() - start
            if elapsed > 0:
                instant = 1.0 / elapsed
                decay = _FPS_SMOOTHING if fps else 0.0
                fps = decay * fps + (1 - decay) * instant
    except KeyboardInterrupt:
        print()
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = _parse_args()
    return _run_once(args) if args.source else _run_camera(args)


if __name__ == '__main__':
    raise SystemExit(main())
