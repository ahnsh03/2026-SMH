#!/usr/bin/env python3
"""Compare OpenCV vs Sungjun YOLO traffic-light backends side by side.

WARNING: enabling the light YOLO while the sign YOLO is also used means
two ONNX forwards per frame — fine for a parked A/B check, not for race.

Usage:

  PYTHONPATH=src/inference python3 scripts/check_traffic_light_ab.py --image frame.jpg
  PYTHONPATH=src/inference python3 scripts/check_traffic_light_ab.py --webcam 0

Env:

  TRAFFIC_LIGHT_BACKEND=opencv|yolo|yolo_then_opencv|opencv_then_yolo
  TRAFFIC_LIGHT_MODEL_PATH=/path/to/sign_light_best_v5b.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'inference'))

import cv2  # noqa: E402

from inference.modules.traffic_sign import detect_signal_both  # noqa: E402


def _label(value: object) -> str:
    return getattr(value, 'name', str(value))


def _annotate(frame, report: dict) -> None:
    lines = [
        f"mode={report['mode']}",
        f"opencv={_label(report['opencv'])}",
        f"yolo={_label(report['yolo'])}",
        f"selected={_label(report['selected'])}",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        y += 22


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--image', type=Path, help='Still BGR image path')
    src.add_argument('--webcam', type=int, nargs='?', const=0, help='Webcam index')
    args = parser.parse_args()

    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            print(f'failed to read {args.image}', file=sys.stderr)
            return 1
        report = detect_signal_both(frame)
        print(report)
        _annotate(frame, report)
        cv2.imshow('traffic_light_ab', frame)
        cv2.waitKey(0)
        return 0

    cap = cv2.VideoCapture(int(args.webcam))
    if not cap.isOpened():
        print(f'failed to open webcam {args.webcam}', file=sys.stderr)
        return 1
    print('q = quit')
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        report = detect_signal_both(frame)
        print(
            f"opencv={_label(report['opencv'])} "
            f"yolo={_label(report['yolo'])} "
            f"selected={_label(report['selected'])} "
            f"mode={report['mode']}",
            flush=True,
        )
        _annotate(frame, report)
        cv2.imshow('traffic_light_ab', frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
