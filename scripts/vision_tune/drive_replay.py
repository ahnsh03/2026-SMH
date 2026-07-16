#!/usr/bin/env python3
"""주행 스택 오프라인 검증 — bag 재생 → 인지+제어(in-process) → 사람 /control 비교.

인지→제어를 lane_drive_node 와 동일하게 import 로 in-process 실행하고, Out 모드
제어 조향을 bag 의 사람 수동조향(/control)과 상관·부호일치로 비교한다. In 모드
상태 전이도 함께 추적한다(노랑/회전교차로 진입 여부).

예:
  python3 scripts/vision_tune/drive_replay.py \
      --bag /home/topst/2026-SMH/external/D-Racer-Kit/bagfile/bag_20260715_204143 --stride 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    sys.exit(f"cv2 필요: {exc}")

_WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WS / "src" / "inference"))
sys.path.insert(0, str(_WS / "src" / "driving"))
VISION_CFG = str(_WS / "config" / "lane_vision.yaml")

from inference.modules.lane_detection import LaneDetector      # noqa: E402
from driving.planner.mission import MissionController          # noqa: E402


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--config", default=VISION_CFG)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    from control_msgs.msg import Control

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(
        topics=["/camera/image/compressed", "/control"]))

    cam, cts, cst, cth = [], [], [], []
    while reader.has_next():
        tp, data, ts = reader.read_next()
        if tp == "/camera/image/compressed":
            cam.append((ts, data))
        else:
            m = deserialize_message(data, Control)
            cts.append(ts)
            cst.append(m.steering)
            cth.append(m.throttle)
    cts = np.array(cts, dtype=np.int64)
    cst = np.array(cst)
    cth = np.array(cth)
    print(f"cam={len(cam)} ctrl={len(cts)}", flush=True)

    def near(ts):
        i = min(max(int(np.searchsorted(cts, ts)), 0), len(cts) - 1)
        return cst[i], cth[i]

    det = LaneDetector(args.config)
    mout = MissionController(dict(course_mode="out"))
    minn = MissionController(dict(course_mode="in", roundabout_lap_time_s=0.0))

    my, hu, ht = [], [], []
    trans, last, prev, yf, n = [], None, None, 0, 0
    for k, (ts, data) in enumerate(cam):
        if k % args.stride:
            continue
        img = cv2.imdecode(np.frombuffer(
            deserialize_message(data, CompressedImage).data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        dt = 0.12 if prev is None else max(1e-3, (ts - prev) * 1e-9)
        prev = ts
        lane = det.detect(img)              # 1회 검출, 두 미션 공유
        co, _ = mout.plan(lane, dt)
        _, st = minn.plan(lane, dt)
        hs, htr = near(ts)
        my.append(co.steering)
        hu.append(hs)
        ht.append(htr)
        if lane.yellow_visible:
            yf += 1
        if st.reason != last:
            trans.append((n, st.reason, int(round(minn._heading * 57.3))))
            last = st.reason
        n += 1
        if n % 200 == 0:
            print(f"...{n} frames", flush=True)

    my = np.array(my)
    hu = np.array(hu)
    ht = np.array(ht)
    mov = np.abs(ht) > 0.01
    print("\n=== OUT-mode 제어 vs 사람 /control ===")
    print(f"frames={len(my)} moving={int(mov.sum())}")
    print(f"steer corr(moving)      : {_corr(my[mov], hu[mov]):+.3f}")
    sig = mov & (np.abs(hu) > 0.1)
    if sig.sum():
        agree = (np.sign(my[sig]) == np.sign(hu[sig])).mean() * 100
        print(f"sign agree(|human|>0.1) : {agree:.1f}%  n={int(sig.sum())}")
    print(f"my/human steer std      : {my.std():.3f} / {hu.std():.3f}")
    print("\n=== IN-mode 상태전이 (frame, state, heading deg) ===")
    print(f"yellow frames: {yf}/{n}")
    for t in trans:
        print("  ", t)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
