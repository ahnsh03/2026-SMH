#!/usr/bin/env python3
"""bag 재생 → metric BEV 차선검출 검증/튜닝 도구 (오프라인, ROS 노드 불필요).

용도
----
- 실차 백파일을 재생해 LaneDetector 검출을 프레임 단위로 확인·튜닝한다.
- HSV 재tap(조명 바뀌면), 검출률/센터라인 통계, 오버레이 이미지·mp4 생성.

예시
----
  # 전체 검증 + 샘플 오버레이 저장
  python3 scripts/vision_tune/lane_replay.py \
      --bag /home/topst/2026-SMH/external/D-Racer-Kit/bagfile/bag_20260715_204143 \
      --out /tmp/lane_out --stride 20

  # HSV 통계만 (밝은/노란 픽셀 분포)
  python3 scripts/vision_tune/lane_replay.py --bag <bag> --hsv-stats --stride 40

  # mp4 로 검출 영상 저장(원본|BEV 오버레이)
  python3 scripts/vision_tune/lane_replay.py --bag <bag> --video /tmp/lane.mp4 --stride 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    sys.exit(f"cv2 필요: {exc}")

# team-new src/inference 를 import 경로에 추가
_HERE = Path(__file__).resolve()
_WS = _HERE.parents[2]                       # 2026-SMH-team-new
sys.path.insert(0, str(_WS / "src" / "inference"))
DEFAULT_CONFIG = _WS / "config" / "lane_vision.yaml"

from inference.modules.lane_detection import LaneDetector  # noqa: E402


def read_camera_frames(bag: str, topic: str, stride: int, limit: int):
    """rosbag2 에서 CompressedImage 프레임을 디코드해 yield."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    i = 0
    yielded = 0
    while reader.has_next():
        tp, data, t = reader.read_next()
        if tp != topic:
            continue
        if i % stride == 0:
            msg = deserialize_message(data, CompressedImage)
            img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                yield i, t, img
                yielded += 1
                if limit and yielded >= limit:
                    return
        i += 1


def _pct(a):
    return np.round(np.percentile(a, [2, 10, 50, 90, 98]), 0) if len(a) else "n/a"


def hsv_stats(frames):
    """밝은(흰 후보)·노란 후보 픽셀 HSV 분포를 프레임 하단 60% 에서 집계."""
    yh, ys, yv = [], [], []
    acc_s, acc_v = [], []
    for _, _, img in frames:
        h, w = img.shape[:2]
        roi = img[int(h * 0.4):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        acc_s.append(S.ravel())
        acc_v.append(V.ravel())
        ym = (H >= 8) & (H <= 45) & (S >= 40) & (V >= 100)
        yh += list(H[ym])
        ys += list(S[ym])
        yv += list(V[ym])
    Ss = np.concatenate(acc_s)
    Vs = np.concatenate(acc_v)
    bright = Vs >= np.percentile(Vs, 94)
    print("=== bright(=흰 후보, V상위6%) ===")
    print("  S", _pct(Ss[bright]), " V", _pct(Vs[bright]))
    print("=== yellow 후보(H8-45,S>=40,V>=100) ===")
    print("  H", _pct(np.array(yh)), " S", _pct(np.array(ys)), " V", _pct(np.array(yv)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="rosbag2 디렉토리")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="lane_vision.yaml")
    ap.add_argument("--topic", default="/camera/image/compressed")
    ap.add_argument("--stride", type=int, default=20, help="N 프레임마다 1장")
    ap.add_argument("--limit", type=int, default=0, help="최대 처리 프레임(0=전체)")
    ap.add_argument("--out", default="", help="샘플 오버레이 저장 디렉토리")
    ap.add_argument("--sample", type=int, default=16, help="--out 에 저장할 샘플 수")
    ap.add_argument("--video", default="", help="검출 오버레이 mp4 경로")
    ap.add_argument("--hsv-stats", action="store_true", help="HSV 분포만 출력")
    args = ap.parse_args()

    if args.hsv_stats:
        hsv_stats(read_camera_frames(args.bag, args.topic, args.stride, args.limit))
        return

    det = LaneDetector(args.config)
    print(f"config={args.config}")
    print(f"BEV {det.bev.bev_width}x{det.bev.bev_height} mpp={det.mpp} "
          f"x=[{det.p.x_min_m},{det.p.x_max_m}] offset_px={det.lane_half_offset_px} "
          f"color={det.color}")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
    writer = None

    n = 0
    n_det = 0
    n_white = 0
    n_yellow = 0
    near_ys = []
    samples = []
    for idx, t, img in read_camera_frames(args.bag, args.topic, args.stride, args.limit):
        r = det.detect(img)
        n += 1
        if r.white_visible:
            n_white += 1
        if r.yellow_visible:
            n_yellow += 1
        cl = r.white_centerline
        if len(cl) >= 3:
            n_det += 1
            near_ys.append(cl[0][1])

        if args.out or args.video:
            bev = det.debug_view(img)
            # 원본(리사이즈)과 BEV 나란히
            oh = bev.shape[0]
            ow = int(img.shape[1] * oh / img.shape[0])
            orig = cv2.resize(img, (ow, oh))
            combo = np.hstack([orig, bev])
            cv2.putText(combo, f"idx{idx} pts{len(cl)} W{int(r.white_visible)}"
                        f"Y{int(r.yellow_visible)}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if args.video:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.video, fourcc, 15.0,
                                             (combo.shape[1], combo.shape[0]))
                writer.write(combo)
            if args.out and len(samples) < args.sample:
                samples.append((idx, combo))

    if writer is not None:
        writer.release()
        print(f"video -> {args.video}")

    if args.out and samples:
        step = max(1, len(samples) // args.sample)
        for idx, combo in samples[::step][:args.sample]:
            cv2.imwrite(f"{args.out}/replay_{idx:05d}.png", combo)
        print(f"{len(samples[::step][:args.sample])} samples -> {args.out}")

    print("\n=== 검증 요약 ===")
    print(f"프레임 처리        : {n}")
    if n:
        print(f"센터라인 검출률    : {n_det/n*100:5.1f}%  ({n_det}/{n})")
        print(f"흰 차선 가시율     : {n_white/n*100:5.1f}%")
        print(f"노랑 차선 가시율   : {n_yellow/n*100:5.1f}%")
    if near_ys:
        a = np.array(near_ys)
        print(f"근거리 y(우측+)    : mean{a.mean():+.3f} std{a.std():.3f} "
              f"[{a.min():+.3f},{a.max():+.3f}] m")


if __name__ == "__main__":
    main()
