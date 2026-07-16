#!/usr/bin/env python3
"""sign_light YOLO ONNX → INT8 정적 양자화 (백파일 프레임 calibration).

aarch64 CPU 추론 가속용. calibration 데이터는 실주행 백파일 프레임을 모델
전처리(letterbox416/RGB//255/CHW)와 동일하게 넣는다 → 활성값 범위를 실제
분포로 잡아 정확도 손실 최소화.

  python3 scripts/quantize_sign_light.py \
      --model /home/topst/2026-SMH/weights/sign_light_best_v5b.onnx \
      --out   weights/sign_light_best_v5b_int8.onnx
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

BAGS = [
    "/home/topst/2026-SMH/external/D-Racer-Kit/bagfile/bag_auto_20260716_004154",
    "/home/topst/2026-SMH/external/D-Racer-Kit/bagfile/bag_20260715_230145",
    "/home/topst/2026-SMH/external/D-Racer-Kit/bagfile/bag_20260715_204143",
]
SIZE = 416


def _letterbox(frame):
    h, w = frame.shape[:2]
    sc = min(SIZE / w, SIZE / h)
    nw, nh = round(w * sc), round(h * sc)
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
    return canvas


def _preprocess(frame):
    lb = _letterbox(frame)
    rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)


def _load_frames(max_frames):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage
    frames, per = [], max(1, max_frames // len(BAGS))
    for bag in BAGS:
        if not Path(bag).is_dir():
            continue
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
               rosbag2_py.ConverterOptions("cdr", "cdr"))
        r.set_filter(rosbag2_py.StorageFilter(topics=["/camera/image/compressed"]))
        raw = []
        while r.has_next():
            t, d, _ = r.read_next()
            if t == "/camera/image/compressed":
                raw.append(d)
        idxs = np.linspace(0, len(raw) - 1, min(per, len(raw))).astype(int)
        for i in idxs:
            m = deserialize_message(raw[int(i)], CompressedImage)
            img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(_preprocess(img))
    return frames


class _Reader(CalibrationDataReader):
    def __init__(self, tensors, name):
        self._it = iter([{name: t} for t in tensors])

    def get_next(self):
        return next(self._it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/topst/2026-SMH/weights/sign_light_best_v5b.onnx")
    ap.add_argument("--out", default="weights/sign_light_best_v5b_int8.onnx")
    ap.add_argument("--calib", type=int, default=90)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inp_name = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"]).get_inputs()[0].name

    print(f"calibration 프레임 로딩(~{args.calib})...", flush=True)
    tensors = _load_frames(args.calib)
    print(f"  {len(tensors)} 프레임", flush=True)

    pre = str(out.with_suffix(".pre.onnx"))
    print("shape inference/preprocess...", flush=True)
    quant_pre_process(args.model, pre)

    print("INT8 정적 양자화 중...", flush=True)
    quantize_static(
        pre, str(out), _Reader(tensors, inp_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,   # per-tensor: 모델 opset<13 의 DequantizeLinear axis 미지원 회피
    )
    Path(pre).unlink(missing_ok=True)
    mb = out.stat().st_size / 1e6
    print(f"완료 → {out}  ({mb:.1f} MB)", flush=True)

    # --- 속도 + 정확도 비교 ---
    def sess(p, th=3):
        o = ort.SessionOptions(); o.intra_op_num_threads = th; o.log_severity_level = 3
        return ort.InferenceSession(p, o, providers=["CPUExecutionProvider"])
    fp, q = sess(args.model), sess(str(out))
    t = tensors[len(tensors) // 2]
    for _ in range(2):
        fp.run(None, {inp_name: t}); q.run(None, {inp_name: t})

    def bench(s):
        st = time.time()
        for _ in range(8):
            r = s.run(None, {inp_name: t})
        return (time.time() - st) / 8 * 1000, np.array(r[0])
    fp_ms, fp_out = bench(fp)
    q_ms, q_out = bench(q)
    a, b = fp_out.ravel(), q_out.ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"\nFP32: {fp_ms:.0f} ms   INT8: {q_ms:.0f} ms   ({fp_ms/q_ms:.2f}x)")
    print(f"출력 상관(FP32 vs INT8): {corr:+.4f}  (1.0에 가까울수록 정확도 보존)")


if __name__ == "__main__":
    main()
