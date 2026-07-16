"""표지판(좌/우) + 신호등(빨강/초록) 인식 — YOLO26n ONNX.

모델: weights/sign_light_best_v5b.onnx (Ultralytics anchor-free, imgsz 416).
  클래스: 0=Left표지판, 1=Right표지판, 2=Red불, 3=Green불.
  출력 output0 (1, 4+nc, N) → 전치 (N, 4box+nc). objectness 없음.

추론이 aarch64 CPU 에서 ~0.4s/frame 로 느리다 → **lane_drive_node 가 별도 스레드**에서
infer() 를 돌리고, 제어 루프는 state() 로 캐시된 판단만 읽는다(30fps 유지).

판단(SignLightState):
  - stop_for_light : 빨강 래치(빨강 보면 정지, 초록 보면 해제). throttle=0.
  - sign_dir       : 0 없음 / 1 좌 / 2 우. (types.SignResult 규약과 동일)
    표지 보이는 동안 감속 + 해당 차선 쪽으로 추종(상위 노드가 사용).

onnxruntime/모델 없으면 조용히 무동작(state 중립) → 주행은 정상 유지.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import onnxruntime as ort
except Exception:  # onnxruntime 없으면 비활성
    ort = None

# 클래스 인덱스 (모델 정의)
CLS_LEFT = 0
CLS_RIGHT = 1
CLS_RED = 2
CLS_GREEN = 3

SIGN_NONE, SIGN_LEFT, SIGN_RIGHT = 0, 1, 2

_INPUT_SIZE = 416
_PAD = 114


@dataclass
class SignLightState:
    stop_for_light: bool = False      # 빨강 래치 → 정지
    sign_dir: int = SIGN_NONE         # 0/1좌/2우 (감속 + 차선쪽 추종)
    light: str = "none"               # "red"/"green"/"none" (디버그)
    light_conf: float = 0.0
    sign_conf: float = 0.0


# ------------------------------------------------------------------ 모델 경로
def _resolve_model_path(path: Optional[str]) -> Optional[Path]:
    cands = []
    if path:
        cands.append(Path(path))
    env = os.environ.get("SIGN_LIGHT_MODEL_PATH")
    if env:
        cands.append(Path(env))
    cands += [
        # INT8 양자화본 우선(1.9× 빠름). 없으면 FP32.
        Path("/home/topst/2026-SMH-team-new/weights/sign_light_best_v5b_int8.onnx"),
        Path("/home/topst/2026-SMH-team-new/weights/sign_light_best_v5b.onnx"),
        Path("/home/topst/2026-SMH/weights/sign_light_best_v5b_int8.onnx"),
        Path("/home/topst/2026-SMH/weights/sign_light_best_v5b.onnx"),
    ]
    for c in cands:
        if c and c.is_file():
            return c
    return None


# ------------------------------------------------------------------ YOLO 추론
class SignLightYolo:
    """ONNX 추론 → 검출 목록. 모델/onnxruntime 없으면 빈 목록."""

    def __init__(self, model_path: Optional[str] = None, conf: float = 0.35,
                 iou: float = 0.45, threads: int = 3):   # 3=4코어 스윗스팟(1개는 제어용)
        self.conf = float(conf)
        self.iou = float(iou)
        self._session = None
        self._input_name = "images"
        path = _resolve_model_path(model_path)
        if ort is not None and cv2 is not None and path is not None:
            try:
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                opts.intra_op_num_threads = int(threads)
                self._session = ort.InferenceSession(
                    str(path), opts, providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
            except Exception:
                self._session = None

    @property
    def available(self) -> bool:
        return self._session is not None

    def _letterbox(self, frame):
        h, w = frame.shape[:2]
        sc = min(_INPUT_SIZE / w, _INPUT_SIZE / h)
        nw, nh = round(w * sc), round(h * sc)
        canvas = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), _PAD, np.uint8)
        px, py = (_INPUT_SIZE - nw) // 2, (_INPUT_SIZE - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
        return canvas, sc, px, py

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            best, rest = order[0], order[1:]
            keep.append(int(best))
            if rest.size == 0:
                break
            x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
            y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
            x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
            y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thr]
        return keep

    def detect(self, frame) -> List[Tuple[int, float, float]]:
        """(cls, score, box_area_frac) 목록. box_area_frac=마커/표지 화면점유(거리 프록시)."""
        if not self.available or frame is None or getattr(frame, "size", 0) == 0:
            return []
        h, w = frame.shape[:2]
        lb, sc, px, py = self._letterbox(frame)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
        try:
            raw = self._session.run(None, {self._input_name: tensor})[0]
        except Exception:
            return []
        pred = np.array(raw)[0].T                    # (N, 4+nc)
        boxes_cxcywh, class_scores = pred[:, :4], pred[:, 4:]
        classes = class_scores.argmax(axis=1)
        confs = class_scores[np.arange(classes.size), classes]
        m = confs >= self.conf
        if not m.any():
            return []
        boxes_cxcywh, classes, confs = boxes_cxcywh[m], classes[m], confs[m]
        cx, cy, bw, bh = boxes_cxcywh.T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - px) / sc
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - py) / sc
        out = []
        for c in np.unique(classes):
            idx = np.flatnonzero(classes == c)
            for k in self._nms(boxes[idx], confs[idx], self.iou):
                i = idx[k]
                bx = boxes[i]
                area_frac = float(max(0, bx[2] - bx[0]) * max(0, bx[3] - bx[1]) / (w * h + 1e-9))
                out.append((int(c), float(confs[i]), area_frac))
        return out


# ------------------------------------------------------------ 판단(래치/디바운스)
class SignLightDetector:
    """YOLO + 신호 래치 + 표지 디바운스. 스레드 세이프(infer/ state)."""

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.yolo = SignLightYolo(
            model_path=p.get("sign_light_model_path"),
            conf=float(p.get("sign_light_conf", 0.35)),
            threads=int(p.get("sign_light_threads", 3)),
        )
        # 신호등 래치: 빨강/초록 이 시간 이상 지속돼야 상태 변경(오탐 억제)
        self.light_enter_s = float(p.get("light_enter_seconds", 0.2))
        # 표지: 감지 후 이 시간 유지(깜빡여도 감속/차선 유지)
        self.sign_hold_s = float(p.get("sign_hold_seconds", 1.0))
        self.sign_min_area = float(p.get("sign_min_area_frac", 0.0))   # 거리 게이팅(옵션)
        self.light_min_area = float(p.get("light_min_area_frac", 0.0))

        self._lock = threading.Lock()
        self._state = SignLightState()
        self._red_since: Optional[float] = None
        self._green_since: Optional[float] = None
        self._sign_last: Optional[float] = None
        self._sign_dir = SIGN_NONE
        self._sign_conf = 0.0
        self._stop_latch = False

    @property
    def available(self) -> bool:
        return self.yolo.available

    def infer(self, frame, now: float | None = None) -> None:
        """(백그라운드 스레드에서 호출) YOLO 실행 → 상태 갱신."""
        self.decide(self.yolo.detect(frame), now)

    def decide(self, dets, now: float | None = None) -> None:
        """검출목록 [(cls, score, area_frac)] → 래치/디바운스 상태 갱신(테스트 가능)."""
        now = time.monotonic() if now is None else now

        red = max([s for c, s, a in dets if c == CLS_RED and a >= self.light_min_area],
                  default=0.0)
        green = max([s for c, s, a in dets if c == CLS_GREEN and a >= self.light_min_area],
                    default=0.0)
        # 표지: 좌/우 중 최고점
        left = max([s for c, s, a in dets if c == CLS_LEFT and a >= self.sign_min_area],
                   default=0.0)
        right = max([s for c, s, a in dets if c == CLS_RIGHT and a >= self.sign_min_area],
                    default=0.0)

        with self._lock:
            self._update_light(red, green, now)
            self._update_sign(left, right, now)
            self._state = SignLightState(
                stop_for_light=self._stop_latch,
                sign_dir=self._sign_dir,
                light=("red" if red >= green and red > 0 else
                       "green" if green > 0 else "none"),
                light_conf=max(red, green),
                sign_conf=self._sign_conf,
            )

    def _update_light(self, red, green, now):
        # 빨강 래치: 빨강 지속 → 정지 / 초록 지속 → 해제
        if red > 0:
            self._green_since = None
            if self._red_since is None:
                self._red_since = now
            if (now - self._red_since) >= self.light_enter_s:
                self._stop_latch = True
        elif green > 0:
            self._red_since = None
            if self._green_since is None:
                self._green_since = now
            if (now - self._green_since) >= self.light_enter_s:
                self._stop_latch = False
        else:
            self._red_since = None
            self._green_since = None   # 신호 안 보이면 마지막 래치 유지

    def _update_sign(self, left, right, now):
        if left > 0 or right > 0:
            self._sign_dir = SIGN_LEFT if left >= right else SIGN_RIGHT
            self._sign_conf = max(left, right)
            self._sign_last = now
        elif self._sign_last is not None and (now - self._sign_last) >= self.sign_hold_s:
            self._sign_dir = SIGN_NONE
            self._sign_conf = 0.0

    def state(self) -> SignLightState:
        """(제어 루프에서 호출) 캐시된 판단."""
        with self._lock:
            return self._state

    def reset(self) -> None:
        with self._lock:
            self._state = SignLightState()
            self._red_since = self._green_since = self._sign_last = None
            self._sign_dir = SIGN_NONE
            self._sign_conf = 0.0
            self._stop_latch = False
