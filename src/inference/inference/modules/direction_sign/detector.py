"""Direction (left/right fork) sign detection — 담당: 장원정."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import onnxruntime as ort

from inference.types import TurnSign

# weights/sign_best.onnx: Ultralytics YOLO26n, imgsz 416, end2end=False.
# 클래스 순서는 모델 메타데이터의 names={0: 'Left Sign', 1: 'Right Sign'}를 따른다.
_CLASS_TO_TURN = (TurnSign.LEFT, TurnSign.RIGHT)

_INPUT_SIZE = 416
_CONF_THRESHOLD = 0.25
_IOU_THRESHOLD = 0.45
_INTRA_OP_THREADS = 4
_PAD_VALUE = 114

_session: ort.InferenceSession | None = None


class Detection(NamedTuple):
    """One surviving detection, in original-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    @property
    def turn(self) -> TurnSign:
        return _CLASS_TO_TURN[self.cls]


def _model_path() -> Path:
    override = os.environ.get('SIGN_MODEL_PATH')
    if override:
        return Path(override)

    # .../src/inference/inference/modules/direction_sign/detector.py -> repo root
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'weights' / 'sign_best.onnx'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'weights/sign_best.onnx not found; set SIGN_MODEL_PATH to override'
    )


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = _INTRA_OP_THREADS
        _session = ort.InferenceSession(
            str(_model_path()), options, providers=['CPUExecutionProvider']
        )
    return _session


def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Scale to fit _INPUT_SIZE keeping aspect ratio, then pad to a square."""
    height, width = frame.shape[:2]
    scale = min(_INPUT_SIZE / width, _INPUT_SIZE / height)
    new_w, new_h = round(width * scale), round(height * scale)

    canvas = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), _PAD_VALUE, dtype=np.uint8)
    pad_x, pad_y = (_INPUT_SIZE - new_w) // 2, (_INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, pad_x, pad_y


def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    canvas, scale, pad_x, pad_y = _letterbox(frame)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), scale, pad_x, pad_y


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
    """Greedy NMS over xyxy boxes. Returns indices into `boxes`, best first."""
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]

    keep: list[int] = []
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
        iou = inter / (areas[best] + areas[rest] - inter)

        order = rest[iou <= _IOU_THRESHOLD]
    return keep


def _postprocess(
    raw: np.ndarray, scale: float, pad_x: int, pad_y: int, width: int, height: int
) -> list[Detection]:
    # raw (1, 6, 3549) -> (3549, 6): cx, cy, w, h, score_left, score_right.
    # end2end=False, so boxes arrive decoded in 416x416 pixel space and class
    # scores are already sigmoid-activated. There is no objectness channel.
    pred = raw[0].T
    boxes_cxcywh, class_scores = pred[:, :4], pred[:, 4:]

    classes = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(classes.size), classes]

    above = confidences >= _CONF_THRESHOLD
    if not above.any():
        return []
    boxes_cxcywh = boxes_cxcywh[above]
    classes, confidences = classes[above], confidences[above]

    cx, cy, box_w, box_h = boxes_cxcywh.T
    boxes = np.stack(
        [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2], axis=1
    )

    # Undo the letterbox: strip padding, rescale, then clamp to the frame.
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

    detections: list[Detection] = []
    for cls in np.unique(classes):
        members = np.flatnonzero(classes == cls)
        for local in _nms(boxes[members], confidences[members]):
            index = members[local]
            detections.append(
                Detection(*boxes[index], float(confidences[index]), int(cls))
            )

    detections.sort(key=lambda detection: detection.score, reverse=True)
    return detections


def detect_signs(frame: np.ndarray) -> list[Detection]:
    """
    Detect left/right direction signs in the frame.

    Returns detections sorted by confidence, highest first (empty if none).
    """
    if frame is None or getattr(frame, 'size', 0) == 0:
        return []

    tensor, scale, pad_x, pad_y = _preprocess(frame)
    raw = _get_session().run(None, {'images': tensor})[0]
    height, width = frame.shape[:2]
    return _postprocess(raw, scale, pad_x, pad_y, width, height)


def detect_turn(frame: np.ndarray) -> TurnSign:
    """Return the turn direction of the most confident sign, else UNKNOWN."""
    detections = detect_signs(frame)
    return detections[0].turn if detections else TurnSign.UNKNOWN
