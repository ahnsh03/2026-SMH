"""Traffic-light detection from Sungjun's 4-class YOLO ONNX (lights only).

Model classes (``sign_light_best_v5b.onnx`` / Ultralytics, imgsz 416):

* 0 Left Sign / 1 Right Sign — **ignored** (use ``direction_sign`` + ``sign_best.onnx``)
* 2 Red Light / 3 Green Light — used here

Weights live at repo ``weights/sign_light_best_v5b.onnx`` or ``TRAFFIC_LIGHT_MODEL_PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from inference.types import TrafficSignal

_CLS_RED = 2
_CLS_GREEN = 3
_CLASS_TO_SIGNAL = {
    _CLS_RED: TrafficSignal.RED,
    _CLS_GREEN: TrafficSignal.GREEN,
}

_INPUT_SIZE = 416
# Match team-new SignLightYolo defaults (lane_control sign_light_conf).
_CONF_THRESHOLD = 0.35
_IOU_THRESHOLD = 0.45
_INTRA_OP_THREADS = 2
_PAD_VALUE = 114

_session = None


class LightDetection(NamedTuple):
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    cls: int

    @property
    def signal(self) -> TrafficSignal:
        return _CLASS_TO_SIGNAL.get(self.cls, TrafficSignal.UNKNOWN)


def _model_path() -> Path:
    override = os.environ.get('TRAFFIC_LIGHT_MODEL_PATH')
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'weights' / 'sign_light_best_v5b.onnx'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'weights/sign_light_best_v5b.onnx not found; '
        'set TRAFFIC_LIGHT_MODEL_PATH to override'
    )


def _get_session():
    global _session
    if ort is None:
        raise RuntimeError('onnxruntime is not installed')
    if _session is None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = _INTRA_OP_THREADS
        _session = ort.InferenceSession(
            str(_model_path()), options, providers=['CPUExecutionProvider']
        )
    return _session


def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    scale = min(_INPUT_SIZE / width, _INPUT_SIZE / height)
    new_w, new_h = round(width * scale), round(height * scale)
    canvas = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), _PAD_VALUE, dtype=np.uint8)
    pad_x, pad_y = (_INPUT_SIZE - new_w) // 2, (_INPUT_SIZE - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(
        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, scale, pad_x, pad_y


def _nms(boxes: np.ndarray, scores: np.ndarray) -> list[int]:
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


def _postprocess_lights(
    raw: np.ndarray, scale: float, pad_x: int, pad_y: int, width: int, height: int
) -> list[LightDetection]:
    pred = raw[0].T
    boxes_cxcywh, class_scores = pred[:, :4], pred[:, 4:]
    classes = class_scores.argmax(axis=1)
    confidences = class_scores[np.arange(classes.size), classes]

    # Drop sign classes 0/1; keep only red/green light.
    light = (
        (confidences >= _CONF_THRESHOLD)
        & ((classes == _CLS_RED) | (classes == _CLS_GREEN))
    )
    if not light.any():
        return []

    boxes_cxcywh = boxes_cxcywh[light]
    classes, confidences = classes[light], confidences[light]
    cx, cy, box_w, box_h = boxes_cxcywh.T
    boxes = np.stack(
        [cx - box_w / 2, cy - box_h / 2, cx + box_w / 2, cy + box_h / 2], axis=1
    )
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)

    detections: list[LightDetection] = []
    for cls in np.unique(classes):
        members = np.flatnonzero(classes == cls)
        for local in _nms(boxes[members], confidences[members]):
            index = members[local]
            detections.append(
                LightDetection(*boxes[index], float(confidences[index]), int(cls))
            )
    # Red vs green mutex: keep highest-score light if they heavily overlap.
    detections.sort(key=lambda d: d.score, reverse=True)
    kept: list[LightDetection] = []
    for detection in detections:
        overlaps = False
        for other in kept:
            x1 = max(detection.x1, other.x1)
            y1 = max(detection.y1, other.y1)
            x2 = min(detection.x2, other.x2)
            y2 = min(detection.y2, other.y2)
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_d = (detection.x2 - detection.x1) * (detection.y2 - detection.y1)
            area_o = (other.x2 - other.x1) * (other.y2 - other.y1)
            if inter / max(area_d + area_o - inter, 1e-6) > _IOU_THRESHOLD:
                overlaps = True
                break
        if not overlaps:
            kept.append(detection)
    return kept


def detect_lights(frame: np.ndarray) -> list[LightDetection]:
    """Return red/green light boxes (sign classes discarded), best score first."""
    if frame is None or getattr(frame, 'size', 0) == 0:
        return []
    tensor_frame, scale, pad_x, pad_y = _letterbox(frame)
    rgb = cv2.cvtColor(tensor_frame, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    tensor = np.ascontiguousarray(tensor)
    raw = _get_session().run(None, {'images': tensor})[0]
    height, width = frame.shape[:2]
    return _postprocess_lights(raw, scale, pad_x, pad_y, width, height)


def detect_signal_yolo(frame: np.ndarray) -> TrafficSignal:
    """Best light class, or UNKNOWN if none / model missing."""
    try:
        detections = detect_lights(frame)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError):
        return TrafficSignal.UNKNOWN
    if not detections:
        return TrafficSignal.UNKNOWN
    return detections[0].signal
