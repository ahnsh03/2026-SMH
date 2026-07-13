"""Direction (left/right fork) sign detection — 담당: 장원정."""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # Rule-based fallback remains available without ONNX Runtime.
    ort = None

from inference.types import TurnSign

# weights/sign_best.onnx: Ultralytics YOLO26n, imgsz 416, end2end=False.
# 클래스 순서는 모델 메타데이터의 names={0: 'Left Sign', 1: 'Right Sign'}를 따른다.
_CLASS_TO_TURN = (TurnSign.LEFT, TurnSign.RIGHT)

_INPUT_SIZE = 416
_CONF_THRESHOLD = 0.25
_IOU_THRESHOLD = 0.45
_INTRA_OP_THREADS = 4
_PAD_VALUE = 114

_session = None

# Rule-based fallback for the competition sign: blue circle + white arrow.
_BLUE_LOWER = np.array([90, 70, 35], dtype=np.uint8)
_BLUE_UPPER = np.array([140, 255, 255], dtype=np.uint8)
_WHITE_LOWER = np.array([0, 0, 170], dtype=np.uint8)
_WHITE_UPPER = np.array([179, 80, 255], dtype=np.uint8)
_MIN_BLUE_AREA_RATIO = 0.001
_MIN_ARROW_PIXELS = 15
_MIN_ARROW_OFFSET_RATIO = 0.012


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


def detect_turn_rule_based(frame: np.ndarray) -> TurnSign:
    """Detect blue-circle white-arrow signs without learned model weights.

    The arrow head occupies the upper-middle band. Its white-pixel centroid is
    left/right of the sign centre for LEFT/RIGHT respectively; the vertical stem
    is deliberately down-weighted by excluding the lower band.
    """
    if frame is None or getattr(frame, 'size', 0) == 0:
        return TurnSign.UNKNOWN

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, _BLUE_LOWER, _BLUE_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame.shape[0] * frame.shape[1])

    candidates: list[tuple[float, TurnSign]] = []
    white = cv2.inRange(hsv, _WHITE_LOWER, _WHITE_UPPER)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < frame_area * _MIN_BLUE_AREA_RATIO:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 8 or height < 8:
            continue
        aspect = width / float(height)
        if not 0.55 <= aspect <= 1.45:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-6)
        if circularity < 0.35:
            continue

        region = np.zeros_like(blue)
        cv2.drawContours(region, [contour], -1, 255, thickness=cv2.FILLED)
        erode_px = max(1, int(round(min(width, height) * 0.03)))
        region = cv2.erode(
            region,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
            ),
        )

        band_top = y + int(round(0.25 * height))
        band_bottom = y + int(round(0.58 * height))
        arrow_mask = cv2.bitwise_and(white, region)
        band = arrow_mask[band_top:band_bottom, x:x + width]
        _, columns = np.nonzero(band)
        if columns.size < _MIN_ARROW_PIXELS:
            continue
        offset = float(np.mean(columns) - (width - 1) / 2.0) / float(width)
        if abs(offset) < _MIN_ARROW_OFFSET_RATIO:
            continue
        turn = TurnSign.LEFT if offset < 0.0 else TurnSign.RIGHT
        score = area * max(circularity, 0.01) * abs(offset)
        candidates.append((score, turn))

    if not candidates:
        return TurnSign.UNKNOWN
    return max(candidates, key=lambda item: item[0])[1]


def detect_turn(frame: np.ndarray) -> TurnSign:
    """Use ONNX when available, otherwise the blue/white rule fallback."""
    try:
        detections = detect_signs(frame)
    except (FileNotFoundError, RuntimeError, ValueError):
        detections = []
    if detections:
        return detections[0].turn
    return detect_turn_rule_based(frame)
