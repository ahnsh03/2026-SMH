"""Rule-based direction-sign fallback tests (no ONNX model required)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_INFERENCE_SRC = Path(__file__).resolve().parents[1]
if str(_INFERENCE_SRC) not in sys.path:
    sys.path.insert(0, str(_INFERENCE_SRC))

from inference.modules.direction_sign.detector import (  # noqa: E402
    detect_turn_rule_based,
)
from inference.types import TurnSign  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _sign_frame(direction: str) -> np.ndarray:
    path = (
        _REPO_ROOT
        / 'src'
        / 'dracer_sim'
        / 'models'
        / f'turn_sign_{direction}'
        / 'materials'
        / 'textures'
        / f'turn_sign_{direction}.png'
    )
    sign = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert sign is not None
    sign = cv2.resize(sign, (90, 90), interpolation=cv2.INTER_AREA)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[35:125, 115:205] = sign
    return frame


def test_rule_fallback_detects_left_sign():
    assert detect_turn_rule_based(_sign_frame('left')) is TurnSign.LEFT


def test_rule_fallback_detects_right_sign():
    assert detect_turn_rule_based(_sign_frame('right')) is TurnSign.RIGHT


def test_rule_fallback_rejects_empty_frame():
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    assert detect_turn_rule_based(frame) is TurnSign.UNKNOWN
