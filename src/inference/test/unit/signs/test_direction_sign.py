"""Rule-based direction-sign fallback tests (no ONNX model required)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from inference.modules.direction_sign.detector import detect_turn_rule_based
from inference.types import TurnSign


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / 'config' / 'main_planner.yaml').is_file():
            return cand
    pytest.skip('board repo root not found')


_REPO_ROOT = _repo_root()


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
    if not path.is_file():
        pytest.skip(f'sign texture missing (sim asset): {path}')
    sign = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if sign is None:
        pytest.skip(f'failed to load sign texture: {path}')
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
