"""SignLightDetector 테스트 — 신호등 래치 / 표지 디바운스 (시간 주입).

decide()에 검출목록 [(cls, score, area_frac)] 를 직접 넣어 판단 로직을 검증한다
(YOLO/모델 없이). 실제 모델 추론은 모델 있을 때만 옵션 테스트.
"""
import sys
from pathlib import Path

import pytest

_WS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_WS / "src" / "inference"))

from inference.modules.sign_light import (   # noqa: E402
    SignLightDetector, SignLightYolo,
    CLS_LEFT, CLS_RIGHT, CLS_RED, CLS_GREEN, SIGN_LEFT, SIGN_RIGHT, SIGN_NONE,
)


def _det():
    return SignLightDetector({"light_enter_seconds": 0.2, "sign_hold_seconds": 1.0})


# ------------------------------------------------------------------ 신호등 래치
def test_red_light_latches_stop():
    d = _det()
    d.decide([(CLS_RED, 0.9, 0.02)], now=0.0)
    assert d.state().stop_for_light is False           # 0.2s 미만 → 아직
    d.decide([(CLS_RED, 0.9, 0.02)], now=0.3)
    assert d.state().stop_for_light is True             # 지속 → 정지 래치


def test_green_releases_stop():
    d = _det()
    d.decide([(CLS_RED, 0.9, 0.02)], now=0.0)
    d.decide([(CLS_RED, 0.9, 0.02)], now=0.3)
    assert d.state().stop_for_light is True
    d.decide([(CLS_GREEN, 0.9, 0.02)], now=0.5)
    d.decide([(CLS_GREEN, 0.9, 0.02)], now=0.8)         # 초록 지속 → 해제
    assert d.state().stop_for_light is False


def test_no_light_keeps_latch():
    """신호 안 보이면 마지막 래치 유지(빨강 유지)."""
    d = _det()
    d.decide([(CLS_RED, 0.9, 0.02)], now=0.0)
    d.decide([(CLS_RED, 0.9, 0.02)], now=0.3)
    d.decide([], now=1.0)                               # 아무 신호 없음
    assert d.state().stop_for_light is True


def test_start_scenario_red_then_green():
    """출발: 빨강 대기 → 초록 출발."""
    d = _det()
    d.decide([(CLS_RED, 0.8, 0.03)], now=0.0)
    d.decide([(CLS_RED, 0.8, 0.03)], now=0.25)
    assert d.state().stop_for_light is True             # 대기
    d.decide([(CLS_GREEN, 0.8, 0.03)], now=1.0)
    d.decide([(CLS_GREEN, 0.8, 0.03)], now=1.25)
    assert d.state().stop_for_light is False            # 출발


# ------------------------------------------------------------------ 표지 방향
def test_right_sign_detected():
    d = _det()
    d.decide([(CLS_RIGHT, 0.7, 0.02)], now=0.0)
    assert d.state().sign_dir == SIGN_RIGHT


def test_left_sign_detected():
    d = _det()
    d.decide([(CLS_LEFT, 0.7, 0.02)], now=0.0)
    assert d.state().sign_dir == SIGN_LEFT


def test_sign_holds_after_disappear():
    """표지 사라져도 hold_seconds 동안 유지."""
    d = _det()
    d.decide([(CLS_RIGHT, 0.7, 0.02)], now=0.0)
    assert d.state().sign_dir == SIGN_RIGHT
    d.decide([], now=0.5)                               # 사라짐(0.5s<1.0s hold)
    assert d.state().sign_dir == SIGN_RIGHT
    d.decide([], now=1.5)                               # hold 초과 → 해제
    assert d.state().sign_dir == SIGN_NONE


def test_higher_score_wins_sign():
    d = _det()
    d.decide([(CLS_LEFT, 0.4, 0.02), (CLS_RIGHT, 0.8, 0.02)], now=0.0)
    assert d.state().sign_dir == SIGN_RIGHT


def test_no_detection_neutral():
    d = _det()
    d.decide([], now=0.0)
    s = d.state()
    assert s.stop_for_light is False and s.sign_dir == SIGN_NONE


# ------------------------------------------------------------------ 모델(옵션)
def test_yolo_graceful_without_model(tmp_path):
    """모델/onnxruntime 없어도 크래시 없이 빈 검출."""
    y = SignLightYolo(model_path=str(tmp_path / "nope.onnx"))
    import numpy as np
    assert y.detect(np.zeros((180, 320, 3), np.uint8)) == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
