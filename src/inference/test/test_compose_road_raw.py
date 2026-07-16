"""Unit tests for pre-PDC / bag SSOT compose_road_raw."""

from __future__ import annotations

import numpy as np

from inference.modules.perception.blob.masks import compose_road_raw, red_road_present


def _blank(h: int = 200, w: int = 200) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _blob(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    out.flat[:n] = 255
    return out


def test_road_is_black_or_red_or_cyan():
    black = _blob(_blank(), 100)
    red = _blob(_blank(), 50)
    cyan1 = _blob(_blank(), 40)
    cyan2 = _blob(_blank(), 30)
    road, cyan, cyan_raw = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        prefer_yellow=False,
        keep_near_fn=lambda m: m,
    )
    assert np.count_nonzero(road) >= 100
    assert np.any(cyan)
    assert np.any(cyan_raw)


def test_cyan2_included_even_on_out():
    """pre-PDC SSOT ORs both cyan channels regardless of prefer_yellow."""

    black = _blank()
    red = _blank()
    cyan1 = _blank()
    cyan2 = _blob(_blank(), 200)
    road_out, _, cyan_raw_out = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        prefer_yellow=False,
        keep_near_fn=lambda m: m,
    )
    assert np.any(road_out)
    assert np.any(cyan_raw_out)


def test_yellow_does_not_drop_cyan1():
    black = _blank()
    red = _blank()
    cyan1 = _blob(_blank(), 200)
    cyan2 = _blank()
    yellow = _blob(_blank(), 200)
    road, _, cyan_raw = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow,
        prefer_yellow=True,
        keep_near_fn=lambda m: m,
    )
    assert np.any(road)
    assert np.any(cyan_raw)


def test_red_present_helper_still_works():
    red = _blank()
    red[120:, :].flat[:300] = 255
    assert red_road_present(red)
