"""Unit tests for compose_road_raw fill gates."""

from __future__ import annotations

import numpy as np

from inference.modules.perception.blob.masks import (
    _MIN_RED_ROAD_PX,
    _RED_BOTTOM_Y_FRAC,
    compose_road_raw,
    red_road_present,
)
from inference.modules.perception.blob.rail_corridor import yellow_lane_present


def _blank(h: int = 200, w: int = 200) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _blob(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    out.flat[:n] = 255
    return out


def _bottom_red(h: int = 200, w: int = 200, n: int = _MIN_RED_ROAD_PX) -> np.ndarray:
    """Paint ``n`` red pixels in the lower BEV band used by ``red_road_present``."""

    out = _blank(h, w)
    y0 = int(_RED_BOTTOM_Y_FRAC * h)
    band = out[y0:, :].reshape(-1)
    band[:n] = 255
    out[y0:, :] = band.reshape(out[y0:, :].shape)
    return out


def test_red_present_uses_red_only():
    black = _blob(_blank(), 300)
    red = _bottom_red(n=_MIN_RED_ROAD_PX)
    cyan1 = _blob(_blank(), 200)
    cyan2 = _blob(_blank(), 200)
    yellow = _blank()
    road, cyan, cyan_raw = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow,
        prefer_yellow=True,
        keep_near_fn=lambda m: m,
    )
    assert red_road_present(red)
    assert np.array_equal(road > 0, red > 0)
    assert not np.any(cyan)
    assert not np.any(cyan_raw)


def test_upper_red_ignored():
    """Glare/false red in upper BEV must not trigger red-only fill."""

    red = _blank()
    red[:40, :].flat[:800] = 255
    assert not red_road_present(red)


def test_sparse_bottom_red_keeps_black_cyan():
    black = _blob(_blank(), 300)
    red = _bottom_red(n=max(1, _MIN_RED_ROAD_PX // 10))
    cyan1 = _blank()
    cyan2 = _blank()
    yellow = _blank()
    road, _, _ = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow,
        prefer_yellow=False,
        keep_near_fn=lambda m: m,
    )
    assert not red_road_present(red)
    assert np.any(road & (black > 0))


def test_cyan2_in_only():
    black = _blank()
    red = _blank()
    cyan1 = _blank()
    cyan2 = _blob(_blank(), 200)
    yellow = _blank()

    road_out, _, cyan_raw_out = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow,
        prefer_yellow=False,
        keep_near_fn=lambda m: m,
    )
    assert not np.any(road_out)
    assert not np.any(cyan_raw_out)

    road_in, _, cyan_raw_in = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow,
        prefer_yellow=True,
        keep_near_fn=lambda m: m,
    )
    assert np.any(road_in)
    assert np.any(cyan_raw_in)


def test_cyan1_skipped_when_yellow_visible():
    black = _blank()
    red = _blank()
    cyan1 = _blob(_blank(), 200)
    cyan2 = _blank()
    yellow = _blob(_blank(), 200)
    assert yellow_lane_present(yellow)

    road, _, cyan_raw = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow,
        prefer_yellow=False,
        keep_near_fn=lambda m: m,
    )
    assert not np.any(road)
    assert not np.any(cyan_raw)

    yellow_off = _blank()
    road2, _, cyan_raw2 = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow_off,
        prefer_yellow=False,
        keep_near_fn=lambda m: m,
    )
    assert np.any(road2)
    assert np.any(cyan_raw2)
