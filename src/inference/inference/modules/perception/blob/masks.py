"""HSV masks on Metric IPM BEV for blob corridor perception."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

_VISION_TUNE: Path | None = None


def _vision_tune_dir() -> Path:
    global _VISION_TUNE
    if _VISION_TUNE is not None:
        return _VISION_TUNE
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'scripts' / 'vision_tune' / 'metric_ipm.py'
        if candidate.is_file():
            _VISION_TUNE = candidate.parent
            if str(_VISION_TUNE) not in sys.path:
                sys.path.insert(0, str(_VISION_TUNE))
            return _VISION_TUNE
    raise ImportError('scripts/vision_tune/metric_ipm.py not found')


def _metric_ipm():
    _vision_tune_dir()
    from metric_ipm import (  # noqa: E402
        DEFAULT_CONFIG_PATH,
        build_ipm_maps,
        load_metric_ipm,
        warp_metric_ipm,
    )

    return DEFAULT_CONFIG_PATH, build_ipm_maps, load_metric_ipm, warp_metric_ipm


_ipm_map_x: np.ndarray | None = None
_ipm_map_y: np.ndarray | None = None
_ipm_map_shape: tuple[int, int] | None = None
_ipm_params = None


def get_ipm_params():
    global _ipm_params
    if _ipm_params is None:
        _, _, load_metric_ipm, _ = _metric_ipm()
        _ipm_params = load_metric_ipm().clamp()
    return _ipm_params


def ensure_ipm_maps(img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    global _ipm_map_x, _ipm_map_y, _ipm_map_shape
    _, build_ipm_maps, _, _ = _metric_ipm()
    shape = (img_w, img_h)
    if (
        _ipm_map_x is None
        or _ipm_map_y is None
        or _ipm_map_shape != shape
    ):
        params = get_ipm_params()
        _ipm_map_x, _ipm_map_y, _valid = build_ipm_maps(img_w, img_h, params)
        _ipm_map_shape = shape
    assert _ipm_map_x is not None and _ipm_map_y is not None
    return _ipm_map_x, _ipm_map_y


def warp_mask(mask: np.ndarray) -> np.ndarray:
    map_x, map_y = ensure_ipm_maps(mask.shape[1], mask.shape[0])
    return cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_bgr(frame: np.ndarray) -> np.ndarray:
    _, _, _, warp_metric_ipm = _metric_ipm()
    return warp_metric_ipm(frame, get_ipm_params())


def _load_hsv_bounds() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    defaults = {
        'white': ((0, 0, 174), (179, 29, 255)),
        'yellow': ((0, 32, 79), (55, 255, 255)),
        'black_road': ((17, 0, 15), (70, 255, 140)),
        'red_road': ((170, 125, 161), (179, 192, 229)),
        # OUT LED billboard cyan wash on asphalt
        'black_cyan': ((90, 200, 190), (100, 215, 238)),
        # Secondary cyan/teal asphalt (IN bag tune 2026-07-15)
        'black_cyan_2': ((97, 240, 105), (105, 255, 180)),
    }
    DEFAULT_CONFIG_PATH, _, _, _ = _metric_ipm()
    try:
        with open(DEFAULT_CONFIG_PATH, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        data = {}
    hsv = data.get('hsv') or {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (lo, hi) in defaults.items():
        block = hsv.get(name) or {}
        lower = np.array(
            [
                int(block.get('h_min', lo[0])),
                int(block.get('s_min', lo[1])),
                int(block.get('v_min', lo[2])),
            ],
            dtype=np.uint8,
        )
        upper = np.array(
            [
                int(block.get('h_max', hi[0])),
                int(block.get('s_max', hi[1])),
                int(block.get('v_max', hi[2])),
            ],
            dtype=np.uint8,
        )
        out[name] = (lower, upper)
    return out


_HSV_BOUNDS: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


def hsv_bounds() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    global _HSV_BOUNDS
    if _HSV_BOUNDS is None:
        _HSV_BOUNDS = _load_hsv_bounds()
    return _HSV_BOUNDS


def reload_hsv_bounds() -> None:
    global _HSV_BOUNDS
    _HSV_BOUNDS = None


def _red_inrange(hsv: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Red wraps hue: also catch low-H side when h_min is high."""

    main = cv2.inRange(hsv, lo, hi)
    h_lo = int(lo[0])
    if h_lo <= 10:
        return main
    wrap_lo = np.array([0, int(lo[1]), int(lo[2])], dtype=np.uint8)
    wrap_hi = np.array([10, int(hi[1]), int(hi[2])], dtype=np.uint8)
    return cv2.bitwise_or(main, cv2.inRange(hsv, wrap_lo, wrap_hi))


# Kept for tests / diagnostics; road fill SSOT no longer uses red-only gating.
_MIN_RED_ROAD_PX = 250
_RED_BOTTOM_Y_FRAC = 0.55


def red_road_present(
    red: np.ndarray,
    *,
    min_px: int = _MIN_RED_ROAD_PX,
    bottom_y_frac: float = _RED_BOTTOM_Y_FRAC,
) -> bool:
    """True when enough red asphalt sits in the lower BEV (near-ego)."""

    if red is None or not getattr(red, 'size', 0):
        return False
    h = int(red.shape[0])
    if h <= 0:
        return False
    y0 = int(np.clip(float(bottom_y_frac), 0.0, 0.95) * h)
    band = red[y0:, :]
    return int(np.count_nonzero(band > 0)) >= int(min_px)


def compose_road_raw(
    black: np.ndarray,
    red: np.ndarray,
    cyan1: np.ndarray,
    cyan2: np.ndarray,
    yellow: np.ndarray | None = None,
    *,
    prefer_yellow: bool = False,
    keep_near_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``road_raw`` = black_near | red | cyan_near (pre-PDC / bag SSOT).

    Cyan1∨cyan2 → near-ego CC before morph. ``prefer_yellow`` / ``yellow`` are
    API-compat only and do not gate fill (camera-retune gates removed).
    """

    del yellow, prefer_yellow
    from inference.modules.perception.blob.morph_blob import keep_near_floor_blob

    near = keep_near_fn or keep_near_floor_blob
    shape = black.shape if getattr(black, 'size', 0) else red.shape
    empty = np.zeros(shape, dtype=np.uint8)

    cyan_sel = empty.copy()
    if cyan1 is not None and getattr(cyan1, 'size', 0):
        cyan_sel = cv2.bitwise_or(cyan_sel, cyan1)
    if cyan2 is not None and getattr(cyan2, 'size', 0):
        cyan_sel = cv2.bitwise_or(cyan_sel, cyan2)
    cyan = near(cyan_sel) if np.any(cyan_sel) else empty

    road = black.copy() if getattr(black, 'size', 0) else empty.copy()
    if getattr(red, 'size', 0) and np.any(red):
        road = cv2.bitwise_or(road, red)
    if getattr(cyan, 'size', 0):
        road = cv2.bitwise_or(road, cyan)
    return road, cyan, cyan_sel


def _bev_channel_mask(
    bev: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    morph_open: bool = True,
) -> np.ndarray:
    """HSV inRange on already-warped BEV (tune_hsv / extract_five)."""

    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    if morph_open:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def course_ego_blob(
    frame: np.ndarray,
    *,
    prefer_yellow: bool,
    open_k: int = 3,
    close_k: int = 13,
    max_hole_px: int = 3000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bag SSOT ego blob: paint|road → morph open3/close13 → bottom ego.

    Matches ``viz_raw_hsv_masks.extract_five`` (pre-PDC / race SSOT). Returns
    ``(white_mask, ego_blob, bev)``.
    """

    from inference.modules.perception.blob.morph_blob import (
        keep_bottom_ego_blob,
        keep_near_floor_blob,
        morph_clean_road,
    )
    from inference.modules.perception.blob.rail_corridor import (
        resolve_course_lane_mask,
    )

    if frame is None or getattr(frame, 'size', 0) == 0:
        empty = np.empty((0, 0), dtype=np.uint8)
        return empty, empty, np.empty((0, 0, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
    ensure_ipm_maps(w, h)
    bounds = hsv_bounds()
    bev = warp_bgr(frame)

    white = _bev_channel_mask(bev, bounds['white'][0], bounds['white'][1])
    yellow = _bev_channel_mask(bev, bounds['yellow'][0], bounds['yellow'][1])
    black_raw = _bev_channel_mask(
        bev, bounds['black_road'][0], bounds['black_road'][1], morph_open=False
    )
    black = keep_near_floor_blob(black_raw)
    red = _bev_channel_mask(bev, bounds['red_road'][0], bounds['red_road'][1])
    cyan1 = _bev_channel_mask(
        bev, bounds['black_cyan'][0], bounds['black_cyan'][1], morph_open=False
    )
    cyan2 = _bev_channel_mask(
        bev, bounds['black_cyan_2'][0], bounds['black_cyan_2'][1], morph_open=False
    )
    cyan = keep_near_floor_blob(cv2.bitwise_or(cyan1, cyan2))
    road = cv2.bitwise_or(cv2.bitwise_or(black, red), cyan)

    paint, _used = resolve_course_lane_mask(
        white, yellow, prefer_yellow=bool(prefer_yellow)
    )
    if paint is None or getattr(paint, 'size', 0) == 0:
        paint = np.zeros_like(road)
    lane_road = cv2.bitwise_or(paint, road)
    cleaned = morph_clean_road(
        lane_road,
        open_k=int(open_k),
        close_k=int(close_k),
        max_hole_px=int(max_hole_px),
    )
    ego = keep_bottom_ego_blob(cleaned)
    return white, ego, bev


def extract_bev_masks(
    frame: np.ndarray,
    *,
    prefer_yellow: bool = False,
) -> dict[str, np.ndarray]:
    """Return BEV uint8 masks: white, yellow, black, red, cyan, road_raw, bev.

    ``road_raw`` = black_near | red | cyan_near — near-robot CC on black and cyan
    *before* road morph (pre-PDC / bag SSOT). ``prefer_yellow`` kept for call-site
    compat; it does not alter road fill.
    """

    del prefer_yellow
    from inference.modules.perception.blob.morph_blob import keep_near_floor_blob

    if frame is None or frame.size == 0:
        empty = np.empty((0, 0), dtype=np.uint8)
        return {
            'bev': np.empty((0, 0, 3), dtype=np.uint8),
            'white': empty,
            'yellow': empty,
            'black': empty,
            'red': empty,
            'cyan': empty,
            'cyan_raw': empty,
            'road_raw': empty,
        }

    h, w = frame.shape[:2]
    ensure_ipm_maps(w, h)
    global _HSV_BOUNDS
    _HSV_BOUNDS = None  # pick up retuned lane_vision.yaml
    bounds = hsv_bounds()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_src = cv2.inRange(hsv, bounds['white'][0], bounds['white'][1])
    yellow_src = cv2.inRange(hsv, bounds['yellow'][0], bounds['yellow'][1])
    black_src = cv2.inRange(hsv, bounds['black_road'][0], bounds['black_road'][1])
    red_src = _red_inrange(hsv, bounds['red_road'][0], bounds['red_road'][1])
    cyan_lo, cyan_hi = bounds.get(
        'black_cyan',
        (
            np.array([72, 40, 80], dtype=np.uint8),
            np.array([110, 255, 230], dtype=np.uint8),
        ),
    )
    cyan2_lo, cyan2_hi = bounds.get(
        'black_cyan_2',
        (
            np.array([97, 240, 105], dtype=np.uint8),
            np.array([105, 255, 180], dtype=np.uint8),
        ),
    )
    cyan_src = cv2.inRange(hsv, cyan_lo, cyan_hi)
    cyan2_src = cv2.inRange(hsv, cyan2_lo, cyan2_hi)

    white = warp_mask(white_src)
    yellow = warp_mask(yellow_src)
    black_raw = warp_mask(black_src)
    black = keep_near_floor_blob(black_raw)
    red = warp_mask(red_src)
    cyan1 = warp_mask(cyan_src)
    cyan2 = warp_mask(cyan2_src)
    road_raw, cyan, cyan_raw = compose_road_raw(
        black,
        red,
        cyan1,
        cyan2,
        yellow,
        prefer_yellow=False,
        keep_near_fn=keep_near_floor_blob,
    )
    bev = warp_bgr(frame)
    return {
        'bev': bev,
        'white': white,
        'yellow': yellow,
        'black': black,
        'black_raw': black_raw,
        'red': red,
        'cyan': cyan,
        'cyan_raw': cyan_raw,
        'cyan1': cyan1,
        'cyan2': cyan2,
        'road_raw': road_raw,
    }
