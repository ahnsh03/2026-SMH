"""IN/OUT fork-*moment* detectors (mask geometry only).

These flags mark *approach frames* (직전), not full L/R branch geometry.
Branch pairs still come from legacy ``fork_lane_pairs_*`` / ``road_split`` /
``yellow_alt`` (see ``docs/fork-moment-detection.md``).

SSOT thresholds: ``docs/lane-occlusion-fork-strategy.md`` §5.1.2–5.1.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# --- Band ratios (v=0 is far / image top in Metric IPM BEV) ---
FAR = (0.05, 0.45)
MID = (0.40, 0.70)
NEAR = (0.70, 0.95)
TOP20 = (0.0, 0.20)

# IN circle keep/exit moment — retuned 2026-07-16 on in_cam (HSV after camera change).
# Keep cluster ~677–711 (roomy road + strong dual); exit tip ~1010–1012 (sparse yellow).
IN_KEEP_FAR_DUALY_MIN = 25.0
IN_KEEP_MID_DUALY_MIN = 30.0
IN_KEEP_SEP_MIN = 120.0
IN_KEEP_YA2_MIN = 0.50
IN_KEEP_NEAR_DUALY_MAX = 30.0
IN_KEEP_FAR_DUALF_MIN = 60.0
IN_KEEP_SPAN_MIN = 1.15
IN_KEEP_SPAN_MAX = 2.60
IN_KEEP_Y_PX_MIN = 300
IN_KEEP_Y_PX_MAX = 1100
IN_KEEP_ROAD_PCT_MIN = 15.0

IN_EXIT_FAR_DUALY_MIN = 5.0
IN_EXIT_FAR_DUALY_MAX = 35.0
IN_EXIT_MID_DUALY_MIN = 12.0
IN_EXIT_SEP_MIN = 125.0
IN_EXIT_YA2_MIN = 0.35
IN_EXIT_NEAR_DUALY_MAX = 5.0
IN_EXIT_FAR_DUALF_MIN = 70.0
IN_EXIT_SPAN_MIN = 1.20
IN_EXIT_SPAN_MAX = 2.00
IN_EXIT_Y_PX_MIN = 250
IN_EXIT_Y_PX_MAX = 550
IN_EXIT_ROAD_PCT_MAX = 12.0

# Soft = approach (either path, slightly looser).
IN_SPAN_SOFT = 1.5
IN_TOP_DUAL_FREE_SOFT = 60.0

# OUT white fork moment (§5.1.3)
OUT_FAR_DUALW_MIN = 90.0
OUT_MID_DUALW_MIN = 70.0
OUT_SEP_W_MIN = 150.0
OUT_WA2_RATIO_MIN = 0.50
OUT_FAR_DUAL_ROAD_MIN = 80.0
OUT_SPAN_ROAD_MIN = 2.2
OUT_ROAD_PCT_MIN = 28.0


def _bin(mask: np.ndarray) -> np.ndarray:
    if mask is None or getattr(mask, 'size', 0) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return (np.asarray(mask) > 0).astype(np.uint8)


def row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive [start, end] runs of nonzero pixels on one row."""

    out: list[tuple[int, int]] = []
    on = False
    start = 0
    vals = np.asarray(row).reshape(-1)
    for i, v in enumerate(vals.tolist()):
        if v and not on:
            on = True
            start = i
        elif not v and on:
            on = False
            out.append((start, i - 1))
    if on:
        out.append((start, int(vals.shape[0]) - 1))
    return out


def _band(h: int, lo: float, hi: float) -> tuple[int, int]:
    return int(h * lo), int(h * hi)


def dual_run_stats(
    mask: np.ndarray, a: int, b: int
) -> tuple[float, float]:
    """Return (dual_run_row_pct, median_left_right_mid_sep_px)."""

    binary = _bin(mask)
    if binary.size == 0:
        return 0.0, 0.0
    dual = 0
    seps: list[float] = []
    n = max(b - a, 1)
    for v in range(a, b):
        rr = row_runs(binary[v])
        if len(rr) >= 2:
            dual += 1
            mids = [(s + e) / 2.0 for s, e in rr]
            seps.append(mids[-1] - mids[0])
    return 100.0 * dual / n, float(np.median(seps)) if seps else 0.0


def median_span_px(mask: np.ndarray, a: int, b: int) -> float:
    binary = _bin(mask)
    spans: list[int] = []
    for v in range(a, b):
        xs = np.where(binary[v] > 0)[0]
        if xs.size:
            spans.append(int(xs.max() - xs.min()))
    return float(np.median(spans)) if spans else 0.0


def top_area_ratio(mask: np.ndarray, a: int, b: int) -> float:
    """a2/a1 of connected components in band (0 if <2 components)."""

    binary = _bin(mask)
    if binary.size == 0:
        return 0.0
    roi = binary[a:b]
    nlab, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    areas = sorted(
        [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, nlab)], reverse=True
    )
    if not areas:
        return 0.0
    if len(areas) < 2 or areas[0] <= 0:
        return 0.0
    return float(areas[1] / areas[0])


def dilate_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    binary = (_bin(mask) * 255).astype(np.uint8)
    if binary.size == 0:
        return binary
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(binary, ker, iterations=1) > 0


def gore_tip_ratio(mask: np.ndarray) -> float:
    """Deepest far dual-run tip as fraction of height (0..1), or -1."""

    binary = _bin(mask)
    if binary.size == 0:
        return -1.0
    h = binary.shape[0]
    tip: int | None = None
    for v in range(0, h):
        if len(row_runs(binary[v])) >= 2:
            tip = v
        elif tip is not None and len(row_runs(binary[v])) <= 1:
            break
    return (float(tip) / float(h)) if tip is not None else -1.0


@dataclass(frozen=True)
class InCircleForkMoment:
    """IN rotate keep/exit approach detector."""

    far_dual_yellow: float
    mid_dual_yellow: float
    near_dual_yellow: float
    far_sep_yellow: float
    ya2_ratio: float
    far_dual_free: float
    span_ratio: float
    top_dual_free: float
    yellow_pct: float
    hard_base: bool
    hard: bool
    soft: bool
    boosted: bool


@dataclass(frozen=True)
class OutForkMoment:
    """OUT true-fork approach detector."""

    far_dual_white: float
    mid_dual_white: float
    near_dual_white: float
    sep_white: float
    wa2_ratio: float
    far_dual_road: float
    span_road: float
    top_dual_road: float
    road_pct: float
    white_pct: float
    gore_pct: float
    frac3_white: float
    hard: bool


def score_in_circle_fork_moment(
    yellow: np.ndarray,
    road: np.ndarray,
    *,
    dilate_k: int = 5,
) -> InCircleForkMoment:
    """Score IN keep/exit fork moment from BEV yellow + road masks."""

    y = _bin(yellow)
    r = _bin(road)
    if y.size == 0 or r.size == 0 or y.shape != r.shape:
        empty = InCircleForkMoment(
            0, 0, 0, 0, 0, 0, 0, 0, 0, False, False, False, False
        )
        return empty

    h = y.shape[0]
    far = _band(h, *FAR)
    mid = _band(h, *MID)
    near = _band(h, *NEAR)
    top = _band(h, *TOP20)

    free = r & ~dilate_mask(y, dilate_k)
    far_dual_y, far_sep = dual_run_stats(y, *far)
    mid_dual_y, _ = dual_run_stats(y, *mid)
    near_dual_y, _ = dual_run_stats(y, *near)
    far_dual_f, _ = dual_run_stats(free, *far)
    top_n = max(8, top[1] - top[0])
    top_dual_f = (
        100.0
        * sum(1 for v in range(top[0], top[1]) if len(row_runs(free[v])) >= 2)
        / top_n
    )
    span = median_span_px(free, *far) / max(median_span_px(free, *near), 1.0)
    ya2 = top_area_ratio(y, *far)
    ypct = 100.0 * float(y.mean())
    y_px = int(np.count_nonzero(y))
    road_pct = 100.0 * float(r.mean())

    # Keep = first circle fork (strong dual + filled road).
    keep = (
        far_dual_y >= IN_KEEP_FAR_DUALY_MIN
        and mid_dual_y >= IN_KEEP_MID_DUALY_MIN
        and far_sep >= IN_KEEP_SEP_MIN
        and ya2 >= IN_KEEP_YA2_MIN
        and near_dual_y <= IN_KEEP_NEAR_DUALY_MAX
        and far_dual_f >= IN_KEEP_FAR_DUALF_MIN
        and IN_KEEP_SPAN_MIN <= span <= IN_KEEP_SPAN_MAX
        and IN_KEEP_Y_PX_MIN <= y_px <= IN_KEEP_Y_PX_MAX
        and road_pct >= IN_KEEP_ROAD_PCT_MIN
    )
    # Exit tip = second circle fork (sparse yellow, large sep, low road fill).
    exit_tip = (
        IN_EXIT_FAR_DUALY_MIN <= far_dual_y < IN_EXIT_FAR_DUALY_MAX
        and mid_dual_y >= IN_EXIT_MID_DUALY_MIN
        and far_sep >= IN_EXIT_SEP_MIN
        and ya2 >= IN_EXIT_YA2_MIN
        and near_dual_y <= IN_EXIT_NEAR_DUALY_MAX
        and far_dual_f >= IN_EXIT_FAR_DUALF_MIN
        and IN_EXIT_SPAN_MIN <= span <= IN_EXIT_SPAN_MAX
        and IN_EXIT_Y_PX_MIN <= y_px <= IN_EXIT_Y_PX_MAX
        and road_pct <= IN_EXIT_ROAD_PCT_MAX
    )
    hard_base = keep
    hard = keep or exit_tip
    soft = (
        far_dual_y >= 5.0
        and mid_dual_y >= 10.0
        and far_sep >= 100.0
        and ya2 >= 0.30
        and far_dual_f >= 60.0
        and (span >= IN_SPAN_SOFT or top_dual_f >= IN_TOP_DUAL_FREE_SOFT)
    )
    boosted = hard and (
        span >= IN_SPAN_SOFT or top_dual_f >= IN_TOP_DUAL_FREE_SOFT
    )
    return InCircleForkMoment(
        far_dual_yellow=far_dual_y,
        mid_dual_yellow=mid_dual_y,
        near_dual_yellow=near_dual_y,
        far_sep_yellow=far_sep,
        ya2_ratio=ya2,
        far_dual_free=far_dual_f,
        span_ratio=span,
        top_dual_free=top_dual_f,
        yellow_pct=ypct,
        hard_base=hard_base,
        hard=hard,
        soft=soft,
        boosted=boosted,
    )


def score_out_fork_moment(
    white: np.ndarray,
    road: np.ndarray,
) -> OutForkMoment:
    """Score OUT true-fork moment from BEV white + road masks."""

    w = _bin(white)
    r = _bin(road)
    if w.size == 0 or r.size == 0 or w.shape != r.shape:
        return OutForkMoment(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, False
        )

    h = w.shape[0]
    far = _band(h, *FAR)
    mid = _band(h, *MID)
    near = _band(h, *NEAR)
    top = _band(h, *TOP20)

    far_dual_w, sep_w = dual_run_stats(w, *far)
    mid_dual_w, _ = dual_run_stats(w, *mid)
    near_dual_w, _ = dual_run_stats(w, *near)
    far_dual_r, _ = dual_run_stats(r, *far)
    top_n = max(8, top[1] - top[0])
    top_dual_r = (
        100.0
        * sum(1 for v in range(top[0], top[1]) if len(row_runs(r[v])) >= 2)
        / top_n
    )
    span_r = median_span_px(r, *far) / max(median_span_px(r, *near), 1.0)
    wa2 = top_area_ratio(w, *far)
    gore = 100.0 * gore_tip_ratio(r)
    run_counts = [len(row_runs(w[v])) for v in range(*far)]
    frac3 = (
        100.0 * sum(1 for c in run_counts if c >= 3) / max(len(run_counts), 1)
    )
    road_pct = 100.0 * float(r.mean())
    white_pct = 100.0 * float(w.mean())

    hard = (
        far_dual_w >= OUT_FAR_DUALW_MIN
        and mid_dual_w >= OUT_MID_DUALW_MIN
        and sep_w >= OUT_SEP_W_MIN
        and wa2 >= OUT_WA2_RATIO_MIN
        and far_dual_r >= OUT_FAR_DUAL_ROAD_MIN
        and span_r >= OUT_SPAN_ROAD_MIN
        and road_pct >= OUT_ROAD_PCT_MIN
    )
    return OutForkMoment(
        far_dual_white=far_dual_w,
        mid_dual_white=mid_dual_w,
        near_dual_white=near_dual_w,
        sep_white=sep_w,
        wa2_ratio=wa2,
        far_dual_road=far_dual_r,
        span_road=span_r,
        top_dual_road=top_dual_r,
        road_pct=road_pct,
        white_pct=white_pct,
        gore_pct=gore,
        frac3_white=frac3,
        hard=hard,
    )


def combine_road_masks(
    black: np.ndarray,
    red: np.ndarray,
    cyan: np.ndarray | None = None,
) -> np.ndarray:
    """``road_raw = black | red | cyan``."""

    out = _bin(black) | _bin(red)
    if cyan is not None:
        out = out | _bin(cyan)
    return out
