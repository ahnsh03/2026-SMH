"""OUT ego-blob *fork stretch* detector (drivable shape only).

Sibling to ``moment.score_out_fork_moment`` (white+road approach flags).
This scores the **Y / dual-lobe far + single near throat** geometry on the
final ``ego_blob`` mask — the long stretch on OUT bag ~1690–1783.

Experimental (v0). Entry / L-R follow is *not* here.
Docs: ``docs/fork-moment-detection.md`` §3.5 · ``docs/out-ego-fork-shape.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# BEV: v=0 is far (image top). Same band idea as moment.py.
FAR = (0.00, 0.40)
NEAR = (0.70, 1.00)
TOP55 = (0.00, 0.55)

# Gate C (default hard) — OUT BEV ego_blob.
# Distant FP = 0 on out.mp4 stride-5; remaining extras are approach/exit bleed.
SEP_FAR_MIN = 130.0
W_FAR_MIN = 220.0
WR_FN_MIN = 2.2
DUAL_FAR_MIN = 50
DUAL_NEAR_MAX = 5
MAX_RUN_MIN = 45
COV_MIN = 30.0
SOLID_MIN = 0.68
SOLID_MAX = 0.82

# Soft / approach (Gate B): earlier onset (~1681), longer exit bleed.
SOFT_SEP_FAR_MIN = 120.0
SOFT_W_FAR_MIN = 200.0
SOFT_WR_FN_MIN = 2.0
SOFT_DUAL_FAR_MIN = 40
SOFT_MAX_RUN_MIN = 40
SOFT_COV_MIN = 25.0
SOFT_SOLID_MIN = 0.65
SOFT_SOLID_MAX = 0.85
SOFT_MAX_SEP_MIN = 140.0


def _bin(mask: np.ndarray) -> np.ndarray:
    if mask is None or getattr(mask, 'size', 0) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return (np.asarray(mask) > 0).astype(np.uint8)


def _band(h: int, lo: float, hi: float) -> slice:
    return slice(int(h * lo), int(h * hi))


def row_segments(row: np.ndarray, gap: int = 2) -> list[tuple[int, int]]:
    """Inclusive [start, end] runs; merges gaps ≤ ``gap`` px."""

    xs = np.flatnonzero(np.asarray(row).reshape(-1) > 0)
    if xs.size == 0:
        return []
    segs: list[tuple[int, int]] = []
    s = int(xs[0])
    prev = int(xs[0])
    for x in xs[1:]:
        x = int(x)
        if x - prev > gap:
            segs.append((s, prev))
            s = x
        prev = x
    segs.append((s, prev))
    return segs


@dataclass(frozen=True)
class OutEgoForkShape:
    """Per-frame ego-blob fork-stretch features + gates."""

    cov: float
    w_far: float
    w_near: float
    wr_fn: float
    dual_far: int
    dual_near: int
    dual_top55: int
    dual_frac: float
    mean_sep_far: float
    max_sep: float
    max_run: int
    solid: float
    hard: bool
    soft: bool


def score_out_ego_fork_shape(ego_blob: np.ndarray) -> OutEgoForkShape:
    """Score Y-fork stretch from a single ``ego_blob`` (white binary/gray)."""

    m = _bin(ego_blob)
    empty = OutEgoForkShape(
        cov=0.0,
        w_far=0.0,
        w_near=0.0,
        wr_fn=0.0,
        dual_far=0,
        dual_near=0,
        dual_top55=0,
        dual_frac=0.0,
        mean_sep_far=0.0,
        max_sep=0.0,
        max_run=0,
        solid=0.0,
        hard=False,
        soft=False,
    )
    if m.size == 0:
        return empty

    h, _w = m.shape
    area = int(m.sum())
    if area < 80:
        return empty

    cov = 100.0 * area / float(m.size)
    widths = np.zeros(h, np.float32)
    dual = np.zeros(h, np.uint8)
    seps = np.full(h, np.nan, np.float32)

    for y in range(h):
        segs = [(a, b) for a, b in row_segments(m[y]) if b - a + 1 >= 3]
        if not segs:
            continue
        left, right = segs[0][0], segs[-1][1]
        widths[y] = float(right - left + 1)
        if len(segs) >= 2:
            dual[y] = 1
            top2 = sorted(segs, key=lambda t: t[1] - t[0] + 1, reverse=True)[:2]
            top2 = sorted(top2, key=lambda t: t[0])
            c0 = 0.5 * (top2[0][0] + top2[0][1])
            c1 = 0.5 * (top2[1][0] + top2[1][1])
            seps[y] = abs(c1 - c0)

    far = _band(h, *FAR)
    near = _band(h, *NEAR)
    top55 = _band(h, *TOP55)

    def _mean_w(sl: slice) -> float:
        ww = widths[sl]
        a = ww[ww > 2]
        return float(a.mean()) if a.size else 0.0

    w_far = _mean_w(far)
    w_near = _mean_w(near)
    wr_fn = w_far / max(w_near, 1e-3)
    dual_far = int(dual[far].sum())
    dual_near = int(dual[near].sum())
    dual_top55 = int(dual[top55].sum())
    active_top = int(np.count_nonzero(widths[top55] > 2))
    dual_frac = float(dual_top55) / max(active_top, 1)

    sep_far = seps[far]
    sep_far = sep_far[np.isfinite(sep_far)]
    mean_sep_far = float(sep_far.mean()) if sep_far.size else 0.0
    sep_all = seps[np.isfinite(seps)]
    max_sep = float(sep_all.max()) if sep_all.size else 0.0

    max_run = 0
    run = 0
    for v in dual[top55]:
        if v:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    solid = 0.0
    cnts, _ = cv2.findContours(m.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c0 = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c0)
        solid = float(cv2.contourArea(c0) / max(cv2.contourArea(hull), 1.0))

    hard = (
        mean_sep_far >= SEP_FAR_MIN
        and w_far >= W_FAR_MIN
        and wr_fn >= WR_FN_MIN
        and dual_far >= DUAL_FAR_MIN
        and dual_near <= DUAL_NEAR_MAX
        and max_run >= MAX_RUN_MIN
        and cov >= COV_MIN
        and SOLID_MIN <= solid <= SOLID_MAX
    )
    soft = (
        mean_sep_far >= SOFT_SEP_FAR_MIN
        and max_sep >= SOFT_MAX_SEP_MIN
        and w_far >= SOFT_W_FAR_MIN
        and wr_fn >= SOFT_WR_FN_MIN
        and dual_far >= SOFT_DUAL_FAR_MIN
        and dual_near == 0
        and max_run >= SOFT_MAX_RUN_MIN
        and cov >= SOFT_COV_MIN
        and SOFT_SOLID_MIN <= solid <= SOFT_SOLID_MAX
    )

    return OutEgoForkShape(
        cov=cov,
        w_far=w_far,
        w_near=w_near,
        wr_fn=wr_fn,
        dual_far=dual_far,
        dual_near=dual_near,
        dual_top55=dual_top55,
        dual_frac=dual_frac,
        mean_sep_far=mean_sep_far,
        max_sep=max_sep,
        max_run=max_run,
        solid=solid,
        hard=hard,
        soft=soft,
    )
