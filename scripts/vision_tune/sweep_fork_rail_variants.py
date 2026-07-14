#!/usr/bin/env python3
"""Sweep fork rail far/curve techniques and log metrics + previews.

Does not change production defaults permanently — monkeypatch / post-process
per variant, write under data/captures/fork_rail_sweeps/<stamp>/.

Variants
--------
A0 baseline          : current code (far-extend + same-row ±w)
A1 no_far_extend     : disable extend_boundary_pair_far_along_marks
A2 far_both_only     : far-extend only when both L/R marks hit
B0 clip_outer_paint  : drop rail rows once outer loses nearby mark
B1 no_onesided_synth : stitch skips single-outer FOV fill
C0 heading_cos       : same-row Δu = w/(mpp·|tx|) post-pass on pairs
C1 frenet_normal     : rebuild inner/center via path-normal ±w
D0 clip_then_frenet  : B0 then C1
D1 both_only_frenet  : A2 detect + C1

Examples::

  python3 scripts/vision_tune/sweep_fork_rail_variants.py
  python3 scripts/vision_tune/sweep_fork_rail_variants.py --scenes in_exit
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_INFER = _ROOT / "src" / "inference"
import sys

if str(_INFER) not in sys.path:
    sys.path.insert(0, str(_INFER))

from inference.modules import lane_detection as ld  # noqa: E402

OUT_ROOT = _ROOT / "data" / "captures" / "fork_rail_sweeps"

SCENES = {
    "in_exit": {
        "frame": _ROOT
        / "data/captures/lane_tune_logs/auto_fork/in_roundabout_exit/runs/20260713_152921/source_frame.png",
        "prefer_yellow": True,
    },
    "out_fork": {
        "frame": _ROOT
        / "data/captures/lane_tune_logs/auto_fork/out_fork/runs/20260713_152900/source_frame.png",
        "prefer_yellow": False,
    },
}


@dataclass
class VariantResult:
    scene: str
    variant: str
    n_pairs: int
    fork_split_source: str
    path_far_x_m: float
    path_near_x_m: float
    valid_center_rows: int
    overextend_rows: int
    overextend_fraction: float
    mean_outer_paint_err_px: float
    mean_same_row_width_m: float
    mean_heading_deg: float
    notes: str = ""


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _pair_as_mutable(pairs: list[ld.ForkLanePair]) -> list[ld.ForkLanePair]:
    return [
        ld.ForkLanePair(
            lateral_rank=int(p.lateral_rank),
            outer_u=np.asarray(p.outer_u, dtype=np.float32).copy(),
            inner_u=np.asarray(p.inner_u, dtype=np.float32).copy(),
            center_u=np.asarray(p.center_u, dtype=np.float32).copy(),
            outer_missing=bool(p.outer_missing),
            inner_missing=bool(p.inner_missing),
            confidence=float(p.confidence),
        )
        for p in pairs
    ]


def _mark_near_u(mask: np.ndarray, row: int, u: float, assoc_px: float) -> bool:
    if mask.size == 0 or row < 0 or row >= mask.shape[0]:
        return False
    if np.isnan(u):
        return False
    cols = np.flatnonzero(mask[row] > 0)
    if cols.size == 0:
        return False
    return bool(np.min(np.abs(cols.astype(np.float32) - float(u))) <= assoc_px)


def clip_pairs_to_outer_paint(
    pairs: list[ld.ForkLanePair],
    mark_mask: np.ndarray,
    *,
    assoc_m: float = 0.12,
) -> list[ld.ForkLanePair]:
    """Walking far→near: once outer loses nearby paint, clear that row (and allow holes)."""

    assoc_px = max(2.0, assoc_m / ld.METERS_PER_PIXEL)
    out: list[ld.ForkLanePair] = []
    for p in _pair_as_mutable(pairs):
        o = p.outer_u
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                continue
            if not _mark_near_u(mark_mask, row, float(o[row]), assoc_px):
                o[row] = np.nan
                p.inner_u[row] = np.nan
                p.center_u[row] = np.nan
        # From farthest: if a gap appears, drop everything farther than last paint-ok streak.
        valid = np.flatnonzero(~np.isnan(o))
        if valid.size:
            # keep only continuous run containing nearest (max row)
            tip_near = int(valid[-1])
            keep_from = int(valid[0])
            for row in range(tip_near, -1, -1):
                if np.isnan(o[row]):
                    keep_from = row + 1
                    break
            for row in range(0, keep_from):
                o[row] = np.nan
                p.inner_u[row] = np.nan
                p.center_u[row] = np.nan
        conf = float(np.clip(np.count_nonzero(~np.isnan(p.center_u)) / ld.BEV_HEIGHT, 0, 1))
        out.append(replace(p, outer_u=o, confidence=conf))
    return out


def apply_heading_cos_width(
    pairs: list[ld.ForkLanePair],
    *,
    width_m: float | None = None,
) -> list[ld.ForkLanePair]:
    """Same-row rail with Δu = (w/mpp) / max(ε, |t_x|) using outer polyline heading."""

    w_m = float(width_m if width_m is not None else ld.FORK_PAIR_WIDTH_M)
    base_px = w_m / ld.METERS_PER_PIXEL
    out: list[ld.ForkLanePair] = []
    for p in _pair_as_mutable(pairs):
        side = "left" if int(p.lateral_rank) == 0 else "right"
        o = p.outer_u
        xy = ld._boundary_u_to_vehicle_points(o)
        if xy.shape[0] < 3:
            out.append(p)
            continue
        # Map row -> unit tangent (tx, ty) by nearest xy sample.
        tangents = np.zeros((xy.shape[0], 2), dtype=np.float32)
        for i in range(xy.shape[0]):
            j0 = max(0, i - 1)
            j1 = min(xy.shape[0] - 1, i + 1)
            d = xy[j1] - xy[j0]
            nrm = float(np.linalg.norm(d))
            tangents[i] = d / nrm if nrm > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)
        row_tx = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
        for i, (x, _y) in enumerate(xy):
            row = int(round((ld.X_MAX_M - float(x)) / ld.METERS_PER_PIXEL))
            if 0 <= row < ld.BEV_HEIGHT:
                row_tx[row] = float(tangents[i, 0])
        # fill row_tx gaps with nearest
        known = np.flatnonzero(~np.isnan(row_tx))
        if known.size == 0:
            out.append(p)
            continue
        for row in range(ld.BEV_HEIGHT):
            if not np.isnan(row_tx[row]):
                continue
            j = int(known[np.argmin(np.abs(known - row))])
            row_tx[row] = row_tx[j]

        inner = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
        center = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                continue
            tx = float(row_tx[row])
            scale = 1.0 / max(0.25, abs(tx))  # |tx|=cosθ in vehicle frame if |t|=1
            du = base_px * float(np.clip(scale, 1.0, 2.5))
            if side == "left":
                inner[row] = float(o[row]) + du
                center[row] = float(o[row]) + 0.5 * du
            else:
                inner[row] = float(o[row]) - du
                center[row] = float(o[row]) - 0.5 * du
        conf = float(np.clip(np.count_nonzero(~np.isnan(center)) / ld.BEV_HEIGHT, 0, 1))
        out.append(
            replace(
                p,
                inner_u=inner,
                center_u=center,
                confidence=conf,
                inner_missing=True,
            )
        )
    return out


def apply_frenet_normal_width(
    pairs: list[ld.ForkLanePair],
    *,
    width_m: float | None = None,
) -> list[ld.ForkLanePair]:
    """Rebuild inner/center by offsetting outer polyline along path normal ±w."""

    w_m = float(width_m if width_m is not None else ld.FORK_PAIR_WIDTH_M)
    half = 0.5 * w_m
    out: list[ld.ForkLanePair] = []
    for p in _pair_as_mutable(pairs):
        side = "left" if int(p.lateral_rank) == 0 else "right"
        o = p.outer_u
        xy = ld._boundary_u_to_vehicle_points(o)
        if xy.shape[0] < 3:
            out.append(p)
            continue
        # signed offset: left outer → lane is to the right of travel (-n_left)
        # right outer → lane is to the left of travel (+n_left)
        sign = -1.0 if side == "left" else 1.0
        inner_xy = np.zeros_like(xy)
        center_xy = np.zeros_like(xy)
        for i in range(xy.shape[0]):
            j0 = max(0, i - 1)
            j1 = min(xy.shape[0] - 1, i + 1)
            d = xy[j1] - xy[j0]
            nrm = float(np.linalg.norm(d))
            if nrm < 1e-6:
                t = np.array([1.0, 0.0], dtype=np.float32)
            else:
                t = (d / nrm).astype(np.float32)
            # left normal in (x, y_left): (-ty, tx)
            n_left = np.array([-t[1], t[0]], dtype=np.float32)
            inner_xy[i] = xy[i] + sign * w_m * n_left
            center_xy[i] = xy[i] + sign * half * n_left

        def rasterize(points_xy: np.ndarray) -> np.ndarray:
            cols = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
            buckets: dict[int, list[float]] = {}
            for x, y in points_xy:
                row = int(round((ld.X_MAX_M - float(x)) / ld.METERS_PER_PIXEL))
                if row < 0 or row >= ld.BEV_HEIGHT:
                    continue
                u = (ld.BEV_WIDTH - 1) / 2.0 - float(y) / ld.METERS_PER_PIXEL
                buckets.setdefault(row, []).append(u)
            for row, us in buckets.items():
                cols[row] = float(np.median(us))
            return cols

        # Keep outer as-is (observed); only rebuild inner/center.
        inner = rasterize(inner_xy)
        center = rasterize(center_xy)
        # Mask to rows where outer exists.
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                inner[row] = np.nan
                center[row] = np.nan
        conf = float(np.clip(np.count_nonzero(~np.isnan(center)) / ld.BEV_HEIGHT, 0, 1))
        out.append(
            replace(
                p,
                inner_u=inner,
                center_u=center,
                confidence=conf,
                inner_missing=True,
            )
        )
    return out


def _xy_to_u(x: float, y: float) -> tuple[int, float]:
    row = int(round((ld.X_MAX_M - float(x)) / ld.METERS_PER_PIXEL))
    u = (ld.BEV_WIDTH - 1) / 2.0 - float(y) / ld.METERS_PER_PIXEL
    return row, u


def _rasterize_polyline_dense(points_xy: np.ndarray) -> np.ndarray:
    """Fill every BEV row crossed by polyline segments (avoids C1 horizontal spikes)."""

    cols = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
    if points_xy.shape[0] < 2:
        return cols
    buckets: dict[int, list[float]] = {}
    for i in range(points_xy.shape[0] - 1):
        x0, y0 = float(points_xy[i, 0]), float(points_xy[i, 1])
        x1, y1 = float(points_xy[i + 1, 0]), float(points_xy[i + 1, 1])
        r0, _ = _xy_to_u(x0, y0)
        r1, _ = _xy_to_u(x1, y1)
        n = max(1, abs(r1 - r0) * 2 + 2)
        for t in np.linspace(0.0, 1.0, n):
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            row, u = _xy_to_u(x, y)
            if 0 <= row < ld.BEV_HEIGHT:
                buckets.setdefault(row, []).append(u)
    for row, us in buckets.items():
        cols[row] = float(np.median(us))
    return cols


def _smooth_polyline_xy(xy: np.ndarray, *, window: int = 9) -> np.ndarray:
    if xy.shape[0] < 3:
        return xy.astype(np.float32, copy=True)
    out = xy.astype(np.float32, copy=True)
    w = max(3, int(window) | 1)
    half = w // 2
    y = out[:, 1].copy()
    for i in range(out.shape[0]):
        a = max(0, i - half)
        b = min(out.shape[0], i + half + 1)
        y[i] = float(np.mean(out[a:b, 1]))
    out[:, 1] = y
    return out


def _resample_arclength(xy: np.ndarray, *, step_m: float = 0.02) -> np.ndarray:
    if xy.shape[0] < 2:
        return xy.astype(np.float32, copy=True)
    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    total = float(s[-1])
    if total < step_m * 2:
        return xy.astype(np.float32, copy=True)
    s_new = np.arange(0.0, total, step_m, dtype=np.float32)
    if s_new[-1] < total:
        s_new = np.append(s_new, total)
    x = np.interp(s_new, s, xy[:, 0])
    y = np.interp(s_new, s, xy[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


def apply_curvature_parallel_rails(
    pairs: list[ld.ForkLanePair],
    *,
    width_m: float | None = None,
    min_radius_m: float = 0.35,
    return_debug: bool = False,
) -> list[ld.ForkLanePair] | tuple[list[ld.ForkLanePair], dict]:
    """Osculating-circle / curvature-aware parallel rails.

    Outer paint stay as observed. Inner & center are parallel curves of the
    smoothed outer path at lateral distance ``width`` / ``width/2``, using local
    curvature radius R=1/κ:

        C = p + n_left / κ
        p_off = C - (R - d_signed) * n_left   # when κ≠0
              = p + d_signed * n_left           # |κ| small → Frenet limit

    ``d_signed`` is toward the lane interior (−w for left outer, +w for right).
    When |κ|·|d| → 1 the offset cusps; we clamp |d| < 0.85/|κ|.
    """

    w_m = float(width_m if width_m is not None else ld.FORK_PAIR_WIDTH_M)
    half = 0.5 * w_m
    dbg: dict = {"pairs": []}
    out: list[ld.ForkLanePair] = []

    for p in _pair_as_mutable(pairs):
        side = "left" if int(p.lateral_rank) == 0 else "right"
        o = p.outer_u
        xy0 = ld._boundary_u_to_vehicle_points(o)
        if xy0.shape[0] < 5:
            out.append(p)
            continue

        xy = _resample_arclength(_smooth_polyline_xy(xy0, window=11), step_m=0.015)
        n_pts = xy.shape[0]
        # tangents / normals / curvature along arc length
        t_hat = np.zeros((n_pts, 2), dtype=np.float32)
        n_left = np.zeros((n_pts, 2), dtype=np.float32)
        kappa = np.zeros(n_pts, dtype=np.float32)
        theta = np.zeros(n_pts, dtype=np.float32)
        for i in range(n_pts):
            j0 = max(0, i - 1)
            j1 = min(n_pts - 1, i + 1)
            d = xy[j1] - xy[j0]
            nrm = float(np.linalg.norm(d))
            if nrm < 1e-6:
                t_hat[i] = (1.0, 0.0)
            else:
                t_hat[i] = d / nrm
            n_left[i] = (-t_hat[i, 1], t_hat[i, 0])
            theta[i] = float(np.arctan2(t_hat[i, 1], t_hat[i, 0]))
        # unwrap heading then κ = dθ/ds
        theta = np.unwrap(theta.astype(np.float64)).astype(np.float32)
        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        ds = np.maximum(ds, 1e-4)
        dtheta = np.diff(theta)
        k_seg = dtheta / ds
        kappa[0] = float(k_seg[0])
        kappa[-1] = float(k_seg[-1])
        for i in range(1, n_pts - 1):
            kappa[i] = float(0.5 * (k_seg[i - 1] + k_seg[i]))
        # mild kappa smooth
        kappa = np.convolve(kappa, np.ones(5) / 5.0, mode="same").astype(np.float32)

        # interior sign along left-normal
        sign = -1.0 if side == "left" else 1.0

        def offset_curve(dist_m: float) -> np.ndarray:
            pts = np.zeros_like(xy)
            for i in range(n_pts):
                k = float(kappa[i])
                d = sign * float(dist_m)
                # cusp guard: |κ d| < 0.85
                if abs(k) > 1e-4:
                    max_d = 0.85 / abs(k)
                    if abs(d) > max_d:
                        d = float(np.sign(d) * max_d)
                    # Osculating circle: C = p + n/κ, radial from C to p is -n when κ>0
                    # p = C - (1/κ) n_left  ⇒  parallel at d along +n_left:
                    # p' = C - (1/κ - d) n = p + d n_left
                    # (same formula; explicit R helps when we want radius logging)
                    R = 1.0 / k
                    C = xy[i] + n_left[i] * R
                    pts[i] = C - n_left[i] * (R - d)
                else:
                    pts[i] = xy[i] + n_left[i] * d
            return pts.astype(np.float32)

        inner_xy = offset_curve(w_m)
        center_xy = offset_curve(half)
        inner = _rasterize_polyline_dense(inner_xy)
        center = _rasterize_polyline_dense(center_xy)
        # Outer: densify observed outer too so gaps from extend don't leave holes
        outer_dense = _rasterize_polyline_dense(xy)
        # Prefer observed outer where available; else densified smooth outer
        outer = o.copy()
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(outer[row]) and not np.isnan(outer_dense[row]):
                # do not invent outer beyond observation — keep nan
                pass
            if np.isnan(outer[row]):
                inner[row] = np.nan
                center[row] = np.nan

        radii = np.abs(1.0 / np.clip(kappa, -50.0, 50.0))
        radii = radii[np.isfinite(radii) & (radii < 50.0)]
        mean_R = float(np.median(radii)) if radii.size else float("nan")
        conf = float(np.clip(np.count_nonzero(~np.isnan(center)) / ld.BEV_HEIGHT, 0, 1))
        out.append(
            replace(
                p,
                outer_u=outer,
                inner_u=inner,
                center_u=center,
                confidence=conf,
                inner_missing=True,
            )
        )
        dbg["pairs"].append(
            {
                "rank": int(p.lateral_rank),
                "mean_abs_radius_m": mean_R,
                "mean_abs_kappa": float(np.mean(np.abs(kappa))),
                "n_samples": int(n_pts),
            }
        )

    if return_debug:
        return out, dbg
    return out


def extend_both_only(
    left: np.ndarray,
    right: np.ndarray,
    boundary_mask: np.ndarray,
    *,
    assoc_m: float = ld.FAR_COURSE_ASSOC_M,
    max_miss_rows: int = ld.FAR_COURSE_MAX_MISS_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Like far-extend but never synthesize a missing side with ±width."""

    left_out = np.asarray(left, dtype=np.float32).copy()
    right_out = np.asarray(right, dtype=np.float32).copy()
    if boundary_mask.size == 0 or left_out.shape[0] != boundary_mask.shape[0]:
        return left_out, right_out
    both = np.flatnonzero(~np.isnan(left_out) & ~np.isnan(right_out))
    if both.size == 0:
        return left_out, right_out
    tip = int(both[0])
    if tip <= 0:
        return left_out, right_out

    assoc_px = float(max(2.0, assoc_m / ld.METERS_PER_PIXEL))
    segs_by_row = ld.find_line_segments_by_row(boundary_mask)
    prev_l = float(left_out[tip])
    prev_r = float(right_out[tip])
    miss = 0

    def nearest(prev_u: float, segments: list[tuple[int, int]]) -> float | None:
        best = None
        best_d = float("inf")
        for s, e in segments:
            for cand in (float(s), float(e)):
                d = abs(cand - prev_u)
                if d < best_d:
                    best_d = d
                    best = cand
        return best if best is not None and best_d <= assoc_px else None

    for row in range(tip - 1, -1, -1):
        if not np.isnan(left_out[row]) and not np.isnan(right_out[row]):
            prev_l = float(left_out[row])
            prev_r = float(right_out[row])
            miss = 0
            continue
        segments = segs_by_row[row]
        if not segments:
            miss += 1
            if miss > int(max_miss_rows):
                break
            continue
        nl = nearest(prev_l, segments)
        nr = nearest(prev_r, segments)
        if nl is None or nr is None or nr <= nl:
            miss += 1
            if miss > int(max_miss_rows):
                break
            continue
        left_out[row] = float(np.clip(nl, 0.0, float(ld.BEV_WIDTH - 1)))
        right_out[row] = float(np.clip(nr, 0.0, float(ld.BEV_WIDTH - 1)))
        prev_l, prev_r = float(left_out[row]), float(right_out[row])
        miss = 0
    return left_out, right_out


def stitch_no_onesided_factory(orig_stitch):
    def stitch_no_onesided(
        pairs: list[ld.ForkLanePair],
        mark_mask: np.ndarray | None = None,
    ) -> list[ld.ForkLanePair]:
        """Wrap stitch then clear rows where this path's outer was missing."""

        stitched = orig_stitch(pairs, mark_mask=mark_mask)
        if len(pairs) < 2 or len(stitched) < 2:
            return stitched
        by_in = {int(p.lateral_rank): p for p in pairs}
        out: list[ld.ForkLanePair] = []
        for sp in stitched:
            src = by_in.get(int(sp.lateral_rank))
            if src is None:
                out.append(sp)
                continue
            o = np.asarray(sp.outer_u, dtype=np.float32).copy()
            i = np.asarray(sp.inner_u, dtype=np.float32).copy()
            c = np.asarray(sp.center_u, dtype=np.float32).copy()
            for row in range(ld.BEV_HEIGHT):
                own = not np.isnan(src.outer_u[row])
                if not own:
                    o[row] = np.nan
                    i[row] = np.nan
                    c[row] = np.nan
            conf = float(
                np.clip(np.count_nonzero(~np.isnan(c)) / ld.BEV_HEIGHT, 0, 1)
            )
            out.append(
                ld.ForkLanePair(
                    lateral_rank=int(sp.lateral_rank),
                    outer_u=o,
                    inner_u=i,
                    center_u=c,
                    outer_missing=bool(sp.outer_missing),
                    inner_missing=bool(sp.inner_missing),
                    confidence=conf,
                )
            )
        return out

    return stitch_no_onesided


def score_pairs(
    pairs: list[ld.ForkLanePair],
    mark_mask: np.ndarray,
    *,
    assoc_m: float = 0.12,
) -> dict[str, float]:
    assoc_px = max(2.0, assoc_m / ld.METERS_PER_PIXEL)
    far_xs: list[float] = []
    near_xs: list[float] = []
    valid_rows = 0
    over = 0
    considered = 0
    paint_err: list[float] = []
    widths: list[float] = []
    headings: list[float] = []

    for p in pairs:
        o = np.asarray(p.outer_u, dtype=np.float32)
        i = np.asarray(p.inner_u, dtype=np.float32)
        c = np.asarray(p.center_u, dtype=np.float32)
        cv = np.flatnonzero(~np.isnan(c))
        valid_rows += int(cv.size)
        if cv.size:
            far_xs.append(float(ld.X_MAX_M - cv[0] * ld.METERS_PER_PIXEL))
            near_xs.append(float(ld.X_MAX_M - cv[-1] * ld.METERS_PER_PIXEL))
        xy = ld._boundary_u_to_vehicle_points(o)
        for k in range(1, max(1, xy.shape[0])):
            d = xy[k] - xy[k - 1]
            if float(np.linalg.norm(d)) < 1e-6:
                continue
            headings.append(float(np.degrees(np.arctan2(d[1], d[0]))))
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                continue
            considered += 1
            cols = np.flatnonzero(mark_mask[row] > 0) if mark_mask.size else np.array([])
            if cols.size == 0:
                over += 1
                continue
            err = float(np.min(np.abs(cols.astype(np.float32) - float(o[row]))))
            paint_err.append(err)
            if err > assoc_px:
                over += 1
            if not np.isnan(i[row]):
                widths.append(abs(float(i[row]) - float(o[row])) * ld.METERS_PER_PIXEL)

    return {
        "path_far_x_m": float(max(far_xs) if far_xs else float("nan")),
        "path_near_x_m": float(min(near_xs) if near_xs else float("nan")),
        "valid_center_rows": float(valid_rows),
        "overextend_rows": float(over),
        "overextend_fraction": float(over / max(1, considered)),
        "mean_outer_paint_err_px": float(np.mean(paint_err) if paint_err else float("nan")),
        "mean_same_row_width_m": float(np.mean(widths) if widths else float("nan")),
        "mean_heading_deg": float(np.mean(np.abs(headings)) if headings else float("nan")),
    }


def make_debug_with_pairs(dbg: ld.LaneDebugFrame, pairs: list[ld.ForkLanePair]) -> ld.LaneDebugFrame:
    return replace(
        dbg,
        fork_lane_pairs=tuple(pairs),
        fork_active=len(pairs) >= 2,
    )


def run_detect(frame: np.ndarray, prefer_yellow: bool) -> tuple[object, ld.LaneDebugFrame]:
    return ld.detect_with_debug(frame, prefer_yellow=prefer_yellow)


def variant_runners() -> dict[str, Callable[..., tuple[ld.LaneDebugFrame, str]]]:
    """Each runner(frame, prefer_yellow) -> (debug, notes)."""

    def A0(frame, prefer_yellow):
        _, dbg = run_detect(frame, prefer_yellow)
        return dbg, "production: far-extend + same-row ±w stitch"

    def A1(frame, prefer_yellow):
        orig = ld.extend_boundary_pair_far_along_marks
        ld.extend_boundary_pair_far_along_marks = (
            lambda l, r, m, **k: (np.asarray(l, np.float32).copy(), np.asarray(r, np.float32).copy())
        )
        try:
            _, dbg = run_detect(frame, prefer_yellow)
        finally:
            ld.extend_boundary_pair_far_along_marks = orig
        return dbg, "no far-extend"

    def A2(frame, prefer_yellow):
        orig = ld.extend_boundary_pair_far_along_marks
        ld.extend_boundary_pair_far_along_marks = extend_both_only
        try:
            _, dbg = run_detect(frame, prefer_yellow)
        finally:
            ld.extend_boundary_pair_far_along_marks = orig
        return dbg, "far-extend both-mark hits only (no ±w synth)"

    def B0(frame, prefer_yellow):
        _, dbg = run_detect(frame, prefer_yellow)
        mask = dbg.yellow_connected_bev if prefer_yellow else dbg.white_bev
        if mask is None or mask.size == 0:
            mask = dbg.yellow_bev if prefer_yellow else dbg.white_bev
        pairs = clip_pairs_to_outer_paint(list(dbg.fork_lane_pairs), mask)
        return make_debug_with_pairs(dbg, pairs), "baseline then clip rails to outer paint"

    def B1(frame, prefer_yellow):
        orig = ld.stitch_fork_stem_continuity
        ld.stitch_fork_stem_continuity = stitch_no_onesided_factory(orig)
        try:
            _, dbg = run_detect(frame, prefer_yellow)
        finally:
            ld.stitch_fork_stem_continuity = orig
        return dbg, "stitch cleared when outer missing"

    def C0(frame, prefer_yellow):
        _, dbg = run_detect(frame, prefer_yellow)
        pairs = apply_heading_cos_width(list(dbg.fork_lane_pairs))
        return make_debug_with_pairs(dbg, pairs), "baseline + heading-cos same-row width"

    def C1(frame, prefer_yellow):
        _, dbg = run_detect(frame, prefer_yellow)
        pairs = apply_frenet_normal_width(list(dbg.fork_lane_pairs))
        return make_debug_with_pairs(dbg, pairs), "baseline + Frenet normal ±w"

    def D0(frame, prefer_yellow):
        _, dbg = run_detect(frame, prefer_yellow)
        mask = dbg.yellow_connected_bev if prefer_yellow else dbg.white_bev
        if mask is None or mask.size == 0:
            mask = dbg.yellow_bev if prefer_yellow else dbg.white_bev
        pairs = clip_pairs_to_outer_paint(list(dbg.fork_lane_pairs), mask)
        pairs = apply_frenet_normal_width(pairs)
        return make_debug_with_pairs(dbg, pairs), "clip outer paint then Frenet"

    def D1(frame, prefer_yellow):
        orig = ld.extend_boundary_pair_far_along_marks
        ld.extend_boundary_pair_far_along_marks = extend_both_only
        try:
            _, dbg = run_detect(frame, prefer_yellow)
        finally:
            ld.extend_boundary_pair_far_along_marks = orig
        pairs = apply_frenet_normal_width(list(dbg.fork_lane_pairs))
        return make_debug_with_pairs(dbg, pairs), "both-only far-extend + Frenet"

    def D2(frame, prefer_yellow):
        """A1 (no extend) + clip + Frenet — conservative stack."""
        orig = ld.extend_boundary_pair_far_along_marks
        ld.extend_boundary_pair_far_along_marks = (
            lambda l, r, m, **k: (np.asarray(l, np.float32).copy(), np.asarray(r, np.float32).copy())
        )
        try:
            _, dbg = run_detect(frame, prefer_yellow)
        finally:
            ld.extend_boundary_pair_far_along_marks = orig
        mask = dbg.yellow_connected_bev if prefer_yellow else dbg.white_bev
        if mask is None or mask.size == 0:
            mask = dbg.yellow_bev if prefer_yellow else dbg.white_bev
        pairs = clip_pairs_to_outer_paint(list(dbg.fork_lane_pairs), mask)
        pairs = apply_frenet_normal_width(pairs)
        return make_debug_with_pairs(dbg, pairs), "no-extend + clip + Frenet"

    def F0(frame, prefer_yellow):
        _, dbg = run_detect(frame, prefer_yellow)
        pairs = apply_curvature_parallel_rails(list(dbg.fork_lane_pairs))
        return make_debug_with_pairs(dbg, pairs), "curvature-radius parallel (osculating)"

    def F1(frame, prefer_yellow):
        _, dbg = run_detect(frame, prefer_yellow)
        mask = dbg.yellow_connected_bev if prefer_yellow else dbg.white_bev
        if mask is None or mask.size == 0:
            mask = dbg.yellow_bev if prefer_yellow else dbg.white_bev
        pairs = clip_pairs_to_outer_paint(list(dbg.fork_lane_pairs), mask)
        pairs = apply_curvature_parallel_rails(pairs)
        return make_debug_with_pairs(dbg, pairs), "clip + curvature-radius parallel"

    def F2(frame, prefer_yellow):
        orig = ld.extend_boundary_pair_far_along_marks
        ld.extend_boundary_pair_far_along_marks = extend_both_only
        try:
            _, dbg = run_detect(frame, prefer_yellow)
        finally:
            ld.extend_boundary_pair_far_along_marks = orig
        pairs = apply_curvature_parallel_rails(list(dbg.fork_lane_pairs))
        return make_debug_with_pairs(dbg, pairs), "both-only far-extend + curvature parallel"

    def F3(frame, prefer_yellow):
        """Smooth observed outer as the base curve, then curvature-parallel rails.

        Unlike F0 (keep jagged observed outer), rebuild outer from the same
        arclength-smoothed polyline used for κ so tip follows path curvature.
        """
        _, dbg = run_detect(frame, prefer_yellow)
        mask = dbg.yellow_connected_bev if prefer_yellow else dbg.white_bev
        if mask is None or mask.size == 0:
            mask = dbg.yellow_bev if prefer_yellow else dbg.white_bev
        base = clip_pairs_to_outer_paint(list(dbg.fork_lane_pairs), mask)
        rebuilt: list[ld.ForkLanePair] = []
        for p in base:
            side = "left" if int(p.lateral_rank) == 0 else "right"
            xy0 = ld._boundary_u_to_vehicle_points(p.outer_u)
            if xy0.shape[0] < 5:
                rebuilt.append(p)
                continue
            xy = _resample_arclength(_smooth_polyline_xy(xy0, window=15), step_m=0.012)
            # Temporary pair with densified smooth outer, then curvature offset.
            outer_s = _rasterize_polyline_dense(xy)
            # Keep only rows that had original outer observation (no invent).
            outer_obs = np.asarray(p.outer_u, dtype=np.float32).copy()
            for row in range(ld.BEV_HEIGHT):
                if np.isnan(outer_obs[row]):
                    outer_s[row] = np.nan
                elif not np.isnan(outer_s[row]):
                    # blend toward smooth but stay near paint
                    outer_s[row] = 0.35 * float(outer_obs[row]) + 0.65 * float(outer_s[row])
            tmp = replace(p, outer_u=outer_s)
            curved = apply_curvature_parallel_rails([tmp])[0]
            rebuilt.append(curved)
        return make_debug_with_pairs(dbg, rebuilt), "clip + smooth-outer + curvature parallel"

    return {
        "A0_baseline": A0,
        "A1_no_far_extend": A1,
        "A2_far_both_only": A2,
        "B0_clip_outer_paint": B0,
        "B1_no_onesided_stitch": B1,
        "C0_heading_cos": C0,
        "C1_frenet_normal": C1,
        "D0_clip_then_frenet": D0,
        "D1_both_only_frenet": D1,
        "D2_noext_clip_frenet": D2,
        "F0_curvature_parallel": F0,
        "F1_clip_curvature": F1,
        "F2_both_curvature": F2,
        "F3_smooth_outer_curvature": F3,
    }


def write_contact_sheet(previews: list[tuple[str, np.ndarray]], path: Path, cols: int = 5) -> None:
    if not previews:
        return
    tiles = []
    h = w = None
    for name, img in previews:
        canvas = img.copy()
        cv2.putText(
            canvas,
            name,
            (4, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if h is None:
            h, w = canvas.shape[:2]
        tiles.append(canvas)
    rows = int(np.ceil(len(tiles) / cols))
    pad = np.zeros((h, w, 3), dtype=np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(pad.copy())
    grid_rows = []
    for r in range(rows):
        grid_rows.append(np.hstack(tiles[r * cols : (r + 1) * cols]))
    sheet = np.vstack(grid_rows)
    cv2.imwrite(str(path), sheet)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="in_exit,out_fork")
    ap.add_argument("--variants", default="")
    args = ap.parse_args()
    scene_names = [s.strip() for s in args.scenes.split(",") if s.strip()]
    runners = variant_runners()
    if args.variants.strip():
        keep = {k.strip() for k in args.variants.split(",") if k.strip()}
        runners = {k: v for k, v in runners.items() if k in keep}

    stamp = _stamp()
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[VariantResult] = []
    summary_lines = [
        f"# Fork rail variant sweep `{stamp}`",
        "",
        f"- X_MAX_M={ld.X_MAX_M} mpp={ld.METERS_PER_PIXEL} width={ld.FORK_PAIR_WIDTH_M}",
        f"- variants={list(runners)}",
        "",
    ]

    t0 = time.time()
    for scene in scene_names:
        spec = SCENES[scene]
        frame = cv2.imread(str(spec["frame"]))
        if frame is None:
            raise SystemExit(f"missing frame {spec['frame']}")
        scene_dir = out_dir / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        previews: list[tuple[str, np.ndarray]] = []
        summary_lines.append(f"## {scene}")
        summary_lines.append("")
        summary_lines.append(
            "| variant | far_x | overext_frac | paint_err_px | width_m | heading_deg | pairs |"
        )
        summary_lines.append("|---|---:|---:|---:|---:|---:|---:|")

        for name, runner in runners.items():
            dbg, notes = runner(frame, bool(spec["prefer_yellow"]))
            pairs = list(dbg.fork_lane_pairs)
            mask = dbg.yellow_connected_bev if spec["prefer_yellow"] else dbg.white_bev
            if mask is None or getattr(mask, "size", 0) == 0:
                mask = dbg.yellow_bev if spec["prefer_yellow"] else dbg.white_bev
            if mask is None or getattr(mask, "size", 0) == 0:
                mask = np.zeros((ld.BEV_HEIGHT, ld.BEV_WIDTH), dtype=np.uint8)
            metrics = score_pairs(pairs, mask)
            preview = ld.make_fork_focus_preview(dbg, focus="all")
            preview_path = scene_dir / f"{name}.png"
            cv2.imwrite(str(preview_path), preview)
            previews.append((name, preview))
            meta = {
                "scene": scene,
                "variant": name,
                "notes": notes,
                "fork_split_source": getattr(dbg, "fork_split_source", ""),
                "n_pairs": len(pairs),
                **metrics,
                "preview": str(preview_path.relative_to(_ROOT)),
            }
            (scene_dir / f"{name}.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            vr = VariantResult(
                scene=scene,
                variant=name,
                n_pairs=len(pairs),
                fork_split_source=str(getattr(dbg, "fork_split_source", "")),
                path_far_x_m=float(metrics["path_far_x_m"]),
                path_near_x_m=float(metrics["path_near_x_m"]),
                valid_center_rows=int(metrics["valid_center_rows"]),
                overextend_rows=int(metrics["overextend_rows"]),
                overextend_fraction=float(metrics["overextend_fraction"]),
                mean_outer_paint_err_px=float(metrics["mean_outer_paint_err_px"]),
                mean_same_row_width_m=float(metrics["mean_same_row_width_m"]),
                mean_heading_deg=float(metrics["mean_heading_deg"]),
                notes=notes,
            )
            results.append(vr)
            summary_lines.append(
                f"| `{name}` | {vr.path_far_x_m:.2f} | {vr.overextend_fraction:.3f} | "
                f"{vr.mean_outer_paint_err_px:.1f} | {vr.mean_same_row_width_m:.3f} | "
                f"{vr.mean_heading_deg:.1f} | {vr.n_pairs} |"
            )
            print(
                f"[{scene}] {name}: far={vr.path_far_x_m:.2f}m "
                f"over={vr.overextend_fraction:.3f} paint_err={vr.mean_outer_paint_err_px:.1f}px "
                f"w={vr.mean_same_row_width_m:.3f}m"
            )

        write_contact_sheet(previews, scene_dir / "contact_sheet.png")
        summary_lines.append("")
        summary_lines.append(f"![contact]({scene}/contact_sheet.png)")
        summary_lines.append("")

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
        if results:
            w.writeheader()
            for r in results:
                w.writerow(asdict(r))

    # Ranking hint: low overextend + paint_err, far_x not necessarily max, width near 0.35
    summary_lines.extend(
        [
            "## Ranking hint (auto, not final)",
            "",
            "Score = `overextend_fraction*2 + (paint_err_px/50) + abs(width_m-0.35)*3 - max(0, far_x-1.0)*0.2`",
            "Lower is better. Prefer low over-extend; far reach only if still on paint.",
            "",
        ]
    )
    for scene in scene_names:
        scene_res = [r for r in results if r.scene == scene]
        scored = []
        for r in scene_res:
            score = (
                r.overextend_fraction * 2.0
                + (r.mean_outer_paint_err_px / 50.0 if np.isfinite(r.mean_outer_paint_err_px) else 1.0)
                + abs((r.mean_same_row_width_m if np.isfinite(r.mean_same_row_width_m) else 0.35) - 0.35) * 3.0
                - max(0.0, (r.path_far_x_m if np.isfinite(r.path_far_x_m) else 0.0) - 1.0) * 0.2
            )
            scored.append((score, r))
        scored.sort(key=lambda t: t[0])
        summary_lines.append(f"### {scene}")
        for score, r in scored:
            summary_lines.append(f"1. `{r.variant}` score={score:.3f} — {r.notes}")
        summary_lines.append("")

    summary_lines.append(f"elapsed_s={time.time() - t0:.1f}")
    (out_dir / "REPORT.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_dir}")
    print(f"Report: {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
