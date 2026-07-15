"""Blob / DT-ridge → base_link centerline."""

from __future__ import annotations

import cv2
import numpy as np


def blob_row_mids(blob: np.ndarray) -> np.ndarray:
    """Per-row center u of foreground; NaN where empty."""

    h, _w = blob.shape[:2]
    mids = np.full(h, np.nan, dtype=np.float32)
    for v in range(h):
        cols = np.flatnonzero(blob[v] > 0)
        if cols.size == 0:
            continue
        mids[v] = 0.5 * (float(cols[0]) + float(cols[-1]))
    return mids


def dt_ridge_mids(blob: np.ndarray) -> np.ndarray:
    """Per-row argmax of distance transform (skeleton ridge)."""

    binary = (blob > 0).astype(np.uint8)
    h = binary.shape[0]
    mids = np.full(h, np.nan, dtype=np.float32)
    if not np.any(binary):
        return mids
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    for v in range(h):
        row = dist[v]
        if float(row.max()) <= 0.0:
            continue
        mids[v] = float(np.argmax(row))
    return mids


def smooth_mids_1d(mids: np.ndarray, win: int = 15) -> np.ndarray:
    """Box smooth finite runs; keep NaN gaps."""

    out = mids.astype(np.float64).copy()
    n = out.size
    if n == 0:
        return mids.astype(np.float32)
    half = max(1, int(win) // 2)
    finite = np.isfinite(out)
    if not np.any(finite):
        return mids.astype(np.float32)
    idx = np.where(finite, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = out.copy()
    filled[~finite] = filled[idx[~finite]]
    first = int(np.flatnonzero(finite)[0])
    filled[:first] = filled[first]
    kernel = np.ones(2 * half + 1, dtype=np.float64)
    kernel /= kernel.size
    pad = np.pad(filled, (half, half), mode='edge')
    smooth = np.convolve(pad, kernel, mode='valid')
    alpha = 0.35
    ema = smooth.copy()
    for i in range(1, n):
        ema[i] = alpha * smooth[i] + (1.0 - alpha) * ema[i - 1]
    for i in range(n - 2, -1, -1):
        ema[i] = alpha * ema[i] + (1.0 - alpha) * ema[i + 1]
    ema[~finite] = np.nan
    return ema.astype(np.float32)


def polyfit_mids(mids: np.ndarray, deg: int = 2) -> np.ndarray:
    """Smooth trajectory with a low-order poly on finite mid rows."""

    out = mids.astype(np.float64).copy()
    rows = np.flatnonzero(np.isfinite(out))
    if rows.size < deg + 2:
        return mids.astype(np.float32)
    coef = np.polyfit(rows.astype(np.float64), out[rows], deg)
    pred = np.polyval(coef, np.arange(out.size, dtype=np.float64))
    lo, hi = int(rows[0]), int(rows[-1])
    pred[:lo] = np.nan
    pred[hi + 1 :] = np.nan
    return pred.astype(np.float32)


def mids_to_vehicle_points(
    mids: np.ndarray,
    *,
    x_max_m: float,
    meters_per_pixel: float,
    bev_width: int,
) -> np.ndarray:
    """BEV row mids → Nx2 base_link [x forward, y left], near-first."""

    rows = np.flatnonzero(np.isfinite(mids))
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    x_forward = float(x_max_m) - rows.astype(np.float32) * float(meters_per_pixel)
    y_left = (
        (float(bev_width) - 1) * 0.5 - mids[rows].astype(np.float32)
    ) * float(meters_per_pixel)
    pts = np.column_stack((x_forward, y_left)).astype(np.float32)
    return pts[np.argsort(pts[:, 0])]


def centerline_from_blob(
    blob: np.ndarray,
    *,
    x_max_m: float,
    meters_per_pixel: float,
    bev_width: int,
    smooth_win: int = 15,
    mode: str = 'dt_ridge',
) -> tuple[np.ndarray, np.ndarray]:
    """Return (centerline_xy base_link, row_mids_u).

    ``mode``: ``dt_ridge`` (default) | ``row_mid`` | ``poly2``.
    """

    key = str(mode or 'dt_ridge').lower()
    if key == 'row_mid':
        raw = blob_row_mids(blob)
        mids = smooth_mids_1d(raw, win=smooth_win)
    else:
        raw = dt_ridge_mids(blob)
        mids = polyfit_mids(smooth_mids_1d(raw, win=max(11, smooth_win)), deg=2)
    pts = mids_to_vehicle_points(
        mids,
        x_max_m=x_max_m,
        meters_per_pixel=meters_per_pixel,
        bev_width=bev_width,
    )
    return pts, mids
