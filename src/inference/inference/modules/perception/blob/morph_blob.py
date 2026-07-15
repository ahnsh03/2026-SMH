"""Morphology + connected-component selection of one road-like blob."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BlobSelectStats:
    n_components: int = 0
    chosen_label: int = 0
    chosen_area: int = 0
    score: float = 0.0


def _odd(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def fill_small_holes(mask: np.ndarray, max_hole_px: int = 400) -> np.ndarray:
    """Fill enclosed holes smaller than ``max_hole_px``."""

    binary = (mask > 0).astype(np.uint8) * 255
    if binary.size == 0:
        return binary
    h, w = binary.shape
    flood = binary.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    holes = cv2.bitwise_and(holes, cv2.bitwise_not(binary))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (holes > 0).astype(np.uint8), connectivity=8
    )
    out = binary.copy()
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if 0 < area <= int(max_hole_px):
            out[labels == lab] = 255
    return out


def morph_clean_road(
    road: np.ndarray,
    *,
    open_k: int = 2,
    close_k: int = 3,
    max_hole_px: int = 400,
) -> np.ndarray:
    """Open → close → fill small holes (softened one step vs 3/5/500)."""

    binary = (road > 0).astype(np.uint8) * 255
    if binary.size == 0:
        return binary
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(open_k), _odd(open_k)))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(close_k), _odd(close_k)))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ko, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kc, iterations=1)
    return fill_small_holes(closed, max_hole_px=max_hole_px)


def select_best_blob(
    candidate: np.ndarray,
    *,
    track_width_m: float,
    meters_per_pixel: float,
    lane_bonus: np.ndarray | None = None,
    prefer_near: bool = True,
    prefer_largest_near: bool = True,
) -> tuple[np.ndarray, BlobSelectStats]:
    """Keep one CC: prefer max **near-band pixel count** among near-touching CCs.

    Total area alone lets tall off-track floors that only graze the bottom win.
    Track-width control is applied separately via row clip.
    """

    del track_width_m, meters_per_pixel  # reserved for callers / API stability
    binary = (candidate > 0).astype(np.uint8)
    h, w = binary.shape
    empty = np.zeros((h, w), dtype=np.uint8)
    if binary.size == 0 or not np.any(binary):
        return empty, BlobSelectStats()

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return empty, BlobSelectStats(n_components=0)

    near_band = max(4, int(h * 0.28))
    near_slice = labels[h - near_band :, :]
    near_labels = {
        int(lab)
        for lab in np.unique(near_slice)
        if int(lab) > 0
    }

    best_lab = 0
    best_score = -1.0

    for lab in range(1, n):
        area = float(stats[lab, cv2.CC_STAT_AREA])
        if area < 40.0:
            continue
        touches_near = lab in near_labels
        if prefer_near and prefer_largest_near and near_labels and not touches_near:
            continue
        near_area = float(np.count_nonzero(near_slice == lab)) if touches_near else 0.0
        lane_overlap = 0.0
        if lane_bonus is not None and lane_bonus.shape == labels.shape:
            lane_overlap = float(np.count_nonzero((labels == lab) & (lane_bonus > 0)))
        # Primary: mass in ego near-band; paint overlap as tie-break.
        score = near_area + 0.35 * lane_overlap + 0.05 * area
        if score > best_score:
            best_score = score
            best_lab = lab

    if best_lab <= 0:
        # Fallback: absolute largest
        best_lab = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        best_score = float(stats[best_lab, cv2.CC_STAT_AREA])

    out = np.zeros((h, w), dtype=np.uint8)
    out[labels == best_lab] = 255
    return out, BlobSelectStats(
        n_components=n - 1,
        chosen_label=best_lab,
        chosen_area=int(stats[best_lab, cv2.CC_STAT_AREA]),
        score=float(best_score),
    )


def drop_top_edge_only_blobs(
    mask: np.ndarray,
    *,
    min_area: int = 350,
    top_band_ratio: float = 0.025,
    near_band_ratio: float = 0.18,
) -> np.ndarray:
    """Remove CCs that touch the BEV **top** edge but not the ego bottom band.

    Black asphalt trial #2: drop large top-only (off-track) blobs *before* morph,
    then morph → bottom ego. CCs touching both top and bottom are kept.
    """

    binary = (mask > 0).astype(np.uint8) * 255
    if binary.size == 0 or not np.any(binary):
        return binary
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary

    top_h = max(2, int(round(h * float(top_band_ratio))))
    near_h = max(2, int(round(h * float(near_band_ratio))))
    top_labs = {int(lab) for lab in np.unique(labels[:top_h, :]) if int(lab) > 0}
    bot_labs = {
        int(lab) for lab in np.unique(labels[h - near_h :, :]) if int(lab) > 0
    }

    out = binary.copy()
    for lab in top_labs:
        if lab in bot_labs:
            continue
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == lab] = 0
    return out


def keep_near_floor_blob(
    mask: np.ndarray,
    *,
    near_band_ratio: float = 0.35,
    min_near_area: int = 80,
    centroid_lower_frac: float = 0.55,
) -> np.ndarray:
    """Keep one near-robot CC **before** road morph (black asphalt / cyan wash).

    Score by **pixels inside the near (ego) band**, not total CC area. Large
    off-track floor that only grazes the bottom band otherwise wins on area
    (OUT curve ~1412–1491).
    """

    binary = (mask > 0).astype(np.uint8) * 255
    if binary.size == 0 or not np.any(binary):
        return binary
    h, w = binary.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)

    near_h = max(2, int(round(h * float(near_band_ratio))))
    near_slice = labels[h - near_h :, :]
    near_labs = {int(lab) for lab in np.unique(near_slice) if int(lab) > 0}

    near_ok: list[tuple[int, int]] = []
    for lab in near_labs:
        near_area = int(np.count_nonzero(near_slice == lab))
        if near_area >= int(min_near_area):
            near_ok.append((lab, near_area))

    if near_ok:
        best = max(near_ok, key=lambda t: t[1])[0]
    else:
        v_cut = float(h) * float(centroid_lower_frac)
        lower: list[tuple[int, int]] = []
        for lab in range(1, n):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            cy = float(cents[lab][1])
            if cy >= v_cut and area >= int(min_near_area):
                # Prefer mass in the lower half when no near-band hit.
                lower_area = int(np.count_nonzero(labels[h // 2 :, :] == lab))
                lower.append((lab, lower_area if lower_area > 0 else area))
        if lower:
            best = max(lower, key=lambda t: t[1])[0]
        else:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    out = np.zeros_like(binary)
    if best > 0:
        out[labels == best] = 255
    return out


# Back-compat alias (cyan path / older call sites).
keep_near_cyan_blob = keep_near_floor_blob


def keep_bottom_ego_blob(
    mask: np.ndarray,
    *,
    near_band_ratio: float = 0.18,
) -> np.ndarray:
    """Keep the CC with max pixels in the BEV **bottom** band (after morph).

    Used by extract_five / trial #2 post-morph ego select. Scoring matches
    ``keep_near_floor_blob``: band mass, not total CC area.
    """

    binary = (mask > 0).astype(np.uint8) * 255
    if binary.size == 0 or not np.any(binary):
        return binary
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)

    near_h = max(2, int(round(h * float(near_band_ratio))))
    near_slice = labels[h - near_h :, :]
    near_labs = {int(lab) for lab in np.unique(near_slice) if int(lab) > 0}
    if not near_labs:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(binary)
        out[labels == best] = 255
        return out

    best = 0
    best_near = -1
    for lab in near_labs:
        near_area = int(np.count_nonzero(near_slice == lab))
        if near_area > best_near:
            best_near = near_area
            best = lab
    out = np.zeros_like(binary)
    if best > 0:
        out[labels == best] = 255
    return out
