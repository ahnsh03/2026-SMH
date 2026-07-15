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
    open_k: int = 3,
    close_k: int = 5,
    max_hole_px: int = 500,
) -> np.ndarray:
    """Open → close → fill small holes."""

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
    """Keep one CC: prefer largest that touches the near (ego) band.

    Complex width scoring was discarding long corridors for tiny near flakes.
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
        lane_overlap = 0.0
        if lane_bonus is not None and lane_bonus.shape == labels.shape:
            lane_overlap = float(np.count_nonzero((labels == lab) & (lane_bonus > 0)))
        # Prefer area; slight bonus for lane paint overlap and near contact.
        score = area + 0.35 * lane_overlap + (0.15 * area if touches_near else 0.0)
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
