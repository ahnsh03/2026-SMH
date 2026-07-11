#!/usr/bin/env python3
"""Shared BEV ROI geometry for vision_tune tools.

Trapezoid convention (docs/lane-drive-strategy.md §4):
  - Drop top crop_top_ratio of the image.
  - Top edge of src = full image width at crop line.
  - Bottom edge half-width may exceed image half-width (virtual corners).
  - All pixels below the crop line participate in the warp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / 'config' / 'lane_vision.yaml'
)


@dataclass
class BevRoiParams:
    crop_top_ratio: float = 0.15
    bottom_half_width_ratio: float = 1.35
    bev_width: int = 320
    bev_height: int = 240

    def clamp(self) -> BevRoiParams:
        return BevRoiParams(
            crop_top_ratio=float(np.clip(self.crop_top_ratio, 0.0, 0.5)),
            bottom_half_width_ratio=float(
                np.clip(self.bottom_half_width_ratio, 0.5, 3.0)
            ),
            bev_width=int(np.clip(self.bev_width, 64, 1280)),
            bev_height=int(np.clip(self.bev_height, 64, 1280)),
        )


def load_bev_roi(path: Path | None = None) -> BevRoiParams:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return BevRoiParams()
    with cfg_path.open(encoding='utf-8') as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    block = data.get('bev_roi') or {}
    return BevRoiParams(
        crop_top_ratio=float(block.get('crop_top_ratio', 0.15)),
        bottom_half_width_ratio=float(
            block.get('bottom_half_width_ratio', 1.35)
        ),
        bev_width=int(block.get('bev_width', 320)),
        bev_height=int(block.get('bev_height', 240)),
    ).clamp()


def save_bev_roi(params: BevRoiParams, path: Path | None = None) -> Path:
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open(encoding='utf-8') as f:
            existing = yaml.safe_load(f) or {}
    existing['bev_roi'] = asdict(params.clamp())
    if 'image' not in existing:
        existing['image'] = {'width': 320, 'height': 180}
    if 'hsv' not in existing:
        existing['hsv'] = {
            'white': None,
            'yellow': None,
            'black_road': None,
            'red_road': None,
        }
    with cfg_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(existing, f, sort_keys=False, allow_unicode=True)
    return cfg_path


def src_dst_points(
    image_w: int,
    image_h: int,
    params: BevRoiParams,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (src 4x2, dst 4x2, top_y_px) for getPerspectiveTransform."""
    p = params.clamp()
    top_y = int(round(image_h * p.crop_top_ratio))
    top_y = int(np.clip(top_y, 0, max(0, image_h - 2)))
    bottom_y = image_h - 1
    half_w = image_w / 2.0
    bottom_half = half_w * p.bottom_half_width_ratio

    src = np.float32(
        [
            [0.0, float(top_y)],
            [float(image_w - 1), float(top_y)],
            [half_w + bottom_half, float(bottom_y)],
            [half_w - bottom_half, float(bottom_y)],
        ]
    )
    dst = np.float32(
        [
            [0.0, 0.0],
            [float(p.bev_width - 1), 0.0],
            [float(p.bev_width - 1), float(p.bev_height - 1)],
            [0.0, float(p.bev_height - 1)],
        ]
    )
    return src, dst, top_y


def warp_bev(frame: np.ndarray, params: BevRoiParams) -> np.ndarray:
    h, w = frame.shape[:2]
    src, dst, _ = src_dst_points(w, h, params)
    matrix = cv2.getPerspectiveTransform(src, dst)
    p = params.clamp()
    return cv2.warpPerspective(
        frame,
        matrix,
        (p.bev_width, p.bev_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def draw_roi_overlay(frame: np.ndarray, params: BevRoiParams) -> np.ndarray:
    """Draw trapezoid; clip drawing to image but keep true src for warp."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    src, _, top_y = src_dst_points(w, h, params)
    # Visible clip of the trapezoid inside the image for the overlay window.
    clipped = src.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h - 1)
    pts = clipped.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    # Mark virtual (outside) bottom corners with arrows on the bottom edge.
    for x, y in src[2:]:
        edge_x = int(np.clip(x, 0, w - 1))
        cv2.circle(overlay, (edge_x, int(y)), 4, (0, 128, 255), -1)
        if x < 0 or x > w - 1:
            cv2.putText(
                overlay,
                'out',
                (edge_x, max(top_y + 15, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 128, 255),
                1,
                cv2.LINE_AA,
            )
    cv2.line(overlay, (0, top_y), (w - 1, top_y), (255, 0, 255), 1)
    cv2.putText(
        overlay,
        f'crop_top={params.crop_top_ratio:.2f}  bottom_r={params.bottom_half_width_ratio:.2f}',
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay
