#!/usr/bin/env python3
"""Shared BEV ROI geometry for vision_tune tools.

Trapezoid convention (docs/lane-drive-strategy.md §4):
  - Drop top crop_top_ratio of the image.
  - Top edge of src = full image width at crop line.
  - Bottom edge half-width may exceed image half-width (virtual corners).
  - All pixels below the crop line participate in the warp.

bev_width / bev_height are destination *resolution* (and aspect), not meters.
Real scale is recovered after lanes look parallel: known road width (m) /
measured width (px) → meters_per_pixel.
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

# Competition CW track: distance between left/right lane markings ≈ 0.35 m (fixed).
DEFAULT_TRACK_WIDTH_M = 0.35


@dataclass
class BevRoiParams:
    crop_top_ratio: float = 0.39
    bottom_half_width_ratio: float = 6.35
    bev_width: int = 500
    bev_height: int = 370
    # Scale-guide half-width in BEV pixels (± from center). Align to lane markings.
    guide_half_width_px: int = 44
    # Fixed physical width between lane markings (m). Not a free tuning knob.
    track_width_m: float = DEFAULT_TRACK_WIDTH_M

    def clamp(self) -> BevRoiParams:
        return BevRoiParams(
            crop_top_ratio=float(np.clip(self.crop_top_ratio, 0.0, 0.5)),
            bottom_half_width_ratio=float(
                np.clip(self.bottom_half_width_ratio, 0.5, 15.0)
            ),
            bev_width=int(np.clip(self.bev_width, 64, 1280)),
            bev_height=int(np.clip(self.bev_height, 64, 1280)),
            guide_half_width_px=int(
                np.clip(self.guide_half_width_px, 5, max(10, self.bev_width // 2 - 1))
            ),
            track_width_m=DEFAULT_TRACK_WIDTH_M,
        )

    def meters_per_pixel_lateral(self) -> float | None:
        """Lateral m/px if ±guide_half spans track_width_m."""
        p = self.clamp()
        full_px = 2 * p.guide_half_width_px
        if full_px <= 0:
            return None
        return p.track_width_m / float(full_px)

    # Back-compat alias used by older call sites.
    def meters_per_pixel(self) -> float | None:
        return self.meters_per_pixel_lateral()



def load_bev_roi(path: Path | None = None) -> BevRoiParams:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return BevRoiParams()
    with cfg_path.open(encoding='utf-8') as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    block = data.get('bev_roi') or {}
    scale = data.get('bev_scale') or {}
    return BevRoiParams(
        crop_top_ratio=float(block.get('crop_top_ratio', 0.39)),
        bottom_half_width_ratio=float(
            block.get('bottom_half_width_ratio', 6.35)
        ),
        bev_width=int(block.get('bev_width', 500)),
        bev_height=int(block.get('bev_height', 370)),
        guide_half_width_px=int(
            scale.get('guide_half_width_px', block.get('guide_half_width_px', 44))
        ),
        track_width_m=DEFAULT_TRACK_WIDTH_M,
    ).clamp()


def save_bev_roi(params: BevRoiParams, path: Path | None = None) -> Path:
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open(encoding='utf-8') as f:
            existing = yaml.safe_load(f) or {}
    p = params.clamp()
    existing['bev_roi'] = {
        'crop_top_ratio': p.crop_top_ratio,
        'bottom_half_width_ratio': p.bottom_half_width_ratio,
        'bev_width': p.bev_width,
        'bev_height': p.bev_height,
    }
    mpp = p.meters_per_pixel_lateral()
    existing['bev_scale'] = {
        'track_width_m': DEFAULT_TRACK_WIDTH_M,
        'guide_half_width_px': p.guide_half_width_px,
        'meters_per_pixel_lateral': None if mpp is None else float(mpp),
        'meters_per_pixel_longitudinal': None,
        'note': (
            'Lateral: align green guides to left/right lane markings on a straight; '
            'm/px_lat = track_width_m / (2 * guide_half_width_px). '
            'Longitudinal is NOT equal to lateral under trapezoid warp — calibrate '
            'separately with known-distance markers, or switch to metric IPM '
            '(height/pitch/HFoV). bev_w/h are pixel resolution only.'
        ),
    }
    # Drop obsolete key if present.
    if 'road_width_m' in (existing.get('bev_scale') or {}):
        existing['bev_scale'].pop('road_width_m', None)
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
    clipped = src.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, w - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, h - 1)
    pts = clipped.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
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


def draw_bev_guides(bev: np.ndarray, params: BevRoiParams) -> np.ndarray:
    """Scale / alignment guides on the BEV image (for tuning, not runtime)."""
    out = bev.copy()
    p = params.clamp()
    h, w = out.shape[:2]
    cx = (w - 1) / 2.0

    # Light grid every 20 px (lateral / longitudinal).
    grid = 20
    for x in range(0, w, grid):
        cv2.line(out, (x, 0), (x, h - 1), (40, 40, 40), 1)
    for y in range(0, h, grid):
        cv2.line(out, (0, y), (w - 1, y), (40, 40, 40), 1)

    # Vehicle center line.
    cv2.line(out, (int(round(cx)), 0), (int(round(cx)), h - 1), (0, 255, 255), 1)

    # Near / mid / far horizontals (image bottom = near).
    for frac, label in ((0.75, 'near'), (0.50, 'mid'), (0.25, 'far')):
        y = int(round((h - 1) * frac))
        cv2.line(out, (0, y), (w - 1, y), (180, 180, 0), 1)
        cv2.putText(
            out,
            label,
            (4, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 0),
            1,
            cv2.LINE_AA,
        )

    # Known track-width guides: align to left/right lane markings (35 cm fixed).
    half = p.guide_half_width_px
    left_x = int(round(cx - half))
    right_x = int(round(cx + half))
    for x in (left_x, right_x):
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 0), 1)
    cv2.putText(
        out,
        f'guide +/-{half}px  track_width={p.track_width_m:.2f}m (fixed)',
        (4, h - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    mpp = p.meters_per_pixel_lateral()
    if mpp is not None:
        cv2.putText(
            out,
            f'lat m/px~{mpp * 1000:.1f}mm  (longitudinal needs markers or metric IPM)',
            (4, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f'{w}x{h}px  (bev_w/h = resolution, not meters)',
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out
