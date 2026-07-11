#!/usr/bin/env python3
"""Metric IPM matching Phase-0 trapezoid fine-tune coverage.

Camera model (C920e / sim_interface):
  height 0.13 m, pitch 10° down, HFoV 70.42°.

Fine-tune equivalence (320×180):
  crop_top_ratio 0.39  ↔  ground x ≈ 1.50 m  (geometric: v/H ≈ 0.389)
  image bottom         ↔  ground x ≈ 0.22 m
  lateral              ↔  ±Y_HALF_WIDTH_M (default = full image width at x_max)

BEV pixels are isotropic meters (same m/px longitudinally and laterally).
Trapezoid bev_w/h / bottom_half are *not* used here — coverage is set in meters.
Default y_half = ±0.77 m (team lock, 2026-07-12). Set range.full_image_width
to true only if you want the geometric full-frame half-width at x_max (~±1.05 m).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / 'config' / 'lane_vision.yaml'
)

# C920e + D-Racer mount (sim_interface.yaml / hardware-camera.md).
DEFAULT_HFOV_DEG = 70.42
DEFAULT_HEIGHT_M = 0.13
DEFAULT_PITCH_DOWN_DEG = 10.0

# Phase-0 fine-tune reproduction targets.
DEFAULT_X_MIN_M = 0.22
DEFAULT_X_MAX_M = 1.50
# Full image half-width on ground at x_max (HFoV 70.42°, x=1.5 m) ≈ ±1.06 m.
# Narrow ±0.40 was planner margin only — too tight vs trapezoid "use all pixels".
DEFAULT_Y_HALF_WIDTH_M = 0.77
DEFAULT_METERS_PER_PIXEL = 0.004
DEFAULT_CROP_TOP_RATIO = 0.39
DEFAULT_TRACK_WIDTH_M = 0.35
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 180
DEFAULT_FULL_IMAGE_WIDTH = False


@dataclass
class MetricIpmParams:
    hfov_deg: float = DEFAULT_HFOV_DEG
    camera_height_m: float = DEFAULT_HEIGHT_M
    pitch_down_deg: float = DEFAULT_PITCH_DOWN_DEG
    x_min_m: float = DEFAULT_X_MIN_M
    x_max_m: float = DEFAULT_X_MAX_M
    y_half_width_m: float = DEFAULT_Y_HALF_WIDTH_M
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL
    crop_top_ratio: float = DEFAULT_CROP_TOP_RATIO
    track_width_m: float = DEFAULT_TRACK_WIDTH_M

    def clamp(self) -> MetricIpmParams:
        mpp = float(np.clip(self.meters_per_pixel, 0.001, 0.05))
        x_min = float(np.clip(self.x_min_m, 0.05, 1.0))
        x_max = float(np.clip(self.x_max_m, x_min + 0.2, 5.0))
        return MetricIpmParams(
            hfov_deg=float(np.clip(self.hfov_deg, 30.0, 120.0)),
            camera_height_m=float(np.clip(self.camera_height_m, 0.05, 0.5)),
            pitch_down_deg=float(np.clip(self.pitch_down_deg, 0.0, 45.0)),
            x_min_m=x_min,
            x_max_m=x_max,
            y_half_width_m=float(np.clip(self.y_half_width_m, 0.15, 2.0)),
            meters_per_pixel=mpp,
            crop_top_ratio=float(np.clip(self.crop_top_ratio, 0.0, 0.6)),
            track_width_m=DEFAULT_TRACK_WIDTH_M,
        )

    def y_half_for_full_image_width(
        self,
        img_w: int = DEFAULT_IMAGE_WIDTH,
        img_h: int = DEFAULT_IMAGE_HEIGHT,
    ) -> float:
        """Ground |y| at image left/right edge for the far plane (x_max).

        Setting y_half_width_m to this value makes the far BEV row use the
        full cropped-frame width — same intent as the wide trapezoid.
        Near rows still show black corner fans (geometry, not a bug).
        """
        p = self.clamp()
        fx, _fy, cx, _cy, theta = _camera_intrinsics(img_w, img_h, p)
        zc = p.camera_height_m * np.sin(theta) + p.x_max_m * np.cos(theta)
        # Use u = W-1 (not W) so the far-right sample stays inside the frame.
        return float(abs((float(img_w - 1) - cx) / fx) * zc)

    def with_full_image_width(
        self,
        img_w: int = DEFAULT_IMAGE_WIDTH,
        img_h: int = DEFAULT_IMAGE_HEIGHT,
    ) -> MetricIpmParams:
        """Return a copy whose y_half spans the full image at x_max."""
        p = self.clamp()
        return MetricIpmParams(
            hfov_deg=p.hfov_deg,
            camera_height_m=p.camera_height_m,
            pitch_down_deg=p.pitch_down_deg,
            x_min_m=p.x_min_m,
            x_max_m=p.x_max_m,
            y_half_width_m=p.y_half_for_full_image_width(img_w, img_h),
            meters_per_pixel=p.meters_per_pixel,
            crop_top_ratio=p.crop_top_ratio,
            track_width_m=p.track_width_m,
        ).clamp()

    @property
    def bev_width(self) -> int:
        p = self.clamp()
        return int(round((2.0 * p.y_half_width_m) / p.meters_per_pixel)) + 1

    @property
    def bev_height(self) -> int:
        p = self.clamp()
        return int(round((p.x_max_m - p.x_min_m) / p.meters_per_pixel)) + 1

    def guide_half_width_px(self) -> int:
        """Half-width in BEV px spanning track_width_m (lane markings)."""
        p = self.clamp()
        return max(1, int(round((p.track_width_m / 2.0) / p.meters_per_pixel)))


def load_metric_ipm(path: Path | None = None) -> MetricIpmParams:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        return MetricIpmParams()
    with cfg_path.open(encoding='utf-8') as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    block = data.get('metric_ipm') or {}
    cam = block.get('camera') or {}
    rng = block.get('range') or {}
    params = MetricIpmParams(
        hfov_deg=float(cam.get('hfov_deg', DEFAULT_HFOV_DEG)),
        camera_height_m=float(cam.get('height_ground_m', DEFAULT_HEIGHT_M)),
        pitch_down_deg=float(cam.get('pitch_down_deg', DEFAULT_PITCH_DOWN_DEG)),
        x_min_m=float(rng.get('x_min_m', DEFAULT_X_MIN_M)),
        x_max_m=float(rng.get('x_max_m', DEFAULT_X_MAX_M)),
        y_half_width_m=float(rng.get('y_half_width_m', DEFAULT_Y_HALF_WIDTH_M)),
        meters_per_pixel=float(
            block.get('meters_per_pixel', DEFAULT_METERS_PER_PIXEL)
        ),
        crop_top_ratio=float(
            block.get('crop_top_ratio', DEFAULT_CROP_TOP_RATIO)
        ),
        track_width_m=DEFAULT_TRACK_WIDTH_M,
    ).clamp()
    img = data.get('image') or {}
    img_w = int(img.get('width', DEFAULT_IMAGE_WIDTH))
    img_h = int(img.get('height', DEFAULT_IMAGE_HEIGHT))
    if bool(rng.get('full_image_width', DEFAULT_FULL_IMAGE_WIDTH)):
        params = params.with_full_image_width(img_w, img_h)
    return params


def save_metric_ipm(params: MetricIpmParams, path: Path | None = None) -> Path:
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open(encoding='utf-8') as f:
            existing = yaml.safe_load(f) or {}
    p = params.clamp()
    existing['metric_ipm'] = {
        'camera': {
            'hfov_deg': p.hfov_deg,
            'height_ground_m': p.camera_height_m,
            'pitch_down_deg': p.pitch_down_deg,
        },
        'range': {
            'x_min_m': p.x_min_m,
            'x_max_m': p.x_max_m,
            'y_half_width_m': p.y_half_width_m,
            'full_image_width': False,
        },
        'meters_per_pixel': p.meters_per_pixel,
        'crop_top_ratio': p.crop_top_ratio,
        'bev_width': p.bev_width,
        'bev_height': p.bev_height,
        'track_width_m': DEFAULT_TRACK_WIDTH_M,
        'guide_half_width_px': p.guide_half_width_px(),
        'note': (
            'Team lock 2026-07-12: Metric IPM provisional SSOT. '
            'y_half_width_m=0.77 by default; set full_image_width only to expand.'
        ),
    }
    with cfg_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(existing, f, sort_keys=False, allow_unicode=True)
    return cfg_path


def _camera_intrinsics(
    img_w: int,
    img_h: int,
    params: MetricIpmParams,
) -> tuple[float, float, float, float, float]:
    """Return fx, fy, cx, cy_full, theta_rad (cy in full-image coords)."""
    p = params.clamp()
    fx = img_w / (2.0 * np.tan(np.deg2rad(p.hfov_deg) / 2.0))
    fy = fx
    cx = img_w / 2.0
    cy_full = img_h / 2.0
    theta = np.deg2rad(p.pitch_down_deg)
    return fx, fy, cx, cy_full, theta


def ground_to_image_uv(
    x_m: float,
    y_m: float,
    img_w: int,
    img_h: int,
    params: MetricIpmParams,
) -> tuple[float, float]:
    """Project ground (x forward, y left) → full-image pixel (u, v)."""
    p = params.clamp()
    fx, fy, cx, cy_full, theta = _camera_intrinsics(img_w, img_h, p)
    yc = p.camera_height_m * np.cos(theta) - x_m * np.sin(theta)
    zc = p.camera_height_m * np.sin(theta) + x_m * np.cos(theta)
    u = fx * (y_m / zc) + cx
    v = fy * (yc / zc) + cy_full
    return float(u), float(v)


def resolve_crop_top_px(img_w: int, img_h: int, params: MetricIpmParams) -> int:
    """Crop so x_max stays inside the kept image.

    Configured crop_top_ratio≈0.39 matches ~1.5 m on 320×180, but rounding can
    put the x_max row 0.2 px above the crop line. Take the min of configured
    crop and floor(v(x_max)) so the far BEV edge remains valid.
    """
    p = params.clamp()
    configured = int(round(img_h * p.crop_top_ratio))
    _, v_xmax = ground_to_image_uv(p.x_max_m, 0.0, img_w, img_h, p)
    geometric = int(np.floor(v_xmax + 1e-6))
    crop_top_px = min(configured, geometric)
    return int(np.clip(crop_top_px, 0, max(0, img_h - 2)))


def build_ipm_maps(
    img_w: int,
    img_h: int,
    params: MetricIpmParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (map_x, map_y, valid) for cv2.remap on the *cropped* image.

    map_* are in cropped-image coordinates (y=0 at crop line).
    """
    p = params.clamp()
    crop_top_px = resolve_crop_top_px(img_w, img_h, p)
    cropped_h = img_h - crop_top_px

    fx, fy, cx, cy_full, theta = _camera_intrinsics(img_w, img_h, p)
    cy = cy_full - crop_top_px

    y_right = np.linspace(
        -p.y_half_width_m,
        p.y_half_width_m,
        p.bev_width,
        dtype=np.float32,
    )
    # Row 0 = far (x_max), last row = near (x_min) — same as WonTae / trapezoid top=far.
    x_forward = np.linspace(
        p.x_max_m,
        p.x_min_m,
        p.bev_height,
        dtype=np.float32,
    )
    y_grid, x_grid = np.meshgrid(y_right, x_forward)

    # OpenCV camera: Xc right, Yc down, Zc forward.
    xc = y_grid
    yc = p.camera_height_m * np.cos(theta) - x_grid * np.sin(theta)
    zc = p.camera_height_m * np.sin(theta) + x_grid * np.cos(theta)

    map_x = (fx * (xc / zc) + cx).astype(np.float32)
    map_y = (fy * (yc / zc) + cy).astype(np.float32)

    valid = (
        (zc > 0.001)
        & (map_x >= 0.0)
        & (map_x < float(img_w))
        & (map_y >= 0.0)
        & (map_y < float(cropped_h))
    )
    map_x = map_x.copy()
    map_y = map_y.copy()
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y, valid


def warp_metric_ipm(frame: np.ndarray, params: MetricIpmParams) -> np.ndarray:
    """Warp BGR (or single-channel) frame to metric BEV."""
    h, w = frame.shape[:2]
    p = params.clamp()
    crop_top_px = resolve_crop_top_px(w, h, p)
    cropped = frame[crop_top_px:, :]
    map_x, map_y, _ = build_ipm_maps(w, h, p)
    return cv2.remap(
        cropped,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0) if frame.ndim == 3 else 0,
    )


def bev_uv_to_xy(
    u: np.ndarray | float,
    v: np.ndarray | float,
    params: MetricIpmParams,
) -> tuple[np.ndarray, np.ndarray]:
    """BEV pixel (u right, v down) → base_link ground (x forward, y left) meters."""
    p = params.clamp()
    u_arr = np.asarray(u, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)
    x = p.x_max_m - v_arr * p.meters_per_pixel
    y = p.y_half_width_m - u_arr * p.meters_per_pixel
    return x, y


def draw_crop_overlay(frame: np.ndarray, params: MetricIpmParams) -> np.ndarray:
    """Mark crop line (= x_max) and note on the original frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    p = params.clamp()
    top_y = resolve_crop_top_px(w, h, p)
    cv2.line(out, (0, top_y), (w - 1, top_y), (0, 255, 255), 2)
    cv2.putText(
        out,
        f'crop {p.crop_top_ratio:.2f}  ~=  x_max {p.x_max_m:.2f} m',
        (8, max(18, top_y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f'bottom ~= x_min {p.x_min_m:.2f} m',
        (8, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_metric_guides(bev: np.ndarray, params: MetricIpmParams) -> np.ndarray:
    """Meter / lane guides on metric BEV (tuning only)."""
    out = bev.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    p = params.clamp()
    h, w = out.shape[:2]
    cx = w // 2
    guide = p.guide_half_width_px()

    cv2.line(out, (cx, 0), (cx, h - 1), (0, 255, 255), 1)
    cv2.line(out, (cx - guide, 0), (cx - guide, h - 1), (0, 220, 0), 1)
    cv2.line(out, (cx + guide, 0), (cx + guide, h - 1), (0, 220, 0), 1)

    for dist in (0.5, 1.0, 1.5):
        if p.x_min_m <= dist <= p.x_max_m:
            row = int(round((p.x_max_m - dist) / p.meters_per_pixel))
            if 0 <= row < h:
                cv2.line(out, (0, row), (w - 1, row), (180, 180, 0), 1)
                cv2.putText(
                    out,
                    f'{dist:.1f}m',
                    (4, max(12, row - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (180, 180, 0),
                    1,
                    cv2.LINE_AA,
                )

    step = max(1, int(round(0.1 / p.meters_per_pixel)))
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h - 1), (40, 40, 40), 1)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w - 1, y), (40, 40, 40), 1)

    cv2.putText(
        out,
        f'metric {w}x{h}  mpp={p.meters_per_pixel*1000:.1f}mm  '
        f'x=[{p.x_min_m:.2f},{p.x_max_m:.2f}]  |y|<={p.y_half_width_m:.2f}',
        (4, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return out
