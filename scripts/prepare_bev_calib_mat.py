#!/usr/bin/env python3
"""Generate a metric grid mat texture for BEV scale calibration in Gazebo.

Physical size and cell spacing are defined in config/bev_calib_mat.yaml.
Output: src/dracer_sim/models/bev_calib_mat/materials/textures/bev_calib_grid.png

  python3 scripts/prepare_bev_calib_mat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

PKG_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'dracer_sim'
CONFIG_PATH = PKG_ROOT / 'config' / 'bev_calib_mat.yaml'
OUT_DIR = PKG_ROOT / 'models' / 'bev_calib_mat' / 'materials' / 'textures'
OUT_PNG = OUT_DIR / 'bev_calib_grid.png'


def _load_cfg() -> dict:
  if not CONFIG_PATH.is_file():
    return {
      'width_m': 4.0,
      'height_m': 2.0,
      'pixels_per_meter': 250,
      'minor_m': 0.1,
      'major_m': 0.5,
      'track_width_m': 0.35,
    }
  with CONFIG_PATH.open(encoding='utf-8') as f:
    data = yaml.safe_load(f) or {}
  return data.get('bev_calib_mat', data)


def generate(cfg: dict) -> np.ndarray:
  width_m = float(cfg['width_m'])
  height_m = float(cfg['height_m'])
  ppm = int(cfg['pixels_per_meter'])
  minor_m = float(cfg['minor_m'])
  major_m = float(cfg['major_m'])
  track_width_m = float(cfg.get('track_width_m', 0.35))

  w_px = max(64, int(round(width_m * ppm)))
  h_px = max(64, int(round(height_m * ppm)))

  # Light floor so lanes/camera contrast well in BEV.
  img = np.full((h_px, w_px, 3), (235, 235, 230), dtype=np.uint8)

  def m_to_x(xm: float) -> int:
    return int(round(xm * ppm))

  def m_to_y(ym: float) -> int:
    # Image y grows downward; put y=0 at bottom (near side) for "forward up" feel in BEV.
    return int(round((height_m - ym) * ppm))

  # Minor grid
  x = 0.0
  while x <= width_m + 1e-9:
    px = m_to_x(x)
    if 0 <= px < w_px:
      cv2.line(img, (px, 0), (px, h_px - 1), (190, 190, 190), 1)
    x += minor_m
  y = 0.0
  while y <= height_m + 1e-9:
    py = m_to_y(y)
    if 0 <= py < h_px:
      cv2.line(img, (0, py), (w_px - 1, py), (190, 190, 190), 1)
    y += minor_m

  # Major grid + labels
  x = 0.0
  while x <= width_m + 1e-9:
    px = m_to_x(x)
    if 0 <= px < w_px:
      cv2.line(img, (px, 0), (px, h_px - 1), (40, 40, 40), 2)
      cv2.putText(
        img,
        f'{x:.1f}m',
        (min(px + 4, w_px - 60), h_px - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
      )
    x += major_m

  y = 0.0
  while y <= height_m + 1e-9:
    py = m_to_y(y)
    if 0 <= py < h_px:
      cv2.line(img, (0, py), (w_px - 1, py), (40, 40, 40), 2)
      cv2.putText(
        img,
        f'{y:.1f}m',
        (6, max(14, py - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
      )
    y += major_m

  # 0.35 m track-width reference bar (lateral) near origin corner.
  bar_y0 = m_to_y(0.15)
  bar_y1 = m_to_y(0.25)
  bar_x0 = m_to_x(0.10)
  bar_x1 = m_to_x(0.10 + track_width_m)
  cv2.rectangle(img, (bar_x0, min(bar_y0, bar_y1)), (bar_x1, max(bar_y0, bar_y1)), (0, 140, 0), -1)
  cv2.putText(
    img,
    f'track {track_width_m:.2f}m',
    (bar_x0, min(bar_y0, bar_y1) - 6),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.4,
    (0, 100, 0),
    1,
    cv2.LINE_AA,
  )

  # Title / legend
  cv2.putText(
    img,
    f'BEV calib mat  {width_m:.1f}x{height_m:.1f}m  '
    f'minor={minor_m:.1f}m  major={major_m:.1f}m',
    (8, 22),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (0, 0, 0),
    1,
    cv2.LINE_AA,
  )
  cv2.putText(
    img,
    'Align car camera; count major cells for longitudinal m/px',
    (8, 42),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.4,
    (60, 60, 60),
    1,
    cv2.LINE_AA,
  )

  # Origin marker at (0,0) corner (bottom-left of image = -X/-Y of plane depends on Gazebo UV;
  # labels still encode absolute meter spacing).
  ox, oy = m_to_x(0.0), m_to_y(0.0)
  cv2.drawMarker(img, (ox, min(oy, h_px - 1)), (0, 0, 220), cv2.MARKER_CROSS, 18, 2)

  return img


def main() -> int:
  cfg = _load_cfg()
  img = generate(cfg)
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  if not cv2.imwrite(str(OUT_PNG), img):
    print(f'[prepare_bev_calib_mat] failed to write {OUT_PNG}', file=sys.stderr)
    return 1
  print(
    f'[prepare_bev_calib_mat] {OUT_PNG.relative_to(PKG_ROOT.parent.parent)} '
    f'({img.shape[1]}x{img.shape[0]} px, '
    f'{cfg["width_m"]}x{cfg["height_m"]} m, '
    f'{cfg["pixels_per_meter"]} px/m)'
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
