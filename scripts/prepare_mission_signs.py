#!/usr/bin/env python3
"""Bake mission sign textures from bundled assets into Gazebo model folders.

Turn signs: Ø20 cm graphic on Ø21 cm white circle (PNG alpha mask, corners transparent).
ArUco: 15 cm marker on 18 cm white square board.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1] / 'src' / 'dracer_sim'
ASSETS_DIR = PKG_ROOT / 'assets' / 'signs'
MODELS_DIR = PKG_ROOT / 'models'

TURN_CONTENT_M = 0.20
TURN_BOARD_M = 0.21
ARUCO_CONTENT_M = 0.15
ARUCO_BOARD_M = 0.18


def _composite_on_white_rgba(src: np.ndarray, content_fraction: float) -> np.ndarray:
  if src.ndim == 2:
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGRA)
  elif src.shape[2] == 3:
    src = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)

  height, width = src.shape[:2]
  scale = 1.0 / content_fraction
  canvas_w = max(1, int(round(width * scale)))
  canvas_h = max(1, int(round(height * scale)))
  canvas = np.full((canvas_h, canvas_w, 4), (255, 255, 255, 255), dtype=np.uint8)

  x0 = (canvas_w - width) // 2
  y0 = (canvas_h - height) // 2
  roi = canvas[y0 : y0 + height, x0 : x0 + width]
  alpha = src[:, :, 3:4].astype(np.float32) / 255.0
  roi[:] = (src.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(
    np.uint8
  )
  return canvas


def _apply_circle_mask(bgra: np.ndarray) -> np.ndarray:
  """Clip to inscribed circle (Ø21 cm board on square texture)."""
  height, width = bgra.shape[:2]
  cx, cy = width / 2.0, height / 2.0
  radius = min(width, height) / 2.0
  ys, xs = np.ogrid[:height, :width]
  inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius**2
  out = bgra.copy()
  out[~inside, 3] = 0
  return out


def _bake_turn_sign(src: np.ndarray) -> np.ndarray:
  fraction = TURN_CONTENT_M / TURN_BOARD_M
  canvas = _composite_on_white_rgba(src, fraction)
  return _apply_circle_mask(canvas)


def _bake_aruco_sign(src: np.ndarray) -> np.ndarray:
  fraction = ARUCO_CONTENT_M / ARUCO_BOARD_M
  return cv2.cvtColor(_composite_on_white_rgba(src, fraction), cv2.COLOR_BGRA2BGR)


def _write_baked(model_name: str, texture_name: str, image: np.ndarray) -> Path:
  out_dir = MODELS_DIR / model_name / 'materials' / 'textures'
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / texture_name
  if not cv2.imwrite(str(out_path), image):
    raise RuntimeError(f'failed to write {out_path}')
  return out_path


def main() -> int:
  turn_jobs = [
    ('turn_sign_left', 'trun_left.png', 'turn_sign_left.png'),
    ('turn_sign_right', 'trun_right.png', 'turn_sign_right.png'),
  ]

  for model_name, src_name, out_name in turn_jobs:
    src_path = ASSETS_DIR / src_name
    if not src_path.exists():
      print(f'[prepare_mission_signs] missing {src_path}', file=sys.stderr)
      return 1
    src = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if src is None:
      print(f'[prepare_mission_signs] cannot read {src_path}', file=sys.stderr)
      return 1
    out = _write_baked(model_name, out_name, _bake_turn_sign(src))
    print(f'[prepare_mission_signs] {src_name} -> {out.relative_to(PKG_ROOT.parent.parent)} (circular Ø21 cm)')

  aruco_src = ASSETS_DIR / 'ArUco_stop.png'
  if not aruco_src.exists():
    print(f'[prepare_mission_signs] missing {aruco_src}', file=sys.stderr)
    return 1
  aruco = cv2.imread(str(aruco_src), cv2.IMREAD_UNCHANGED)
  if aruco is None:
    print(f'[prepare_mission_signs] cannot read {aruco_src}', file=sys.stderr)
    return 1
  out = _write_baked('aruco_stop_sign', 'aruco_stop_sign.png', _bake_aruco_sign(aruco))
  print(f'[prepare_mission_signs] ArUco_stop.png -> {out.relative_to(PKG_ROOT.parent.parent)}')

  print(
    '[prepare_mission_signs] done — turn Ø20 cm on circular white Ø21 cm, '
    f'ArUco {ARUCO_CONTENT_M * 100:.0f} cm on {ARUCO_BOARD_M * 100:.0f} cm square'
  )
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
