"""Metric Inverse-Perspective-Mapping (IPM) → 등거리(metric) BEV.

카메라 프레임(BGR, 320x180)을 지면 평면으로 역투영해, 픽셀이 실제 미터에
1:1 대응하는 Bird's-Eye-View 를 만든다. planner/제어가 곧바로 미터 좌표로
쓸 수 있게 하는 것이 목적이다.

파라미터 SSOT
-------------
config/lane_vision.yaml 의 ``metric_ipm`` 블록. 값은 팀 참조본
(/home/topst/2026-SMH/config/lane_vision.yaml, 2026-07-12 lock)과 동일하며,
scripts/vision_tune/metric_ipm.py 의 카메라 모델을 그대로 옮겨왔다.

좌표 규약 (중요, 안전 직결)
--------------------------
BEV 이미지: row 0 = 원거리(x_max), 마지막 row = 근거리(x_min).
            column 0 = 월드 좌측,  마지막 column = 월드 우측 (미러 아님).
지면 base_link 출력: x = 전방(+),  **y = 우측(+)**.

  ⚠ 이 코드베이스의 centerline y 부호는 "우측 +" 이다 (일반 ROS y-left 와 반대).
  후속 LaneController(Pure Pursuit)가 δ = +atan(L·2y/d²) 로 (부호반전 없이)
  동작하도록 맞춘 규약이다. 백파일 검증: 사람이 우회전 → centerline_y>0.
  이 규약이 깨지면 실차가 반대로 조향한다. bev_uv_to_xy / xy_to_bev_uv 가 SSOT.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover - yaml 은 런타임에 항상 존재
    yaml = None


# --- C920e + D-Racer 마운트 기본값 (config 없을 때 폴백) -------------------
DEFAULT_HFOV_DEG = 70.42
DEFAULT_HEIGHT_M = 0.13
DEFAULT_PITCH_DOWN_DEG = 10.0
DEFAULT_X_MIN_M = 0.22
DEFAULT_X_MAX_M = 1.30
DEFAULT_Y_HALF_WIDTH_M = 0.77
DEFAULT_METERS_PER_PIXEL = 0.004
DEFAULT_CROP_TOP_RATIO = 0.39
DEFAULT_TRACK_WIDTH_M = 0.35
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 180


@dataclass
class MetricBevParams:
    """metric IPM 카메라·범위 파라미터 (config/lane_vision.yaml: metric_ipm)."""

    hfov_deg: float = DEFAULT_HFOV_DEG
    camera_height_m: float = DEFAULT_HEIGHT_M
    pitch_down_deg: float = DEFAULT_PITCH_DOWN_DEG
    x_min_m: float = DEFAULT_X_MIN_M
    x_max_m: float = DEFAULT_X_MAX_M
    y_half_width_m: float = DEFAULT_Y_HALF_WIDTH_M
    meters_per_pixel: float = DEFAULT_METERS_PER_PIXEL
    crop_top_ratio: float = DEFAULT_CROP_TOP_RATIO
    track_width_m: float = DEFAULT_TRACK_WIDTH_M

    def clamp(self) -> "MetricBevParams":
        mpp = float(np.clip(self.meters_per_pixel, 0.001, 0.05))
        x_min = float(np.clip(self.x_min_m, 0.05, 1.0))
        x_max = float(np.clip(self.x_max_m, x_min + 0.2, 5.0))
        return MetricBevParams(
            hfov_deg=float(np.clip(self.hfov_deg, 30.0, 120.0)),
            camera_height_m=float(np.clip(self.camera_height_m, 0.05, 0.5)),
            pitch_down_deg=float(np.clip(self.pitch_down_deg, 0.0, 45.0)),
            x_min_m=x_min,
            x_max_m=x_max,
            y_half_width_m=float(np.clip(self.y_half_width_m, 0.15, 2.0)),
            meters_per_pixel=mpp,
            crop_top_ratio=float(np.clip(self.crop_top_ratio, 0.0, 0.6)),
            track_width_m=float(np.clip(self.track_width_m, 0.2, 0.6)),
        )

    @property
    def bev_width(self) -> int:
        p = self.clamp()
        return int(round((2.0 * p.y_half_width_m) / p.meters_per_pixel)) + 1

    @property
    def bev_height(self) -> int:
        p = self.clamp()
        return int(round((p.x_max_m - p.x_min_m) / p.meters_per_pixel)) + 1

    def guide_half_width_px(self) -> int:
        """track_width_m/2 에 해당하는 BEV 가로 픽셀 수 (단측선 오프셋용)."""
        p = self.clamp()
        return max(1, int(round((p.track_width_m / 2.0) / p.meters_per_pixel)))


def load_bev_params(path: Optional[Path | str] = None) -> MetricBevParams:
    """config/lane_vision.yaml 의 metric_ipm 블록에서 파라미터 로드.

    path 가 없거나 파일이 없으면 기본값(팀 lock 값)으로 폴백한다.
    """
    if path is None or yaml is None:
        return MetricBevParams().clamp()
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return MetricBevParams().clamp()
    with cfg_path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    block = data.get("metric_ipm") or {}
    cam = block.get("camera") or {}
    rng = block.get("range") or {}
    return MetricBevParams(
        hfov_deg=float(cam.get("hfov_deg", DEFAULT_HFOV_DEG)),
        camera_height_m=float(cam.get("height_ground_m", DEFAULT_HEIGHT_M)),
        pitch_down_deg=float(cam.get("pitch_down_deg", DEFAULT_PITCH_DOWN_DEG)),
        x_min_m=float(rng.get("x_min_m", DEFAULT_X_MIN_M)),
        x_max_m=float(rng.get("x_max_m", DEFAULT_X_MAX_M)),
        y_half_width_m=float(rng.get("y_half_width_m", DEFAULT_Y_HALF_WIDTH_M)),
        meters_per_pixel=float(block.get("meters_per_pixel", DEFAULT_METERS_PER_PIXEL)),
        crop_top_ratio=float(block.get("crop_top_ratio", DEFAULT_CROP_TOP_RATIO)),
        track_width_m=float(block.get("track_width_m", DEFAULT_TRACK_WIDTH_M)),
    ).clamp()


def _camera_intrinsics(
    img_w: int, img_h: int, p: MetricBevParams
) -> Tuple[float, float, float, float, float]:
    """fx, fy, cx, cy_full(전체 이미지 기준), theta(rad) 반환."""
    fx = img_w / (2.0 * np.tan(np.deg2rad(p.hfov_deg) / 2.0))
    fy = fx
    cx = img_w / 2.0
    cy_full = img_h / 2.0
    theta = np.deg2rad(p.pitch_down_deg)
    return fx, fy, cx, cy_full, theta


def _ground_to_image_v(x_m: float, img_w: int, img_h: int, p: MetricBevParams) -> float:
    """지면 x(전방) → 전체 이미지의 세로 픽셀 v (crop 계산용, y=0 중심선)."""
    fx, fy, cx, cy_full, theta = _camera_intrinsics(img_w, img_h, p)
    yc = p.camera_height_m * np.cos(theta) - x_m * np.sin(theta)
    zc = p.camera_height_m * np.sin(theta) + x_m * np.cos(theta)
    return float(fy * (yc / zc) + cy_full)


def resolve_crop_top_px(img_w: int, img_h: int, p: MetricBevParams) -> int:
    """x_max 지면점이 잘려나가지 않게 crop 상단 픽셀을 정한다."""
    configured = int(round(img_h * p.crop_top_ratio))
    v_xmax = _ground_to_image_v(p.x_max_m, img_w, img_h, p)
    geometric = int(np.floor(v_xmax + 1e-6))
    crop_top_px = min(configured, geometric)
    return int(np.clip(crop_top_px, 0, max(0, img_h - 2)))


class MetricBev:
    """카메라 프레임 → 등거리 BEV 변환기 (remap 맵 캐시).

    사용::

        bev = MetricBev(load_bev_params(cfg_path))
        top = bev.warp(frame)             # (H, W[, 3]) metric BEV
        xs, ys = bev.bev_uv_to_xy(u, v)   # BEV px → base_link (x전방, y우측)
    """

    def __init__(self, params: MetricBevParams | None = None):
        self.params = (params or MetricBevParams()).clamp()
        self._maps_key: tuple[int, int] | None = None
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self._crop_top_px = 0

    # -- 형상 프로퍼티 -------------------------------------------------------
    @property
    def bev_width(self) -> int:
        return self.params.bev_width

    @property
    def bev_height(self) -> int:
        return self.params.bev_height

    @property
    def u_center(self) -> float:
        """y=0(월드 중앙)에 해당하는 BEV column."""
        return (self.bev_width - 1) / 2.0

    # -- remap 맵 (입력 해상도별 캐시) --------------------------------------
    def _ensure_maps(self, img_w: int, img_h: int) -> None:
        key = (img_w, img_h)
        if self._maps_key == key:
            return
        p = self.params
        crop_top_px = resolve_crop_top_px(img_w, img_h, p)
        cropped_h = img_h - crop_top_px

        fx, fy, cx, cy_full, theta = _camera_intrinsics(img_w, img_h, p)
        cy = cy_full - crop_top_px

        # column 0 = 월드 좌측(-y_half), 마지막 = 월드 우측(+y_half). 미러 아님.
        y_left_to_right = np.linspace(
            -p.y_half_width_m, p.y_half_width_m, self.bev_width, dtype=np.float32
        )
        # row 0 = 원거리(x_max), 마지막 = 근거리(x_min).
        x_forward = np.linspace(
            p.x_max_m, p.x_min_m, self.bev_height, dtype=np.float32
        )
        y_grid, x_grid = np.meshgrid(y_left_to_right, x_forward)

        # OpenCV 카메라계: Xc 우, Yc 하, Zc 전방.
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
        map_x[~valid] = -1.0
        map_y[~valid] = -1.0

        self._map_x, self._map_y = map_x, map_y
        self._crop_top_px = crop_top_px
        self._maps_key = key

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """BGR(또는 단일채널) 프레임 → metric BEV."""
        h, w = frame.shape[:2]
        self._ensure_maps(w, h)
        cropped = frame[self._crop_top_px:, :]
        border = (0, 0, 0) if frame.ndim == 3 else 0
        return cv2.remap(
            cropped,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    # -- 좌표 변환 (y = 우측 +) ---------------------------------------------
    def bev_uv_to_xy(self, u, v):
        """BEV px (u 우측, v 하) → base_link (x 전방+, y 우측+) 미터."""
        p = self.params
        u_arr = np.asarray(u, dtype=np.float32)
        v_arr = np.asarray(v, dtype=np.float32)
        x = p.x_max_m - v_arr * p.meters_per_pixel
        y = (u_arr - self.u_center) * p.meters_per_pixel  # 우측 +
        return x, y

    def xy_to_bev_uv(self, x, y):
        """base_link (x 전방+, y 우측+) → BEV px (u, v)."""
        p = self.params
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        u = y_arr / p.meters_per_pixel + self.u_center
        v = (p.x_max_m - x_arr) / p.meters_per_pixel
        return u, v
