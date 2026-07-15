"""색상별 좌우 도로 경계와 전체 주행 가능 영역을 검출한다.

이 모듈은 주행 모드 선택, 중심선 계획, 조향 및 장애물 판단을 하지 않는다.
출력 좌표계는 ``base_link`` 관례인 x 전방, y 왼쪽이며 단위는 m이다.
"""

from __future__ import annotations

import bisect
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


# =========================================================
# scripts/vision_tune Metric IPM (팀 SSOT, config/lane_vision.yaml)
# =========================================================
def _locate_vision_tune() -> Path:
    """상위 디렉터리를 거슬러 scripts/vision_tune/metric_ipm.py를 찾는다."""

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts" / "vision_tune" / "metric_ipm.py"
        if candidate.is_file():
            return candidate.parent
    raise ImportError(
        "scripts/vision_tune/metric_ipm.py를 찾을 수 없습니다. "
        "저장소 루트에 config/lane_vision.yaml과 함께 있어야 합니다."
    )


_VISION_TUNE_DIR = _locate_vision_tune()
if str(_VISION_TUNE_DIR) not in sys.path:
    sys.path.insert(0, str(_VISION_TUNE_DIR))

from metric_ipm import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MetricIpmParams,
    build_ipm_maps,
    load_metric_ipm,
    resolve_crop_top_px,
    warp_metric_ipm,
)


# =========================================================
# Runtime visualization
# =========================================================
VISUALIZE_OFF = "off"
VISUALIZE_CONTROL = "control"
VISUALIZE_ON = "on"
VISUALIZE_TUNE = "tune"

# ┌──────────────────────────────────────────────────────────┐
# │  기본은 off. 시뮬 갈림 실험은 lane_preview_node 한 창만. │
# │                                                          │
# │    "off"      창 없음                                    │
# │    "control"  주행용 1창 (Lane drive)                     │
# │    "on"       Lane drive + HSV masks 2창                 │
# │    "tune"     흰 차선 생성 프리뷰를 웹 토픽으로 발행      │
# │               (cv2 창 대신 CompressedImage → 모니터 패널) │
# │  환경변수: LANE_VISUALIZE=off|control|on|tune            │
# └──────────────────────────────────────────────────────────┘
VISUALIZE_MODE = "off"

# 보드/SSH/headless에서는 창을 띄우면 죽는다. 코드를 안 고치고 끄려면
# 환경변수로 덮어쓴다(있을 때만 우선).
#   LANE_VISUALIZE=off ros2 run inference inference_node
# 헤드리스 보드에서 시각화가 필요하면 imshow가 아니라 tune(웹 발행)을 쓴다.
#   LANE_VISUALIZE=tune ros2 run inference lane_preview_node -p use_compressed:=true
_VISUALIZE_ALIASES = {
    "off": VISUALIZE_OFF,
    "0": VISUALIZE_OFF,
    "false": VISUALIZE_OFF,
    "none": VISUALIZE_OFF,
    "control": VISUALIZE_CONTROL,
    "ctrl": VISUALIZE_CONTROL,
    "drive": VISUALIZE_CONTROL,
    "on": VISUALIZE_ON,
    "1": VISUALIZE_ON,
    "true": VISUALIZE_ON,
    "all": VISUALIZE_ON,
    "debug": VISUALIZE_ON,
    "tune": VISUALIZE_TUNE,
    "web": VISUALIZE_TUNE,
}


def resolve_visualize_mode(raw: str | None) -> str:
    """시각화 모드 문자열을 정규화한다. 모르는 값은 안전하게 OFF."""

    return _VISUALIZE_ALIASES.get((raw or "").strip().lower(), VISUALIZE_OFF)


VISUALIZE_MODE = resolve_visualize_mode(
    os.environ.get("LANE_VISUALIZE") or VISUALIZE_MODE
)
# 로컬 cv2 창(imshow) 경로를 켜는 플래그. tune은 웹 발행이므로 여기서 제외해
# 헤드리스 보드에서 inference_node가 imshow로 죽지 않게 한다.
VISUALIZE = VISUALIZE_MODE in (VISUALIZE_CONTROL, VISUALIZE_ON)


# CONTROL: 주행용 통합 1창. ON: 같은 주행 창 + HSV 마스크 1창만.
CONTROL_WINDOWS = ("Lane drive",)
ON_EXTRA_WINDOWS = ("HSV masks",)


def window_enabled(name: str) -> bool:
    """현재 모드에서 이 창을 띄울지 결정한다."""

    if VISUALIZE_MODE == VISUALIZE_CONTROL:
        return name in CONTROL_WINDOWS
    if VISUALIZE_MODE == VISUALIZE_ON:
        return name in CONTROL_WINDOWS or name in ON_EXTRA_WINDOWS
    return False


VISUALIZATION_SCALE = 2.0

# OpenCV BGR 색상: 왼쪽 경계 빨강, 오른쪽 경계 파랑
LEFT_BOUNDARY_COLOR = (0, 0, 255)
RIGHT_BOUNDARY_COLOR = (255, 0, 0)
DRIVABLE_COLOR = (0, 150, 0)
INTERPOLATED_LINE_COLOR = (0, 255, 255)
CENTERLINE_COLOR = (255, 0, 255)


@dataclass(frozen=True)
class LaneBoundary:
    """차량 기준 경계점: Nx2 [x 전방 m, y 왼쪽 m]."""

    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    detected: bool = False
    confidence: float = 0.0


@dataclass(frozen=True)
class LaneMarking:
    """LaneMarking.msg와 1:1로 대응하는 인지 모듈 출력."""

    COLOR_UNKNOWN = 0
    COLOR_WHITE = 1
    COLOR_YELLOW = 2

    SIDE_UNKNOWN = 0
    SIDE_LEFT = 1
    SIDE_RIGHT = 2
    SIDE_CENTER = 3

    id: int = 0
    color: int = COLOR_UNKNOWN
    side_hint: int = SIDE_UNKNOWN
    confidence: float = 0.0
    length: float = 0.0
    heading: float = 0.0
    curvature: float = 0.0
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )


@dataclass(frozen=True)
class LaneDetections:
    """LaneDetections.msg에 바로 복사할 수 있는 프레임 단위 결과."""

    # ROS Header의 stamp는 publisher가 ROS clock으로 채운다.
    header: object | None = None
    lanes: tuple[LaneMarking, ...] = ()
    white_visible: bool = False
    yellow_visible: bool = False
    left_visible: bool = False
    right_visible: bool = False
    white_confidence: float = 0.0
    yellow_confidence: float = 0.0
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    # LaneDetections.msg에는 없으므로 별도 주행가능영역 토픽용으로 유지한다.
    drivable_area: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    # 흰/노란 차선 센터라인(좌우 경계 중점, base_link Nx2 [x 전방, y 왼쪽]).
    white_centerline: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    yellow_centerline: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float32)
    )
    # 노란 가로 실선(정지선/원형교차로 진입선 등) 등장 여부.
    yellow_crossing_line: bool = False
    # Out 갈림 / In 탈출 분기: 갈림 활성 여부와 갈래 목록.
    # RoadBranch.lateral_rank: 0=왼쪽 갈래, 1=오른쪽. points=base_link 센터라인.
    # 갈림이 아니면 branches는 단일 경로 1개. 용어 SSOT: docs/lane-occlusion-fork-strategy.md §0
    fork_active: bool = False
    branches: tuple["RoadBranch", ...] = ()
    # Active-lane policy after fork lock (see modules/active_lane.py).
    active_branch_rank: int | None = None
    lane_policy: str = "explore"
    # 기존 pipeline이 즉시 AttributeError를 내지 않도록 남긴 읽기 전용 호환값.
    # 이 모듈은 더 이상 조향이나 주행 신뢰도를 계산하지 않는다.
    steering_offset: float = 0.0
    confidence: float = 0.0


@dataclass
class LaneDebugFrame:
    """Intermediate masks/boundaries for mode tuners (not on the ROS wire)."""

    bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0, 3), dtype=np.uint8)
    )
    white_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    yellow_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    red_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    black_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    road_clean: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    road_raw: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    yellow_dash_points_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    yellow_connected_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_dash_points_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_dash_connected_bev: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    crossing_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_crossing_mask: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    white_left: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    white_right: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    yellow_left: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    yellow_right: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )
    road_cells: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.uint8)
    )
    road_branches: tuple = ()
    ego_road_color: str | None = None
    fork_active: bool = False
    yellow_crossing_line: bool = False
    white_crossing_line: bool = False
    red_coverage: float = 0.0
    red_pixel_count: int = 0
    # Out 갈림 / In 탈출: 차로 쌍(outer+inner) → 갈래 검증용.
    # fork_split_source: road_split_marks | yellow_alt_marks | yellow_marks |
    #   white_marks | white_alt_marks | cells  (§0 용어표)
    fork_lane_pairs: tuple = ()
    fork_mark_tracks: tuple = ()
    fork_split_source: str = ""
    # Active-lane policy (see modules/active_lane.py): explore | locked | ego_only
    # Course contract: Out→prefer_yellow=False (white only), In→True (yellow first).
    prefer_yellow: bool | None = None
    active_branch_rank: int | None = None
    lane_policy: str = "explore"


# =========================================================
# Metric IPM geometry (config/lane_vision.yaml → metric_ipm)
# =========================================================
METRIC_IPM_PARAMS: MetricIpmParams = load_metric_ipm()

BEV_WIDTH = METRIC_IPM_PARAMS.bev_width
BEV_HEIGHT = METRIC_IPM_PARAMS.bev_height
METERS_PER_PIXEL = float(METRIC_IPM_PARAMS.meters_per_pixel)
X_MAX_M = float(METRIC_IPM_PARAMS.x_max_m)
X_MIN_M = float(METRIC_IPM_PARAMS.x_min_m)

# remap 캐시 (입력 해상도별). map_*는 crop된 프레임 좌표.
_ipm_map_x: np.ndarray | None = None
_ipm_map_y: np.ndarray | None = None
_ipm_map_shape: tuple[int, int] | None = None
_bev_observable: np.ndarray | None = None


def _ensure_ipm_maps(img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """입력 해상도에 맞는 Metric IPM remap 맵을 준비한다."""

    global _ipm_map_x, _ipm_map_y, _ipm_map_shape, _bev_observable
    shape = (img_w, img_h)
    if (
        _ipm_map_x is None
        or _ipm_map_y is None
        or _ipm_map_shape != shape
    ):
        _ipm_map_x, _ipm_map_y, valid = build_ipm_maps(
            img_w, img_h, METRIC_IPM_PARAMS
        )
        _bev_observable = valid.astype(bool)
        _ipm_map_shape = shape
    return _ipm_map_x, _ipm_map_y


def bev_observable_mask() -> np.ndarray | None:
    """BEV에서 카메라가 실제로 '본' 픽셀(True)만 참인 마스크.

    BEV 아래·양옆의 검은 쐐기는 지면이 카메라 화각 밖이라 비어 있는 것이지
    '도로가 없다'는 뜻이 아니다. 이 둘을 구분하지 않으면, 시야 밖으로 뻗는
    차로 가설이 전부 '도로 겹침 부족'으로 탈락한다.
    """

    return _bev_observable


def _load_hsv_thresholds() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """lane_vision.yaml hsv 블록을 OpenCV inRange 하한/상한으로 읽는다."""

    defaults = {
        # Fallbacks only — YAML hsv block is SSOT (bag-tuned 2026-07-15).
        "white": ((0, 0, 210), (179, 20, 255)),
        "yellow": ((15, 50, 160), (50, 150, 255)),
        "black_road": ((17, 0, 50), (70, 255, 140)),
        "red_road": ((0, 155, 120), (9, 255, 255)),
    }
    try:
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        data = {}
    hsv_block = data.get("hsv") or {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, (lo_d, hi_d) in defaults.items():
        block = hsv_block.get(key)
        if isinstance(block, dict):
            lo = (
                int(block.get("h_min", lo_d[0])),
                int(block.get("s_min", lo_d[1])),
                int(block.get("v_min", lo_d[2])),
            )
            hi = (
                int(block.get("h_max", hi_d[0])),
                int(block.get("s_max", hi_d[1])),
                int(block.get("v_max", hi_d[2])),
            )
        else:
            lo, hi = lo_d, hi_d
        out[key] = (
            np.array(lo, dtype=np.uint8),
            np.array(hi, dtype=np.uint8),
        )
    return out


_HSV = _load_hsv_thresholds()

# =========================================================
# HSV thresholds
# =========================================================
WHITE_LOWER, WHITE_UPPER = _HSV["white"]
YELLOW_LOWER, YELLOW_UPPER = _HSV["yellow"]
BLACK_LOWER, BLACK_UPPER = _HSV["black_road"]
RED_ROAD_LOWER, RED_ROAD_UPPER = _HSV["red_road"]

# Hue wrap for red (OpenCV H wraps at 0/179). 0 = disabled.
# When >0, OR inRange(H∈[0, wrap], S/V from red_road) with the high band.
# 실차 빨간 매트는 H 3~9 (저역)이라 wrap 이 꺼져 있으면 0.1% 밖에 못 잡는다.
# lane_vision.yaml 의 detect_tune.red_h_low_wrap 이 이 값을 덮어쓴다.
RED_H_LOW_WRAP = 15


def apply_hsv_thresholds(
    ranges: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Live-tune HSV inRange bounds (tuner). Keys: white/yellow/black_road/red_road."""

    global WHITE_LOWER, WHITE_UPPER
    global YELLOW_LOWER, YELLOW_UPPER
    global BLACK_LOWER, BLACK_UPPER
    global RED_ROAD_LOWER, RED_ROAD_UPPER
    if "white" in ranges:
        WHITE_LOWER, WHITE_UPPER = ranges["white"]
    if "yellow" in ranges:
        YELLOW_LOWER, YELLOW_UPPER = ranges["yellow"]
    if "black_road" in ranges:
        BLACK_LOWER, BLACK_UPPER = ranges["black_road"]
    if "red_road" in ranges:
        RED_ROAD_LOWER, RED_ROAD_UPPER = ranges["red_road"]


def apply_detect_tune(
    *,
    crossing_coverage_ratio: float | None = None,
    crossing_min_rows: int | None = None,
    min_branch_separation_m: float | None = None,
    dash_max_lateral_error_m: float | None = None,
    dash_max_forward_gap_m: float | None = None,
    dash_max_heading_diff_deg: float | None = None,
    dash_min_component_area_px: int | None = None,
    dash_branch_assoc_m: float | None = None,
    red_h_low_wrap: int | None = None,
    fork_track_assoc_m: float | None = None,
    fork_track_min_rows: int | None = None,
    fork_pair_width_m: float | None = None,
    fork_far_zone_ratio: float | None = None,
    fork_track_max_row_gap: int | None = None,
    fork_near_zone_ratio: float | None = None,
) -> None:
    """Live-tune detection scalars exposed by tune_lane_detect."""

    global CROSSING_COVERAGE_RATIO, CROSSING_MIN_ROWS
    global MIN_BRANCH_SEPARATION_M, MIN_BRANCH_SEPARATION_ROWS
    global DASH_MAX_LATERAL_ERROR_M, DASH_MAX_FORWARD_GAP_M
    global DASH_MAX_HEADING_DIFF_DEG, DASH_MIN_COMPONENT_AREA_PX
    global DASH_BRANCH_ASSOC_M, RED_H_LOW_WRAP
    global FORK_TRACK_ASSOC_M, FORK_TRACK_MIN_ROWS, FORK_PAIR_WIDTH_M
    global FORK_FAR_ZONE_RATIO, FORK_TRACK_MAX_ROW_GAP, FORK_NEAR_ZONE_RATIO
    if crossing_coverage_ratio is not None:
        CROSSING_COVERAGE_RATIO = float(np.clip(crossing_coverage_ratio, 0.05, 1.0))
    if crossing_min_rows is not None:
        CROSSING_MIN_ROWS = max(1, int(crossing_min_rows))
    if min_branch_separation_m is not None:
        MIN_BRANCH_SEPARATION_M = float(max(0.02, min_branch_separation_m))
        MIN_BRANCH_SEPARATION_ROWS = max(
            1, int(round(MIN_BRANCH_SEPARATION_M / METERS_PER_PIXEL))
        )
    if dash_max_lateral_error_m is not None:
        DASH_MAX_LATERAL_ERROR_M = float(max(0.005, dash_max_lateral_error_m))
    if dash_max_forward_gap_m is not None:
        DASH_MAX_FORWARD_GAP_M = float(max(0.05, dash_max_forward_gap_m))
    if dash_max_heading_diff_deg is not None:
        DASH_MAX_HEADING_DIFF_DEG = float(
            np.clip(dash_max_heading_diff_deg, 5.0, 90.0)
        )
    if dash_min_component_area_px is not None:
        DASH_MIN_COMPONENT_AREA_PX = max(3, int(dash_min_component_area_px))
    if dash_branch_assoc_m is not None:
        DASH_BRANCH_ASSOC_M = float(max(0.05, dash_branch_assoc_m))
    if red_h_low_wrap is not None:
        RED_H_LOW_WRAP = int(np.clip(red_h_low_wrap, 0, 30))
    if fork_track_assoc_m is not None:
        FORK_TRACK_ASSOC_M = float(max(0.02, fork_track_assoc_m))
    if fork_track_min_rows is not None:
        FORK_TRACK_MIN_ROWS = max(5, int(fork_track_min_rows))
    if fork_pair_width_m is not None:
        FORK_PAIR_WIDTH_M = float(max(0.15, fork_pair_width_m))
    if fork_far_zone_ratio is not None:
        FORK_FAR_ZONE_RATIO = float(np.clip(fork_far_zone_ratio, 0.15, 0.75))
    if fork_track_max_row_gap is not None:
        FORK_TRACK_MAX_ROW_GAP = max(2, int(fork_track_max_row_gap))
    if fork_near_zone_ratio is not None:
        FORK_NEAR_ZONE_RATIO = float(np.clip(fork_near_zone_ratio, 0.10, 0.55))


def get_detect_tune() -> dict[str, float | int]:
    """Snapshot of scalars the lane-detect tuner may edit."""

    return {
        "crossing_coverage_ratio": float(CROSSING_COVERAGE_RATIO),
        "crossing_min_rows": int(CROSSING_MIN_ROWS),
        "min_branch_separation_m": float(MIN_BRANCH_SEPARATION_M),
        "dash_max_lateral_error_m": float(DASH_MAX_LATERAL_ERROR_M),
        "dash_max_forward_gap_m": float(DASH_MAX_FORWARD_GAP_M),
        "dash_max_heading_diff_deg": float(DASH_MAX_HEADING_DIFF_DEG),
        "dash_min_component_area_px": int(DASH_MIN_COMPONENT_AREA_PX),
        "dash_branch_assoc_m": float(DASH_BRANCH_ASSOC_M),
        "red_h_low_wrap": int(RED_H_LOW_WRAP),
        "fork_track_assoc_m": float(FORK_TRACK_ASSOC_M),
        "fork_track_min_rows": int(FORK_TRACK_MIN_ROWS),
        "fork_pair_width_m": float(FORK_PAIR_WIDTH_M),
        "fork_far_zone_ratio": float(FORK_FAR_ZONE_RATIO),
        "fork_track_max_row_gap": int(FORK_TRACK_MAX_ROW_GAP),
        "fork_near_zone_ratio": float(FORK_NEAR_ZONE_RATIO),
    }


def _red_inrange(hsv_source: np.ndarray) -> np.ndarray:
    """Red road mask; optional low-H wrap band ORed in."""

    mask = cv2.inRange(hsv_source, RED_ROAD_LOWER, RED_ROAD_UPPER)
    if RED_H_LOW_WRAP > 0:
        lo = np.array(
            [0, int(RED_ROAD_LOWER[1]), int(RED_ROAD_LOWER[2])],
            dtype=np.uint8,
        )
        hi = np.array(
            [RED_H_LOW_WRAP, int(RED_ROAD_UPPER[1]), int(RED_ROAD_UPPER[2])],
            dtype=np.uint8,
        )
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv_source, lo, hi))
    return mask


# =========================================================
# Track width (YAML metric_ipm.track_width_m, default 0.35 m)
# =========================================================
ROAD_WIDTH_M = float(METRIC_IPM_PARAMS.track_width_m)

ROAD_WIDTH_PX = int(
    round(
        ROAD_WIDTH_M
        / METERS_PER_PIXEL
    )
)

ROAD_WIDTH_TOLERANCE_M = 0.08

ROAD_WIDTH_TOLERANCE_PX = int(
    round(
        ROAD_WIDTH_TOLERANCE_M
        / METERS_PER_PIXEL
    )
)

# =========================================================
# Black road hole filling
# =========================================================
# 색상선 픽셀로 확인된 구멍만 채우도록, 선 마스크를 이만큼만 부풀려 쓴다.
ROAD_LINE_SUPPORT_DILATION_M = 0.01
# 도로를 가로지르는 실선(원형교차로 진입선 등)을 방향에 관계없이 잇기 위한
# 등방 커널. 실선 두께(~0.1 m)보다 크고, 분기 섬 간격(>0.3 m)보다 작아야
# 실선만 메우고 갈래 사이는 그대로 둔다.
ROAD_MARKING_BRIDGE_M = 0.16


# =========================================================
# Boundary tracking and interpolation
# =========================================================

# 점선 사이에서 같은 경계를 추적할 최대 전후 간격
MAX_BOUNDARY_TRACK_GAP_M = 0.35

MAX_BOUNDARY_TRACK_GAP_PX = int(
    round(
        MAX_BOUNDARY_TRACK_GAP_M
        / METERS_PER_PIXEL
    )
)

# 점선 사이에서 허용할 최대 좌우 이동량.
# In 탈출·Out 갈림에서 갈래가 벌어질 때 행당 ~8 px(≈3 cm) 개방이 필요해
# 0.12 m / 0.45 px·row 이면 DP가 x≈1.0 m에서 끊김 → BEV X_MAX(1.5 m)까지 못 감.
MAX_BOUNDARY_SHIFT_M = 0.20

MAX_BOUNDARY_SHIFT_PX = (
    MAX_BOUNDARY_SHIFT_M
    / METERS_PER_PIXEL
)

# 한 행씩 이어질 때 기본 허용 이동량
BOUNDARY_BASE_SHIFT_M = 0.025

BOUNDARY_BASE_SHIFT_PX = (
    BOUNDARY_BASE_SHIFT_M
    / METERS_PER_PIXEL
)

BOUNDARY_SHIFT_PER_ROW_PX = 2.5

# DP tip 너머 far 마킹으로 코스를 이어 붙일 때 허용 횡오차 (Metric IPM X_MAX까지).
FAR_COURSE_ASSOC_M = 0.28
FAR_COURSE_MAX_MISS_ROWS = int(
    round(0.20 / METERS_PER_PIXEL)
)  # 연속 미스 ≈0.20 m 이면 far 연장 중단
# Tip-only skate detection: outer pinned at the absolute FOV edge (not soft).
# Large soft margins cut good mid-curve fits on in_roundabout_exit.
SIDE_WALL_HARD_MARGIN_PX = 1.5

# 보간할 최대 점선 간격
MAX_BOUNDARY_INTERPOLATION_GAP_M = 0.32

MAX_BOUNDARY_INTERPOLATION_GAP_PX = int(
    round(
        MAX_BOUNDARY_INTERPOLATION_GAP_M
        / METERS_PER_PIXEL
    )
)

# 너무 짧고 고립된 경계 덩어리를 제거하기 위한 최소 길이
MIN_COURSE_RUN_LENGTH_M = 0.08

MIN_COURSE_RUN_ROWS = int(
    round(
        MIN_COURSE_RUN_LENGTH_M
        / METERS_PER_PIXEL
    )
)

# 노란색 인코스 후보는 흰색 경로보다 오른쪽이어야 함
INNER_REFERENCE_MARGIN_M = 0.01

INNER_REFERENCE_MARGIN_PX = (
    INNER_REFERENCE_MARGIN_M
    / METERS_PER_PIXEL
)

# 교차로에서 행별 후보를 즉시 확정하지 않고
# 여러 행에 걸친 하나의 경로로 선택할 때 사용한다.
PATH_GAP_PENALTY = 0.12
PATH_CENTER_SHIFT_PENALTY = 4.0
PATH_BOUNDARY_SHIFT_PENALTY = 2.5
PATH_SLOPE_CHANGE_PENALTY = 7.0
PATH_REFERENCE_PENALTY = 0.35
# 두 선을 본 뒤 한 선만 남았을 때 기존 left/right ID를 도로 마스크보다
# 우선 유지한다. 두 가정의 중심은 약 350 mm만큼 달라지므로 강한 패널티가 필요하다.
PATH_TEMPORAL_PENALTY = 8.0
PATH_BOUNDARY_ID_PENALTY = 20.0
TEMPORAL_ID_ROW_RADIUS = 10
TEMPORAL_ID_MATCH_M = 0.08
TEMPORAL_ID_MATCH_PX = TEMPORAL_ID_MATCH_M / METERS_PER_PIXEL
# required_side가 지정된 노란 인코스 선택은 이전 프레임 temporal lock보다
# 우선해야 하므로 반대편(아웃코스) 후보에 강한 패널티를 준다.
PATH_WRONG_SIDE_PENALTY = 30.0
PATH_SOURCE_SWITCH_PENALTY = 4.0
# 도로 HSV 점수 대신 모든 관측 후보에 양의 기본 점수를 줘 긴 경로가 누적
# 점수에서 유리하게 한다. 실제 두 선을 모두 본 PAIR는 단일선보다 우선한다.
PATH_CANDIDATE_BASE_SCORE = 8.0
PATH_PAIR_BONUS = 4.0
PAIR_MAX_HEADING_DIFF_DEG = 20.0
MAX_PATH_CANDIDATES_PER_ROW = 10
MAX_PATH_PREVIOUS_ROWS = 3

# 노란 연결선이 하나뿐이면 끝점 방향을 좌/우 후보의 보조 점수로 쓴다.
# BEV 화면에서 '/'는 음의 각도이며 LEFT 후보, '\\'는 양의 각도이며
# RIGHT 후보에 보너스를 준다. 수직(0도)에 가까우면 빠르게 0으로 수렴한다.
SINGLE_LINE_SIDE_BIAS_MAX_SCORE = 100.0
SINGLE_LINE_SIDE_BIAS_FULL_ANGLE_DEG = 1.2
SINGLE_LINE_SIDE_BIAS_POWER = 2.0
SINGLE_LINE_ENDPOINT_BAND_RATIO = 0.10
SINGLE_LINE_MIN_ROW_SPAN_M = 0.08
SINGLE_LINE_MIN_ROW_SPAN_PX = max(
    2, int(round(SINGLE_LINE_MIN_ROW_SPAN_M / METERS_PER_PIXEL))
)
SINGLE_LINE_CENTER_DEADBAND_M = 0.04
SINGLE_LINE_CENTER_DEADBAND_PX = (
    SINGLE_LINE_CENTER_DEADBAND_M / METERS_PER_PIXEL
)
# 단일선 좌우 판정: 차로는 도로(주행가능영역)가 있는 쪽에 있다. 선의 한쪽 도로
# 픽셀이 반대쪽의 이 배수 이상이고 최소 픽셀을 넘으면 그 방향으로 확정한다.
# 이진 영역(drivable)의 '좌/우 한 표'로만 쓰고 경계 기하에는 관여시키지 않는다.
SINGLE_LINE_ROAD_SIDE_RATIO = 1.5
SINGLE_LINE_ROAD_SIDE_MIN_PX = 8
# 단일선 각도가 지지하는 후보 안에 반대색 선이 있어도 즉시 탈락시키지 않고
# 이 점수만 감점한다. 합류 구간에서는 실제 노란 LEFT 경계 오른쪽에 흰선이
# 함께 보일 수 있다. 각도 지지가 없는 후보에는 기존 hard reject를 유지한다.
SINGLE_LINE_OPPOSITE_LINE_PENALTY = 0.0

# 화면 가장자리에서는 추정한 반대 경계가 BEV 밖에 있어도 관측선을
# 버리지 않는다. 단, 도로 방향을 확인할 수 있는 최소 폭은 필요하다.
MIN_VISIBLE_CANDIDATE_WIDTH_M = 0.05
MIN_VISIBLE_CANDIDATE_WIDTH_PX = int(
    round(MIN_VISIBLE_CANDIDATE_WIDTH_M / METERS_PER_PIXEL)
)
PARTIAL_CANDIDATE_PENALTY = 1.5

# 한쪽 선만 보일 때는 '이 선이 왼쪽 경계' / '이 선이 오른쪽 경계' 두 가설을 다
# 만들고 점수로 고른다. 선의 양쪽 모두에 도로가 있으면(교차로에서 흰 도로와
# 노란 도로가 맞닿는 지점) 도로 겹침 점수가 양쪽 다 만점이라, center_error가
# 승부를 가르며 '도로는 나와 저 선 사이에 있다'는 쪽으로 항상 기울어버린다.
#
# 트랙의 도로는 같은 색 선으로만 둘러싸인다(흰 도로=흰선, 노란 도로=노란선).
# 그러니 지어낸 차로 '안쪽'에 다른 색 차선이 들어앉았다면 그건 이 색 도로가
# 아니다. 그 가설에 페널티를 줘서 위 편향을 이긴다.
# 관측한 선 자신은 제외하고 차로 안쪽만 본다(선 두께·워프 번짐 여유).
OPPOSITE_LINE_MARGIN_M = 0.03
OPPOSITE_LINE_MARGIN_PX = max(
    1, int(round(OPPOSITE_LINE_MARGIN_M / METERS_PER_PIXEL))
)
OPPOSITE_LINE_MIN_PX = 2

# 도로 폭 350 mm는 진행방향에 수직인 폭이다. BEV의 같은 행에서 좌우로 재면
# 도로가 기운 만큼 넓게 잘리므로, 관측선의 국소 기울기로 폭을 보정한다.
SLOPE_ROW_DELTA = max(1, int(round(0.05 / METERS_PER_PIXEL)))  # 위/아래 50 mm
SLOPE_MATCH_TOLERANCE_PX = 2.0 * SLOPE_ROW_DELTA               # |기울기| <= 2
# 상한이 너무 크면(2.5) 선이 BEV에서 거의 수평일 때 지어낸 차로가 0.87 m까지
# 부풀어 도로 밖으로 나가고, 그 후보가 통째로 탈락해 검출이 끊긴다. 1.6이면
# 45도에서 -6 mm, 55도에서 -31 mm로 우회전 차선 폭 보정은 살리면서 폭발은 막는다.
MAX_WIDTH_SCALE = 1.6                                          # 약 58도에서 포화

# 점수 정규화 분모. 후보마다 max()를 다시 부르면 프레임당 수십만 번이 된다.
ROAD_WIDTH_NORM = float(max(1, ROAD_WIDTH_PX))

BOUNDARY_SOURCE_PAIR = 0
BOUNDARY_SOURCE_LEFT = 1
BOUNDARY_SOURCE_RIGHT = 2

# FOLLOW_YELLOW 중 검출 공백을 흰색 코스로 대체하지 않고
# 노란 경계의 위치/기울기로 복원할 최대 거리다.
YELLOW_SPATIAL_GAP_M = 0.20
YELLOW_SPATIAL_GAP_ROWS = int(
    round(YELLOW_SPATIAL_GAP_M / METERS_PER_PIXEL)
)


PLANNING_OUTLIER_SIGMA = 1.5
# 점선 블록 가장자리의 좌우 흔들림을 곡선 피팅에 포함하지 않도록
# 기존 8 px보다 엄격하게 제거한다.
PLANNING_MIN_OUTLIER_THRESHOLD_PX = 3.0
PLANNING_FIT_ITERATIONS = 3

# 차량 바로 앞까지 최종 곡선으로 재생성한다. 기존 0.20 m 제한 때문에
# 화면 아래쪽만 원시 점선 보간이 남아 구불거렸다.
BOUNDARY_SMOOTH_X_MIN_M = X_MIN_M
BOUNDARY_SMOOTH_X_MAX_M = 1.40
BOUNDARY_SMOOTH_MIN_VALID_ROWS = 12
BOUNDARY_SMOOTH_CENTER_DEGREE = 2
BOUNDARY_SMOOTH_WIDTH_DEGREE = 1


# =========================================================
# Inner-course transition
# =========================================================

YELLOW_MIN_VALID_LENGTH_M = 0.06

YELLOW_MIN_VALID_ROWS = int(
    round(
        YELLOW_MIN_VALID_LENGTH_M
        / METERS_PER_PIXEL
    )
)

# =========================================================
# Runtime variables
# =========================================================
cached_shape: tuple[int, int] | None = None

last_yellow_left: np.ndarray | None = None
last_yellow_right: np.ndarray | None = None
last_white_left: np.ndarray | None = None
last_white_right: np.ndarray | None = None

# 노란선 플래그는 잡음 한 프레임에 켜지거나 검출 누락 한 프레임에
# 꺼지지 않도록 짧은 히스테리시스를 둔다.
YELLOW_FLAG_ON_FRAMES = 3
YELLOW_FLAG_OFF_FRAMES = 8
yellow_flag_on_count = 0
yellow_flag_off_count = 0
yellow_flag = False


def make_odd(value: int) -> int:
    """커널 크기를 1 이상의 홀수로 만든다."""

    value = max(
        1,
        int(value),
    )

    if value % 2 == 0:
        value += 1

    return value


# 원근 워프는 화면 위쪽(먼 곳)을 BEV에서 크게 확대하고 아래쪽(가까운 곳)은
# 오히려 압축한다. 측정하면 소스의 8x8 점 하나가 화면 40% 높이에서는 BEV
# 366px, 95% 높이에서는 6px이 된다 — 먼 픽셀이 가까운 픽셀보다 BEV에서 약
# 57배 넓다. 그래서 먼 곳의 작은 하양 오검출 몇 점이 BEV에서는 근거리 차선
# 보다 큰 덩어리가 되고, 경계 추적이 그쪽으로 통째로 끌려간다.
#
# 걸러내려면 반드시 '워프 전'이어야 한다. 워프 후에는 크기 관계가 뒤집혀
# 노이즈와 진짜 차선을 크기로 구분할 수 없다.
FAR_REGION_ROW_RATIO = 0.55
FAR_SPECK_MAX_AREA_PX = 120


def remove_far_specks(mask: np.ndarray) -> np.ndarray:
    """화면 위쪽(먼 곳)에만 있는 작은 덩어리를 워프 전에 지운다.

    근거리 차선은 성분이 크고, 근거리 점선은 먼 영역 밖이라 둘 다 살아남는다.
    성분이 먼 영역과 가까운 영역에 걸쳐 있으면(멀리까지 이어지는 실제 차선)
    지우지 않는다.
    """

    far_limit = int(round(mask.shape[0] * FAR_REGION_ROW_RATIO))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = mask.copy()
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        bottom = int(
            stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT]
        )
        if area <= FAR_SPECK_MAX_AREA_PX and bottom <= far_limit:
            cleaned[labels == label] = 0
    return cleaned


def warp_mask(mask: np.ndarray) -> np.ndarray:
    """원본 프레임 마스크를 Metric IPM BEV로 워프해 이진 마스크로 되돌린다."""

    h, w = mask.shape[:2]
    crop_top_px = resolve_crop_top_px(w, h, METRIC_IPM_PARAMS)
    cropped = mask[crop_top_px:, :]
    map_x, map_y = _ensure_ipm_maps(w, h)
    warped = cv2.remap(
        cropped,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY)
    return binary


def enclosed_road_holes(road_bev: np.ndarray) -> np.ndarray:
    """도로로 완전히 둘러싸인 내부 구멍만 255로 반환한다.

    점선 자국이나 표면 노이즈처럼 도로에 둘러싸인 구멍은 실제로 주행
    가능한 노면이므로 채울 대상이다. 이미지 경계(도로 바깥)와 연결된
    배경은 채우지 않으므로 도로 외곽 경계와 도로 밖 영역은 그대로 남는다.
    """

    # 경계를 배경(0)으로 한 칸 덧대어 flood 시작점이 항상 배경이 되게 한다.
    padded = cv2.copyMakeBorder(
        road_bev,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    flood = padded.copy()
    flood_mask = np.zeros(
        (padded.shape[0] + 2, padded.shape[1] + 2),
        dtype=np.uint8,
    )
    # 도로 바깥과 연결된 배경만 255로 칠한다. 둘러싸인 구멍은 0으로 남는다.
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    enclosed = cv2.bitwise_not(flood)
    return enclosed[1:-1, 1:-1]


def fill_road_surface_holes(
    road_bev: np.ndarray,
    line_bev: np.ndarray,
) -> np.ndarray:
    """
    도로 envelope 내부 구멍 중 흰색/노란색 선과 겹치는 부분만 메운다.

    색상선을 단순 OR하지 않으므로 도로 밖의 점선과 경계선은
    새로운 도로 영역으로 추가되지 않는다.
    """

    if road_bev.shape != line_bev.shape:
        raise ValueError("road_bev and line_bev must have the same shape")

    # 등방 커널로 닫아 방향에 관계없이 실선 두께만큼의 끊김을 잇는다.
    # (기존 가로 커널은 도로를 가로지르는 가로 실선을 잇지 못했다.)
    bridge_size_px = make_odd(
        int(
            round(
                ROAD_MARKING_BRIDGE_M
                / METERS_PER_PIXEL
            )
        )
    )

    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            bridge_size_px,
            bridge_size_px,
        ),
    )

    road_envelope = cv2.morphologyEx(
        road_bev,
        cv2.MORPH_CLOSE,
        bridge_kernel,
    )

    road_holes = cv2.bitwise_and(
        road_envelope,
        cv2.bitwise_not(road_bev),
    )

    support_size_px = make_odd(
        int(
            round(
                ROAD_LINE_SUPPORT_DILATION_M
                / METERS_PER_PIXEL
            )
        )
    )
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            support_size_px,
            support_size_px,
        ),
    )
    line_support = cv2.dilate(
        line_bev,
        support_kernel,
        iterations=1,
    )
    validated_line_holes = cv2.bitwise_and(
        road_holes,
        line_support,
    )

    # 도로로 완전히 둘러싸인 내부 구멍(점선 자국, 노면 노이즈)은 무조건 메운다.
    # 도로 밖과 연결된 영역은 포함되지 않으므로 외곽 경계는 유지된다.
    interior_holes = enclosed_road_holes(road_bev)

    return cv2.bitwise_or(
        road_bev,
        cv2.bitwise_or(
            validated_line_holes,
            interior_holes,
        ),
    )


def find_line_segments(
    row: np.ndarray,
) -> list[tuple[int, int]]:
    """한 행에 있는 연속된 색상 픽셀을 개별 선 구간으로 분리한다."""

    columns = np.flatnonzero(row)
    if columns.size == 0:
        return []

    # np.split은 구간마다 배열을 새로 할당한다. 시작/끝 인덱스만 뽑으면 된다.
    breaks = np.flatnonzero(np.diff(columns) > 1)
    starts = columns[np.concatenate(([0], breaks + 1))]
    ends = columns[np.concatenate((breaks, [columns.size - 1]))]
    return list(zip(starts.tolist(), ends.tolist()))


def find_line_segments_by_row(
    mask: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """이진 마스크 모든 행의 연속 구간을 전체 배열 연산으로 한 번에 구한다."""

    binary = mask != 0
    padded = np.pad(binary, ((0, 0), (1, 1)), constant_values=False)
    transitions = np.diff(padded.astype(np.int8), axis=1)
    start_rows, starts = np.nonzero(transitions == 1)
    end_rows, ends_after = np.nonzero(transitions == -1)

    segments_by_row: list[list[tuple[int, int]]] = [
        [] for _ in range(mask.shape[0])
    ]
    for start_row, start, end_row, end_after in zip(
        start_rows,
        starts,
        end_rows,
        ends_after,
    ):
        if start_row != end_row:
            continue
        segments_by_row[int(start_row)].append(
            (int(start), int(end_after) - 1)
        )
    return segments_by_row


def boundary_candidate_is_continuous(
    left_u: float,
    right_u: float,
    previous_left: float | None,
    previous_right: float | None,
    current_v: int,
    previous_v: int | None,
) -> bool:
    """후보 좌우 경계가 이전 경계에서 갑자기 점프하는지 검사한다."""

    if (
        previous_left is None
        or previous_right is None
        or previous_v is None
    ):
        return True

    row_gap = abs(
        current_v - previous_v
    )

    if (
        row_gap
        > MAX_BOUNDARY_TRACK_GAP_PX
    ):
        return False

    allowed_shift = min(
        MAX_BOUNDARY_SHIFT_PX,
        (
            BOUNDARY_BASE_SHIFT_PX
            + BOUNDARY_SHIFT_PER_ROW_PX
            * row_gap
        ),
    )

    return (
        abs(
            left_u
            - previous_left
        )
        <= allowed_shift
        and abs(
            right_u
            - previous_right
        )
        <= allowed_shift
    )


def candidate_matches_reference_side(
    candidate_center: float,
    reference_centerline: np.ndarray | None,
    row_v: int,
    required_side: str | None,
) -> bool:
    """노란 인코스 후보가 흰색 경로의 지정 방향에 있는지 검사한다."""

    if (
        required_side is None
        or reference_centerline is None
    ):
        return True

    reference_center = (
        reference_centerline[row_v]
    )

    if np.isnan(
        reference_center
    ):
        return True

    if required_side == "right":
        return (
            candidate_center
            >= float(reference_center)
            + INNER_REFERENCE_MARGIN_PX
        )

    if required_side == "left":
        return (
            candidate_center
            <= float(reference_center)
            - INNER_REFERENCE_MARGIN_PX
        )

    return True


def horizontal_width_scale(slope: float) -> float:
    """국소 기울기(du/dv)에서 '수평 폭 / 수직 폭' 배율을 구한다.

    도로 폭 350 mm는 진행방향에 '수직'인 폭이다. 그런데 후보는 BEV의 같은
    행에서 좌우로 재므로, 도로가 기울면 수평으로 자른 폭이 더 넓어진다.

        수평폭 = 수직폭 / cos(theta) = 수직폭 * sqrt(1 + slope^2)

    이 보정을 빼면 우회전 차선처럼 비스듬한 도로에서 차로가 cos(theta)배로
    좁아지고(30도에서 -49 mm), 각이 커지면 폭 허용오차를 벗어나 아예 검출이
    끊긴다. 기울기 추정이 튀는 것을 막으려 배율에 상한을 둔다.
    """

    return float(min(MAX_WIDTH_SCALE, math.sqrt(1.0 + slope * slope)))


def estimate_segment_slopes(
    segments_by_row: list[list[tuple[int, int]]],
    height: int,
) -> list[list[float]]:
    """행마다 각 세그먼트의 국소 기울기 du/dv를 추정한다.

    위/아래 SLOPE_ROW_DELTA행에서 중심이 가장 가까운 세그먼트를 찾아 잇는다.
    한쪽만 찾으면 그 한쪽으로, 둘 다 못 찾으면 0(수직)으로 둔다.
    """

    def nearest_center(row: int, center: float) -> float | None:
        if row < 0 or row >= height:
            return None
        best: float | None = None
        best_distance = SLOPE_MATCH_TOLERANCE_PX
        for segment in segments_by_row[row]:
            candidate = segment_center(segment)
            distance = abs(candidate - center)
            if distance <= best_distance:
                best_distance = distance
                best = candidate
        return best

    slopes: list[list[float]] = []
    for row in range(height):
        row_slopes: list[float] = []
        for segment in segments_by_row[row]:
            center = segment_center(segment)
            above = nearest_center(row - SLOPE_ROW_DELTA, center)
            below = nearest_center(row + SLOPE_ROW_DELTA, center)
            if above is not None and below is not None:
                slope = (below - above) / (2.0 * SLOPE_ROW_DELTA)
            elif above is not None:
                slope = (center - above) / SLOPE_ROW_DELTA
            elif below is not None:
                slope = (below - center) / SLOPE_ROW_DELTA
            else:
                slope = 0.0
            row_slopes.append(float(slope))
        slopes.append(row_slopes)
    return slopes


def scores_tied(score: float, best: float) -> bool:
    """두 경로 점수가 사실상 같은지 본다(np.isclose 기본 허용오차와 동일).

    스칼라 두 개에 np.isclose를 쓰면 numpy 디스패치 비용이 붙어, 프레임당 수천
    번 호출되는 이 자리에서만 수십 ms가 샌다. 같은 판정을 순수 파이썬으로 한다.
    """

    if math.isinf(score) or math.isinf(best) or math.isnan(score) or math.isnan(best):
        return score == best
    return abs(score - best) <= 1e-8 + 1e-5 * abs(best)


def single_line_component_angles_deg(
    boundary_mask: np.ndarray,
    side_debug: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """각 연결조각의 가까운 끝→먼 끝 수직 기준 각도를 반환한다.

    음수는 BEV 화면의 '/', 양수는 '\\', 0은 수직이다. 끝 한 픽셀의 잡음에
    흔들리지 않도록 위·아래 10% 구간의 열 좌표 중앙값으로 두 끝을 잡는다.
    연결이 완벽하지 않아 조각이 2~3개여도 각 조각의 방향 점수를 살린다.
    """

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        boundary_mask,
        connectivity=8,
    )
    valid_labels = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= DASH_MIN_COMPONENT_AREA_PX
    ]
    if side_debug is not None:
        side_debug["components"] = len(valid_labels)
    component_angles: dict[int, float] = {}
    component_spans: dict[int, int] = {}
    for label in valid_labels:
        rows, columns = np.nonzero(labels == label)
        if rows.size == 0:
            continue
        far_v = int(np.min(rows))
        near_v = int(np.max(rows))
        row_span = near_v - far_v
        if row_span < SINGLE_LINE_MIN_ROW_SPAN_PX:
            continue

        band_rows = max(
            2,
            int(round(row_span * SINGLE_LINE_ENDPOINT_BAND_RATIO)),
        )
        far_columns = columns[rows <= far_v + band_rows]
        near_columns = columns[rows >= near_v - band_rows]
        if far_columns.size == 0 or near_columns.size == 0:
            continue

        far_u = float(np.median(far_columns))
        near_u = float(np.median(near_columns))
        delta_u = near_u - far_u
        component_angles[label] = float(
            np.degrees(np.arctan2(delta_u, float(row_span)))
        )
        component_spans[label] = row_span

    if side_debug is not None:
        angles = tuple(component_angles.values())
        side_debug["component_angles"] = angles
        side_debug["angled_components"] = len(angles)
        side_debug["angle_deg"] = (
            float(np.median(np.asarray(angles, dtype=np.float64)))
            if angles
            else None
        )
        side_debug["row_span"] = max(component_spans.values(), default=0)
    return labels, component_angles


def single_line_side_bonus(
    endpoint_angle_deg: float | None,
    source: int,
) -> float:
    """한 줄의 기울기가 가리키는 LEFT/RIGHT 후보에 부드러운 보너스를 준다."""

    if endpoint_angle_deg is None:
        return 0.0
    favors_left = endpoint_angle_deg < 0.0 and source == BOUNDARY_SOURCE_LEFT
    favors_right = endpoint_angle_deg > 0.0 and source == BOUNDARY_SOURCE_RIGHT
    if not (favors_left or favors_right):
        return 0.0
    normalized = min(
        1.0,
        abs(endpoint_angle_deg) / SINGLE_LINE_SIDE_BIAS_FULL_ANGLE_DEG,
    )
    return SINGLE_LINE_SIDE_BIAS_MAX_SCORE * (
        normalized ** SINGLE_LINE_SIDE_BIAS_POWER
    )


def _row_cumsum(mask: np.ndarray) -> np.ndarray:
    """행마다 왼쪽부터의 누적합. [v, u] = 0..u-1 픽셀 수 (앞에 0 한 칸)."""

    counts = np.cumsum(mask.astype(np.int32), axis=1, dtype=np.int32)
    zero = np.zeros((counts.shape[0], 1), dtype=np.int32)
    return np.concatenate([zero, counts], axis=1)


def opposite_line_inside_lane(
    opposite_cum_row: np.ndarray,
    left_u: float,
    right_u: float,
    source: int,
) -> bool:
    """한쪽 선만 보고 지어낸 차로 안에 '다른 색' 차선이 들어앉았는지 본다.

    양 끝을 모두 제외하고 '엄밀히 안쪽'만 본다. 끝을 안 빼면 두 경우가 뒤섞인다.

    - 관측선 쪽 끝: 관측선 자신이 잡힌다 → OPPOSITE_LINE_MARGIN_PX만큼 뺀다.
    - 지어낸 쪽 끝: 여기 다른 색 선이 있다는 건 '내 차로 안에 있다'가 아니라
      '내 차로의 반대편 경계가 그 색이다'라는 뜻이다. 350 mm 가정의 오차가
      ROAD_WIDTH_TOLERANCE_PX만큼 있으므로 그만큼 뺀다.

    행 누적합을 받아 O(1)로 센다.
    """

    if source == BOUNDARY_SOURCE_LEFT:
        # 관측선 = 왼쪽 경계 → 오른쪽으로 도로를 지어냈다.
        start = left_u + OPPOSITE_LINE_MARGIN_PX
        end = right_u - ROAD_WIDTH_TOLERANCE_PX
    else:
        # 관측선 = 오른쪽 경계 → 왼쪽으로 도로를 지어냈다.
        start = left_u + ROAD_WIDTH_TOLERANCE_PX
        end = right_u - OPPOSITE_LINE_MARGIN_PX

    width = int(opposite_cum_row.shape[0]) - 1
    low = min(max(0, int(round(start))), width)
    high = min(max(0, int(round(end)) + 1), width)
    if high - low <= 0:
        return False
    return (
        int(opposite_cum_row[high] - opposite_cum_row[low])
        >= OPPOSITE_LINE_MIN_PX
    )


def score_boundary_candidate(
    left_u: float,
    right_u: float,
    width_error: float,
    reference_center: float,
    previous_left: float | None,
    previous_right: float | None,
) -> float:
    """차로 폭, 기준 중심과 이전 경계 연속성만으로 후보를 채점한다."""

    center = (
        left_u + right_u
    ) / 2.0

    center_error = abs(center - reference_center) / ROAD_WIDTH_NORM

    width_error_normalized = width_error / ROAD_WIDTH_NORM

    continuity_error = 0.0

    if (
        previous_left is not None
        and previous_right is not None
    ):
        continuity_error = (
            abs(left_u - previous_left) + abs(right_u - previous_right)
        ) / (2.0 * ROAD_WIDTH_NORM)

    return (
        PATH_CANDIDATE_BASE_SCORE
        - width_error_normalized * 3.0
        - center_error * 2.0
        - continuity_error * 5.0
    )


def enumerate_boundary_candidates(
    segments: list[tuple[int, int]],
    row_v: int,
    reference_centerline: np.ndarray | None,
    temporal_centerline: np.ndarray | None,
    temporal_left: np.ndarray | None,
    temporal_right: np.ndarray | None,
    required_side: str | None,
    opposite_cum_row: np.ndarray | None = None,
    segment_slopes: list[float] | None = None,
    is_ego_course: bool = False,
    opposite_segments: list[tuple[int, int]] | None = None,
    single_line_angle_deg: float | None = None,
    side_debug: dict[str, object] | None = None,
    drivable_cum_row: np.ndarray | None = None,
    single_line_side_hint: int | None = None,
) -> list[tuple[float, float, float, int]]:
    """한 행의 가능한 모든 (왼쪽, 오른쪽, 지역 점수) 후보를 반환한다.

    교차로에서는 한 행의 최적 후보가 잘못된 가지일 수 있으므로
    여기서 하나를 확정하지 않고 전체 경로 추적에 넘긴다.

    검정·빨강 도로 HSV는 사용하지 않는다. 두 선 PAIR를 먼저 만들고,
    해당 행에 유효 PAIR가 없을 때만 단일선 후보를 만든다.
    """

    if (
        reference_centerline is not None
        and not np.isnan(reference_centerline[row_v])
    ):
        reference_center = float(reference_centerline[row_v])
    else:
        reference_center = BEV_WIDTH / 2.0

    prefer_inner_course = (
        required_side is not None
        and reference_centerline is not None
        and not np.isnan(reference_centerline[row_v])
    )

    candidates: list[tuple[float, float, float, int]] = []

    def debug_bump(name: str) -> None:
        if side_debug is None:
            return
        side_debug[name] = int(side_debug.get(name, 0)) + 1

    def debug_range(name: str, value: float) -> None:
        if side_debug is None:
            return
        minimum_key = f"{name}_min"
        maximum_key = f"{name}_max"
        side_debug[minimum_key] = min(
            float(side_debug.get(minimum_key, value)),
            value,
        )
        side_debug[maximum_key] = max(
            float(side_debug.get(maximum_key, value)),
            value,
        )

    if side_debug is not None:
        side_debug["rows_seen"] = int(side_debug.get("rows_seen", 0)) + 1
        row_kind = "rows_single" if len(segments) == 1 else "rows_multi"
        side_debug[row_kind] = int(side_debug.get(row_kind, 0)) + 1
        if single_line_angle_deg is not None:
            side_debug["rows_angle"] = int(side_debug.get("rows_angle", 0)) + 1

    temporal_center_v = (
        float(temporal_centerline[row_v])
        if temporal_centerline is not None
        and not np.isnan(temporal_centerline[row_v])
        else None
    )

    def nearby_values(boundary: np.ndarray | None) -> list[float] | None:
        """직전 경계의 인접 전후 행 값(유효한 것만)을 행마다 한 번만 뽑는다."""

        if boundary is None:
            return None
        start = max(0, row_v - TEMPORAL_ID_ROW_RADIUS)
        end = min(len(boundary), row_v + TEMPORAL_ID_ROW_RADIUS + 1)
        nearby = boundary[start:end]
        nearby = nearby[~np.isnan(nearby)]
        if nearby.size == 0:
            return None
        # 후보마다 21개를 선형 탐색하지 않도록 정렬해 두고 이분탐색한다.
        return sorted(nearby.tolist())

    # 후보마다 슬라이스·isnan을 다시 돌면 프레임당 수만 번이 된다. 행마다 한 번.
    nearby_left = nearby_values(temporal_left)
    nearby_right = nearby_values(temporal_right)

    def temporal_distance(
        nearby: list[float] | None,
        observed_u: float,
    ) -> float | None:
        """차량 이동을 고려해 직전 경계의 인접 전후 행과 비교한다.

        nearby는 정렬돼 있으므로 삽입 위치 양옆만 보면 최근접이 나온다.
        """

        if not nearby:
            return None
        index = bisect.bisect_left(nearby, observed_u)
        best = float("inf")
        if index < len(nearby):
            best = nearby[index] - observed_u
        if index > 0:
            best = min(best, observed_u - nearby[index - 1])
        return best

    def append_if_valid(
        left_u: float,
        right_u: float,
        width_error: float,
        pair_bonus: float,
        source: int,
    ) -> None:
        source_name = (
            "pair"
            if source == BOUNDARY_SOURCE_PAIR
            else "left"
            if source == BOUNDARY_SOURCE_LEFT
            else "right"
        )

        def bump(reason: str) -> None:
            debug_bump(f"{source_name}_{reason}")

        bump("attempt")
        if right_u <= left_u:
            bump("geometry")
            return

        visible_left = max(0.0, left_u)
        visible_right = min(float(BEV_WIDTH - 1), right_u)
        visible_width = visible_right - visible_left
        if visible_width < MIN_VISIBLE_CANDIDATE_WIDTH_PX:
            bump("view")
            return

        opposite_line_conflict = (
            opposite_cum_row is not None
            and source in (BOUNDARY_SOURCE_LEFT, BOUNDARY_SOURCE_RIGHT)
            and opposite_line_inside_lane(
                opposite_cum_row,
                left_u,
                right_u,
                source,
            )
        )
        angle_supports_source = (
            len(segments) == 1
            and single_line_side_bonus(single_line_angle_deg, source) > 0.0
        )
        if opposite_line_conflict and not angle_supports_source:
            bump("white")
            return


        full_width = right_u - left_u
        visible_ratio = min(1.0, visible_width / max(1.0, full_width))

        center = (left_u + right_u) / 2.0

        identity_errors: list[float] = []
        if source in (BOUNDARY_SOURCE_PAIR, BOUNDARY_SOURCE_LEFT):
            left_identity_error = temporal_distance(nearby_left, left_u)
            if left_identity_error is not None:
                identity_errors.append(left_identity_error)
        if source in (BOUNDARY_SOURCE_PAIR, BOUNDARY_SOURCE_RIGHT):
            right_identity_error = temporal_distance(nearby_right, right_u)
            if right_identity_error is not None:
                identity_errors.append(right_identity_error)
        identity_mean = (
            sum(identity_errors) / len(identity_errors)
            if identity_errors
            else 0.0
        )
        matches_preferred_side = not prefer_inner_course or (
            candidate_matches_reference_side(
                center,
                reference_centerline,
                row_v,
                required_side,
            )
        )

        score = score_boundary_candidate(
            left_u,
            right_u,
            width_error,
            reference_center,
            None,
            None,
        )
        score -= PARTIAL_CANDIDATE_PENALTY * (1.0 - visible_ratio)

        reference_error = abs(center - reference_center) / ROAD_WIDTH_NORM
        score -= PATH_REFERENCE_PENALTY * reference_error
        if temporal_center_v is not None:
            temporal_error = (
                abs(center - temporal_center_v) / ROAD_WIDTH_NORM
            )
            score -= PATH_TEMPORAL_PENALTY * temporal_error

        # 중심선뿐 아니라 실제로 관측된 선 자체의 ID를 직전 프레임과
        # 비교한다. 같은 선을 LEFT에서 RIGHT로 바꾸면 약 350 mm의
        # 불일치가 생기므로 도로 겹침 점수가 좋아도 쉽게 전환하지 않는다.
        if identity_errors:
            score -= PATH_BOUNDARY_ID_PENALTY * (
                identity_mean / ROAD_WIDTH_NORM
            )

        # P1: 카메라=전방 중앙 → 근거리(BEV 하단)에서는 ego 축 중앙 차로를 선호.
        near_row0 = int(round(BEV_HEIGHT * (1.0 - FORK_NEAR_ZONE_RATIO)))
        if row_v >= near_row0:
            ego_u = (BEV_WIDTH - 1) / 2.0
            ego_err = abs(center - ego_u) / ROAD_WIDTH_NORM
            score += EGO_NEAR_CENTER_BONUS * max(0.0, 1.0 - min(1.5, ego_err))
        if not matches_preferred_side:
            # 차량이 회전교차로 한쪽으로 치우치면 정상 노란 코스가
            # 흰 중심선의 반대쪽에 보일 수 있다. 절대 탈락시키지 않고
            # 우선순위만 낮춘다.
            score -= PATH_WRONG_SIDE_PENALTY
        if opposite_line_conflict:
            score -= SINGLE_LINE_OPPOSITE_LINE_PENALTY
            bump("white_penalty")
        score += pair_bonus
        candidates.append((left_u, right_u, score, source))
        bump("valid")

    # 기울기에 따른 수평 폭 배율. 도로가 기울면 같은 행에서 자른 폭이 넓어진다.
    if segment_slopes is None or len(segment_slopes) != len(segments):
        scales = [1.0] * len(segments)
    else:
        scales = [horizontal_width_scale(s) for s in segment_slopes]

    # 실제 노란선 두 개로 이루어진 후보
    for left_index in range(len(segments)):
        for right_index in range(left_index + 1, len(segments)):
            left_u = float(segments[left_index][1])
            right_u = float(segments[right_index][0])
            measured_width = right_u - left_u
            pair_width = ROAD_WIDTH_PX * (
                (scales[left_index] + scales[right_index]) / 2.0
            )
            width_error = abs(measured_width - pair_width)

            if side_debug is not None:
                debug_bump("pair_tested")
                debug_range("pair_measured_width", measured_width)
                debug_range("pair_expected_width", pair_width)
                debug_range("pair_width_error", width_error)

            if measured_width <= 0.0 or width_error > ROAD_WIDTH_TOLERANCE_PX:
                debug_bump("pair_width_reject")
                continue

            left_heading_deg = math.degrees(
                math.atan(
                    0.0
                    if segment_slopes is None
                    or len(segment_slopes) != len(segments)
                    else segment_slopes[left_index]
                )
            )
            right_heading_deg = math.degrees(
                math.atan(
                    0.0
                    if segment_slopes is None
                    or len(segment_slopes) != len(segments)
                    else segment_slopes[right_index]
                )
            )
            heading_diff_deg = abs(left_heading_deg - right_heading_deg)
            debug_range("pair_heading_diff", heading_diff_deg)
            if heading_diff_deg > PAIR_MAX_HEADING_DIFF_DEG:
                debug_bump("pair_heading_reject")
                continue

            append_if_valid(
                left_u,
                right_u,
                width_error,
                PATH_PAIR_BONUS,
                BOUNDARY_SOURCE_PAIR,
            )

    if opposite_segments is None or not is_ego_course:
        # 옆 도로(내가 안 달리는 코스)까지 혼색으로 짝지으면, 나란히 달리는
        # 흰 도로와 노란 도로 사이에 있지도 않은 차로가 생긴다. 내가 실제로
        # 달리는 코스에서만 쓴다.
        opposite_segments = []

    # 좌우 색이 다른 차로. 연결로가 직선 도로에 합류하는 구간이 그렇다:
    # 왼쪽은 연결로에서 이어진 노란 점선, 오른쪽은 직선 도로의 흰 실선이다.
    # 같은 색끼리만 짝지으면 이 차로는 절대 못 만든다. 그래서 코드는 노란선
    # 오른쪽 350 mm에 노란 경계를 '지어내는데', 그게 카메라 사각(검은 쐐기)에
    # 떨어져 점수를 못 받고 결국 뒤쪽 아스팔트에 차로를 얹는 오답에 진다.
    # 두 선을 실제로 관측한 짝이므로 지어낸 후보보다 근거가 강하다.
    for index, (own_start, own_end) in enumerate(segments):
        for other_start, other_end in opposite_segments:
            pair_width = ROAD_WIDTH_PX * scales[index]
            for left_u, right_u in (
                (float(own_end), float(other_start)),
                (float(other_end), float(own_start)),
            ):
                measured_width = right_u - left_u
                width_error = abs(measured_width - pair_width)
                if side_debug is not None:
                    debug_bump("pair_tested")
                    debug_range("pair_measured_width", measured_width)
                    debug_range("pair_expected_width", pair_width)
                    debug_range("pair_width_error", width_error)
                if (
                    measured_width <= 0.0
                    or width_error > ROAD_WIDTH_TOLERANCE_PX
                ):
                    debug_bump("pair_width_reject")
                    continue
                append_if_valid(
                    left_u,
                    right_u,
                    width_error,
                    PATH_PAIR_BONUS,
                    BOUNDARY_SOURCE_PAIR,
                )

    # 실제 두 선을 모두 본 행에서는 추정 단일선이 경쟁하지 못하게 한다.
    # PAIR가 없는 행에서만 직전 프레임의 경계 ID를 먼저 보고, temporal 정보가
    # 없으면 기준 중심의 좌우 위치로 관측선을 LEFT/RIGHT로 분류한다.
    has_pair_candidate = any(
        source == BOUNDARY_SOURCE_PAIR
        for _, _, _, source in candidates
    )
    if not has_pair_candidate:
        row_single_line_angle = (
            single_line_angle_deg if len(segments) == 1 else None
        )
        for index, (segment_start, segment_end) in enumerate(segments):
            lane_width_px = ROAD_WIDTH_PX * scales[index]
            detected_as_left = float(segment_end)
            detected_as_right = float(segment_start)
            segment_center_u = (segment_start + segment_end) / 2.0

            left_identity_error = temporal_distance(
                nearby_left,
                detected_as_left,
            )
            right_identity_error = temporal_distance(
                nearby_right,
                detected_as_right,
            )
            allowed_sources: tuple[int, ...]
            temporal_matches: list[tuple[float, int]] = []
            if (
                left_identity_error is not None
                and left_identity_error <= TEMPORAL_ID_MATCH_PX
            ):
                temporal_matches.append(
                    (left_identity_error, BOUNDARY_SOURCE_LEFT)
                )
            if (
                right_identity_error is not None
                and right_identity_error <= TEMPORAL_ID_MATCH_PX
            ):
                temporal_matches.append(
                    (right_identity_error, BOUNDARY_SOURCE_RIGHT)
                )

            # 도로(주행가능영역)가 어느 쪽에 있는지로 좌우를 정한다: 차로는 도로
            # 쪽에 있다. 이진 영역의 '좌/우 한 표'로만 쓰고 경계 기하에는 관여시키지
            # 않는다(경계 모양은 선이 그대로 만든다). 차량 중심 기준보다 물리적으로
            # 견고해서, 차가 치우쳐 한 선만 보일 때의 좌우 뒤집힘을 바로잡는다.
            road_side: int | None = None
            if drivable_cum_row is not None and drivable_cum_row.shape[0] > 1:
                width_px = drivable_cum_row.shape[0] - 1
                start_u = max(0, min(int(segment_start), width_px))
                end_u = max(0, min(int(segment_end) + 1, width_px))
                left_road = float(drivable_cum_row[start_u])
                right_road = float(drivable_cum_row[width_px] - drivable_cum_row[end_u])
                if (
                    right_road >= SINGLE_LINE_ROAD_SIDE_MIN_PX
                    and right_road >= left_road * SINGLE_LINE_ROAD_SIDE_RATIO
                ):
                    road_side = BOUNDARY_SOURCE_LEFT  # 도로가 오른쪽 → 선은 왼쪽 경계
                elif (
                    left_road >= SINGLE_LINE_ROAD_SIDE_MIN_PX
                    and left_road >= right_road * SINGLE_LINE_ROAD_SIDE_RATIO
                ):
                    road_side = BOUNDARY_SOURCE_RIGHT  # 도로가 왼쪽 → 선은 오른쪽 경계

            if temporal_matches:
                allowed_sources = (min(temporal_matches)[1],)
                debug_bump(
                    "single_temporal_left"
                    if allowed_sources[0] == BOUNDARY_SOURCE_LEFT
                    else "single_temporal_right"
                )
            elif single_line_side_hint is not None:
                # 라인 전체 도로 분포로 정한 좌우(행별 투표보다 견고).
                allowed_sources = (single_line_side_hint,)
                debug_bump(
                    "single_agg_left"
                    if single_line_side_hint == BOUNDARY_SOURCE_LEFT
                    else "single_agg_right"
                )
            elif road_side is not None:
                allowed_sources = (road_side,)
                debug_bump(
                    "single_road_left"
                    if road_side == BOUNDARY_SOURCE_LEFT
                    else "single_road_right"
                )
            elif segment_center_u < (
                reference_center - SINGLE_LINE_CENTER_DEADBAND_PX
            ):
                allowed_sources = (BOUNDARY_SOURCE_LEFT,)
                debug_bump("single_center_left")
            elif segment_center_u > (
                reference_center + SINGLE_LINE_CENTER_DEADBAND_PX
            ):
                allowed_sources = (BOUNDARY_SOURCE_RIGHT,)
                debug_bump("single_center_right")
            else:
                # 중심 바로 앞의 선은 위치만으로 좌우를 정할 수 없다. 이 경우만
                # 두 후보를 열어두고 기존 각도/DP 점수가 결정하게 한다.
                allowed_sources = (
                    BOUNDARY_SOURCE_LEFT,
                    BOUNDARY_SOURCE_RIGHT,
                )
                debug_bump("single_ambiguous")

            if BOUNDARY_SOURCE_LEFT in allowed_sources:
                append_if_valid(
                    detected_as_left,
                    detected_as_left + lane_width_px,
                    0.0,
                    single_line_side_bonus(
                        row_single_line_angle,
                        BOUNDARY_SOURCE_LEFT,
                    ),
                    BOUNDARY_SOURCE_LEFT,
                )

            if BOUNDARY_SOURCE_RIGHT in allowed_sources:
                append_if_valid(
                    detected_as_right - lane_width_px,
                    detected_as_right,
                    0.0,
                    single_line_side_bonus(
                        row_single_line_angle,
                        BOUNDARY_SOURCE_RIGHT,
                    ),
                    BOUNDARY_SOURCE_RIGHT,
                )

    # 두께운 마스크 가장자리에서 사실상 같은 후보가 여러 번
    # 생길 수 있으므로 1 px 단위로 중복을 제거한다.
    deduplicated: dict[tuple[int, int], tuple[float, float, float, int]] = {}
    for left_u, right_u, score, source in candidates:
        key = (int(round(left_u)), int(round(right_u)))
        previous = deduplicated.get(key)
        if previous is None or score > previous[2]:
            deduplicated[key] = (left_u, right_u, score, source)

    ranked_candidates = sorted(
        deduplicated.values(),
        key=lambda candidate: candidate[2],
        reverse=True,
    )
    return ranked_candidates[:MAX_PATH_CANDIDATES_PER_ROW]



def _solve_boundary_dp(
    candidates_by_row: dict[int, list[tuple[float, float, float, int]]],
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Connect per-row L/R candidates into one continuous corridor (DP)."""

    raw_left = np.full(height, np.nan, dtype=np.float32)
    raw_right = np.full(height, np.nan, dtype=np.float32)
    left_observed = np.zeros(height, dtype=bool)
    right_observed = np.zeros(height, dtype=bool)
    if not candidates_by_row:
        return raw_left, raw_right, left_observed, right_observed, float("-inf")

    # key -> (누적 점수, 이전 key, 직전 중심선 기울기, 연결 노드 수)
    states: dict[
        tuple[int, int],
        tuple[float, tuple[int, int] | None, float, int],
    ] = {}

    processed_rows: list[int] = []

    for v in sorted(candidates_by_row, reverse=True):
        current_candidates = candidates_by_row[v]
        previous_rows = [
            previous_v
            for previous_v in reversed(processed_rows)
            if previous_v - v <= MAX_BOUNDARY_TRACK_GAP_PX
        ][:MAX_PATH_PREVIOUS_ROWS]

        for candidate_index, (left_u, right_u, local_score, source) in enumerate(
            current_candidates
        ):
            key = (v, candidate_index)
            best_score = local_score
            best_previous: tuple[int, int] | None = None
            best_slope = 0.0
            best_length = 1
            center = (left_u + right_u) / 2.0

            for previous_v in previous_rows:
                row_gap = previous_v - v
                if row_gap <= 0:
                    continue

                for previous_index, (
                    previous_left,
                    previous_right,
                    _,
                    previous_source,
                ) in enumerate(candidates_by_row[previous_v]):
                    previous_key = (previous_v, previous_index)
                    previous_state = states.get(previous_key)
                    if previous_state is None:
                        continue

                    if not boundary_candidate_is_continuous(
                        left_u,
                        right_u,
                        previous_left,
                        previous_right,
                        v,
                        previous_v,
                    ):
                        continue

                    previous_score, _, previous_slope, previous_length = (
                        previous_state
                    )
                    previous_center = (previous_left + previous_right) / 2.0
                    slope = (center - previous_center) / float(row_gap)

                    center_shift = abs(center - previous_center) / max(
                        1.0, MAX_BOUNDARY_SHIFT_PX
                    )
                    boundary_shift = (
                        abs(left_u - previous_left)
                        + abs(right_u - previous_right)
                    ) / (2.0 * max(1.0, MAX_BOUNDARY_SHIFT_PX))
                    slope_change = abs(slope - previous_slope)

                    transition_penalty = (
                        PATH_GAP_PENALTY * max(0, row_gap - 1)
                        + PATH_CENTER_SHIFT_PENALTY * center_shift
                        + PATH_BOUNDARY_SHIFT_PENALTY * boundary_shift
                    )
                    if previous_length >= 2:
                        transition_penalty += (
                            PATH_SLOPE_CHANGE_PENALTY * slope_change
                        )
                    if source != previous_source:
                        transition_penalty += PATH_SOURCE_SWITCH_PENALTY

                    accumulated_score = (
                        previous_score + local_score - transition_penalty
                    )
                    accumulated_length = previous_length + 1

                    if (
                        accumulated_score > best_score
                        or (
                            scores_tied(accumulated_score, best_score)
                            and accumulated_length > best_length
                        )
                    ):
                        best_score = accumulated_score
                        best_previous = previous_key
                        best_slope = slope
                        best_length = accumulated_length

            states[key] = (
                best_score,
                best_previous,
                best_slope,
                best_length,
            )

        processed_rows.append(v)

    # 짧은 오검출보다 길게 이어진 경로를 선호한다.
    best_key = max(
        states,
        key=lambda state_key: (
            states[state_key][0],
            states[state_key][3],
            state_key[0],
        ),
    )

    cursor: tuple[int, int] | None = best_key
    while cursor is not None:
        v, candidate_index = cursor
        left_u, right_u, _, source = candidates_by_row[v][candidate_index]
        raw_left[v] = left_u
        raw_right[v] = right_u
        left_observed[v] = source in (
            BOUNDARY_SOURCE_PAIR,
            BOUNDARY_SOURCE_LEFT,
        )
        right_observed[v] = source in (
            BOUNDARY_SOURCE_PAIR,
            BOUNDARY_SOURCE_RIGHT,
        )
        cursor = states[cursor][1]

    best_score = float(states[best_key][0]) if states else float("-inf")
    return raw_left, raw_right, left_observed, right_observed, best_score


def _aggregate_single_line_side(
    segments_by_row: list[list[tuple[int, int]]],
    drivable_cums: np.ndarray | None,
) -> int | None:
    """단일선 좌우를 라인 전체 도로 분포로 1회 결정한다(행별 투표보다 견고).

    단일 세그먼트 행들의 도로(주행가능) 좌/우 픽셀을 모두 합산해 우세한 쪽으로
    정한다. 도로가 희박해도 전 행을 합치면 신호가 서고, 프레임 내 좌우 흔들림이
    사라진다. 반환: BOUNDARY_SOURCE_LEFT / RIGHT / None(우열 없음).
    """
    if drivable_cums is None or drivable_cums.shape[0] == 0:
        return None
    width = drivable_cums.shape[1] - 1
    total_left = 0.0
    total_right = 0.0
    for v, segments in enumerate(segments_by_row):
        if v >= drivable_cums.shape[0] or len(segments) != 1:
            continue
        cum = drivable_cums[v]
        start_u = max(0, min(int(segments[0][0]), width))
        end_u = max(0, min(int(segments[0][1]) + 1, width))
        total_left += float(cum[start_u])
        total_right += float(cum[width] - cum[end_u])
    if (
        total_left < SINGLE_LINE_ROAD_SIDE_MIN_PX
        and total_right < SINGLE_LINE_ROAD_SIDE_MIN_PX
    ):
        return None
    if total_right >= total_left * SINGLE_LINE_ROAD_SIDE_RATIO:
        return BOUNDARY_SOURCE_LEFT  # 도로가 오른쪽에 더 많다 → 선은 왼쪽 경계
    if total_left >= total_right * SINGLE_LINE_ROAD_SIDE_RATIO:
        return BOUNDARY_SOURCE_RIGHT  # 도로가 왼쪽에 더 많다 → 선은 오른쪽 경계
    return None


def track_boundary_path(
    boundary_mask: np.ndarray,
    reference_centerline: np.ndarray | None,
    temporal_centerline: np.ndarray | None,
    temporal_left: np.ndarray | None,
    temporal_right: np.ndarray | None,
    required_side: str | None,
    opposite_line_mask: np.ndarray | None = None,
    is_ego_course: bool = False,
    use_single_line_angle_bias: bool = False,
    side_debug: dict[str, object] | None = None,
    boundary_segments_by_row: list[list[tuple[int, int]]] | None = None,
    opposite_segments_by_row: list[list[tuple[int, int]]] | None = None,
    find_alternate: bool = False,
    drivable_mask: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """모든 행의 후보를 연결해 위치/방향/곡률이 연속인 경로를 찾는다.

    ``find_alternate=True``이면 주 경계 코스가 쓴 후보를 제외하고 DP를 한 번
    더 풀어 **보조 경계 코스**(코드명 ``*_alt_*``)를 반환한다. 주+보조 = 경계
    4선 → Out 갈림·In 탈출의 갈래 2개용 ``ForkLanePair``. 용어: strategy.md §0.
    """

    height = boundary_mask.shape[0]
    empty = np.full(height, np.nan, dtype=np.float32)
    empty_flags = np.zeros(height, dtype=bool)

    # 다른 색 선이 단일선으로 추정한 차로 내부에 들어오는지만 O(1)로 검사한다.
    # 이 마스크는 도로 HSV가 아니라 실제 흰색/노란색 차선 검출 결과다.
    opposite_cums = (
        None
        if opposite_line_mask is None
        else _row_cumsum(opposite_line_mask > 0)
    )
    # 단일선 좌우 판정용 도로(주행가능영역) 누적합. 좌/우 한 표로만 쓴다.
    drivable_cums = (
        None if drivable_mask is None else _row_cumsum(drivable_mask > 0)
    )
    segments_by_row = (
        boundary_segments_by_row
        if boundary_segments_by_row is not None
        else find_line_segments_by_row(boundary_mask)
    )
    # 단일선 좌우를 라인 전체 집계로 1회 결정(행별 투표보다 견고, 프레임 내 일관).
    single_line_side_hint = _aggregate_single_line_side(
        segments_by_row, drivable_cums
    )
    slopes_by_row = estimate_segment_slopes(segments_by_row, height)
    if use_single_line_angle_bias:
        component_labels, component_angles = single_line_component_angles_deg(
            boundary_mask,
            side_debug,
        )
    else:
        component_labels = np.zeros_like(boundary_mask, dtype=np.int32)
        component_angles = {}
    if side_debug is not None:
        side_debug.setdefault("components", 0)
        side_debug.setdefault("angle_deg", None)

    def row_single_line_angle_deg(
        row_v: int,
        segments: list[tuple[int, int]],
    ) -> float | None:
        if len(segments) != 1 or not component_angles:
            return None
        start_u, end_u = segments[0]
        row_labels = component_labels[row_v, start_u:end_u + 1]
        angle_labels = np.array(
            [label for label in row_labels if int(label) in component_angles],
            dtype=np.int32,
        )
        if angle_labels.size == 0:
            return None
        labels_found, counts = np.unique(angle_labels, return_counts=True)
        dominant_label = int(labels_found[int(np.argmax(counts))])
        return component_angles.get(dominant_label)

    opposite_by_row = opposite_segments_by_row
    if opposite_by_row is None and opposite_line_mask is not None:
        opposite_by_row = find_line_segments_by_row(opposite_line_mask)

    def build_candidates_by_row(
        *,
        use_temporal: bool,
        write_side_debug: bool,
    ) -> dict[int, list[tuple[float, float, float, int]]]:
        out: dict[int, list[tuple[float, float, float, int]]] = {}
        dbg = side_debug if write_side_debug else None
        for v in range(height - 1, -1, -1):
            segments = segments_by_row[v]
            if not segments:
                continue
            row_angle_deg = row_single_line_angle_deg(v, segments)
            row_candidates = enumerate_boundary_candidates(
                segments,
                v,
                reference_centerline,
                temporal_centerline if use_temporal else None,
                temporal_left if use_temporal else None,
                temporal_right if use_temporal else None,
                required_side,
                None if opposite_cums is None else opposite_cums[v],
                slopes_by_row[v],
                is_ego_course,
                None if opposite_by_row is None else opposite_by_row[v],
                row_angle_deg,
                dbg,
                None if drivable_cums is None else drivable_cums[v],
                single_line_side_hint,
            )
            if row_candidates:
                out[v] = row_candidates
        return out

    candidates_by_row = build_candidates_by_row(
        use_temporal=True, write_side_debug=True
    )
    raw_left, raw_right, left_observed, right_observed, _ = _solve_boundary_dp(
        candidates_by_row, height
    )
    if side_debug is not None:
        side_debug["selected_left"] = int(np.count_nonzero(left_observed))
        side_debug["selected_right"] = int(np.count_nonzero(right_observed))

    if not find_alternate:
        return (
            raw_left,
            raw_right,
            left_observed,
            right_observed,
            empty.copy(),
            empty.copy(),
            empty_flags.copy(),
            empty_flags.copy(),
        )

    # Alt without temporal bias so the non-ego fork corridor is not suppressed.
    alt_candidates_by_row = build_candidates_by_row(
        use_temporal=False, write_side_debug=False
    )

    def is_duplicate_of_primary(v: int, left_u: float, right_u: float) -> bool:
        primary_left = raw_left[v]
        primary_right = raw_right[v]
        if np.isnan(primary_left) or np.isnan(primary_right):
            return False
        return (
            abs(left_u - primary_left) <= ROAD_WIDTH_TOLERANCE_PX
            and abs(right_u - primary_right) <= ROAD_WIDTH_TOLERANCE_PX
        )

    remaining_candidates_by_row = {
        v: remaining
        for v, candidates in alt_candidates_by_row.items()
        if (
            remaining := [
                candidate
                for candidate in candidates
                if not is_duplicate_of_primary(v, candidate[0], candidate[1])
            ]
        )
    }
    alt_left, alt_right, alt_left_observed, alt_right_observed, _ = (
        _solve_boundary_dp(remaining_candidates_by_row, height)
    )
    return (
        raw_left,
        raw_right,
        left_observed,
        right_observed,
        alt_left,
        alt_right,
        alt_left_observed,
        alt_right_observed,
    )



def interpolate_boundary_pair(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """좌우 경계를 한 쌍으로 유지하면서 짧은 점선 간격만 보간한다."""

    left = raw_left.copy()
    right = raw_right.copy()

    valid_rows = np.flatnonzero(
        ~np.isnan(raw_left)
        & ~np.isnan(raw_right)
    )

    if valid_rows.size < 2:
        return (
            left,
            right,
        )

    for (
        start_v,
        end_v,
    ) in zip(
        valid_rows[:-1],
        valid_rows[1:],
    ):
        missing_rows = (
            end_v
            - start_v
            - 1
        )

        if missing_rows <= 0:
            continue

        if (
            missing_rows
            > MAX_BOUNDARY_INTERPOLATION_GAP_PX
        ):
            continue

        left_shift = abs(
            float(
                raw_left[end_v]
            )
            - float(
                raw_left[start_v]
            )
        )

        right_shift = abs(
            float(
                raw_right[end_v]
            )
            - float(
                raw_right[start_v]
            )
        )

        if (
            left_shift
            > MAX_BOUNDARY_SHIFT_PX
        ):
            continue

        if (
            right_shift
            > MAX_BOUNDARY_SHIFT_PX
        ):
            continue

        left[
            start_v:end_v + 1
        ] = np.linspace(
            float(
                raw_left[start_v]
            ),
            float(
                raw_left[end_v]
            ),
            end_v - start_v + 1,
            dtype=np.float32,
        )

        right[
            start_v:end_v + 1
        ] = np.linspace(
            float(
                raw_right[start_v]
            ),
            float(
                raw_right[end_v]
            ),
            end_v - start_v + 1,
            dtype=np.float32,
        )

    return (
        left,
        right,
    )


def keep_nearest_continuous_run(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    차량에 가장 가까운 충분한 길이의 연속 경계 구간만 유지한다.

    이 단계가 멀리 떨어진 별도 yellow_drivable 덩어리를 제거한다.
    """

    valid_rows = np.flatnonzero(
        ~np.isnan(left)
        & ~np.isnan(right)
    )

    if valid_rows.size == 0:
        return (
            left,
            right,
        )

    split_indices = (
        np.where(
            np.diff(valid_rows) > 1
        )[0]
        + 1
    )

    runs = [
        run
        for run in np.split(
            valid_rows,
            split_indices,
        )
        if run.size > 0
    ]

    long_runs = [
        run
        for run in runs
        if run.size
        >= MIN_COURSE_RUN_ROWS
    ]

    candidate_runs = (
        long_runs
        if long_runs
        else runs
    )

    # 가장 아래쪽, 즉 차량에 가장 가까운 연속 구간 선택
    selected_run = max(
        candidate_runs,
        key=lambda run: (
            int(run[-1]),
            int(run.size),
        ),
    )

    kept_left = np.full_like(
        left,
        np.nan,
    )

    kept_right = np.full_like(
        right,
        np.nan,
    )

    kept_left[
        selected_run
    ] = left[
        selected_run
    ]

    kept_right[
        selected_run
    ] = right[
        selected_run
    ]

    return (
        kept_left,
        kept_right,
    )


def centerline_from_boundaries(
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    *,
    synthesize_missing: bool = True,
    lane_width_m: float | None = None,
) -> np.ndarray:
    """좌우 경계의 중간을 중심선으로 계산한다.

    한쪽만 관측되면 ``lane_width_m``(기본 track_width) prior로 반대편을
    가정해 중앙을 유지한다 (P2 소실 보정).
    """

    centerline = np.full_like(left_boundary, np.nan)
    both = ~np.isnan(left_boundary) & ~np.isnan(right_boundary)
    centerline[both] = (
        left_boundary[both] + right_boundary[both]
    ) / 2.0

    if synthesize_missing:
        half_w_px = (
            0.5 * float(lane_width_m if lane_width_m is not None else ROAD_WIDTH_M)
        ) / METERS_PER_PIXEL
        only_left = ~np.isnan(left_boundary) & np.isnan(right_boundary)
        only_right = np.isnan(left_boundary) & ~np.isnan(right_boundary)
        centerline[only_left] = left_boundary[only_left] + half_w_px
        centerline[only_right] = right_boundary[only_right] - half_w_px

    return centerline


# =========================================================
# 센터라인 흔들림 확정-진단 계측 (기본 off · 0비용)
# ---------------------------------------------------------
# LANE_CENTERLINE_DEBUG=1 일 때만 흰 경로 후보 카운터를 모아 ~1초마다 출력한다.
# 직선에서 흰 센터라인이 좌우로 튀는 원인이 아래 중 무엇인지 실차 숫자로 가른다.
#   (1) pair↔single 전환      → rows_single 이 크거나 프레임마다 요동
#   (2) 단일선 좌/우 판정 뒤집힘 → single_center_left ↔ right 가 번갈아 큼
#   (3) PAIR 게이트 간헐 탈락    → pair_heading_reject / pair_width_reject > 0
#   (4) DP corridor 교체        → rows_multi 인데 near_y max_jump 가 큼
# 수집·출력만 하며 인지 로직·조향에는 전혀 영향이 없다(off면 기존과 동일).
# =========================================================
CENTERLINE_DEBUG = (os.environ.get("LANE_CENTERLINE_DEBUG") or "").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)
CENTERLINE_DEBUG_EVERY = 30  # 프레임 (≈30fps → 약 1초마다 한 줄 요약)

_CENTERLINE_DEBUG_KEYS = (
    "rows_multi",
    "rows_single",
    "pair_tested",
    "pair_heading_reject",
    "pair_width_reject",
    "single_center_left",
    "single_center_right",
    "single_temporal_left",
    "single_temporal_right",
    "selected_left",
    "selected_right",
)
_centerline_debug_frame = 0
_centerline_debug_prev_near_y: float | None = None
_centerline_debug_max_jump = 0.0
_centerline_debug_near_lo: float | None = None
_centerline_debug_near_hi: float | None = None
_centerline_debug_acc: dict[str, int] = {}


def _log_centerline_debug(
    white_centerline: np.ndarray,
    side_debug: dict[str, object] | None,
) -> None:
    """흰 센터라인 근거리 오프셋 점프와 후보 카운터를 ~1초마다 출력한다."""

    global _centerline_debug_frame, _centerline_debug_prev_near_y
    global _centerline_debug_max_jump, _centerline_debug_near_lo
    global _centerline_debug_near_hi, _centerline_debug_acc

    near_y: float | None = None
    if white_centerline is not None:
        valid = np.flatnonzero(~np.isnan(white_centerline))
        if valid.size:
            near_row = int(valid.max())  # 큰 v = 카메라에 가장 가까운 행
            near_y = (
                float(white_centerline[near_row]) - 0.5 * (BEV_WIDTH - 1)
            ) * METERS_PER_PIXEL

    if near_y is not None:
        if _centerline_debug_prev_near_y is not None:
            jump = abs(near_y - _centerline_debug_prev_near_y)
            _centerline_debug_max_jump = max(_centerline_debug_max_jump, jump)
        _centerline_debug_prev_near_y = near_y
        _centerline_debug_near_lo = (
            near_y
            if _centerline_debug_near_lo is None
            else min(_centerline_debug_near_lo, near_y)
        )
        _centerline_debug_near_hi = (
            near_y
            if _centerline_debug_near_hi is None
            else max(_centerline_debug_near_hi, near_y)
        )

    for key in _CENTERLINE_DEBUG_KEYS:
        value = side_debug.get(key, 0) if side_debug else 0
        if isinstance(value, (int, float)):
            _centerline_debug_acc[key] = _centerline_debug_acc.get(key, 0) + int(value)

    _centerline_debug_frame += 1
    if _centerline_debug_frame < CENTERLINE_DEBUG_EVERY:
        return

    acc = _centerline_debug_acc
    span = (
        _centerline_debug_near_hi - _centerline_debug_near_lo
        if _centerline_debug_near_lo is not None
        else float("nan")
    )
    print(
        "[centerline/{n}f] rows multi={rm} single={rs} | "
        "pair tested={pt} heading_reject={phr} width_reject={pwr} | "
        "single L={scl} R={scr} tempL={stl} tempR={str_} | "
        "selected_rows L/R={sl}/{sr} | "
        "near_y span={span:.3f}m max_jump={mj:.3f}m".format(
            n=_centerline_debug_frame,
            rm=acc.get("rows_multi", 0),
            rs=acc.get("rows_single", 0),
            pt=acc.get("pair_tested", 0),
            phr=acc.get("pair_heading_reject", 0),
            pwr=acc.get("pair_width_reject", 0),
            scl=acc.get("single_center_left", 0),
            scr=acc.get("single_center_right", 0),
            stl=acc.get("single_temporal_left", 0),
            str_=acc.get("single_temporal_right", 0),
            sl=acc.get("selected_left", 0),
            sr=acc.get("selected_right", 0),
            span=span,
            mj=_centerline_debug_max_jump,
        ),
        flush=True,
    )

    _centerline_debug_frame = 0
    _centerline_debug_max_jump = 0.0
    _centerline_debug_near_lo = None
    _centerline_debug_near_hi = None
    _centerline_debug_acc = {}


def boundary_pair_contains_vehicle(
    left_boundary: np.ndarray | None,
    right_boundary: np.ndarray | None,
) -> bool:
    """가까운 구간의 좌우 차선 사이에 차량 중심 열이 있는지 확인한다.

    검정 도로 HSV로 현재 코스를 정하지 않고, 직전 프레임에서 검출한 차선
    기하만으로 혼색 PAIR 허용 여부와 노란 기준선 선택을 결정한다.
    """

    if left_boundary is None or right_boundary is None:
        return False
    valid_rows = np.flatnonzero(
        ~np.isnan(left_boundary) & ~np.isnan(right_boundary)
    )
    if valid_rows.size == 0:
        return False
    near_rows = valid_rows[-min(10, valid_rows.size):]
    left_u = float(np.median(left_boundary[near_rows]))
    right_u = float(np.median(right_boundary[near_rows]))
    vehicle_u = (BEV_WIDTH - 1) / 2.0
    return left_u <= vehicle_u <= right_u


def fit_robust_polynomial(
    x_values: np.ndarray,
    y_values: np.ndarray,
    degree: int,
    minimum_rows: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """경계 픽셀 이상점을 제거하며 다항식과 인라이어 마스크를 반환한다."""

    if x_values.size < max(minimum_rows, degree + 1):
        return None

    x_fit = x_values.astype(np.float64)
    y_fit = y_values.astype(np.float64)
    kept = np.ones(x_fit.size, dtype=bool)

    for _ in range(PLANNING_FIT_ITERATIONS):
        if np.count_nonzero(kept) < max(minimum_rows, degree + 1):
            return None

        coefficients = np.polyfit(
            x_fit[kept],
            y_fit[kept],
            degree,
        )
        residuals = y_fit - np.polyval(coefficients, x_fit)
        residual_median = float(np.median(residuals[kept]))
        median_absolute_deviation = float(
            np.median(np.abs(residuals[kept] - residual_median))
        )
        threshold = max(
            PLANNING_MIN_OUTLIER_THRESHOLD_PX,
            PLANNING_OUTLIER_SIGMA
            * 1.4826
            * median_absolute_deviation,
        )
        next_kept = kept & (
            np.abs(residuals - residual_median) <= threshold
        )
        if np.count_nonzero(next_kept) < max(minimum_rows, degree + 1):
            break
        if np.array_equal(next_kept, kept):
            kept = next_kept
            break
        kept = next_kept

    coefficients = np.polyfit(
        x_fit[kept],
        y_fit[kept],
        degree,
    )
    return coefficients, kept


def smooth_boundary_pair(
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    선택된 경로의 중심선과 폭을 따로 평활화해 톱니 경계를 제거한다.

    결과는 이후 road_clean과 교집합을 취하므로 도로 밖을
    주행가능영역으로 새로 만들지 않는다.
    """

    smoothed_left = left_boundary.copy()
    smoothed_right = right_boundary.copy()
    x_values = np.linspace(
        X_MAX_M,
        X_MIN_M,
        len(left_boundary),
        dtype=np.float32,
    )
    valid = (
        ~np.isnan(left_boundary)
        & ~np.isnan(right_boundary)
        & (x_values >= BOUNDARY_SMOOTH_X_MIN_M)
        & (x_values <= BOUNDARY_SMOOTH_X_MAX_M)
    )
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size < BOUNDARY_SMOOTH_MIN_VALID_ROWS:
        return smoothed_left, smoothed_right

    fit_x = x_values[valid_rows]
    centers = (
        left_boundary[valid_rows]
        + right_boundary[valid_rows]
    ) / 2.0
    widths = (
        right_boundary[valid_rows]
        - left_boundary[valid_rows]
    )

    center_fit = fit_robust_polynomial(
        fit_x,
        centers,
        BOUNDARY_SMOOTH_CENTER_DEGREE,
        BOUNDARY_SMOOTH_MIN_VALID_ROWS,
    )
    width_fit = fit_robust_polynomial(
        fit_x,
        widths,
        BOUNDARY_SMOOTH_WIDTH_DEGREE,
        BOUNDARY_SMOOTH_MIN_VALID_ROWS,
    )
    if center_fit is None or width_fit is None:
        return smoothed_left, smoothed_right

    center_coefficients, center_kept = center_fit
    width_coefficients, width_kept = width_fit
    kept_x = fit_x[center_kept & width_kept]
    if kept_x.size < BOUNDARY_SMOOTH_MIN_VALID_ROWS:
        return smoothed_left, smoothed_right

    evaluation_rows = np.flatnonzero(
        (x_values >= float(np.min(kept_x)))
        & (x_values <= float(np.max(kept_x)))
        & (x_values >= BOUNDARY_SMOOTH_X_MIN_M)
        & (x_values <= BOUNDARY_SMOOTH_X_MAX_M)
    )
    predicted_centers = np.polyval(
        center_coefficients,
        x_values[evaluation_rows],
    )
    predicted_widths = np.clip(
        np.polyval(width_coefficients, x_values[evaluation_rows]),
        ROAD_WIDTH_PX - ROAD_WIDTH_TOLERANCE_PX,
        ROAD_WIDTH_PX + ROAD_WIDTH_TOLERANCE_PX,
    )
    predicted_left = predicted_centers - predicted_widths / 2.0
    predicted_right = predicted_centers + predicted_widths / 2.0
    inside = (
        (predicted_left >= 0.0)
        & (predicted_right < BEV_WIDTH)
        & (predicted_right > predicted_left)
    )
    rows_inside = evaluation_rows[inside]
    smoothed_left[rows_inside] = predicted_left[inside].astype(np.float32)
    smoothed_right[rows_inside] = predicted_right[inside].astype(np.float32)
    return smoothed_left, smoothed_right


def extend_boundary_pair_far_along_marks(
    left: np.ndarray,
    right: np.ndarray,
    boundary_mask: np.ndarray,
    *,
    assoc_m: float = FAR_COURSE_ASSOC_M,
    max_miss_rows: int = FAR_COURSE_MAX_MISS_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """DP tip보다 먼 BEV 행을 마킹에 스냅해 X_MAX(~1.5 m) 쪽으로 연장한다.

    갈림 far에서는 바깥/안쪽 중 한 페인트만 남는 경우가 많다. 관측된 쪽에
    붙이고 반대쪽은 직전 차로폭(또는 track_width) prior로 합성한다.
    """

    left_out = np.asarray(left, dtype=np.float32).copy()
    right_out = np.asarray(right, dtype=np.float32).copy()
    if boundary_mask.size == 0 or left_out.shape[0] != boundary_mask.shape[0]:
        return left_out, right_out

    both = np.flatnonzero(~np.isnan(left_out) & ~np.isnan(right_out))
    if both.size == 0:
        return left_out, right_out

    tip = int(both[0])  # smallest row = farthest valid
    if tip <= 0:
        return left_out, right_out

    assoc_px = float(max(2.0, assoc_m / METERS_PER_PIXEL))
    rail_w_px = float(ROAD_WIDTH_M / METERS_PER_PIXEL)
    min_w_px = float(0.45 * rail_w_px)
    max_w_px = float(2.2 * rail_w_px)
    segments_by_row = find_line_segments_by_row(boundary_mask)

    prev_l = float(left_out[tip])
    prev_r = float(right_out[tip])
    miss = 0

    def nearest_edge(prev_u: float, segments: list[tuple[int, int]]) -> tuple[float, float] | None:
        best_u: float | None = None
        best_d = float("inf")
        for start_u, end_u in segments:
            for cand in (float(start_u), float(end_u)):
                dist = abs(cand - prev_u)
                if dist < best_d:
                    best_d = dist
                    best_u = cand
        if best_u is None or best_d > assoc_px:
            return None
        return best_u, best_d

    for row in range(tip - 1, -1, -1):
        if not np.isnan(left_out[row]) and not np.isnan(right_out[row]):
            prev_l = float(left_out[row])
            prev_r = float(right_out[row])
            miss = 0
            continue

        segments = segments_by_row[row]
        if not segments:
            miss += 1
            if miss > int(max_miss_rows):
                break
            continue

        width = float(np.clip(prev_r - prev_l, min_w_px, max_w_px))
        hit_l = nearest_edge(prev_l, segments)
        hit_r = nearest_edge(prev_r, segments)

        next_l: float | None = None
        next_r: float | None = None
        if hit_l is not None and hit_r is not None:
            next_l, next_r = float(hit_l[0]), float(hit_r[0])
            if next_r <= next_l:
                # Same paint blob claimed both → keep rail from better hit.
                if hit_l[1] <= hit_r[1]:
                    next_r = next_l + width
                else:
                    next_l = next_r - width
        elif hit_l is not None:
            next_l = float(hit_l[0])
            next_r = next_l + width
        elif hit_r is not None:
            next_r = float(hit_r[0])
            next_l = next_r - width
        else:
            # Corridor mid between two marks (stem → open fork recovery).
            mid = 0.5 * (prev_l + prev_r)
            centers = sorted(
                (0.5 * (float(s) + float(e)), float(s), float(e))
                for s, e in segments
            )
            leftish = [c for c in centers if c[0] <= mid + assoc_px]
            rightish = [c for c in centers if c[0] >= mid - assoc_px]
            if leftish and rightish and leftish[-1][2] < rightish[0][1]:
                next_l = leftish[-1][2]  # right edge of left mark
                next_r = rightish[0][1]  # left edge of right mark
                if next_r <= next_l:
                    next_l = next_r = None

        if (
            next_l is None
            or next_r is None
            or next_r <= next_l
            or not (min_w_px <= (next_r - next_l) <= max_w_px)
        ):
            miss += 1
            if miss > int(max_miss_rows):
                break
            continue

        # Only refuse values outside the image. Soft side-margin cuts destroy
        # good mid-curve fits on in_roundabout_exit — tip skate is trimmed later
        # by paint association (see finalize_fork_lane_pair_tips).
        if next_l < 0.0 or next_r > float(BEV_WIDTH - 1):
            break

        left_out[row] = float(np.clip(next_l, 0.0, float(BEV_WIDTH - 1)))
        right_out[row] = float(np.clip(next_r, 0.0, float(BEV_WIDTH - 1)))
        prev_l = float(left_out[row])
        prev_r = float(right_out[row])
        miss = 0

    return left_out, right_out


def _column_hard_wall_skate_cut(
    columns_u: np.ndarray,
    *,
    margin_px: float = SIDE_WALL_HARD_MARGIN_PX,
    min_skate_rows: int = 4,
) -> np.ndarray:
    """Nan far rows only when u is pinned on the absolute FOV edge (≥N rows)."""

    out = np.asarray(columns_u, dtype=np.float32).copy()
    valid = np.flatnonzero(~np.isnan(out))
    if valid.size < min_skate_rows + 1:
        return out
    lo = float(margin_px)
    hi = float(BEV_WIDTH - 1) - float(margin_px)
    # Walk near→far; remember first hard-wall contact, then require a run of
    # farther rows still on the wall (true skate), else leave alone.
    first_wall: int | None = None
    for row in valid[::-1]:
        u = float(out[int(row)])
        if u <= lo or u >= hi:
            first_wall = int(row)
            break
    if first_wall is None:
        return out
    skate = valid[valid < first_wall]
    if skate.size < int(min_skate_rows):
        return out
    # Keep the first wall contact as tip; drop everything farther.
    out[:first_wall] = np.nan
    return out


def _paint_segments_on_row(mark_mask: np.ndarray, row: int) -> list[tuple[int, int]]:
    if mark_mask.size == 0 or row < 0 or row >= int(mark_mask.shape[0]):
        return []
    return find_line_segments(mark_mask[row] > 0)


def _pick_paint_u_near(
    segments: list[tuple[int, int]],
    prev_u: float,
    *,
    assoc_px: float,
    mode: str,
) -> float | None:
    """Pick a paint column near ``prev_u`` (``left_edge``/``right_edge``/``inner``)."""

    if not segments:
        return None
    best_seg: tuple[int, int] | None = None
    best_d = float("inf")
    for start_u, end_u in segments:
        for cand in (
            float(start_u),
            float(end_u),
            0.5 * (float(start_u) + float(end_u)),
        ):
            dist = abs(cand - float(prev_u))
            if dist < best_d:
                best_d = dist
                best_seg = (int(start_u), int(end_u))
    if best_seg is None or best_d > float(assoc_px):
        return None
    s, e = best_seg
    if mode == "left_edge":
        return float(s)
    if mode == "right_edge":
        return float(e)
    if mode == "inner":
        return 0.5 * (float(s) + float(e))
    return float(s) if abs(float(s) - prev_u) <= abs(float(e) - prev_u) else float(e)


def _extend_column_tip_along_paint(
    columns_u: np.ndarray,
    mark_mask: np.ndarray | None,
    *,
    mode: str,
    assoc_m: float = 0.18,
    max_miss_rows: int = 6,
    band_lo: np.ndarray | None = None,
    band_hi: np.ndarray | None = None,
) -> np.ndarray:
    """Grow tip toward far by following paint; stop when paint is gone."""

    out = np.asarray(columns_u, dtype=np.float32).copy()
    if mark_mask is None or mark_mask.size == 0:
        return out
    if mark_mask.shape[:2] != (BEV_HEIGHT, BEV_WIDTH):
        return out
    assoc_px = float(max(2.0, assoc_m / METERS_PER_PIXEL))
    valid = np.flatnonzero(~np.isnan(out))
    if valid.size == 0:
        return out
    tip = int(valid[0])
    prev = float(out[tip])
    miss = 0
    for row in range(tip - 1, -1, -1):
        segs = _paint_segments_on_row(mark_mask, row)
        segs = _filter_wall_noise_paint_segments(segs)
        clipped: list[tuple[int, int]] = []
        for s, e in segs:
            cs, ce = int(s), int(e)
            if band_lo is not None and not np.isnan(band_lo[row]):
                cs = max(cs, int(np.floor(float(band_lo[row]))))
            if band_hi is not None and not np.isnan(band_hi[row]):
                ce = min(ce, int(np.ceil(float(band_hi[row]))))
            if ce >= cs:
                clipped.append((cs, ce))
        hit = _pick_paint_u_near(clipped, prev, assoc_px=assoc_px, mode=mode)
        if hit is None:
            miss += 1
            if miss > int(max_miss_rows):
                break
            continue
        # Reject lateral teleport between paint blobs (zigzag tip).
        max_step = max(float(assoc_px) * 2.0, 0.14 / METERS_PER_PIXEL)
        if abs(float(hit) - prev) > max_step:
            miss += 1
            if miss > int(max_miss_rows):
                break
            continue
        out[row] = float(np.clip(hit, 0.0, float(BEV_WIDTH - 1)))
        prev = float(out[row])
        miss = 0
    return out


def _smooth_rail_u_heading_curve(
    columns_u: np.ndarray,
    *,
    window: int = 13,
    tip_hold: bool = True,
) -> np.ndarray:
    """Smooth a BEV u(row) rail as a heading-aware curve in vehicle XY.

    Fit a low-order polynomial y(x) so the rail is a soft continuous curve
    rather than a row-jittered polyline. Tip samples are gently projected
    onto the terminal tangent so outer FOV exits keep lateral heading.
    """

    src = np.asarray(columns_u, dtype=np.float32)
    xy = _boundary_u_to_vehicle_points(src)
    if xy.shape[0] < 6:
        return _nan_moving_average(src, window=max(5, window // 2))

    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    n = int(x.shape[0])
    # Degree scales with support; keep <=3 so we don't invent wiggles.
    deg = 3 if n >= 18 else (2 if n >= 10 else 1)
    deg = min(deg, max(1, n // 6))

    xs_s = x.copy()
    ys_s = y.copy()
    if float(np.ptp(x)) > 0.08:
        try:
            coef = np.polyfit(x, y, deg)
            ys_s = np.polyval(coef, x).astype(np.float64)
            if tip_hold:
                # Soft tip heading: project last samples onto poly tangent ray.
                tip_n = max(3, min(8, n // 5))
                i0 = max(0, n - tip_n - 4)
                i1 = max(i0 + 1, n - tip_n)
                dcoef = np.polyder(coef)
                x_ref = float(x[i1])
                y_ref = float(np.polyval(coef, x_ref))
                slope = float(np.polyval(dcoef, x_ref))
                t = np.array([1.0, slope], dtype=np.float64)
                nrm = float(np.linalg.norm(t))
                if nrm > 1e-6:
                    t_hat = t / nrm
                    for i in range(i1, n):
                        along = float(np.dot(
                            np.array([xs_s[i] - x_ref, ys_s[i] - y_ref]),
                            t_hat,
                        ))
                        along = max(0.0, along)
                        target = np.array([x_ref, y_ref]) + t_hat * along
                        alpha = 0.78
                        xs_s[i] = (1.0 - alpha) * float(xs_s[i]) + alpha * float(target[0])
                        ys_s[i] = (1.0 - alpha) * float(ys_s[i]) + alpha * float(target[1])
        except (np.linalg.LinAlgError, ValueError):
            w = max(5, int(window) | 1)
            half = w // 2
            for i in range(n):
                a = max(0, i - half)
                b = min(n, i + half + 1)
                xs_s[i] = float(np.mean(x[a:b]))
                ys_s[i] = float(np.mean(y[a:b]))
    else:
        w = max(5, int(window) | 1)
        half = w // 2
        for i in range(n):
            a = max(0, i - half)
            b = min(n, i + half + 1)
            xs_s[i] = float(np.mean(x[a:b]))
            ys_s[i] = float(np.mean(y[a:b]))

    # Rasterize smoothed polyline back to u[row].
    out = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    buckets: dict[int, list[float]] = {}
    for i in range(n - 1):
        x0, y0 = float(xs_s[i]), float(ys_s[i])
        x1, y1 = float(xs_s[i + 1]), float(ys_s[i + 1])
        r0 = int(round((X_MAX_M - x0) / METERS_PER_PIXEL))
        r1 = int(round((X_MAX_M - x1) / METERS_PER_PIXEL))
        steps = max(2, abs(r1 - r0) * 2 + 1)
        for t in np.linspace(0.0, 1.0, steps):
            xv = x0 + (x1 - x0) * float(t)
            yv = y0 + (y1 - y0) * float(t)
            row = int(round((X_MAX_M - xv) / METERS_PER_PIXEL))
            if row < 0 or row >= BEV_HEIGHT:
                continue
            u = (BEV_WIDTH - 1) / 2.0 - yv / METERS_PER_PIXEL
            if u < -1.0 or u > float(BEV_WIDTH):
                continue
            buckets.setdefault(row, []).append(float(np.clip(u, 0.0, float(BEV_WIDTH - 1))))
    for row, us in buckets.items():
        out[row] = float(np.median(us))

    # Keep support only where source had samples (don't invent stem length).
    src_valid = ~np.isnan(src)
    if np.any(src_valid):
        first = int(np.flatnonzero(src_valid)[0])
        last = int(np.flatnonzero(src_valid)[-1])
        keep = np.zeros(BEV_HEIGHT, dtype=bool)
        keep[first : last + 1] = True
        out[~keep] = np.nan
        out[:first] = np.nan
        for row in range(first, last + 1):
            if not np.isnan(out[row]) or np.isnan(src[row]):
                continue
            filled = np.nan
            for d in range(1, 6):
                for cand in (row - d, row + d):
                    if 0 <= cand < BEV_HEIGHT and not np.isnan(out[cand]):
                        filled = float(out[cand])
                        break
                if not np.isnan(filled):
                    break
            if np.isnan(filled):
                filled = float(src[row])
            else:
                filled = 0.70 * filled + 0.30 * float(src[row])
            out[row] = float(np.clip(filled, 0.0, float(BEV_WIDTH - 1)))
    return out


def _filter_wall_noise_paint_segments(
    segments: list[tuple[int, int]],
    *,
    wall_margin_px: int = 5,
    min_wall_width: int = 3,
) -> list[tuple[int, int]]:
    """Drop FOV-edge paint flecks that cause outer tip wall-skating."""

    out: list[tuple[int, int]] = []
    left_lim = int(wall_margin_px)
    right_lim = int(BEV_WIDTH - 1 - wall_margin_px)
    for s, e in segments:
        cs, ce = int(s), int(e)
        width = ce - cs + 1
        on_left = cs <= left_lim
        on_right = ce >= right_lim
        # Thin edge flecks OR any segment entirely inside the wall margin.
        if on_left and ce <= left_lim:
            continue
        if on_right and cs >= right_lim:
            continue
        if (on_left or on_right) and width < int(min_wall_width):
            continue
        # Truncate residual wall overhang so outer edge is interior.
        if on_left:
            cs = left_lim + 1
        if on_right:
            ce = right_lim - 1
        if ce >= cs:
            out.append((cs, ce))
    return out


def _track_outer_paint_tip(
    mark_mask: np.ndarray,
    *,
    side: str,
    wall_margin_px: int = 5,
    jump_px: float = 30.0,
    seed_u: float | None = None,
) -> tuple[int, float] | None:
    """Near→far tip of continuous outer (L=leftmost / R=rightmost) paint edge.

    Scopes candidates to the matching lateral half so gore/opposite paint
    cannot steal the track (matters on out_fork white dashes).
    """

    if mark_mask is None or mark_mask.size == 0:
        return None
    if mark_mask.shape[:2] != (BEV_HEIGHT, BEV_WIDTH):
        return None
    mid_u = 0.5 * float(BEV_WIDTH - 1)
    prev: float | None = None
    last: tuple[int, float] | None = None
    miss = 0

    def _side_segs(segs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if side == "right":
            kept = [(s, e) for s, e in segs if float(e) >= mid_u - 8.0]
        else:
            kept = [(s, e) for s, e in segs if float(s) <= mid_u + 8.0]
        return kept

    for row in range(BEV_HEIGHT - 1, -1, -1):
        segs = _side_segs(
            _filter_wall_noise_paint_segments(
                _paint_segments_on_row(mark_mask, row),
                wall_margin_px=wall_margin_px,
            )
        )
        if not segs:
            miss += 1
            if last is not None and miss > 5:
                break
            continue
        miss = 0
        if side == "right":
            if prev is None:
                if seed_u is not None:
                    scored = sorted(
                        segs,
                        key=lambda se: abs(float(se[1]) - float(seed_u)),
                    )
                    s, e = scored[0]
                else:
                    s, e = max(segs, key=lambda se: se[1])
                edge = float(e)
            else:
                cands = [
                    (s, e)
                    for s, e in segs
                    if abs(float(e) - prev) <= float(jump_px)
                    or abs(0.5 * (float(s) + float(e)) - prev) <= float(jump_px)
                ]
                if not cands:
                    break
                s, e = max(cands, key=lambda se: se[1])
                edge = float(e)
        else:
            if prev is None:
                if seed_u is not None:
                    scored = sorted(
                        segs,
                        key=lambda se: abs(float(se[0]) - float(seed_u)),
                    )
                    s, e = scored[0]
                else:
                    s, e = min(segs, key=lambda se: se[0])
                edge = float(s)
            else:
                cands = [
                    (s, e)
                    for s, e in segs
                    if abs(float(s) - prev) <= float(jump_px)
                    or abs(0.5 * (float(s) + float(e)) - prev) <= float(jump_px)
                ]
                if not cands:
                    break
                s, e = min(cands, key=lambda se: se[0])
                edge = float(s)
        prev = edge
        last = (int(row), edge)
    return last


def _blend_tip_to_paint_edge(
    columns_u: np.ndarray,
    *,
    paint_row: int,
    paint_u: float,
    blend_rows: int = 12,
) -> np.ndarray:
    """Ease the last tip samples onto the paint edge to avoid a hard kink."""

    out = np.asarray(columns_u, dtype=np.float32).copy()
    valid = np.flatnonzero(~np.isnan(out))
    if valid.size < 3:
        return out
    tip = int(valid[0])
    # Ego-ward anchor a few rows from tip.
    anchor_idx = min(len(valid) - 1, max(2, int(blend_rows)))
    anchor_row = int(valid[anchor_idx])
    target_row = int(np.clip(paint_row, 0, BEV_HEIGHT - 1))
    target_u = float(np.clip(paint_u, 0.0, float(BEV_WIDTH - 1)))
    # Prefer keeping existing tip row support: only reshape samples tipward of
    # (or near) the paint tip without inventing deeper tip length.
    for k, row in enumerate(valid):
        r = int(row)
        if r > anchor_row:
            continue
        if r < tip:
            continue
        # Blend factor grows tipward.
        span = max(1, anchor_row - tip)
        t = float(anchor_row - r) / float(span)
        t = float(np.clip(t, 0.0, 1.0))
        # Smoothstep.
        w = t * t * (3.0 - 2.0 * t)
        out[r] = (1.0 - w) * float(out[r]) + w * target_u
    # Ensure a sample exists at paint tip when rail already covers it.
    if tip <= target_row <= int(valid[-1]):
        out[target_row] = target_u
    return out


def _clip_outer_tip_past_paint(
    columns_u: np.ndarray,
    mark_mask: np.ndarray | None,
    *,
    side: str,
    wall_px: float = 10.0,
    slack_rows: int = 1,
    seed_u: float | None = None,
) -> np.ndarray:
    """If outer tip wall-skates past tracked paint, cut back to the paint tip.

    Left outers that already stop at a side exit (tip less tipward than paint)
    are left alone — only tipward overshoot is corrected.
    """

    out = np.asarray(columns_u, dtype=np.float32).copy()
    if mark_mask is None or mark_mask.size == 0:
        return out
    # Seed tracker from a near-ego outer sample so opposite-side paint loses.
    if seed_u is None:
        valid0 = np.flatnonzero(~np.isnan(out))
        if valid0.size:
            seed_u = float(out[int(valid0[min(len(valid0) - 1, valid0.size // 2)])])
    paint = _track_outer_paint_tip(mark_mask, side=side, seed_u=seed_u)
    if paint is None:
        return out
    paint_row, paint_u = paint
    valid = np.flatnonzero(~np.isnan(out))
    if valid.size == 0:
        return out
    tip_row = int(valid[0])
    tip_u = float(out[tip_row])
    on_wall = (
        tip_u >= float(BEV_WIDTH - 1) - float(wall_px)
        if side == "right"
        else tip_u <= float(wall_px)
    )
    # Already at/inside paint tip: optionally ease onto paint edge.
    if tip_row >= int(paint_row) - int(slack_rows):
        if abs(tip_u - float(paint_u)) > 4.0 and tip_row <= int(paint_row) + 4:
            return _blend_tip_to_paint_edge(
                out, paint_row=int(paint_row), paint_u=float(paint_u)
            )
        return out
    # Only retract tipward *wall-skate* past paint. Forward top tips without
    # paint (out_fork) must remain — paint often fades before FOV top.
    if not on_wall:
        return out
    cut = max(0, int(paint_row) - int(slack_rows))
    out[:cut] = np.nan
    tip_now = np.flatnonzero(~np.isnan(out))
    if tip_now.size == 0:
        out[cut] = float(np.clip(paint_u, 0.0, float(BEV_WIDTH - 1)))
        return out
    return _blend_tip_to_paint_edge(
        out, paint_row=int(paint_row), paint_u=float(paint_u)
    )


def _polish_outer_wall_tip_hook(
    columns_u: np.ndarray,
    *,
    wall_px: float = 10.0,
    max_drop: int = 6,
) -> np.ndarray:
    """Remove a short vertical hook after an outer already hit the FOV side."""

    out = np.asarray(columns_u, dtype=np.float32).copy()
    valid = np.flatnonzero(~np.isnan(out))
    if valid.size < 4:
        return out
    tip_u = float(out[int(valid[0])])
    on_left = tip_u <= float(wall_px)
    on_right = tip_u >= float(BEV_WIDTH - 1) - float(wall_px)
    if not (on_left or on_right):
        return out
    k = 0
    limit = min(int(max_drop), len(valid) - 2)
    while k < limit:
        u0 = float(out[int(valid[k])])
        u1 = float(out[int(valid[k + 1])])
        if abs(u0 - u1) > 1.5:
            break
        if on_left and min(u0, u1) > float(wall_px) + 2.0:
            break
        if on_right and max(u0, u1) < float(BEV_WIDTH - 1) - float(wall_px) - 2.0:
            break
        k += 1
    if k >= 2:
        cut = int(valid[k])
        out[:cut] = np.nan
    return out


def _trim_outer_tip_stick_up(
    columns_u: np.ndarray,
    *,
    min_run: int = 6,
    wall_px: float = 10.0,
    tip_row_max: int = 12,
) -> np.ndarray:
    """Drop *top-wall skate* only: tip near BEV top with u glued to FOV sides.

    Do not trim normal side exits whose tips sit mid-far (e.g. row 20–60);
    those are the desired L/R FOV exits on curved fork outs.
    """

    out = np.asarray(columns_u, dtype=np.float32).copy()
    valid = np.flatnonzero(~np.isnan(out))
    if valid.size < min_run + 1:
        return out
    tip_row = int(valid[0])
    tip_u = float(out[tip_row])
    if tip_row > int(tip_row_max):
        return out
    on_left = tip_u <= float(wall_px)
    on_right = tip_u >= float(BEV_WIDTH - 1) - float(wall_px)
    if not (on_left or on_right):
        return out
    k = 0
    while k + 1 < len(valid):
        u0 = float(out[int(valid[k])])
        u1 = float(out[int(valid[k + 1])])
        if abs(u0 - u1) > 1.0:
            break
        if on_left and min(u0, u1) > float(wall_px) + 2.0:
            break
        if on_right and max(u0, u1) < float(BEV_WIDTH - 1) - float(wall_px) - 2.0:
            break
        k += 1
    if k >= int(min_run) - 1:
        cut = int(valid[min(k, len(valid) - 1)])
        out[:cut] = np.nan
    return out


def _fork_tip_mode_for_mark_color(mark_color: str) -> str:
    """Map fork source color → tip finalize profile.

    * ``in_curve`` — yellow In-course curved exits (side FOV outers).
    * ``out_forward`` — white/road_split Out-course forward tips (top FOV).
    """

    color = str(mark_color or "")
    if color.endswith("_marks"):
        color = color[: -len("_marks")]
    if color in ("yellow", "yellow_alt"):
        return "in_curve"
    return "out_forward"


def _rebuild_fork_centers_from_rails(
    lo: np.ndarray,
    li: np.ndarray,
    ro: np.ndarray,
    ri: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * (FORK_PAIR_WIDTH_M / METERS_PER_PIXEL)
    c0 = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    c1 = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    for row in range(BEV_HEIGHT):
        if not np.isnan(lo[row]) and not np.isnan(li[row]):
            c0[row] = 0.5 * (float(lo[row]) + float(li[row]))
        elif not np.isnan(lo[row]):
            c0[row] = float(lo[row]) + half
        if not np.isnan(ro[row]) and not np.isnan(ri[row]):
            c1[row] = 0.5 * (float(ro[row]) + float(ri[row]))
        elif not np.isnan(ro[row]):
            c1[row] = float(ro[row]) - half
        if np.isnan(lo[row]):
            c0[row] = np.nan
        if np.isnan(ro[row]):
            c1[row] = np.nan
    return c0, c1


def _pack_fork_pairs_from_rails(
    by: dict[int, ForkLanePair],
    lo: np.ndarray,
    li: np.ndarray,
    ro: np.ndarray,
    ri: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
) -> list[ForkLanePair]:
    out: list[ForkLanePair] = []
    for rank, outer, inner, center in (
        (0, lo, li, c0),
        (1, ro, ri, c1),
    ):
        valid = int(np.count_nonzero(~np.isnan(center)))
        if valid == 0:
            valid = int(np.count_nonzero(~np.isnan(outer)))
        conf = float(np.clip(valid / float(BEV_HEIGHT), 0.0, 1.0))
        src = by[rank]
        out.append(
            ForkLanePair(
                lateral_rank=rank,
                outer_u=outer.astype(np.float32, copy=False),
                inner_u=inner.astype(np.float32, copy=False),
                center_u=center.astype(np.float32, copy=False),
                outer_missing=bool(src.outer_missing),
                inner_missing=bool(np.any(np.isnan(inner) & ~np.isnan(outer))),
                confidence=conf,
            )
        )
    return out


def _extend_fork_rails_on_paint(
    lo: np.ndarray,
    li: np.ndarray,
    ro: np.ndarray,
    ri: np.ndarray,
    mark_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    for row in range(BEV_HEIGHT):
        if not np.isnan(lo[row]) and not np.isnan(ro[row]) and ro[row] > lo[row]:
            mid[row] = 0.5 * (float(lo[row]) + float(ro[row]))

    lo = _extend_column_tip_along_paint(lo, mark_mask, mode="nearest", band_hi=mid)
    ro = _extend_column_tip_along_paint(ro, mark_mask, mode="nearest", band_lo=mid)
    for row in range(BEV_HEIGHT):
        if not np.isnan(lo[row]) and not np.isnan(ro[row]) and ro[row] > lo[row]:
            mid[row] = 0.5 * (float(lo[row]) + float(ro[row]))
        elif not np.isnan(lo[row]) and np.isnan(mid[row]):
            mid[row] = float(lo[row]) + 0.5 * (FORK_PAIR_WIDTH_M / METERS_PER_PIXEL)
        elif not np.isnan(ro[row]) and np.isnan(mid[row]):
            mid[row] = float(ro[row]) - 0.5 * (FORK_PAIR_WIDTH_M / METERS_PER_PIXEL)

    li = _extend_column_tip_along_paint(
        li, mark_mask, mode="nearest", band_lo=lo, band_hi=mid
    )
    ri = _extend_column_tip_along_paint(
        ri, mark_mask, mode="nearest", band_lo=mid, band_hi=ro
    )
    return lo, li, ro, ri


def _finalize_fork_tips_in_curve(
    pairs: list[ForkLanePair],
    mark_mask: np.ndarray,
) -> list[ForkLanePair]:
    """In-course (yellow): side-exit outers + heading/paint tip surgery."""

    by = {int(p.lateral_rank): p for p in pairs}
    lo = np.asarray(by[0].outer_u, dtype=np.float32).copy()
    li = np.asarray(by[0].inner_u, dtype=np.float32).copy()
    ro = np.asarray(by[1].outer_u, dtype=np.float32).copy()
    ri = np.asarray(by[1].inner_u, dtype=np.float32).copy()

    lo, li, ro, ri = _extend_fork_rails_on_paint(lo, li, ro, ri, mark_mask)

    # Heading-aware curve smooth (poly y(x) + tip tangent).
    lo = _smooth_rail_u_heading_curve(lo, window=15)
    ro = _smooth_rail_u_heading_curve(ro, window=15)
    li = _smooth_rail_u_heading_curve(li, window=15)
    ri = _smooth_rail_u_heading_curve(ri, window=15)
    lo = _trim_outer_tip_stick_up(lo, min_run=6)
    ro = _trim_outer_tip_stick_up(ro, min_run=6)
    lo = _polish_outer_wall_tip_hook(lo, max_drop=6)
    ro = _polish_outer_wall_tip_hook(ro, max_drop=6)
    lo = _clip_outer_tip_past_paint(lo, mark_mask, side="left")
    ro = _clip_outer_tip_past_paint(ro, mark_mask, side="right")
    lo = _smooth_rail_u_heading_curve(lo, window=9, tip_hold=False)
    ro = _smooth_rail_u_heading_curve(ro, window=9, tip_hold=False)
    lo = _clip_outer_tip_past_paint(lo, mark_mask, side="left")
    ro = _clip_outer_tip_past_paint(ro, mark_mask, side="right")

    c0, c1 = _rebuild_fork_centers_from_rails(lo, li, ro, ri)
    c0 = _smooth_rail_u_heading_curve(c0, window=11)
    c1 = _smooth_rail_u_heading_curve(c1, window=11)
    for row in range(BEV_HEIGHT):
        if np.isnan(lo[row]):
            c0[row] = np.nan
        if np.isnan(ro[row]):
            c1[row] = np.nan
    return _pack_fork_pairs_from_rails(by, lo, li, ro, ri, c0, c1)


def _finalize_fork_tips_out_forward(
    pairs: list[ForkLanePair],
    mark_mask: np.ndarray,
) -> list[ForkLanePair]:
    """Out-course (white/road_split): keep stem straight, forward top tips.

    Matches the P0/H0/A0 family: light MA + soft side-wall skate cut only.
    Preserve stitch shared-stem *centers* (do not rebuild mid(outer,inner) —
    that re-splits the stem into an early Y / zig-zag).
    """

    del mark_mask  # reserved for future out-specific paint polish
    by = {int(p.lateral_rank): p for p in pairs}
    lo = np.asarray(by[0].outer_u, dtype=np.float32).copy()
    li = np.asarray(by[0].inner_u, dtype=np.float32).copy()
    ro = np.asarray(by[1].outer_u, dtype=np.float32).copy()
    ri = np.asarray(by[1].inner_u, dtype=np.float32).copy()
    c0 = np.asarray(by[0].center_u, dtype=np.float32).copy()
    c1 = np.asarray(by[1].center_u, dtype=np.float32).copy()

    lo = _nan_moving_average(lo, window=5)
    ro = _nan_moving_average(ro, window=5)
    li = _nan_moving_average(li, window=5)
    ri = _nan_moving_average(ri, window=5)
    c0 = _nan_moving_average(c0, window=5)
    c1 = _nan_moving_average(c1, window=5)
    lo = _column_hard_wall_skate_cut(lo)
    ro = _column_hard_wall_skate_cut(ro)
    for row in range(BEV_HEIGHT):
        if np.isnan(lo[row]):
            c0[row] = np.nan
        if np.isnan(ro[row]):
            c1[row] = np.nan
    return _pack_fork_pairs_from_rails(by, lo, li, ro, ri, c0, c1)


def finalize_fork_lane_pair_tips(
    pairs: list[ForkLanePair],
    mark_mask: np.ndarray | None = None,
    *,
    lateral_heading_deg: float = 40.0,
    tip_mode: str = "in_curve",
) -> list[ForkLanePair]:
    """Finalize fork rail tips with a course-specific profile.

    * ``tip_mode="in_curve"`` — yellow In exits (side FOV, heading/paint tips).
    * ``tip_mode="out_forward"`` — white/road_split Out forks (top FOV tips).
    Without a mark mask, return pairs unchanged.
    """

    del lateral_heading_deg
    if not pairs:
        return pairs
    if mark_mask is None or getattr(mark_mask, "size", 0) == 0:
        return pairs
    if mark_mask.shape[:2] != (BEV_HEIGHT, BEV_WIDTH):
        return pairs

    by = {int(p.lateral_rank): p for p in pairs}
    if 0 not in by or 1 not in by:
        return pairs

    mode = str(tip_mode or "in_curve")
    if mode == "out_forward":
        return _finalize_fork_tips_out_forward(pairs, mark_mask)
    return _finalize_fork_tips_in_curve(pairs, mark_mask)


# Backward-compatible aliases (older sweep scripts).
def clip_boundary_u_at_side_wall(
    columns_u: np.ndarray,
    *,
    margin_px: float | None = None,
    keep_wall_tip: bool = True,
) -> np.ndarray:
    del keep_wall_tip
    margin = float(
        SIDE_WALL_HARD_MARGIN_PX if margin_px is None else margin_px
    )
    return _column_hard_wall_skate_cut(columns_u, margin_px=margin)


def clip_fork_lane_pairs_at_side_wall(
    pairs: list[ForkLanePair],
    *,
    margin_px: float | None = None,
) -> list[ForkLanePair]:
    """Deprecated alias — prefer :func:`finalize_fork_lane_pair_tips`."""

    del margin_px
    return finalize_fork_lane_pair_tips(pairs, mark_mask=None)


def interpolate_yellow_boundary_pair(
    raw_left: np.ndarray,
    raw_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """전역 추적으로 같은 가지로 판단된 노란 경계의 짧은 공백만 메운다."""

    left = raw_left.copy()
    right = raw_right.copy()
    valid_rows = np.flatnonzero(
        ~np.isnan(raw_left) & ~np.isnan(raw_right)
    )

    if valid_rows.size < 2:
        return left, right

    for start_v, end_v in zip(valid_rows[:-1], valid_rows[1:]):
        missing_rows = int(end_v - start_v - 1)
        if missing_rows <= 0 or missing_rows > YELLOW_SPATIAL_GAP_ROWS:
            continue

        left_shift = abs(float(raw_left[end_v]) - float(raw_left[start_v]))
        right_shift = abs(float(raw_right[end_v]) - float(raw_right[start_v]))
        allowed_shift = min(
            MAX_BOUNDARY_SHIFT_PX,
            BOUNDARY_BASE_SHIFT_PX
            + BOUNDARY_SHIFT_PER_ROW_PX * (end_v - start_v),
        )
        if left_shift > allowed_shift or right_shift > allowed_shift:
            continue

        left[start_v:end_v + 1] = np.linspace(
            float(raw_left[start_v]),
            float(raw_left[end_v]),
            end_v - start_v + 1,
            dtype=np.float32,
        )
        right[start_v:end_v + 1] = np.linspace(
            float(raw_right[start_v]),
            float(raw_right[end_v]),
            end_v - start_v + 1,
            dtype=np.float32,
        )

    return left, right


def build_global_boundary_course(
    boundary_mask: np.ndarray,
    reference_centerline: np.ndarray | None = None,
    temporal_centerline: np.ndarray | None = None,
    temporal_left: np.ndarray | None = None,
    temporal_right: np.ndarray | None = None,
    required_side: str | None = None,
    use_yellow_gap_limit: bool = True,
    smooth_course: bool = False,
    opposite_line_mask: np.ndarray | None = None,
    is_ego_course: bool = False,
    use_single_line_angle_bias: bool = False,
    side_debug: dict[str, object] | None = None,
    boundary_segments_by_row: list[list[tuple[int, int]]] | None = None,
    opposite_segments_by_row: list[list[tuple[int, int]]] | None = None,
    find_alternate: bool = False,
    drivable_mask: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """행별 후보를 전체 경로로 연결해 교차로에서도 연속적인 경계를 만든다.

    Returns
    -------
    left_observed, right_observed, left, right, alt_left, alt_right
        ``alt_*`` are empty (NaN) unless ``find_alternate`` is True.
    """

    (
        raw_left,
        raw_right,
        left_observed,
        right_observed,
        alt_raw_left,
        alt_raw_right,
        _,
        _,
    ) = track_boundary_path(
        boundary_mask,
        reference_centerline,
        temporal_centerline,
        temporal_left,
        temporal_right,
        required_side,
        opposite_line_mask,
        is_ego_course,
        use_single_line_angle_bias,
        side_debug,
        boundary_segments_by_row,
        opposite_segments_by_row,
        find_alternate=find_alternate,
        drivable_mask=drivable_mask,
    )

    def gap_fill(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if use_yellow_gap_limit:
            return interpolate_yellow_boundary_pair(left, right)
        return interpolate_boundary_pair(left, right)

    def finalize(
        left: np.ndarray,
        right: np.ndarray,
        *,
        keep_nearest: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if keep_nearest:
            left, right = keep_nearest_continuous_run(left, right)
        # Metric BEV goes to X_MAX(~1.5 m); DP tip often dies earlier when the
        # fork opens — snap the remaining far rows onto the same mark mask.
        left, right = extend_boundary_pair_far_along_marks(
            left, right, boundary_mask
        )
        if smooth_course:
            left, right = smooth_boundary_pair(left, right)
        return left, right

    filled_left, filled_right = gap_fill(raw_left, raw_right)
    interpolated_left, interpolated_right = finalize(filled_left, filled_right)

    alt_interpolated_left = np.full_like(raw_left, np.nan)
    alt_interpolated_right = np.full_like(raw_right, np.nan)
    if find_alternate:
        # Keep far fork segment (do not drop with keep_nearest); stitch stem
        # onto primary after the last alt-valid row (WonJung merge tip).
        alt_filled_left, alt_filled_right = gap_fill(alt_raw_left, alt_raw_right)
        alt_valid_rows = np.flatnonzero(
            ~np.isnan(alt_filled_left) & ~np.isnan(alt_filled_right)
        )
        if alt_valid_rows.size > 0:
            merge_row = int(alt_valid_rows[-1])
            tail_rows = np.arange(merge_row + 1, len(alt_filled_left))
            fillable = tail_rows[
                ~np.isnan(filled_left[tail_rows])
                & ~np.isnan(filled_right[tail_rows])
            ]
            alt_filled_left[fillable] = filled_left[fillable]
            alt_filled_right[fillable] = filled_right[fillable]
        alt_interpolated_left, alt_interpolated_right = finalize(
            alt_filled_left, alt_filled_right, keep_nearest=False
        )

    return (
        left_observed,
        right_observed,
        interpolated_left,
        interpolated_right,
        alt_interpolated_left,
        alt_interpolated_right,
    )


def boundary_to_vehicle_points(boundary: np.ndarray) -> np.ndarray:
    """BEV 행별 경계를 차량 기준 [x 전방, y 왼쪽] 점으로 변환한다."""

    rows = np.flatnonzero(~np.isnan(boundary))
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    x_forward = X_MAX_M - rows.astype(np.float32) * METERS_PER_PIXEL
    y_left = (
        (BEV_WIDTH - 1) / 2.0 - boundary[rows].astype(np.float32)
    ) * METERS_PER_PIXEL
    points = np.column_stack((x_forward, y_left)).astype(np.float32)

    # Path 소비자가 차량 가까운 점부터 받도록 x 오름차순으로 정렬한다.
    return points[np.argsort(points[:, 0])]


def make_boundary_result(
    boundary: np.ndarray,
    observed_rows_mask: np.ndarray,
) -> LaneBoundary:
    """보간 경계와 실제 관측 행을 함께 요약한다."""

    points = boundary_to_vehicle_points(boundary)
    interpolated_rows = int(np.count_nonzero(~np.isnan(boundary)))
    observed_rows = int(
        np.count_nonzero(observed_rows_mask & ~np.isnan(boundary))
    )
    detected = observed_rows >= 3

    if interpolated_rows == 0:
        confidence = 0.0
    else:
        observed_ratio = min(1.0, observed_rows / interpolated_rows)
        length_score = min(1.0, interpolated_rows / max(1, MIN_COURSE_RUN_ROWS))
        confidence = 0.7 * observed_ratio + 0.3 * length_score

    return LaneBoundary(
        points=points,
        detected=detected,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
    )


def make_lane_marking(
    marking_id: int,
    color: int,
    boundary: LaneBoundary,
) -> LaneMarking | None:
    """검출 경계를 LaneMarking.msg 의미의 후보 하나로 변환한다."""

    if not boundary.detected or len(boundary.points) < 2:
        return None

    points_xy = boundary.points.astype(np.float32)
    median_y = float(np.median(points_xy[:, 1]))
    center_threshold_m = 0.05
    if median_y > center_threshold_m:
        side_hint = LaneMarking.SIDE_LEFT
    elif median_y < -center_threshold_m:
        side_hint = LaneMarking.SIDE_RIGHT
    else:
        side_hint = LaneMarking.SIDE_CENTER

    differences = np.diff(points_xy, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    length = float(np.sum(segment_lengths))

    direction = points_xy[-1] - points_xy[0]
    heading = float(np.arctan2(direction[1], direction[0]))

    curvature = 0.0
    x_values = points_xy[:, 0].astype(np.float64)
    y_values = points_xy[:, 1].astype(np.float64)
    if len(points_xy) >= 6 and float(np.ptp(x_values)) >= 0.10:
        coefficients = np.polyfit(x_values, y_values, 2)
        evaluation_x = float(np.median(x_values))
        first_derivative = (
            2.0 * coefficients[0] * evaluation_x + coefficients[1]
        )
        second_derivative = 2.0 * coefficients[0]
        curvature = float(
            second_derivative
            / ((1.0 + first_derivative**2) ** 1.5)
        )

    points_xyz = np.column_stack(
        (points_xy, np.zeros(len(points_xy), dtype=np.float32))
    ).astype(np.float32)
    return LaneMarking(
        id=marking_id,
        color=color,
        side_hint=side_hint,
        confidence=boundary.confidence,
        length=length,
        heading=heading,
        curvature=curvature,
        points=points_xyz,
    )


def aggregate_confidence(
    lanes: list[LaneMarking],
    *,
    color: int | None = None,
    side: int | None = None,
) -> float:
    """지정 색상 또는 차량 기준 위치 후보 중 최고 신뢰도를 반환한다."""

    values = [
        lane.confidence
        for lane in lanes
        if (color is None or lane.color == color)
        and (side is None or lane.side_hint == side)
    ]
    return float(max(values, default=0.0))


def update_yellow_flag(observed_rows: int) -> bool:
    """노란선 검출 플래그를 매 프레임 안정적으로 갱신한다."""

    global yellow_flag_on_count
    global yellow_flag_off_count
    global yellow_flag

    visible = observed_rows >= YELLOW_MIN_VALID_ROWS
    if visible:
        yellow_flag_on_count += 1
        yellow_flag_off_count = 0
        if yellow_flag_on_count >= YELLOW_FLAG_ON_FRAMES:
            yellow_flag = True
    else:
        yellow_flag_on_count = 0
        yellow_flag_off_count += 1
        if yellow_flag_off_count > YELLOW_FLAG_OFF_FRAMES:
            yellow_flag = False

    return yellow_flag


def reset_tracking_state() -> None:
    """영상 크기나 IPM이 바뀌었을 때 경계 ID 추적 상태를 초기화한다."""

    global last_white_left
    global last_white_right
    global last_yellow_left
    global last_yellow_right
    global yellow_flag_on_count
    global yellow_flag_off_count
    global yellow_flag
    global last_ego_course_color

    last_white_left = None
    last_white_right = None
    last_yellow_left = None
    last_yellow_right = None
    yellow_flag_on_count = 0
    yellow_flag_off_count = 0
    yellow_flag = False
    last_ego_course_color = None


def draw_boundary(
    image: np.ndarray,
    boundary: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """행별 경계 배열을 BEV 영상 위에 그린다."""

    rows = np.flatnonzero(~np.isnan(boundary))
    if rows.size < 2:
        return

    # 실제 배열에 NaN 공백이 남아 있으면 화면에서 임의로 이어 그리지 않는다.
    # 따라서 보간 함수가 메운 구간만 연속선으로 보인다.
    split_indices = np.where(np.diff(rows) > 1)[0] + 1
    for run in np.split(rows, split_indices):
        if run.size < 2:
            continue
        points = np.column_stack(
            (
                np.rint(boundary[run]).astype(np.int32),
                run.astype(np.int32),
            )
        ).reshape((-1, 1, 2))
        cv2.polylines(
            image,
            [points],
            isClosed=False,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )


# 가로 실선(정지선/원형교차로 진입선) 검출 파라미터 (행별 가로 커버리지)
CROSSING_TOP_EXCLUDE_RATIO = 0.25   # 원거리 warp 왜곡 구간(상단) 제외 비율
CROSSING_MIN_SPAN_M = 0.15          # 이보다 좁은 도로 폭 행은 무시
CROSSING_COVERAGE_RATIO = 0.40      # 색이 도로 폭의 이 비율 이상 덮으면 가로선 행
CROSSING_MIN_ROWS = 3

# 검출한 가로 마킹을 도로로 되돌릴 때, 마킹 가장자리의 안티에일리어싱 픽셀
# (흑색도 노랑도 아니라 어디에도 안 잡힌다)까지 덮도록 조금 넓힌다.
CROSSING_FILL_M = 0.04
CROSSING_FILL_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        make_odd(max(3, int(round(CROSSING_FILL_M / METERS_PER_PIXEL)))),
        make_odd(max(3, int(round(2 * CROSSING_FILL_M / METERS_PER_PIXEL)))),
    ),
)               # 최소 이만큼 행이 모여야 실선으로 인정
CROSSING_REMOVAL_MARGIN_M = 0.04    # 세로 경계 추적에서 가로선 위·아래 제거 여유

YELLOW_TEMPORAL_SMOOTH_ALPHA = 0.35
YELLOW_TEMPORAL_SMOOTH_MAX_SHIFT_M = 0.10

DASH_MIN_COMPONENT_AREA_PX = 12
DASH_MAX_FORWARD_GAP_M = 0.3
# 끝점을 잇는 선이 성분의 국소 접선에서 벗어나도 되는 수직 이탈 한계.
# lateral_error = |gap| x sin(정렬오차) 라서 픽셀 단위로 환산된다.
# 너무 작으면(0.001m = 0.25px, 서브픽셀) 모든 연결이 차단되고, 너무 크면
# 노이즈에서 엉뚱한 긴 링크가 생긴다. 0.05m(≈13px)에서 점선+실선이 하나로
# 이어지고 다른 차선과의 교차 연결도 발생하지 않는다.
DASH_MAX_LATERAL_ERROR_M = 0.05
DASH_MAX_HEADING_DIFF_DEG = 30
DASH_MIN_VISIBLE_SUPPORT_RATIO = 0.0005
DASH_MAX_LINE_THICKNESS_PX = 8
DASH_ENDPOINT_TANGENT_LENGTH_M = 0.08
DASH_DIRECTIONAL_EIGEN_RATIO = 2.0
# Max lateral distance (m) from a RoadBranch centerline to keep a dash blob
# for that fork path (tune_lane_detect dash_left / dash_right).
DASH_BRANCH_ASSOC_M = 0.22
# Reject dash links whose endpoint column jump exceeds this (blocks adjacent-lane ties).
DASH_MAX_LATERAL_JUMP_M = 0.18
# Components longer than this (along BEV rows) are treated as solid outers:
# painted into the connected mask but not used as dash-link endpoints.
DASH_LINKABLE_MAX_ROW_SPAN_M = 0.22

# Yellow/white fork marking split (roundabout exit / dashed gore).
# Row-to-row association gate for mark polylines (m).
FORK_TRACK_ASSOC_M = 0.08
# Min rows a mark track must cover to be kept.
FORK_TRACK_MIN_ROWS = 18
# Max row gap when continuing a track through a dashed break.
FORK_TRACK_MAX_ROW_GAP = 12
# Expected path width when synthesizing a missing outer from an inner (m).
FORK_PAIR_WIDTH_M = float(ROAD_WIDTH_M)
# Far-zone fraction of BEV height (top of image = ahead) used to decide "forked".
FORK_FAR_ZONE_RATIO = 0.45
# Near-zone fraction (bottom of BEV = ego) for L/R seed anchors (P1).
FORK_NEAR_ZONE_RATIO = 0.28
# Score bonus for near-zone boundary pairs centered on the ego axis (P1).
EGO_NEAR_CENTER_BONUS = 1.8

# 끝점 링크로 안전성이 확인된 한 체인 안에서만 중심점을 2차 곡선으로 잇는다.
# 튜너의 Residual 25 px는 2배 확대 미리보기 기준이므로 Metric BEV에서는
# 약 12.5 px = 0.05 m다.
DASH_CHAIN_CURVE_DEGREE = 2
DASH_CHAIN_CURVE_MAX_RESIDUAL_M = 0.05
DASH_CHAIN_CURVE_FIT_ITERATIONS = 3
DASH_CHAIN_CURVE_MIN_COMPONENTS = 3


def temporally_smooth_boundary(
    current: np.ndarray,
    previous: np.ndarray | None,
) -> np.ndarray:
    """같은 경계로 볼 수 있는 행만 이전 프레임과 EMA 평활화한다."""

    if previous is None or previous.shape != current.shape:
        return current
    smoothed = current.copy()
    valid = ~np.isnan(current) & ~np.isnan(previous)
    max_shift_px = (
        YELLOW_TEMPORAL_SMOOTH_MAX_SHIFT_M / METERS_PER_PIXEL
    )
    stable = valid & (np.abs(current - previous) <= max_shift_px)
    smoothed[stable] = (
        YELLOW_TEMPORAL_SMOOTH_ALPHA * current[stable]
        + (1.0 - YELLOW_TEMPORAL_SMOOTH_ALPHA) * previous[stable]
    )
    return smoothed


@dataclass(frozen=True)
class DashComponent:
    """노란 점선 연결요소의 중심과 원시 픽셀."""

    label: int
    area: int
    centroid_x: float
    centroid_y: float
    points_xy: np.ndarray
    row_span: int
    thickness: int


def extract_dash_point_mask(boundary_mask: np.ndarray) -> np.ndarray:
    """연결 알고리즘이 받는 유효 노란 성분의 원시 픽셀만 남긴다.

    점들이 서로 닿아 긴 성분이 되어도 숨기지 않는다. 따라서 이 결과와
    연결 결과를 비교하면 HSV/성분 검출 문제인지 링크 판정 문제인지 바로
    구분할 수 있다.
    """

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        boundary_mask,
        connectivity=8,
    )
    point_mask = np.zeros_like(boundary_mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < DASH_MIN_COMPONENT_AREA_PX:
            continue
        row_span = int(stats[label, cv2.CC_STAT_HEIGHT])
        column_span = int(stats[label, cv2.CC_STAT_WIDTH])
        # 가로 정지선의 잔여 조각은 점선 후보에서 제외한다.
        if column_span > max(6, row_span * 2):
            continue
        point_mask[labels == label] = 255
    return point_mask


def connect_dashed_components(
    boundary_mask: np.ndarray,
    visibility_mask: np.ndarray | None = None,
) -> np.ndarray:
    """인접 성분의 실제 마주 보는 끝점을 찾아 노란 점선을 잇는다.

    행 범위가 조금 겹치는 대각선 점선도 연결하며, 둥근 점에는 불안정한
    기울기 검사를 적용하지 않는다. 긴 성분은 연결 끝 주변의 국소 PCA
    방향만 검사해 곡선 실선의 전체 평균 방향 때문에 끊기는 것을 막는다.
    """

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        boundary_mask,
        connectivity=8,
    )
    components: list[DashComponent] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < DASH_MIN_COMPONENT_AREA_PX:
            continue
        rows, columns = np.nonzero(labels == label)
        if rows.size == 0:
            continue
        min_row = int(np.min(rows))
        max_row = int(np.max(rows))
        row_span = max_row - min_row + 1
        column_span = int(np.max(columns) - np.min(columns) + 1)

        # 가로선 잔여물은 세로 점선 연결 대상에서 제외한다.
        if column_span > max(6, row_span * 2):
            continue

        points_xy = np.column_stack((columns, rows)).astype(np.float32)

        _, per_row_counts = np.unique(rows, return_counts=True)
        thickness = int(
            np.clip(
                round(float(np.median(per_row_counts))),
                2,
                DASH_MAX_LINE_THICKNESS_PX,
            )
        )
        components.append(
            DashComponent(
                label=label,
                area=area,
                centroid_x=float(np.mean(columns)),
                centroid_y=float(np.mean(rows)),
                points_xy=points_xy,
                row_span=row_span,
                thickness=thickness,
            )
        )

    max_gap_px = DASH_MAX_FORWARD_GAP_M / METERS_PER_PIXEL
    max_lateral_px = DASH_MAX_LATERAL_ERROR_M / METERS_PER_PIXEL
    max_heading_diff = np.deg2rad(DASH_MAX_HEADING_DIFF_DEG)
    tangent_radius_px = max(
        3.0,
        DASH_ENDPOINT_TANGENT_LENGTH_M / METERS_PER_PIXEL,
    )
    if visibility_mask is None:
        visibility_mask = np.full_like(boundary_mask, 255)
    visibility_support = cv2.dilate(
        visibility_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    def local_axis(
        points_xy: np.ndarray,
        endpoint_xy: np.ndarray,
    ) -> np.ndarray | None:
        """끝 주변이 충분히 길쭉할 때만 신뢰 가능한 국소 축을 반환한다."""

        distances = np.linalg.norm(points_xy - endpoint_xy, axis=1)
        local = points_xy[distances <= tangent_radius_px]
        if local.shape[0] < 6:
            return None
        centered = local - np.mean(local, axis=0)
        covariance = centered.T @ centered / max(1, local.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[-1] <= 1e-6:
            return None
        ratio = eigenvalues[-1] / max(eigenvalues[-2], 1e-6)
        if ratio < DASH_DIRECTIONAL_EIGEN_RATIO:
            return None
        axis = eigenvectors[:, -1]
        return axis / max(float(np.linalg.norm(axis)), 1e-6)

    def fit_chain_curve(
        component_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """연결이 확정된 성분 중심들을 튜너와 같은 방식으로 robust fit한다."""

        if len(component_indices) < DASH_CHAIN_CURVE_MIN_COMPONENTS:
            return None

        ys = np.array(
            [components[index].centroid_y for index in component_indices],
            dtype=np.float64,
        )
        xs = np.array(
            [components[index].centroid_x for index in component_indices],
            dtype=np.float64,
        )
        degree = min(DASH_CHAIN_CURVE_DEGREE, len(component_indices) - 1)
        kept = np.ones(len(component_indices), dtype=bool)
        max_residual_px = DASH_CHAIN_CURVE_MAX_RESIDUAL_M / METERS_PER_PIXEL

        for _ in range(DASH_CHAIN_CURVE_FIT_ITERATIONS):
            if np.count_nonzero(kept) < degree + 1:
                return None
            coefficients = np.polyfit(ys[kept], xs[kept], degree)
            residuals = np.abs(xs - np.polyval(coefficients, ys))
            next_kept = kept & (residuals <= max_residual_px)
            if np.count_nonzero(next_kept) < degree + 1:
                break
            if np.array_equal(next_kept, kept):
                kept = next_kept
                break
            kept = next_kept

        if np.count_nonzero(kept) < degree + 1:
            return None
        return np.polyfit(ys[kept], xs[kept], degree), kept

    # score, lower index, upper index, lower endpoint, upper endpoint
    links: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
    for lower_index, lower in enumerate(components):
        for upper_index, upper in enumerate(components):
            # 중심이 더 위에 있는 성분만 다음 전방 점으로 본다. 기존처럼
            # 두 성분의 min/max 행이 겹친다는 이유만으로 버리지는 않는다.
            if upper.centroid_y >= lower.centroid_y - 1.0:
                continue

            center_delta = np.array(
                (
                    upper.centroid_x - lower.centroid_x,
                    upper.centroid_y - lower.centroid_y,
                ),
                dtype=np.float32,
            )
            center_distance = float(np.linalg.norm(center_delta))
            if center_distance <= 1e-6:
                continue
            forward_axis = center_delta / center_distance

            # 중심 연결 방향으로 가장 돌출된 두 픽셀 = 서로 마주 보는 끝점.
            lower_center = np.array(
                (lower.centroid_x, lower.centroid_y),
                dtype=np.float32,
            )
            upper_center = np.array(
                (upper.centroid_x, upper.centroid_y),
                dtype=np.float32,
            )
            lower_projection = (
                lower.points_xy - lower_center
            ) @ forward_axis
            upper_projection = (
                upper.points_xy - upper_center
            ) @ forward_axis
            lower_endpoint = lower.points_xy[int(np.argmax(lower_projection))]
            upper_endpoint = upper.points_xy[int(np.argmin(upper_projection))]
            gap_vector = upper_endpoint - lower_endpoint
            endpoint_gap = float(np.linalg.norm(gap_vector))
            if endpoint_gap <= 1e-6 or endpoint_gap > max_gap_px:
                continue
            # Reject mostly-sideways hops onto a parallel strand (not forward
            # progress along a curve — those have |Δrow| ≳ |Δcol|).
            jump_px = DASH_MAX_LATERAL_JUMP_M / METERS_PER_PIXEL
            if abs(float(gap_vector[0])) > jump_px and abs(
                float(gap_vector[0])
            ) > 1.15 * abs(float(gap_vector[1])):
                continue
            gap_axis = gap_vector / endpoint_gap

            direction_penalty = 0.0
            lateral_penalty = 0.0
            rejected = False
            for component, endpoint in (
                (lower, lower_endpoint),
                (upper, upper_endpoint),
            ):
                axis = local_axis(component.points_xy, endpoint)
                if axis is None:
                    continue
                alignment = float(
                    np.clip(abs(np.dot(axis, gap_axis)), 0.0, 1.0)
                )
                heading_diff = float(np.arccos(alignment))
                lateral_error = abs(
                    float(
                        axis[0] * gap_vector[1]
                        - axis[1] * gap_vector[0]
                    )
                )
                if (
                    heading_diff > max_heading_diff
                    or lateral_error > max_lateral_px
                ):
                    rejected = True
                    break
                direction_penalty = max(
                    direction_penalty,
                    heading_diff / max(max_heading_diff, 1e-6),
                )
                lateral_penalty = max(
                    lateral_penalty,
                    lateral_error / max(max_lateral_px, 1.0),
                )
            if rejected:
                continue

            sample_count = max(2, int(np.ceil(endpoint_gap)) + 1)
            sample_x = np.rint(
                np.linspace(lower_endpoint[0], upper_endpoint[0], sample_count)
            ).astype(np.int32)
            sample_y = np.rint(
                np.linspace(lower_endpoint[1], upper_endpoint[1], sample_count)
            ).astype(np.int32)
            inside = (
                (sample_x >= 0)
                & (sample_x < boundary_mask.shape[1])
                & (sample_y >= 0)
                & (sample_y < boundary_mask.shape[0])
            )
            if not np.any(inside):
                continue
            support_ratio = float(
                np.count_nonzero(
                    visibility_support[sample_y[inside], sample_x[inside]]
                )
            ) / float(np.count_nonzero(inside))
            if support_ratio < DASH_MIN_VISIBLE_SUPPORT_RATIO:
                continue

            # 실제 끝점 거리를 가장 크게 반영해 중간 점을 건너뛴 먼 링크가
            # 가까운 이웃보다 먼저 선택되지 않도록 한다.
            score = (
                endpoint_gap / max(max_gap_px, 1.0)
                + 0.25 * direction_penalty
                + 0.25 * lateral_penalty
            )
            links.append(
                (
                    score,
                    lower_index,
                    upper_index,
                    lower_endpoint,
                    upper_endpoint,
                )
            )

    # 한 블록이 서로 다른 두 차선으로 연결되지 않도록 1:1 링크만 선택한다.
    used_lower: set[int] = set()
    used_upper: set[int] = set()
    selected_links: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    connected = np.zeros_like(boundary_mask)
    for component in components:
        connected[labels == component.label] = 255
    for _, lower_index, upper_index, lower_endpoint, upper_endpoint in sorted(
        links,
        key=lambda link: link[0],
    ):
        if lower_index in used_lower or upper_index in used_upper:
            continue
        lower = components[lower_index]
        upper = components[upper_index]
        thickness = int(round((lower.thickness + upper.thickness) / 2.0))
        cv2.line(
            connected,
            tuple(np.rint(lower_endpoint).astype(np.int32)),
            tuple(np.rint(upper_endpoint).astype(np.int32)),
            255,
            thickness=max(2, thickness),
            lineType=cv2.LINE_8,
        )
        used_lower.add(lower_index)
        used_upper.add(upper_index)
        selected_links.append(
            (lower_index, upper_index, lower_endpoint, upper_endpoint)
        )

    # 선택 링크는 성분당 전방/후방 하나이고 항상 화면 위쪽으로 향하므로
    # 분기나 순환이 없는 체인이다. 체인별 중심에 2차 곡선을 fit해 원시 성분과
    # 기존 끝점 직선 위에 덧그린다.
    outgoing = {
        lower_index: upper_index
        for lower_index, upper_index, _, _ in selected_links
    }
    incoming = {
        upper_index
        for _, upper_index, _, _ in selected_links
    }
    chain_starts = sorted(set(outgoing) - incoming)
    for start_index in chain_starts:
        chain = [start_index]
        while chain[-1] in outgoing:
            chain.append(outgoing[chain[-1]])

        fitted = fit_chain_curve(chain)
        if fitted is None:
            continue
        coefficients, kept = fitted
        inlier_indices = [
            index
            for index, accepted in zip(chain, kept)
            if accepted
        ]
        y_min = max(
            0,
            int(np.floor(min(components[index].centroid_y for index in inlier_indices))),
        )
        y_max = min(
            boundary_mask.shape[0] - 1,
            int(np.ceil(max(components[index].centroid_y for index in inlier_indices))),
        )
        sample_y = np.arange(y_min, y_max + 1, dtype=np.float64)
        sample_x = np.polyval(coefficients, sample_y)
        valid = (sample_x >= 0.0) & (sample_x < boundary_mask.shape[1])
        curve = np.column_stack(
            (
                np.rint(sample_x[valid]).astype(np.int32),
                sample_y[valid].astype(np.int32),
            )
        )
        if curve.shape[0] < 2:
            continue
        curve_thickness = int(
            round(np.median([components[index].thickness for index in inlier_indices]))
        )
        cv2.polylines(
            connected,
            [curve],
            False,
            255,
            thickness=max(2, curve_thickness),
            lineType=cv2.LINE_8,
        )

    return connected


def make_boundary_preview(
    bev: np.ndarray,
    road_clean: np.ndarray,
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    label: str,
    debug_lines: tuple[str, ...] = (),
) -> np.ndarray:
    """road_clean과 좌우 경계 ID를 한 화면에 겹친다."""

    preview = bev.copy()
    road_overlay = np.zeros_like(preview)
    road_overlay[road_clean > 0] = DRIVABLE_COLOR
    preview = cv2.addWeighted(preview, 1.0, road_overlay, 0.45, 0.0)

    draw_boundary(preview, left_boundary, LEFT_BOUNDARY_COLOR)
    draw_boundary(preview, right_boundary, RIGHT_BOUNDARY_COLOR)

    # 좌우 경계의 중간 = 차선 센터라인
    centerline = centerline_from_boundaries(left_boundary, right_boundary)
    draw_boundary(preview, centerline, CENTERLINE_COLOR)

    cv2.putText(
        preview,
        f"{label}  LEFT=RED  RIGHT=BLUE  CENTER=MAGENTA",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, debug_line in enumerate(debug_lines):
        cv2.putText(
            preview,
            debug_line,
            (4, 32 + 14 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def make_drive_preview(
    bev: np.ndarray,
    road_clean: np.ndarray,
    *,
    white_left: np.ndarray,
    white_right: np.ndarray,
    yellow_left: np.ndarray | None = None,
    yellow_right: np.ndarray | None = None,
    prefer_yellow: bool | None = False,
    fork_active: bool = False,
    fork_lane_pairs: tuple | list = (),
    road_branches: tuple | list = (),
    road_cells: np.ndarray | None = None,
    fork_split_source: str = "",
    ego_road_color: str | None = None,
) -> np.ndarray:
    """Single driving canvas: course centerline + road + fork only when active.

    OUT (prefer_yellow=False): white rails. IN: yellow when present else white.
    Fork rails overlay only while ``fork_active`` — not a permanent branch panel.
    """

    use_yellow = bool(prefer_yellow) and yellow_left is not None and yellow_right is not None
    if use_yellow:
        y_obs = np.isfinite(np.asarray(yellow_left, dtype=np.float32)).sum()
        if y_obs < 5:
            use_yellow = False
    if use_yellow:
        preview = make_boundary_preview(
            bev, road_clean, yellow_left, yellow_right, "YELLOW (IN)"
        )
    else:
        preview = make_boundary_preview(
            bev, road_clean, white_left, white_right, "WHITE (OUT)"
        )

    if fork_active and (fork_lane_pairs or len(list(road_branches or ())) >= 2):
        # Compact overlay — not a third full panel.
        if fork_lane_pairs:
            dbg = LaneDebugFrame(
                bev=bev,
                road_clean=road_clean,
                fork_lane_pairs=tuple(fork_lane_pairs),
                fork_split_source=fork_split_source,
                road_branches=tuple(road_branches or ()),
                ego_road_color=ego_road_color,
                fork_active=True,
            )
            fork_overlay = make_fork_lane_pair_preview(dbg, focus="all")
        else:
            fork_overlay = make_course_cell_preview(
                bev,
                road_cells if road_cells is not None else np.zeros_like(road_clean),
                list(road_branches or ()),
                ego_road_color,
            )
        # Blend fork cues on the right third so rails stay readable.
        w = preview.shape[1]
        x0 = int(w * 0.55)
        blend = preview.copy()
        fo = fork_overlay
        if fo.shape[:2] != preview.shape[:2]:
            fo = cv2.resize(fo, (preview.shape[1], preview.shape[0]))
        blend[:, x0:] = cv2.addWeighted(
            preview[:, x0:], 0.35, fo[:, x0:], 0.65, 0.0
        )
        preview = blend
        cv2.putText(
            preview,
            f"FORK ON  src={fork_split_source or '?'}",
            (4, preview.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def detect(
    frame: np.ndarray,
    *,
    active_branch_rank: int | None = None,
    prefer_yellow: bool | None = None,
    enable_fork: bool = True,
) -> LaneDetections:
    """색상별 좌우 경계, 노란선 플래그와 road_clean을 반환한다.

    ``prefer_yellow`` — 코스 계약. False=Out(흰 갈래·흰 추종), True=In(노란 우선).
    None이면 레거시(ego 색·양쪽 후보). ``active_branch_rank``는 선택 갈래 잠금.
    ``enable_fork`` — False면 marking/cell 갈림을 발행하지 않음 (Out 표지 게이트).
    """

    detections, _debug = detect_with_debug(
        frame,
        active_branch_rank=active_branch_rank,
        prefer_yellow=prefer_yellow,
        enable_fork=enable_fork,
    )
    return detections


def detect_with_debug(
    frame: np.ndarray,
    *,
    active_branch_rank: int | None = None,
    prefer_yellow: bool | None = None,
    enable_fork: bool = True,
) -> tuple[LaneDetections, LaneDebugFrame]:
    """Runtime detections plus intermediate masks for mode tuners."""

    global cached_shape
    global last_white_left
    global last_white_right
    global last_yellow_left
    global last_yellow_right

    if frame is None or frame.size == 0:
        return LaneDetections(), LaneDebugFrame()

    # Alt DP is needed for marking-fork pairs; otherwise only pay for it when viz
    # wants opposing-path overlays (main latency gate).
    find_alternate = bool(enable_fork) or VISUALIZE_MODE in (
        VISUALIZE_CONTROL,
        VISUALIZE_ON,
    )

    original_h, original_w = frame.shape[:2]
    current_shape = (original_w, original_h)

    # BEV 크기는 YAML metric_ipm 기준 고정. 입력 해상도가 바뀌면
    # remap 맵·추적 상태를 갱신한다.
    if cached_shape != current_shape:
        cached_shape = current_shape
        _ensure_ipm_maps(original_w, original_h)
        reset_tracking_state()

    # Metric IPM이 crop_top을 적용하므로 원본 프레임을 그대로 넘긴다.
    hsv_source = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 먼 곳의 하양 오검출은 워프에서 크게 부풀어 근거리 차선을 이겨버린다.
    # 노란 점선은 원래 작은 조각이라 건드리지 않는다.
    white_source = remove_far_specks(
        cv2.inRange(hsv_source, WHITE_LOWER, WHITE_UPPER)
    )
    yellow_source = cv2.inRange(hsv_source, YELLOW_LOWER, YELLOW_UPPER)
    black_source = cv2.inRange(hsv_source, BLACK_LOWER, BLACK_UPPER)
    red_source = _red_inrange(hsv_source)

    white_bev = warp_mask(white_source)
    yellow_bev = warp_mask(yellow_source)
    black_bev = warp_mask(black_source)
    red_bev = warp_mask(red_source)

    # 빨강은 도로/차선 생성 입력에서 완전히 분리한다. red_bev는 이벤트 통계와
    # 독립적인 red_hsv 시각화에만 남기며 road_raw에는 절대 합치지 않는다.
    road_raw = black_bev.copy()
    course_lines = cv2.bitwise_or(white_bev, yellow_bev)
    road_clean = fill_road_surface_holes(road_raw, course_lines)

    # 카메라 화각 밖(BEV 검은 쐐기)은 '도로 없음'이 아니라 '모름'이다.
    # 점선 연결도 도로 HSV가 아니라 실제 카메라 관측 가능 영역만 확인한다.
    observable_bev = bev_observable_mask()
    visibility_bev = (
        observable_bev.astype(np.uint8) * 255
        if observable_bev is not None
        else np.full_like(white_bev, 255)
    )

    # 가로선(crossing) 검출 제거됨 — 정지선/진입선을 경계에서 걸러내지 않는다.

    # 단일선 좌우 판정(도로-방향 투표)용 신호. 빨강(아루코 매트)도 주행가능으로
    # 포함한다 — 좌/우 한 표로만 쓰므로 경계 기하엔 영향 없다. road_clean(검정)만
    # 쓰면 빨강 구간에서 도로 신호가 비어 중심 기준으로 떨어져 좌우가 뒤집혔다.
    drivable_for_vote = cv2.bitwise_or(road_clean, red_bev)

    yellow_boundary_raw_bev = yellow_bev
    yellow_dash_points_bev = (
        extract_dash_point_mask(yellow_boundary_raw_bev)
        if window_enabled("yellow_dash_points")
        else None
    )
    yellow_boundary_bev = connect_dashed_components(
        yellow_boundary_raw_bev,
        visibility_bev,
    )

    white_cut_bev = white_bev
    # 세로형 흰 점선(갈림/합류 가이드) — 가로 진입선은 extract에서 걸러진다.
    white_dash_points_bev = extract_dash_point_mask(white_cut_bev)
    white_dash_connected_bev = connect_dashed_components(
        white_cut_bev,
        visibility_bev,
    )

    # 같은 마스크를 흰/노란 추적, 반대색 검사와 도로 기준선 계산에서
    # 반복 분리하지 않도록 프레임당 한 번만 행 구간 목록을 만든다.
    white_segments_by_row = find_line_segments_by_row(
        white_dash_connected_bev
    )
    yellow_segments_by_row = find_line_segments_by_row(yellow_boundary_bev)

    previous_white_center = None
    if last_white_left is not None and last_white_right is not None:
        previous_white_center = centerline_from_boundaries(
            last_white_left, last_white_right
        )

    # 검정 도로 셀이 아니라 직전 프레임의 실제 좌우 차선으로 ego 코스를 정한다.
    temporal_white_is_ego = boundary_pair_contains_vehicle(
        last_white_left,
        last_white_right,
    )
    temporal_yellow_is_ego = boundary_pair_contains_vehicle(
        last_yellow_left,
        last_yellow_right,
    )

    # 센터라인 흔들림 확정-진단: 켜졌을 때만 흰 경로 후보 카운터를 수집한다.
    white_side_debug: dict[str, object] | None = {} if CENTERLINE_DEBUG else None

    (
        white_left_observed,
        white_right_observed,
        white_left,
        white_right,
        white_alt_left,
        white_alt_right,
    ) = build_global_boundary_course(
        boundary_mask=white_dash_connected_bev,
        temporal_centerline=previous_white_center,
        temporal_left=last_white_left,
        temporal_right=last_white_right,
        side_debug=white_side_debug,
        use_yellow_gap_limit=False,
        smooth_course=True,
        # 흰 차로 안에 노란 차선이 들어앉으면 그건 흰 도로가 아니다.
        # 가로선(정지선/진입선)은 코스 경계가 아니므로 제거한 마스크를 쓴다.
        opposite_line_mask=yellow_boundary_bev,
        is_ego_course=temporal_white_is_ego,
        boundary_segments_by_row=white_segments_by_row,
        opposite_segments_by_row=yellow_segments_by_row,
        find_alternate=find_alternate,
        # 단일선 좌우 판정용 도로 신호(좌/우 한 표만; 경계 기하엔 미사용). 빨강 포함.
        drivable_mask=drivable_for_vote,
    )
    white_centerline = centerline_from_boundaries(white_left, white_right)

    if CENTERLINE_DEBUG:
        _log_centerline_debug(white_centerline, white_side_debug)

    previous_yellow_center = None
    if last_yellow_left is not None and last_yellow_right is not None:
        previous_yellow_center = centerline_from_boundaries(
            last_yellow_left, last_yellow_right
        )

    # 인코스 판정 기준선은 실제 검출한 흰 차선 중심만 사용한다. 흰 선이 없으면
    # 도로 HSV로 대신 만들지 않고 None으로 두어 BEV 중앙/temporal 경로를 쓴다.
    inner_course_reference = (
        white_centerline
        if np.any(~np.isnan(white_centerline))
        else None
    )

    # 흰 경계 둘 다 관측된 행 = 흰 차로 확정. 노란 코스는 그 위에 못 얹는다.
    # inner_course_reference(=흰 중심선)가 center_error/PATH_REFERENCE_PENALTY로
    # 노란 후보를 흰 차로 쪽으로 당기기 때문에, 이 금지 구간이 없으면 오른쪽에
    # 노란선 하나만 보일 때 그 선을 '오른쪽 경계'로 읽어 노란 차로를 흰 차로
    # 위에 통째로 올려놓는다(= 왼쪽선을 오른쪽선으로 인식).
    yellow_side_debug: dict[str, object] | None = (
        {} if window_enabled("yellow_boundaries") else None
    )
    (
        yellow_left_observed,
        yellow_right_observed,
        yellow_left,
        yellow_right,
        yellow_alt_left,
        yellow_alt_right,
    ) = build_global_boundary_course(
        boundary_mask=yellow_boundary_bev,
        # 이미 노란 코스 위를 달리고 있다면 '흰 도로 중심선'은 기준이 아니다.
        # 그걸 기준으로 두면 center_error가 노란 차로를 흰 도로 쪽으로 끌어당겨,
        # 노란선 하나만 보일 때 그 선을 반대쪽 경계로 읽게 만든다.
        # 기준을 None으로 두면 BEV 중앙(=차량)이 기준이 되고, 이건 참이다.
        # 인코스 우선(required_side)도 '진입 전에 어느 노란 코스냐'를 고르는
        # 규칙이지, 이미 그 위에 있을 때 쓰는 규칙이 아니다.
        reference_centerline=(
            None if temporal_yellow_is_ego else inner_course_reference
        ),
        temporal_centerline=previous_yellow_center,
        temporal_left=last_yellow_left,
        temporal_right=last_yellow_right,
        # 시계방향 회전교차로: 도로 중심선 오른쪽의 노란 인코스를 우선한다.
        required_side="right",
        use_yellow_gap_limit=True,
        smooth_course=True,
        # 노란 차로 안에 흰 차선이 들어앉으면 그건 노란 도로가 아니다.
        # (흰 도로 주행 중 옆 노란선 하나만 보일 때 좌/우 모호성을 푼다)
        opposite_line_mask=white_dash_connected_bev,
        is_ego_course=temporal_yellow_is_ego,
        use_single_line_angle_bias=True,
        side_debug=yellow_side_debug,
        boundary_segments_by_row=yellow_segments_by_row,
        opposite_segments_by_row=white_segments_by_row,
        find_alternate=find_alternate,
        # 단일선 좌우 판정용 도로 신호(좌/우 한 표만; 경계 기하엔 미사용). 빨강 포함.
        drivable_mask=drivable_for_vote,
    )

    yellow_left = temporally_smooth_boundary(
        yellow_left,
        last_yellow_left,
    )
    yellow_right = temporally_smooth_boundary(
        yellow_right,
        last_yellow_right,
    )

    # 다음 프레임에서 한 선만 남더라도 직전 경계 쌍의 중심과 연결하여
    # 기존 left/right ID를 우선 유지한다. 도로 HSV는 이 판단에 관여하지 않는다.
    if np.any(~np.isnan(white_left)):
        last_white_left = white_left.copy()
        last_white_right = white_right.copy()
    if np.any(~np.isnan(yellow_left)):
        last_yellow_left = yellow_left.copy()
        last_yellow_right = yellow_right.copy()

    yellow_observed_rows = max(
        int(np.count_nonzero(yellow_left_observed)),
        int(np.count_nonzero(yellow_right_observed)),
    )

    yellow_is_detected = update_yellow_flag(yellow_observed_rows)

    # 갈림길 분기 경로(판단제어 출력용): 차선을 도로에서 빼내 코스별 셀로
    # 자른 뒤, 차량이 있는 셀을 행 간 겹침으로 연결 추적한다. 현재 코스와
    # 같은 색 경계로 갈라질 때만 분기로 본다(흰 도로 주행 중 노란 인코스는
    # 분기가 아니다). 갈림길이 아니면 단일 경로 1개다.
    # 노란선은 점선이므로 연결 처리한 yellow_boundary_bev를 커터로 쓴다.
    # 흰/노란 가로 마킹은 둘 다 제거한 마스크를 쓴다(도로를 끊으면 안 된다).
    # 갈림길 검출 비활성화 — 전체 검출의 ~47%(select_course_fork_pairs) + 셀
    # 추적(build_road_branches_cells)을 건너뛴다. 정상 주행은 centerline 을 쓰고
    # planner 의 fork 로직은 모두 len(branches)>=2 / fork_active 게이트라, branches
    # 를 비워도 안전하게 centerline 주행으로 폴백한다.
    road_branches: list = []
    road_cells = None
    ego_road_color = None
    fork_active = False
    fork_split_source = ""
    fork_lane_pairs: list = []
    fork_mark_tracks: list = []

    # 흰/노란 차선 센터라인(좌우 경계 중점) → base_link 점열
    white_centerline_points = boundary_to_vehicle_points(
        centerline_from_boundaries(white_left, white_right)
    )
    yellow_centerline_points = boundary_to_vehicle_points(
        centerline_from_boundaries(yellow_left, yellow_right)
    )
    # 가로선 검출 제거됨 — crossing 이벤트는 항상 False.
    yellow_crossing_line = False
    white_crossing_line = False

    observable = observable_bev
    if observable is not None and observable.size == red_bev.size:
        obs_count = int(np.count_nonzero(observable))
        red_in_view = int(np.count_nonzero(red_bev[observable]))
        red_coverage = (
            float(red_in_view) / float(obs_count) if obs_count > 0 else 0.0
        )
        red_pixel_count = red_in_view
    else:
        total = max(1, int(red_bev.size))
        red_pixel_count = int(np.count_nonzero(red_bev))
        red_coverage = float(red_pixel_count) / float(total)

    bev_color = warp_metric_ipm(frame, METRIC_IPM_PARAMS)

    boundary_candidates = (
        (
            LaneMarking.COLOR_WHITE,
            make_boundary_result(white_left, white_left_observed),
        ),
        (
            LaneMarking.COLOR_WHITE,
            make_boundary_result(white_right, white_right_observed),
        ),
        (
            LaneMarking.COLOR_YELLOW,
            make_boundary_result(yellow_left, yellow_left_observed),
        ),
        (
            LaneMarking.COLOR_YELLOW,
            make_boundary_result(yellow_right, yellow_right_observed),
        ),
    )
    lanes: list[LaneMarking] = []
    for color, boundary in boundary_candidates:
        lane = make_lane_marking(len(lanes), color, boundary)
        if lane is not None:
            lanes.append(lane)

    white_confidence = aggregate_confidence(
        lanes, color=LaneMarking.COLOR_WHITE
    )
    yellow_confidence = aggregate_confidence(
        lanes, color=LaneMarking.COLOR_YELLOW
    )
    left_confidence = aggregate_confidence(
        lanes, side=LaneMarking.SIDE_LEFT
    )
    right_confidence = aggregate_confidence(
        lanes, side=LaneMarking.SIDE_RIGHT
    )

    detections = LaneDetections(
        lanes=tuple(lanes),
        white_visible=white_confidence > 0.0,
        yellow_visible=yellow_is_detected,
        left_visible=left_confidence > 0.0,
        right_visible=right_confidence > 0.0,
        white_confidence=white_confidence,
        yellow_confidence=yellow_confidence,
        left_confidence=left_confidence,
        right_confidence=right_confidence,
        drivable_area=road_clean.copy(),
        white_centerline=white_centerline_points,
        yellow_centerline=yellow_centerline_points,
        yellow_crossing_line=yellow_crossing_line,
        fork_active=fork_active,
        branches=tuple(road_branches),
    )
    debug = LaneDebugFrame(
        bev=bev_color,
        white_bev=white_bev,
        yellow_bev=yellow_bev,
        red_bev=red_bev,
        black_bev=black_bev,
        road_clean=road_clean,
        road_raw=road_raw,
        yellow_dash_points_bev=yellow_dash_points_bev,
        yellow_connected_bev=yellow_boundary_bev,
        white_dash_points_bev=white_dash_points_bev,
        white_dash_connected_bev=white_dash_connected_bev,
        white_left=white_left,
        white_right=white_right,
        yellow_left=yellow_left,
        yellow_right=yellow_right,
        road_cells=road_cells,
        road_branches=tuple(road_branches),
        ego_road_color=ego_road_color,
        fork_active=fork_active,
        yellow_crossing_line=yellow_crossing_line,
        white_crossing_line=white_crossing_line,
        red_coverage=red_coverage,
        red_pixel_count=red_pixel_count,
        fork_lane_pairs=tuple(fork_lane_pairs),
        fork_mark_tracks=tuple(fork_mark_tracks),
        fork_split_source=fork_split_source,
        prefer_yellow=prefer_yellow,
    )
    # Late import avoids circular dependency (active_lane → this module).
    from inference.modules.active_lane import apply_active_lane_policy

    return apply_active_lane_policy(detections, debug, active_branch_rank)


# =============================================================
# TEMP (A 검증용): 삭제됐던 2갈래 분기 검출을 구멍 메운 마스크 위에서
# 그대로 재확인하기 위한 복원. 확인 뒤 (B) 슬림 재구현 시 정리한다.
# =============================================================

BRANCH_COLORS = (
    (0, 0, 255),
    (255, 0, 0),
    (255, 0, 255),
    (255, 255, 0),
)

# =========================================================
# Drivable-area branch extraction
# =========================================================
MIN_BRANCH_LENGTH_M = 0.15
MIN_BRANCH_WIDTH_M = 0.10

MIN_BRANCH_LENGTH_ROWS = int(round(MIN_BRANCH_LENGTH_M / METERS_PER_PIXEL))
MIN_BRANCH_WIDTH_PX = int(round(MIN_BRANCH_WIDTH_M / METERS_PER_PIXEL))

@dataclass(frozen=True)
class RoadBranch:
    """한 갈래의 센터라인 후보 (코드명 branch; 문서 용어=갈래)."""

    lateral_rank: int = 0
    confidence: float = 0.0
    width: float = 0.0
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )


@dataclass(frozen=True)
class ForkLanePair:
    """한 갈래의 차로 쌍: outer+inner (BEV u) + center.

    문서 용어=차로 쌍. ``lateral_rank`` 0 = 왼쪽 갈래. 배열 길이 BEV_HEIGHT,
    결측은 NaN. ``outer_missing`` / ``inner_missing`` = 폭 prior로 합성.
    """

    lateral_rank: int
    outer_u: np.ndarray
    inner_u: np.ndarray
    center_u: np.ndarray
    outer_missing: bool = False
    inner_missing: bool = False
    confidence: float = 0.0


def track_marking_polylines(mark_mask: np.ndarray) -> list[np.ndarray]:
    """Connect per-row yellow/white mark centers into polylines (near→far).

    Each returned array has shape (BEV_HEIGHT,) with NaN gaps. Designed for
    solid outers and dashed inners after ``connect_dashed_components``.

    Near-zone (BEV bottom) seeds prefer ego L/R anchors: camera is front-center,
    so the first two durable strands should sit near ± half track width.
    """

    if mark_mask.size == 0 or mark_mask.shape != (BEV_HEIGHT, BEV_WIDTH):
        return []

    assoc_px = max(2.0, FORK_TRACK_ASSOC_M / METERS_PER_PIXEL)
    min_width_px = max(1, int(round(0.006 / METERS_PER_PIXEL)))
    max_gap = max(2, int(FORK_TRACK_MAX_ROW_GAP))
    near_row0 = int(round(BEV_HEIGHT * (1.0 - FORK_NEAR_ZONE_RATIO)))
    ego_u = (BEV_WIDTH - 1) / 2.0
    half_w_px = (0.5 * FORK_PAIR_WIDTH_M) / METERS_PER_PIXEL
    seed_targets = (ego_u - half_w_px, ego_u + half_w_px)

    tracks: list[dict] = []
    for row in range(BEV_HEIGHT - 1, -1, -1):
        centers: list[float] = []
        for left, right in find_line_segments(mark_mask[row]):
            if right - left + 1 < min_width_px:
                continue
            centers.append(0.5 * (left + right))
        if not centers:
            continue

        # In the ego near zone, prefer associating / seeding toward L/R anchors
        # so side identity does not flip when one outer leaves the FOV.
        if row >= near_row0 and len(centers) >= 2:
            centers = sorted(
                centers,
                key=lambda c: min(abs(c - seed_targets[0]), abs(c - seed_targets[1])),
            )

        used: set[int] = set()
        for track in tracks:
            if track["last_row"] - row > max_gap:
                continue
            best_i: int | None = None
            best_d = assoc_px
            for idx, center in enumerate(centers):
                if idx in used:
                    continue
                dist = abs(center - float(track["last_u"]))
                if dist <= best_d:
                    best_d = dist
                    best_i = idx
            if best_i is None:
                continue
            used.add(best_i)
            center = centers[best_i]
            track["cols"][row] = center
            track["last_u"] = center
            track["last_row"] = row
            track["hits"] += 1

        for idx, center in enumerate(centers):
            if idx in used:
                continue
            cols = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
            cols[row] = center
            seed_side = ""
            if row >= near_row0:
                d_l = abs(center - seed_targets[0])
                d_r = abs(center - seed_targets[1])
                if min(d_l, d_r) <= assoc_px * 2.5:
                    seed_side = "left" if d_l <= d_r else "right"
            tracks.append(
                {
                    "cols": cols,
                    "last_u": center,
                    "last_row": row,
                    "hits": 1,
                    "seed_side": seed_side,
                }
            )

    kept_meta = [
        track
        for track in tracks
        if int(track["hits"]) >= int(FORK_TRACK_MIN_ROWS)
    ]
    kept_meta.sort(key=lambda track: float(np.nanmedian(track["cols"])))
    return [track["cols"] for track in kept_meta]


def _track_far_median(cols: np.ndarray) -> float:
    far_end = max(1, int(round(BEV_HEIGHT * FORK_FAR_ZONE_RATIO)))
    vals = cols[:far_end]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        vals = cols[~np.isnan(cols)]
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def _boundary_u_to_vehicle_points(columns_u: np.ndarray) -> np.ndarray:
    """BEV column-per-row → base_link Nx2 sorted by increasing x."""

    rows = np.flatnonzero(~np.isnan(columns_u))
    if rows.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    x_forward = X_MAX_M - rows.astype(np.float32) * METERS_PER_PIXEL
    y_left = (
        (BEV_WIDTH - 1) / 2.0 - columns_u[rows].astype(np.float32)
    ) * METERS_PER_PIXEL
    points = np.column_stack((x_forward, y_left)).astype(np.float32)
    return points[np.argsort(points[:, 0])]


def _pair_center_u(
    outer_u: np.ndarray,
    inner_u: np.ndarray,
    *,
    side: str,
    outer_weight: float = 0.5,
) -> tuple[np.ndarray, bool, bool]:
    """Lane center for a **parallel-rail** corridor (fixed track width).

    Prefer ``outer ± half_width`` (11-rail). Equal midpoint only as fallback when
    outer is missing. Do **not** outer-bias the midpoint — that put centers at
    the outer 1/3 and split stem centers from ego.
    """

    del outer_weight  # kept for call-site compat; rails ignore bias
    half_w_px = (0.5 * FORK_PAIR_WIDTH_M) / METERS_PER_PIXEL
    center = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    outer_miss_count = 0
    inner_miss_count = 0
    for row in range(BEV_HEIGHT):
        o = outer_u[row]
        i = inner_u[row]
        o_ok = not np.isnan(o)
        i_ok = not np.isnan(i)
        if o_ok:
            # Parallel-rail center anchored on the stable outer.
            if side == "left":
                center[row] = float(o) + half_w_px
            else:
                center[row] = float(o) - half_w_px
            if not i_ok:
                inner_miss_count += 1
        elif i_ok:
            outer_miss_count += 1
            if side == "left":
                center[row] = float(i) - half_w_px
            else:
                center[row] = float(i) + half_w_px
    return (
        center,
        outer_miss_count > 0 and outer_miss_count >= inner_miss_count,
        inner_miss_count > 0 and inner_miss_count > outer_miss_count,
    )


def _far_zone_track_count(tracks: list[np.ndarray]) -> int:
    far_end = max(1, int(round(BEV_HEIGHT * FORK_FAR_ZONE_RATIO)))
    count = 0
    for cols in tracks:
        if np.any(~np.isnan(cols[:far_end])):
            count += 1
    return count


def _tracks_diverge_ahead(tracks: list[np.ndarray]) -> bool:
    """True when two mark tracks are closer near ego and separate ahead."""

    if len(tracks) != 2:
        return False
    far_end = max(1, int(round(BEV_HEIGHT * FORK_FAR_ZONE_RATIO)))
    near0 = int(round(BEV_HEIGHT * (1.0 - FORK_NEAR_ZONE_RATIO)))
    far_meds = []
    near_meds = []
    for cols in tracks:
        far_vals = cols[:far_end]
        far_vals = far_vals[~np.isnan(far_vals)]
        near_vals = cols[near0:]
        near_vals = near_vals[~np.isnan(near_vals)]
        if far_vals.size == 0 or near_vals.size == 0:
            return False
        far_meds.append(float(np.median(far_vals)))
        near_meds.append(float(np.median(near_vals)))
    far_sep = abs(far_meds[0] - far_meds[1]) * METERS_PER_PIXEL
    near_sep = abs(near_meds[0] - near_meds[1]) * METERS_PER_PIXEL
    return far_sep >= max(0.12, 0.6 * MIN_BRANCH_SEPARATION_M) and far_sep > near_sep + 0.04


def build_fork_lane_pairs_from_tracks(
    tracks: list[np.ndarray],
    mark_mask: np.ndarray | None = None,
    *,
    tip_mode: str = "in_curve",
) -> list[ForkLanePair]:
    """Group sorted mark tracks into left/right (outer, inner) pairs (P3)."""

    if len(tracks) < 2:
        return []

    # Prefer far-zone lateral order so near-zone merge noise does not reorder.
    order = sorted(
        range(len(tracks)),
        key=lambda i: _track_far_median(tracks[i]),
    )
    ordered = [tracks[i] for i in order]

    far_n = _far_zone_track_count(ordered)
    # Need a clear split ahead (3+ marks), 4 polylines, or 2 diverging strands.
    if far_n < 3 and len(ordered) < 4 and not _tracks_diverge_ahead(ordered):
        return []

    pairs: list[ForkLanePair] = []
    if len(ordered) >= 4:
        # Leftmost two + rightmost two (drop middle clutter if 5+).
        left_outer, left_inner = ordered[0], ordered[1]
        right_inner, right_outer = ordered[-2], ordered[-1]
    elif len(ordered) == 3:
        left_outer, left_inner = ordered[0], ordered[1]
        right_inner, right_outer = ordered[1], ordered[2]
    else:
        # Two diverging strands: decide outers-vs-inners by far separation.
        left_a, right_a = ordered[0], ordered[1]
        full_w_px = FORK_PAIR_WIDTH_M / METERS_PER_PIXEL
        far_sep_m = abs(
            _track_far_median(left_a) - _track_far_median(right_a)
        ) * METERS_PER_PIXEL
        left_outer = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
        left_inner = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
        right_inner = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
        right_outer = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
        if far_sep_m >= 1.15 * FORK_PAIR_WIDTH_M:
            # Wide pair (typical white out-fork): observed strands are outers.
            # Stem (outers ~1 lane apart): both path centers share the mid —
            # inner = opposite outer. Forked (outers wider): each path is one
            # full lane from its outer; clamp inners at the gore mid so they
            # do not cross. Never use half_w as a fake inner (that put centers
            # at the 1/4 mark).
            for row in range(BEV_HEIGHT):
                lo = left_a[row]
                ro = right_a[row]
                if np.isnan(lo) or np.isnan(ro) or ro <= lo:
                    if not np.isnan(lo):
                        left_outer[row] = lo
                        left_inner[row] = lo + full_w_px
                    if not np.isnan(ro):
                        right_outer[row] = ro
                        right_inner[row] = ro - full_w_px
                    continue
                left_outer[row] = lo
                right_outer[row] = ro
                sep = ro - lo
                if sep <= full_w_px * 1.2:
                    left_inner[row] = ro
                    right_inner[row] = lo
                else:
                    # Parallel ±w; allow temporary X if parallel would cross.
                    # Mid-clamp used to hang a flat shelf across the gore apex.
                    left_inner[row] = lo + full_w_px
                    right_inner[row] = ro - full_w_px
        else:
            # Tight pair: observed strands are inners; synthesize outers at full width.
            for row in range(BEV_HEIGHT):
                if not np.isnan(left_a[row]):
                    left_inner[row] = left_a[row]
                    left_outer[row] = left_a[row] - full_w_px
                if not np.isnan(right_a[row]):
                    right_inner[row] = right_a[row]
                    right_outer[row] = right_a[row] + full_w_px

    for rank, side, outer, inner in (
        (0, "left", left_outer, left_inner),
        (1, "right", right_outer, right_inner),
    ):
        center, outer_missing, inner_missing = _pair_center_u(
            outer, inner, side=side
        )
        valid = int(np.count_nonzero(~np.isnan(center)))
        if valid < max(5, FORK_TRACK_MIN_ROWS // 2):
            continue
        conf = float(np.clip(valid / float(BEV_HEIGHT), 0.0, 1.0))
        # Width consistency bonus baked into confidence when both sides seen.
        both = ~np.isnan(outer) & ~np.isnan(inner)
        if np.any(both):
            width_m = float(np.nanmedian(np.abs(outer[both] - inner[both]))) * (
                METERS_PER_PIXEL
            )
            width_err = abs(width_m - FORK_PAIR_WIDTH_M) / max(
                0.05, FORK_PAIR_WIDTH_M
            )
            conf *= float(np.clip(1.0 - 0.35 * width_err, 0.55, 1.0))
        pairs.append(
            ForkLanePair(
                lateral_rank=rank,
                outer_u=outer.astype(np.float32, copy=False),
                inner_u=inner.astype(np.float32, copy=False),
                center_u=center,
                outer_missing=outer_missing,
                inner_missing=inner_missing,
                confidence=conf,
            )
        )

    if len(pairs) < 2:
        return []

    pairs = stitch_fork_stem_continuity(
        pairs, mark_mask=mark_mask, tip_mode=tip_mode
    )
    pairs = finalize_fork_lane_pair_tips(
        pairs, mark_mask=mark_mask, tip_mode=tip_mode
    )

    # Reject if centers never separate. Stem shares one mid — do NOT use the
    # all-row median (that falsely rejects). Use peak |c0-c1| (and far if any).
    c0 = pairs[0].center_u
    c1 = pairs[1].center_u
    far_end = max(1, int(round(BEV_HEIGHT * FORK_FAR_ZONE_RATIO)))
    both = ~np.isnan(c0) & ~np.isnan(c1)
    if not np.any(both):
        return []
    diffs = np.abs(c0.astype(np.float32) - c1.astype(np.float32))
    sep_peak_m = float(np.nanmax(diffs[both])) * METERS_PER_PIXEL
    both_far = both.copy()
    both_far[far_end:] = False
    sep_far_m = (
        float(np.nanmedian(diffs[both_far])) * METERS_PER_PIXEL
        if np.any(both_far)
        else 0.0
    )
    sep_m = max(sep_peak_m, sep_far_m)
    if sep_m < max(0.08, 0.5 * MIN_BRANCH_SEPARATION_M):
        return []
    return refine_fork_lane_pairs(pairs, mark_mask=mark_mask)


def _nan_moving_average(series: np.ndarray, window: int = 5) -> np.ndarray:
    """Nan-aware 1D moving average along BEV rows (near↔far). Vectorized.

    각 i의 창 [max(0,i-half), min(n,i+half+1)) 에서 non-nan 평균. 창에 유효값이
    없으면 원래 값 유지. cumsum 으로 O(n) — 원소별 np.mean 루프를 제거한다.
    """

    out = series.astype(np.float32, copy=True)
    n = len(out)
    if n == 0:
        return out
    half = max(1, window // 2)
    valid = ~np.isnan(series)
    vals = np.where(valid, series, 0.0).astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(vals)))
    ccnt = np.concatenate(([0.0], np.cumsum(valid.astype(np.float64))))
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + half + 1)
    wsum = csum[hi] - csum[lo]
    wcnt = ccnt[hi] - ccnt[lo]
    have = wcnt > 0
    out[have] = (wsum[have] / wcnt[have]).astype(np.float32)
    return out


def _interpolate_nans_1d(series: np.ndarray) -> np.ndarray:
    """Fill interior NaNs by linear interpolation in row index; tip NaNs stay."""

    out = series.astype(np.float32, copy=True)
    idx = np.flatnonzero(~np.isnan(out))
    if idx.size < 2:
        return out
    missing = np.isnan(out)
    # Only fill between first and last valid sample.
    interior = missing & (np.arange(len(out)) > idx[0]) & (
        np.arange(len(out)) < idx[-1]
    )
    if not np.any(interior):
        return out
    out[interior] = np.interp(
        np.flatnonzero(interior).astype(np.float32),
        idx.astype(np.float32),
        out[idx],
    ).astype(np.float32)
    return out


def stitch_fork_stem_continuity(
    pairs: list[ForkLanePair],
    mark_mask: np.ndarray | None = None,
    *,
    tip_mode: str = "in_curve",
) -> list[ForkLanePair]:
    """Stem share + fork corridors; fork prefers observed inners over ±w.

    Stem (``sep ≈ lane_width``): shared mid / opposite-outer inners.
    Fork: use observed course/mark inners when present so tips can exit
    left/right/top with paint (finalize_fork_lane_pair_tips extends further).

    ``tip_mode="out_forward"``: stem X → parallel fork with a continuous blend;
    never snap to apex mid-collapsed paint inners (gore shelf). In-course keeps
    classic share/X plus observed fork inners after ``fork_t``.
    """

    del mark_mask
    if len(pairs) < 2:
        return pairs
    by_rank = {int(p.lateral_rank): p for p in pairs}
    if 0 not in by_rank or 1 not in by_rank:
        return pairs

    left = by_rank[0]
    right = by_rank[1]
    lo = _nan_moving_average(
        _interpolate_nans_1d(left.outer_u.astype(np.float32, copy=True)),
        window=7,
    )
    ro = _nan_moving_average(
        _interpolate_nans_1d(right.outer_u.astype(np.float32, copy=True)),
        window=7,
    )
    obs_li = left.inner_u.astype(np.float32, copy=True)
    obs_ri = right.inner_u.astype(np.float32, copy=True)
    full_w = FORK_PAIR_WIDTH_M / METERS_PER_PIXEL
    half_w = 0.5 * full_w
    far_end = max(1, int(round(BEV_HEIGHT * FORK_FAR_ZONE_RATIO)))
    out_forward = str(tip_mode or "") == "out_forward"
    fork_t = 0.55

    li = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    ri = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    c0 = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    c1 = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)

    # Per-row outer sep and the last ego-side stem row (sep still ~1 lane).
    sep_row = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    for row in range(BEV_HEIGHT):
        o_l, o_r = lo[row], ro[row]
        if np.isnan(o_l) or np.isnan(o_r) or o_r <= o_l:
            continue
        sep_row[row] = float(o_r - o_l)
    stem_end = None  # largest row index still stem (closest to ego among stem)
    for row in range(BEV_HEIGHT - 1, -1, -1):
        s = sep_row[row]
        if np.isnan(s):
            continue
        if s <= full_w * 1.25:
            stem_end = row
            break
    # First clearly forked row walking ego→far after stem_end.
    fork_start = None
    if stem_end is not None:
        for row in range(stem_end, -1, -1):
            s = sep_row[row]
            if not np.isnan(s) and s >= full_w * 1.55:
                fork_start = row
                break

    for row in range(BEV_HEIGHT):
        o_l = lo[row]
        o_r = ro[row]
        if np.isnan(o_l) and np.isnan(o_r):
            continue
        # Single-outer FOV: keep that path's 11-rail only.
        if np.isnan(o_l) and not np.isnan(o_r):
            ri[row] = float(o_r) - full_w
            c1[row] = float(o_r) - half_w
            continue
        if np.isnan(o_r) and not np.isnan(o_l):
            li[row] = float(o_l) + full_w
            c0[row] = float(o_l) + half_w
            continue
        if o_r <= o_l:
            continue

        sep = float(o_r - o_l)
        mid = 0.5 * (float(o_l) + float(o_r))
        # Force shared stem until the detected fork apex (r06-like split start).
        if stem_end is not None and row >= stem_end:
            t = 0.0
        elif fork_start is not None and stem_end is not None and row <= fork_start:
            t = 1.0
        elif fork_start is not None and stem_end is not None and stem_end > fork_start:
            t = float(
                np.clip(
                    (stem_end - row) / max(1.0, float(stem_end - fork_start)),
                    0.0,
                    1.0,
                )
            )
        else:
            t = float(
                np.clip((sep - full_w * 1.05) / max(1.0, 1.15 * full_w), 0.0, 1.0)
            )

        # Inners / centers. Stem = opposite-outer X (A0); fork = parallel ±w.
        parallel_li = float(o_l) + full_w
        parallel_ri = float(o_r) - full_w
        stem_li = float(o_r)
        stem_ri = float(o_l)
        fork_c0 = float(o_l) + half_w
        fork_c1 = float(o_r) - half_w

        if out_forward:
            # Smooth X→parallel only. Apex paint often collapses both inners
            # onto mid (sep≈0) — snapping to that creates the gore "shelf".
            li[row] = (1.0 - t) * stem_li + t * parallel_li
            ri[row] = (1.0 - t) * stem_ri + t * parallel_ri
            c0[row] = (1.0 - t) * mid + t * fork_c0
            c1[row] = (1.0 - t) * mid + t * fork_c1
            raw_l = obs_li[row]
            raw_r = obs_ri[row]
            if (
                t >= 0.55
                and not np.isnan(raw_l)
                and not np.isnan(raw_r)
                and float(raw_l) < float(raw_r)
                and (float(raw_r) - float(raw_l)) >= 0.20 * full_w
            ):
                li[row] = 0.65 * float(li[row]) + 0.35 * float(raw_l)
                ri[row] = 0.65 * float(ri[row]) + 0.35 * float(raw_r)
                c0[row] = (1.0 - t) * mid + t * (
                    0.5 * (float(o_l) + float(li[row]))
                )
                c1[row] = (1.0 - t) * mid + t * (
                    0.5 * (float(o_r) + float(ri[row]))
                )
        elif t >= fork_t:
            if not np.isnan(obs_li[row]):
                li[row] = float(obs_li[row])
            else:
                li[row] = parallel_li
            if not np.isnan(obs_ri[row]):
                ri[row] = float(obs_ri[row])
            else:
                ri[row] = parallel_ri
            c0[row] = 0.5 * (float(o_l) + float(li[row]))
            c1[row] = 0.5 * (float(o_r) + float(ri[row]))
        else:
            li[row] = (1.0 - t) * stem_li + t * parallel_li
            ri[row] = (1.0 - t) * stem_ri + t * parallel_ri
            c0[row] = (1.0 - t) * mid + t * fork_c0
            c1[row] = (1.0 - t) * mid + t * fork_c1
            if row < far_end and t >= 0.45:
                raw_l = obs_li[row]
                raw_r = obs_ri[row]
                if (
                    not np.isnan(raw_l)
                    and abs(float(raw_l) - parallel_li) <= 0.30 * full_w
                ):
                    li[row] = 0.70 * float(li[row]) + 0.30 * float(raw_l)
                if (
                    not np.isnan(raw_r)
                    and abs(float(raw_r) - parallel_ri) <= 0.30 * full_w
                ):
                    ri[row] = 0.70 * float(ri[row]) + 0.30 * float(raw_r)
                c0[row] = 0.5 * (float(o_l) + float(li[row]))
                c1[row] = 0.5 * (float(o_r) + float(ri[row]))
    li = _nan_moving_average(li, window=5)
    ri = _nan_moving_average(ri, window=5)
    c0 = _nan_moving_average(c0, window=5)
    c1 = _nan_moving_average(c1, window=5)

    rebuilt: list[ForkLanePair] = []
    for rank, side, outer, inner, center in (
        (0, "left", lo, li, c0),
        (1, "right", ro, ri, c1),
    ):
        valid = int(np.count_nonzero(~np.isnan(center)))
        if valid < max(5, FORK_TRACK_MIN_ROWS // 2):
            continue
        # If center series incomplete, fall back to rail formula.
        if valid < int(0.5 * BEV_HEIGHT):
            center, outer_missing, inner_missing = _pair_center_u(
                outer, inner, side=side
            )
        else:
            outer_missing = bool(np.any(np.isnan(outer) & ~np.isnan(inner)))
            inner_missing = bool(np.any(np.isnan(inner) & ~np.isnan(outer)))
        conf = float(np.clip(valid / float(BEV_HEIGHT), 0.0, 1.0))
        rebuilt.append(
            ForkLanePair(
                lateral_rank=rank,
                outer_u=outer,
                inner_u=inner,
                center_u=center.astype(np.float32, copy=False),
                outer_missing=outer_missing,
                inner_missing=inner_missing,
                confidence=conf,
            )
        )
    if len(rebuilt) < 2:
        return pairs
    return rebuilt


def refine_fork_lane_pairs(
    pairs: list[ForkLanePair],
    mark_mask: np.ndarray | None = None,
) -> list[ForkLanePair]:
    """Keep parallel-rail / shared-stem centers from stitch; light near FOV only."""

    del mark_mask
    full_w = FORK_PAIR_WIDTH_M / METERS_PER_PIXEL
    half_w = 0.5 * full_w
    near0 = int(round(BEV_HEIGHT * (1.0 - FORK_NEAR_ZONE_RATIO)))
    refined: list[ForkLanePair] = []

    for pair in pairs:
        side = "left" if int(pair.lateral_rank) == 0 else "right"
        outer = _interpolate_nans_1d(pair.outer_u.astype(np.float32, copy=True))
        inner = pair.inner_u.astype(np.float32, copy=True)
        center = pair.center_u.astype(np.float32, copy=True)

        for row in range(near0, BEV_HEIGHT):
            o = outer[row]
            if np.isnan(o):
                continue
            if np.isnan(inner[row]):
                inner[row] = (o + full_w) if side == "left" else (o - full_w)
            if np.isnan(center[row]):
                center[row] = (o + half_w) if side == "left" else (o - half_w)

        outer = _nan_moving_average(outer, window=5)
        inner = _nan_moving_average(inner, window=5)
        center = _nan_moving_average(center, window=5)
        valid = int(np.count_nonzero(~np.isnan(center)))
        if valid < max(5, FORK_TRACK_MIN_ROWS // 2):
            refined.append(pair)
            continue
        refined.append(
            ForkLanePair(
                lateral_rank=int(pair.lateral_rank),
                outer_u=outer.astype(np.float32, copy=False),
                inner_u=inner.astype(np.float32, copy=False),
                center_u=center.astype(np.float32, copy=False),
                outer_missing=bool(np.any(np.isnan(outer) & ~np.isnan(inner))),
                inner_missing=bool(np.any(np.isnan(inner) & ~np.isnan(outer))),
                confidence=float(np.clip(valid / float(BEV_HEIGHT), 0.0, 1.0)),
            )
        )
    refined.sort(key=lambda p: int(p.lateral_rank))
    return refined if len(refined) >= 2 else pairs


def _limit_column_jumps_zoned(
    series: np.ndarray,
    *,
    far_end: int,
    near0: int,
    max_far: float,
    max_near: float,
) -> np.ndarray:
    """Clamp row-to-row jumps with looser far / tighter near limits."""

    out = series.astype(np.float32, copy=True)

    def _pass(rows: range) -> None:
        last: float | None = None
        last_row: int | None = None
        for row in rows:
            v = out[row]
            if np.isnan(v):
                continue
            limit = max_near if row >= near0 else max_far
            if last is not None and last_row is not None:
                # Scale allowance slightly by row gap.
                gap = abs(row - last_row)
                step = limit * max(1.0, float(gap))
                if abs(v - last) > step:
                    out[row] = last + np.sign(v - last) * step
                    v = out[row]
            last = float(v)
            last_row = row

    _pass(range(BEV_HEIGHT - 1, -1, -1))
    _pass(range(BEV_HEIGHT))
    return out


def fork_lane_pairs_to_road_branches(
    pairs: list[ForkLanePair],
) -> list[RoadBranch]:
    """Convert marking pairs to planner RoadBranch centerlines."""

    branches: list[RoadBranch] = []
    for pair in pairs:
        points_xy = _boundary_u_to_vehicle_points(pair.center_u)
        if len(points_xy) < 2:
            continue
        points_xyz = np.column_stack(
            (
                points_xy[:, 0],
                points_xy[:, 1],
                np.zeros(len(points_xy), dtype=np.float32),
            )
        ).astype(np.float32)
        branches.append(
            RoadBranch(
                lateral_rank=int(pair.lateral_rank),
                confidence=float(pair.confidence),
                width=float(FORK_PAIR_WIDTH_M),
                points=points_xyz,
            )
        )
    branches.sort(key=lambda b: int(b.lateral_rank))
    return branches


def extract_marking_fork_lane_pairs(
    mark_connected_bev: np.ndarray,
    *,
    tip_mode: str = "in_curve",
) -> tuple[list[ForkLanePair], list[np.ndarray]]:
    """Track marking polylines and split into left/right fork lane pairs."""

    tracks = track_marking_polylines(mark_connected_bev)
    pairs = build_fork_lane_pairs_from_tracks(
        tracks, mark_mask=mark_connected_bev, tip_mode=tip_mode
    )
    return pairs, tracks


def extract_road_split_fork_lane_pairs(
    road_clean: np.ndarray,
    mark_mask: np.ndarray | None = None,
) -> tuple[list[ForkLanePair], list[np.ndarray]]:
    """Build L/R pairs from dual road_clean corridors (white V with weak inners).

    Dual-segment rows define the fork. Stem rows (single corridor nearer ego)
    are filled afterward so outers continue and inners meet the 0.35 m stem
    edges. Optional ``mark_mask`` snaps outers to nearby paint.
    """

    if road_clean.size == 0 or road_clean.shape != (BEV_HEIGHT, BEV_WIDTH):
        return [], []

    far_end = max(1, int(round(BEV_HEIGHT * FORK_FAR_ZONE_RATIO)))
    dual_rows = [
        row
        for row in range(BEV_HEIGHT)
        if len(find_drivable_segments(road_clean[row])) >= 2
    ]
    if sum(1 for row in dual_rows if row < far_end) < max(
        5, FORK_TRACK_MIN_ROWS // 3
    ):
        return [], []

    left_outer = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    left_inner = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    right_inner = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    right_outer = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)

    snap_px = max(4.0, 0.10 / METERS_PER_PIXEL)

    def _snap_edges(row: int, lo: float, li: float, ri: float, ro: float):
        if mark_mask is None or mark_mask.shape != road_clean.shape:
            return lo, li, ri, ro
        for a, b in find_line_segments(mark_mask[row]):
            c = 0.5 * (a + b)
            if abs(c - lo) <= snap_px:
                lo = c
            elif abs(c - ro) <= snap_px:
                ro = c
            elif abs(c - li) <= snap_px:
                li = c
            elif abs(c - ri) <= snap_px:
                ri = c
        return lo, li, ri, ro

    # Pass 1: dual corridors, near → far for continuity.
    prev_l: float | None = None
    prev_r: float | None = None
    for row in range(BEV_HEIGHT - 1, -1, -1):
        segs = find_drivable_segments(road_clean[row])
        if len(segs) < 2:
            continue
        if prev_l is not None and prev_r is not None:
            left_seg = min(segs, key=lambda s: abs(segment_center(s) - prev_l))
            remain = [s for s in segs if s is not left_seg] or segs
            right_seg = min(remain, key=lambda s: abs(segment_center(s) - prev_r))
            if segment_center(left_seg) > segment_center(right_seg):
                left_seg, right_seg = right_seg, left_seg
        else:
            ego = (BEV_WIDTH - 1) / 2.0
            nearest = sorted(segs, key=lambda s: abs(segment_center(s) - ego))[:2]
            nearest.sort(key=segment_center)
            left_seg, right_seg = nearest[0], nearest[1]
        prev_l = segment_center(left_seg)
        prev_r = segment_center(right_seg)
        lo, li = float(left_seg[0]), float(left_seg[1])
        ri, ro = float(right_seg[0]), float(right_seg[1])
        lo, li, ri, ro = _snap_edges(row, lo, li, ri, ro)
        left_outer[row] = lo
        left_inner[row] = li
        right_inner[row] = ri
        right_outer[row] = ro

    # Pass 2: stem nearer than the closest dual row → extend outers + stem inners.
    nearest_dual = max(dual_rows)  # largest row index = closest to ego among duals
    stem_lo = left_outer[nearest_dual]
    stem_ro = right_outer[nearest_dual]
    for row in range(nearest_dual + 1, BEV_HEIGHT):
        segs = find_drivable_segments(road_clean[row])
        if not segs:
            continue
        # Prefer the segment closest to the dual-corridor mid.
        ref = (
            0.5 * (stem_lo + stem_ro)
            if not (np.isnan(stem_lo) or np.isnan(stem_ro))
            else (BEV_WIDTH - 1) / 2.0
        )
        seg = min(segs, key=lambda s: abs(segment_center(s) - ref))
        lo, ro = float(seg[0]), float(seg[1])
        lo, _, _, ro = _snap_edges(row, lo, 0.5 * (lo + ro), 0.5 * (lo + ro), ro)
        left_outer[row] = lo
        right_outer[row] = ro
        # Stem inners: opposite-outer share (classic 11자 X / A0). Stitch
        # out_forward then blends X→parallel so the gore does not flatten.
        left_inner[row] = ro
        right_inner[row] = lo
        stem_lo, stem_ro = lo, ro

    # Width-parallel cleanup on all filled rows (also uncrosses bad inners).
    tracks = [left_outer, left_inner, right_inner, right_outer]
    pairs = build_fork_lane_pairs_from_tracks(
        tracks, mark_mask=mark_mask, tip_mode="out_forward"
    )
    if len(pairs) < 2:
        pairs = []
        for rank, side, outer, inner in (
            (0, "left", left_outer, left_inner),
            (1, "right", right_outer, right_inner),
        ):
            center, outer_missing, inner_missing = _pair_center_u(
                outer, inner, side=side
            )
            valid = int(np.count_nonzero(~np.isnan(center)))
            if valid < max(5, FORK_TRACK_MIN_ROWS // 2):
                continue
            pairs.append(
                ForkLanePair(
                    lateral_rank=rank,
                    outer_u=outer,
                    inner_u=inner,
                    center_u=center,
                    outer_missing=outer_missing,
                    inner_missing=inner_missing,
                    confidence=float(np.clip(valid / float(BEV_HEIGHT), 0.0, 1.0)),
                )
            )
        if len(pairs) < 2:
            return [], tracks
        pairs = stitch_fork_stem_continuity(
            pairs, mark_mask=mark_mask, tip_mode="out_forward"
        )
        pairs = finalize_fork_lane_pair_tips(
            pairs, mark_mask=mark_mask, tip_mode="out_forward"
        )
        pairs = refine_fork_lane_pairs(pairs, mark_mask=mark_mask)
        c0, c1 = pairs[0].center_u, pairs[1].center_u
        far_end_i = max(1, int(round(BEV_HEIGHT * FORK_FAR_ZONE_RATIO)))
        both = ~np.isnan(c0[:far_end_i]) & ~np.isnan(c1[:far_end_i])
        if not np.any(both):
            return [], tracks
        sep_m = abs(float(np.nanmedian(c0[:far_end_i][both] - c1[:far_end_i][both]))) * (
            METERS_PER_PIXEL
        )
        if sep_m < max(0.08, 0.5 * MIN_BRANCH_SEPARATION_M):
            return [], tracks
    return pairs, tracks


def fork_source_allowed_for_course(
    fork_mark_color: str,
    *,
    prefer_yellow: bool | None,
    yellow_is_detected: bool,
    ego_road_color: str | None,
) -> bool:
    """Whether a candidate fork source may become planner ``branches``.

    Out (``prefer_yellow=False``): white / road_split only — never yellow_*.
    In (``True``): yellow preferred; white/road_split allowed as fallback.
    """

    color = str(fork_mark_color or "")
    if prefer_yellow is False:
        return color in ("white", "white_alt", "road_split")
    if prefer_yellow is True:
        if color in ("yellow", "yellow_alt"):
            # In already chose yellow candidates; do not require HSV flag again.
            return True
        return color in ("white", "white_alt", "road_split")
    # Legacy auto (None): ego color gate.
    if color in ("yellow", "yellow_alt"):
        return bool(yellow_is_detected) and ego_road_color in ("yellow", None)
    if color in ("white", "white_alt"):
        return ego_road_color in ("white", None)
    if color == "road_split":
        return ego_road_color in ("white", "yellow", None)
    return False


def select_course_fork_pairs(
    *,
    prefer_yellow: bool | None,
    yellow_left: np.ndarray,
    yellow_right: np.ndarray,
    yellow_alt_left: np.ndarray,
    yellow_alt_right: np.ndarray,
    yellow_boundary_bev: np.ndarray,
    white_left: np.ndarray,
    white_right: np.ndarray,
    white_alt_left: np.ndarray,
    white_alt_right: np.ndarray,
    white_dash_connected_bev: np.ndarray,
    road_clean: np.ndarray,
) -> tuple[list, list, list, str]:
    """Pick L/R fork pairs for the active course color contract.

    Returns ``(pairs, tracks, branches, mark_color)``.
    """

    pairs: list = []
    tracks: list = []
    branches: list = []
    mark_color = ""

    def take(candidate_pairs, candidate_tracks, color: str) -> bool:
        nonlocal pairs, tracks, branches, mark_color
        converted = fork_lane_pairs_to_road_branches(candidate_pairs)
        if len(converted) < 2:
            return False
        pairs = list(candidate_pairs)
        tracks = list(candidate_tracks or [])
        branches = converted
        mark_color = color
        return True

    def try_white_then_split() -> None:
        nonlocal pairs, tracks, branches, mark_color
        wp, wt = extract_marking_fork_lane_pairs(
            white_dash_connected_bev, tip_mode="out_forward"
        )
        take(wp, wt, "white")
        if mark_color == "white" and len(tracks) <= 2 and len(branches) >= 2:
            rp, rt = extract_road_split_fork_lane_pairs(
                road_clean, white_dash_connected_bev
            )
            take(rp, rt, "road_split")
        if len(branches) < 2:
            rp, rt = extract_road_split_fork_lane_pairs(
                road_clean, white_dash_connected_bev
            )
            if not take(rp, rt, "road_split"):
                rp, rt = extract_road_split_fork_lane_pairs(
                    road_clean, yellow_boundary_bev
                )
                take(rp, rt, "road_split")
        if len(branches) < 2:
            take(
                fork_lane_pairs_from_dual_courses(
                    white_left,
                    white_right,
                    white_alt_left,
                    white_alt_right,
                    mark_mask=white_dash_connected_bev,
                    tip_mode="out_forward",
                ),
                [],
                "white_alt",
            )

    if prefer_yellow is True:
        if not take(
            fork_lane_pairs_from_dual_courses(
                yellow_left,
                yellow_right,
                yellow_alt_left,
                yellow_alt_right,
                mark_mask=yellow_boundary_bev,
                tip_mode="in_curve",
            ),
            [],
            "yellow_alt",
        ):
            yp, yt = extract_marking_fork_lane_pairs(
                yellow_boundary_bev, tip_mode="in_curve"
            )
            take(yp, yt, "yellow")
        if len(branches) < 2:
            try_white_then_split()
        return pairs, tracks, branches, mark_color

    # Out (False) and legacy (None): white/road_split first.
    # Out never stores yellow_* candidates (would be gated anyway).
    if prefer_yellow is False:
        try_white_then_split()
        return pairs, tracks, branches, mark_color

    # Legacy None: try yellow then white; ego gate decides in caller.
    if not take(
        fork_lane_pairs_from_dual_courses(
            yellow_left,
            yellow_right,
            yellow_alt_left,
            yellow_alt_right,
            mark_mask=yellow_boundary_bev,
            tip_mode="in_curve",
        ),
        [],
        "yellow_alt",
    ):
        yp, yt = extract_marking_fork_lane_pairs(
            yellow_boundary_bev, tip_mode="in_curve"
        )
        take(yp, yt, "yellow")
    if len(branches) < 2:
        try_white_then_split()
    return pairs, tracks, branches, mark_color


def extract_yellow_fork_lane_pairs(
    yellow_connected_bev: np.ndarray,
) -> tuple[list[ForkLanePair], list[np.ndarray]]:
    """Track yellow markings and split into left/right fork lane pairs."""

    return extract_marking_fork_lane_pairs(yellow_connected_bev)


def fork_lane_pairs_from_dual_courses(
    primary_left: np.ndarray,
    primary_right: np.ndarray,
    alt_left: np.ndarray,
    alt_right: np.ndarray,
    *,
    min_valid_rows: int | None = None,
    mark_mask: np.ndarray | None = None,
    tip_mode: str = "in_curve",
) -> list[ForkLanePair]:
    """주+보조 경계 코스 → 좌/우 ``ForkLanePair`` (갈래 2개).

    각 DP 코스 = 한 갈래의 L/R 페인트. ``alt_*``는 코드명=보조 코스
    (alternate). far mid-u로 rank 0=왼쪽. 용어: strategy.md §0.
    """

    min_rows = (
        int(min_valid_rows)
        if min_valid_rows is not None
        else max(8, FORK_TRACK_MIN_ROWS // 2)
    )
    primary_left = np.asarray(primary_left, dtype=np.float32)
    primary_right = np.asarray(primary_right, dtype=np.float32)
    alt_left = np.asarray(alt_left, dtype=np.float32)
    alt_right = np.asarray(alt_right, dtype=np.float32)
    if primary_left.shape != (BEV_HEIGHT,) or alt_left.shape != (BEV_HEIGHT,):
        return []

    def course_valid(left: np.ndarray, right: np.ndarray) -> int:
        return int(np.count_nonzero(~np.isnan(left) & ~np.isnan(right)))

    if course_valid(primary_left, primary_right) < min_rows:
        return []
    if course_valid(alt_left, alt_right) < min_rows:
        return []

    far_end = max(1, int(round(BEV_HEIGHT * FORK_FAR_ZONE_RATIO)))

    def far_mid(left: np.ndarray, right: np.ndarray) -> float:
        both = ~np.isnan(left[:far_end]) & ~np.isnan(right[:far_end])
        if np.any(both):
            return float(
                np.nanmedian(0.5 * (left[:far_end][both] + right[:far_end][both]))
            )
        both = ~np.isnan(left) & ~np.isnan(right)
        if not np.any(both):
            return float("nan")
        return float(np.nanmedian(0.5 * (left[both] + right[both])))

    mid_p = far_mid(primary_left, primary_right)
    mid_a = far_mid(alt_left, alt_right)
    if np.isnan(mid_p) or np.isnan(mid_a):
        return []
    if abs(mid_p - mid_a) * METERS_PER_PIXEL < max(
        0.08, 0.5 * MIN_BRANCH_SEPARATION_M
    ):
        # Alt collapsed onto primary — not a real fork.
        return []

    if mid_p <= mid_a:
        left_l, left_r = primary_left, primary_right
        right_l, right_r = alt_left, alt_right
    else:
        left_l, left_r = alt_left, alt_right
        right_l, right_r = primary_left, primary_right

    pairs: list[ForkLanePair] = []
    for rank, outer, inner, side in (
        (0, left_l, left_r, "left"),
        (1, right_r, right_l, "right"),
    ):
        center, outer_missing, inner_missing = _pair_center_u(
            outer, inner, side=side
        )
        valid = int(np.count_nonzero(~np.isnan(center)))
        if valid < min_rows:
            return []
        pairs.append(
            ForkLanePair(
                lateral_rank=rank,
                outer_u=outer.astype(np.float32, copy=True),
                inner_u=inner.astype(np.float32, copy=True),
                center_u=center,
                outer_missing=outer_missing,
                inner_missing=inner_missing,
                confidence=float(np.clip(valid / float(BEV_HEIGHT), 0.0, 1.0)),
            )
        )
    pairs = stitch_fork_stem_continuity(
        pairs, mark_mask=mark_mask, tip_mode=tip_mode
    )
    pairs = finalize_fork_lane_pair_tips(
        pairs, mark_mask=mark_mask, tip_mode=tip_mode
    )
    pairs = refine_fork_lane_pairs(pairs, mark_mask=mark_mask)
    return pairs if len(pairs) >= 2 else []

def find_drivable_segments(row: np.ndarray) -> list[tuple[int, int]]:
    """한 BEV 행에서 최소 폭을 만족하는 주행 가능 구간을 찾는다."""

    return [
        (left, right)
        for left, right in find_line_segments(row)
        if right - left + 1 >= MIN_BRANCH_WIDTH_PX
    ]


def segment_center(segment: tuple[int, int]) -> float:
    return (segment[0] + segment[1]) / 2.0













def draw_vehicle_polyline(
    image: np.ndarray,
    points_xy: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """base_link polyline을 BEV 픽셀로 되돌려 후보 ID와 함께 그린다."""

    if len(points_xy) < 2:
        return
    columns = (BEV_WIDTH - 1) / 2.0 - points_xy[:, 1] / METERS_PER_PIXEL
    rows = (X_MAX_M - points_xy[:, 0]) / METERS_PER_PIXEL
    valid = (
        (columns >= 0.0)
        & (columns < BEV_WIDTH)
        & (rows >= 0.0)
        & (rows < BEV_HEIGHT)
    )
    if np.count_nonzero(valid) < 2:
        return
    points = np.column_stack((columns[valid], rows[valid]))
    points = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        image,
        [points],
        isClosed=False,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    label_point = tuple(int(value) for value in points[0, 0])
    cv2.putText(
        image,
        label,
        label_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def _smooth_center_track(
    points: list[tuple[int, float, float]],
) -> list[tuple[int, float, float]]:
    """연속 경로의 중심열을 국소 Gaussian으로 완만화한다.

    전역 다항식은 '직선 stem → 갈래' 급전환을 표현 못 해 stem을 옆으로
    휘게 만들지만, 국소 Gaussian은 stem의 중앙값을 그대로 두고 분기점
    부근만 둥글게 휜다(곡률 연속이라 코너도 없다).
    """

    n = len(points)
    if n < 5:
        return points
    columns = np.array([[p[1] for p in points]], dtype=np.float32)
    sigma = max(1.0, 0.10 / METERS_PER_PIXEL)
    smoothed = cv2.GaussianBlur(
        columns,
        (0, 0),
        sigmaX=sigma,
        sigmaY=0.0,
        borderType=cv2.BORDER_REPLICATE,
    ).reshape(-1)
    return [
        (points[i][0], float(smoothed[i]), points[i][2])
        for i in range(n)
    ]


def _branch_paths_to_road_branches(
    paths: list[list[tuple[int, float, float]]],
) -> list[RoadBranch]:
    """(row, center, width) 경로를 base_link RoadBranch로 변환·정렬한다."""

    branches: list[RoadBranch] = []
    for path in paths:
        if len(path) < MIN_COURSE_RUN_ROWS:
            continue
        rows = np.array([p[0] for p in path], dtype=np.float32)
        columns = np.array([p[1] for p in path], dtype=np.float32)
        widths_px = np.array([p[2] for p in path], dtype=np.float32)
        x_forward = X_MAX_M - rows * METERS_PER_PIXEL
        y_left = ((BEV_WIDTH - 1) / 2.0 - columns) * METERS_PER_PIXEL
        order = np.argsort(x_forward)
        points_xyz = np.column_stack(
            (
                x_forward[order],
                y_left[order],
                np.zeros(len(order), dtype=np.float32),
            )
        ).astype(np.float32)
        row_span = max(1.0, float(np.max(rows) - np.min(rows) + 1.0))
        confidence = float(
            np.clip(len(np.unique(rows)) / row_span, 0.0, 1.0)
        )
        branches.append(
            RoadBranch(
                confidence=confidence,
                width=float(np.median(widths_px) * METERS_PER_PIXEL),
                points=points_xyz,
            )
        )

    branches.sort(
        key=lambda branch: float(np.median(branch.points[:, 1])),
        reverse=True,
    )
    return [
        RoadBranch(
            lateral_rank=rank,
            confidence=branch.confidence,
            width=branch.width,
            points=branch.points,
        )
        for rank, branch in enumerate(branches)
    ]


# =============================================================
# 코스 셀(cell) 기반 갈래 추출
#
# road_clean은 fill_road_surface_holes가 흰/노란 차선 픽셀을 도로로 메워
# 넣기 때문에 아웃코스와 인코스가 하나의 검은 덩어리로 붙는다. 그 상태에서는
#   (a) 주행영역 세그먼트의 좌/우 끝이 차선이 아니라 도로 바깥 경계라
#       코스 색을 알 수 없고(노란 점선은 덩어리 '내부'에 있어 절대 안 잡힌다),
#   (b) 갈래가 실제로 분리되지 않아 고정 수직 midline으로 반을 가르는 수밖에
#       없어, 도로가 휘면 원거리에서 갈래가 옆 코스로 넘어가 버린다.
#
# 차선을 도로에서 다시 빼내면 각 코스가 '차선으로 둘러싸인 셀'이 된다.
# 그러면 셀 가장자리 색 = 그 코스의 색이고, 행 간 겹침으로 셀을 연결 추적할
# 수 있어 갈래가 옆 코스로 새지 않는다.
# =============================================================

# 얇은 차선이 도로를 확실히 끊도록 살짝만 부풀린다. 크게 잡으면 분기 직전
# 갈래가 뾰족해지는 구간(점선이 만나는 곳)의 셀을 통째로 갉아먹는다.
LANE_CUT_DILATE_M = 0.008
LANE_CUT_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        make_odd(max(3, int(round(2 * LANE_CUT_DILATE_M / METERS_PER_PIXEL)))),
        make_odd(max(3, int(round(2 * LANE_CUT_DILATE_M / METERS_PER_PIXEL)))),
    ),
)

# 점선 dash 간격을 메우는 커널들. 세로 커널 하나로는 '세로로 흐르는' 점선만
# 이어진다. 갈림길 고어 섬의 점선은 대각선이라 세로 커널로는 안 메워지고, 그
# 틈으로 컷이 새서 양옆 셀이 한 덩어리로 붙는다. dash 위상은 차가 움직일 때마다
# 달라지므로 이게 분기 검출을 프레임마다 깜빡이게 만든다.
#
# 그래서 여러 방향의 '선형' 커널로 각각 닫고 합집합을 쓴다. 선형이라 진행
# 방향으로만 잇고 선을 옆으로 굵히지 않는다.
LANE_CUT_CLOSE_LENGTH_M = 0.14
LANE_CUT_CLOSE_LENGTH_PX = make_odd(
    max(3, int(round(LANE_CUT_CLOSE_LENGTH_M / METERS_PER_PIXEL)))
)
# 90도 = BEV 세로(진행방향). 차선은 진행방향을 따라 흐르므로 그 근처 각도만
# 쓴다. 가로 방향까지 닫으면 두 가지가 망가진다.
#   1) 도로를 가로지르는 마킹(정지선/진입선)이 두껍게 부풀어 셀을 끊는다.
#   2) 갈림길 고어 섬의 점선 V자에서, 두 변이 가까워지는 꼭짓점 부근을 '가로질러'
#      메워 큰 덩어리가 된다. 그 덩어리가 도로 폭 전체를 덮어 분기점 바로 그
#      자리의 셀을 지워버린다(실측: 그 행들의 셀 개수가 0).
# close는 '한 선의 dash를 그 선 방향으로 잇는' 용도지, '다른 선과 잇는' 게 아니다.
LANE_CUT_CLOSE_ANGLES_DEG = (50, 70, 90, 110, 130)


def _line_kernel(length_px: int, angle_deg: float) -> np.ndarray:
    """중심을 지나는 angle_deg 방향 선분 커널."""

    kernel = np.zeros((length_px, length_px), dtype=np.uint8)
    center = length_px // 2
    half = length_px / 2.0
    radians = math.radians(angle_deg)
    dx = math.cos(radians) * half
    dy = math.sin(radians) * half
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1,
        1,
    )
    return kernel


LANE_CUT_CLOSE_KERNELS = tuple(
    _line_kernel(LANE_CUT_CLOSE_LENGTH_PX, angle)
    for angle in LANE_CUT_CLOSE_ANGLES_DEG
)

# 셀을 '자르는' 해상도와 '갈래로 인정하는' 기준은 다르다. 분기점 부근에서
# 갈래는 뾰족하게 시작하므로 셀 최소 폭을 갈래 최소 폭(10cm)으로 잡으면
# 분기 직전에 ego 셀이 사라져 추적이 분기점에 닿기도 전에 끊긴다.
MIN_CELL_WIDTH_M = 0.03
MIN_CELL_WIDTH_PX = max(2, int(round(MIN_CELL_WIDTH_M / METERS_PER_PIXEL)))

# 셀이 뾰족해지거나(분기 시작점) 점선이 뭉쳐 잠깐 막혀도 추적이 살아남도록
# 행 방향 끊김을 넉넉히 허용한다.
MAX_CELL_ROW_GAP_M = 0.15
MAX_CELL_ROW_GAP_ROWS = max(
    1, int(round(MAX_CELL_ROW_GAP_M / METERS_PER_PIXEL))
)

# 셀 바깥에서 차선 색을 훑을 폭. 차선을 부풀린 만큼보다 넉넉해야 한다.
COURSE_COLOR_MARGIN_PX = max(
    4, int(round(0.06 / METERS_PER_PIXEL))
)
# 현재 주행 코스 색을 투표할 근거리 길이
EGO_COLOR_LENGTH_M = 0.60
EGO_COLOR_ROWS = max(1, int(round(EGO_COLOR_LENGTH_M / METERS_PER_PIXEL)))
# 각 갈래 색을 투표할 분기 직후 길이
BRANCH_COLOR_LENGTH_M = 0.40
BRANCH_COLOR_ROWS = max(1, int(round(BRANCH_COLOR_LENGTH_M / METERS_PER_PIXEL)))
# 행 간 셀 연결로 인정할 최대 좌우 이격(겹치지 않아도 이만큼은 붙은 것으로 본다)
CELL_TRACK_GAP_PX = max(1, int(round(0.04 / METERS_PER_PIXEL)))

# 막다른 셀(고어 주머니 끝 등)에서 점선 너머 갈래로 건너뛸 때 쓰는 좌우 반경.
# 부풀린 차선 두께보다 넉넉해야 건너뛸 수 있다.
#
# 이 값을 '끊긴 행 수에 비례해 서서히 키우면' 안 된다. 허용치가 자라는 도중
# 가까운 쪽 갈래가 먼저 반경 안에 들어오는 순간 그것만 후속으로 잡히고 끊김
# 카운터가 리셋돼, 반대쪽 갈래를 영영 못 본다. 좌우 거리가 프레임마다 미세
# 하게 달라지므로 분기 검출이 프레임마다 깜빡인다. 처음부터 대칭으로 넓게
# 열어 두 갈래가 같은 행에서 함께 잡히게 한다.
DEAD_END_REACH_M = 0.18
DEAD_END_REACH_PX = max(1, int(round(DEAD_END_REACH_M / METERS_PER_PIXEL)))
# 갈래의 '시작 폭'을 잴 구간(고어 섬은 한 점에서 시작하므로 여기서 걸러진다)
BRANCH_START_ROWS = max(1, int(round(0.05 / METERS_PER_PIXEL)))
# 다른 갈래와 같은 셀을 밟는 행이 이 비율을 넘으면 '다시 합쳐진 가짜 갈래'다.
# 갈래로 인정하려면 갈라진 직후 이만큼은 다른 셀로 따로 가야 한다.
# 진짜 Y 갈림길은 즉시 크게 벌어지고, 가짜 갈래(노면 마킹·그림자로 도로가
# 잠깐 두 조각 난 것)는 몇 행 만에 같은 셀로 되돌아온다.
MIN_BRANCH_SEPARATION_M = 0.15
MIN_BRANCH_SEPARATION_ROWS = max(
    1, int(round(MIN_BRANCH_SEPARATION_M / METERS_PER_PIXEL))
)


def _apply_detect_tune_from_yaml() -> None:
    """Optional ``detect_tune:`` block from lane_vision.yaml (tuner saves here)."""

    try:
        with open(DEFAULT_CONFIG_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except OSError:
        return
    block = data.get("detect_tune")
    if not isinstance(block, dict):
        return
    kwargs: dict = {}
    if "crossing_coverage_ratio" in block:
        kwargs["crossing_coverage_ratio"] = float(block["crossing_coverage_ratio"])
    if "crossing_min_rows" in block:
        kwargs["crossing_min_rows"] = int(block["crossing_min_rows"])
    if "min_branch_separation_m" in block:
        kwargs["min_branch_separation_m"] = float(block["min_branch_separation_m"])
    if "dash_max_lateral_error_m" in block:
        kwargs["dash_max_lateral_error_m"] = float(block["dash_max_lateral_error_m"])
    if "dash_max_forward_gap_m" in block:
        kwargs["dash_max_forward_gap_m"] = float(block["dash_max_forward_gap_m"])
    if "dash_max_heading_diff_deg" in block:
        kwargs["dash_max_heading_diff_deg"] = float(block["dash_max_heading_diff_deg"])
    if "dash_min_component_area_px" in block:
        kwargs["dash_min_component_area_px"] = int(block["dash_min_component_area_px"])
    if "dash_branch_assoc_m" in block:
        kwargs["dash_branch_assoc_m"] = float(block["dash_branch_assoc_m"])
    if "red_h_low_wrap" in block:
        kwargs["red_h_low_wrap"] = int(block["red_h_low_wrap"])
    if "fork_track_assoc_m" in block:
        kwargs["fork_track_assoc_m"] = float(block["fork_track_assoc_m"])
    if "fork_track_min_rows" in block:
        kwargs["fork_track_min_rows"] = int(block["fork_track_min_rows"])
    if "fork_pair_width_m" in block:
        kwargs["fork_pair_width_m"] = float(block["fork_pair_width_m"])
    if "fork_far_zone_ratio" in block:
        kwargs["fork_far_zone_ratio"] = float(block["fork_far_zone_ratio"])
    if "fork_track_max_row_gap" in block:
        kwargs["fork_track_max_row_gap"] = int(block["fork_track_max_row_gap"])
    if "fork_near_zone_ratio" in block:
        kwargs["fork_near_zone_ratio"] = float(block["fork_near_zone_ratio"])
    if kwargs:
        apply_detect_tune(**kwargs)


_apply_detect_tune_from_yaml()

# 가짜 분기(노이즈)를 건너뛰며 재탐색할 최대 횟수
MAX_FORK_PROBES = 8

# 공통 진입부(stem)는 흰 코스와 노란 코스가 붙어 있어 한쪽 경계는 흰선,
# 반대쪽은 노란선인 경우가 많다. 그런 행은 색을 판정할 수 없으므로, 마지막
# 으로 명확했던 코스 색을 프레임 간 래치해 분기 필터의 기준으로 쓴다.
last_ego_course_color: str | None = None

# 분기 판정이 어느 단계에서 갈래를 떨어뜨렸는지 프리뷰에 찍기 위한 진단 문자열.
# split@<행> cand=<후보 셀 수> like=<길이·폭 통과>[색] sep=<재합류 통과> color=<색 통과>
last_fork_debug: str = "-"


def build_course_cells(
    road_clean: np.ndarray,
    white_line: np.ndarray,
    yellow_line: np.ndarray,
) -> np.ndarray:
    """차선을 도로에서 빼내 코스별로 분리된 셀 마스크를 만든다."""

    lines = cv2.bitwise_or(white_line, yellow_line)
    # 점선의 dash 간격을 먼저 메운다. 한 행이라도 dash가 비면 그 행에서 선이
    # 도로를 못 끊어 양옆 셀이 한 덩어리로 붙고, 갈래 추적이 옆 코스로 샌다.
    # dash 위상은 프레임마다 달라지므로 이게 분기 검출을 깜빡이게 만든다.
    # 방향별 선형 커널로 각각 닫고 합쳐, 대각선 점선(고어 섬 외곽)도 잇는다.
    closed = lines
    for kernel in LANE_CUT_CLOSE_KERNELS:
        closed = cv2.bitwise_or(
            closed,
            cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel),
        )
    closed = cv2.dilate(closed, LANE_CUT_KERNEL)
    return cv2.bitwise_and(road_clean, cv2.bitwise_not(closed))


def is_branch_like(path: list[tuple[int, tuple[int, int]]]) -> bool:
    """갈래로 인정할 만큼 길고 넓은 경로인지 본다.

    폭 기준은 셀 하나가 아니라 경로의 중앙값에 건다. 셀 단위로 거르면 분기
    직전 뾰족해지는 구간에서 진짜 갈래도 같이 날아간다.

    시작 폭도 함께 본다. 진짜 갈래는 갈라지는 순간 이미 도로 폭의 절반쯤
    되지만, 두 갈래 사이 포장된 고어(gore) 섬은 한 점에서 시작해 서서히
    벌어지므로 시작 폭으로 걸러진다.
    """

    if len(path) < MIN_BRANCH_LENGTH_ROWS:
        return False
    widths = [cell[1] - cell[0] + 1 for _, cell in path]
    if float(np.median(widths)) < MIN_BRANCH_WIDTH_PX:
        return False
    return float(np.median(widths[:BRANCH_START_ROWS])) >= MIN_BRANCH_WIDTH_PX


def dominant_line_color(white: int, yellow: int) -> str | None:
    """한 방향에서 본 흰/노란 픽셀 수로 그쪽 경계선 색을 정한다."""

    if white == yellow:
        return None
    return "white" if white > yellow else "yellow"


def cell_row_color(
    white_line: np.ndarray,
    yellow_line: np.ndarray,
    row: int,
    cell: tuple[int, int],
) -> str | None:
    """한 행에서 셀 좌/우 '바깥'에 맞닿은 차선 색으로 그 셀의 코스 색을 본다.

    양쪽 색이 다르면(흰선과 노란선 사이에 낀 공통 진입부) 판정 불가로 None을
    돌려준다. 이걸 픽셀 수 합산으로 뭉개면 선 굵기/점선 여부에 따라 무작위로
    한쪽이 이겨버린다.
    """

    left, right = cell
    margin = COURSE_COLOR_MARGIN_PX
    left_slice = slice(max(0, left - margin), left)
    right_slice = slice(right + 1, right + 1 + margin)

    left_color = dominant_line_color(
        int(np.count_nonzero(white_line[row, left_slice])),
        int(np.count_nonzero(yellow_line[row, left_slice])),
    )
    right_color = dominant_line_color(
        int(np.count_nonzero(white_line[row, right_slice])),
        int(np.count_nonzero(yellow_line[row, right_slice])),
    )
    if left_color is not None and right_color is not None:
        return left_color if left_color == right_color else None
    # 한쪽 경계만 보이면(반대쪽이 시야 밖이거나 선이 끊김) 그 색을 쓴다.
    return left_color if left_color is not None else right_color


def vote_course_color(
    white_line: np.ndarray,
    yellow_line: np.ndarray,
    path: list[tuple[int, tuple[int, int]]],
) -> str | None:
    """경로가 지나는 행들의 코스 색을 다수결한다(판정 불가 행은 기권)."""

    votes = {"white": 0, "yellow": 0}
    for row, cell in path:
        color = cell_row_color(white_line, yellow_line, row, cell)
        if color is not None:
            votes[color] += 1
    if votes["white"] == votes["yellow"]:
        return None
    return "white" if votes["white"] > votes["yellow"] else "yellow"


def cells_connect(
    current: tuple[int, int],
    candidate: tuple[int, int],
    gap_rows: int = 0,
) -> bool:
    """두 행의 셀이 같은 코스로 이어지는지(겹침/근접) 본다.

    셀이 이어지는 동안은 좁게 붙은 것만 인정한다. 셀이 아예 끊겼다가(고어
    주머니 끝처럼 막다른 곳) 다시 나타나는 경우에만 DEAD_END_REACH_PX만큼
    대칭으로 넓혀, 점선 너머 좌우 갈래가 같은 행에서 함께 잡히게 한다.
    """

    tolerance = CELL_TRACK_GAP_PX if gap_rows <= 1 else DEAD_END_REACH_PX
    overlap = min(current[1], candidate[1]) - max(current[0], candidate[0]) + 1
    return overlap >= -tolerance


def follow_cell(
    cells_by_row: list[list[tuple[int, int]]],
    rows: list[int],
    start: int,
    cell: tuple[int, int],
    stop_on_split: bool = True,
) -> tuple[list[tuple[int, tuple[int, int]]], int | None, list[tuple[int, int]]]:
    """근거리→원거리로 셀을 행 간 겹침으로 이어붙인다.

    stop_on_split이면 후속 셀이 2개 이상으로 갈라지는 행에서 멈추고 그
    행 인덱스와 후속 셀들을 함께 돌려준다. 아니면 가장 가까운 후속 셀을
    골라 계속 따라간다.
    """

    path: list[tuple[int, tuple[int, int]]] = [(rows[start], cell)]
    current = cell
    gap_rows = 0
    for index in range(start + 1, len(rows)):
        row = rows[index]
        gap_rows += rows[index - 1] - row
        successors = [
            candidate
            for candidate in cells_by_row[row]
            if cells_connect(current, candidate, gap_rows)
        ]
        if not successors:
            if gap_rows > MAX_CELL_ROW_GAP_ROWS:
                break
            continue
        gap_rows = 0
        if len(successors) >= 2:
            if stop_on_split:
                return path, index, sorted(successors)
            successors = [
                min(
                    successors,
                    key=lambda candidate: abs(
                        segment_center(candidate) - segment_center(current)
                    ),
                )
            ]
        current = successors[0]
        path.append((row, current))
    return path, None, []


def describe_branch_candidate(path: list[tuple[int, tuple[int, int]]]) -> str:
    """진단용: 분기 후보의 '길이(행)x중앙폭(px)'. 어느 기준에 걸렸는지 보인다."""

    if not path:
        return "0"
    widths = [cell[1] - cell[0] + 1 for _, cell in path]
    return f"{len(path)}x{int(np.median(widths))}"


def separated_prefix_length(
    path: list[tuple[int, tuple[int, int]]],
    others: list[list[tuple[int, tuple[int, int]]]],
) -> int:
    """갈라진 직후부터 다른 갈래와 '다른 셀'로 계속 가는 행 수."""

    other_cells = [dict(other) for other in others]
    length = 0
    for row, cell in path:
        if any(cells.get(row) == cell for cells in other_cells):
            break
        length += 1
    return length


def drop_reconverging_branches(
    followed: list[tuple[list[tuple[int, tuple[int, int]]], str | None]],
) -> list[tuple[list[tuple[int, tuple[int, int]]], str | None]]:
    """갈라진 직후 곧바로 다시 같은 셀로 합쳐지는 '가짜 갈래'를 버린다.

    도로가 잠깐 두 조각으로 끊겼다가(점선 뭉침, 노면 마킹, 그림자) 다시
    붙는 구간은 분기가 아니다. 그런 후보들은 합류 지점 이후로는 완전히 같은
    셀을 따라가므로 길이·폭 검사를 그대로 통과해 버린다(그래서 두 갈래가
    거의 겹쳐 그려진다).

    판별 기준은 '얼마나 떨어져 가는가'가 아니라 '대부분의 구간을 같이 가는가'
    다. 거리로 재면 임계값이 BEV가 분기 이후로 볼 수 있는 거리와 비슷해져,
    분기점이 조금만 멀어도 진짜 갈래가 탈락한다.

    판별 기준은 '전체 중 얼마나 겹치는가'가 아니라 '갈라진 직후 실제로 얼마나
    따로 가는가'다. 겹침 비율로 재면, 진짜 갈래가 먼 곳에서 딱 한 번 붙기만
    해도(원거리 dash는 원근 압축으로 얇아져 커터가 자주 샌다) 비율이 임계를
    넘어 두 갈래가 통째로 탈락한다. 실제로 분기 검출이 프레임마다 깜빡인
    주된 원인이 이거였다.

    진짜 갈래는 갈라진 즉시 상당 거리를 따로 간다. 가짜 갈래는 몇 행 만에
    같은 셀로 되돌아온다. 그래서 '갈라진 직후 따로 가는 행 수'로 가르고,
    합류하는 지점에서 경로를 잘라낸다(합류 이후는 갈래가 아니므로 센터라인이
    옆 갈래로 넘어가는 것도 함께 막는다).
    """

    paths = [path for path, _ in followed]
    kept: list[tuple[list[tuple[int, tuple[int, int]]], str | None]] = []
    for index, (path, color) in enumerate(followed):
        others = [
            other
            for other_index, other in enumerate(paths)
            if other_index != index
        ]
        length = separated_prefix_length(path, others)
        if length >= MIN_BRANCH_SEPARATION_ROWS:
            kept.append((path[:length], color))
    return kept


def same_course_branches(
    followed: list[tuple[list[tuple[int, tuple[int, int]]], str | None]],
    ego_color: str | None,
) -> list[list[tuple[int, tuple[int, int]]]]:
    """현재 주행 코스와 '다른' 색 경계로 확인된 갈래를 버린다.

    버리는 근거는 '다른 색임이 확인됨'이지 '같은 색임이 확인 안 됨'이 아니다.
    색을 못 읽은 갈래(None)는 남긴다. 점선 간격이나 시야 밖 경계 때문에 한
    프레임 색이 안 잡혔다는 이유로 진짜 갈래가 통째로 사라지면 안 된다.

    반대로 모든 갈래가 ego와 '다른 색으로 확인'되면 빈 리스트를 돌려준다.
    그건 갈림길이 아니라 옆 코스(흰 도로 주행 중 보이는 노란 도로)가 시야에
    붙은 것이므로, 분기로 세면 안 된다.
    """

    if ego_color is None:
        return [path for path, _ in followed]
    return [
        path
        for path, color in followed
        if color is None or color == ego_color
    ]


def trace_ego_course(
    cells_by_row: list[list[tuple[int, int]]],
    rows: list[int],
    white_line: np.ndarray,
    yellow_line: np.ndarray,
) -> tuple[list[tuple[int, tuple[int, int]]], list[list[tuple[int, tuple[int, int]]]], str | None]:
    """차량이 있는 셀을 따라가며 '같은 색 갈래 2개 이상'인 진짜 분기를 찾는다.

    색이 다른 갈래(흰 도로 주행 중 노란 인코스)로 갈라지는 지점은 분기가
    아니라 '옆 코스가 붙었다 떨어지는 것'이므로, 같은 색 갈래를 stem으로
    계속 이어붙이고 더 앞쪽에서 진짜 분기를 다시 찾는다.
    """

    global last_ego_course_color
    global last_fork_debug

    vehicle_center = (BEV_WIDTH - 1) / 2.0
    cell = min(
        cells_by_row[rows[0]],
        key=lambda candidate: abs(segment_center(candidate) - vehicle_center),
    )
    stem: list[tuple[int, tuple[int, int]]] = []
    index = 0
    measured_color: str | None = None
    ego_color: str | None = None
    probes: list[str] = []

    for _ in range(MAX_FORK_PROBES):
        path, split_index, successors = follow_cell(
            cells_by_row, rows, index, cell
        )
        stem.extend(path)
        if measured_color is None:
            measured_color = vote_course_color(
                white_line, yellow_line, stem[:EGO_COLOR_ROWS]
            )
            if measured_color is not None:
                last_ego_course_color = measured_color
        # 공통 진입부는 한쪽이 흰선·반대쪽이 노란선이라 색이 안 잡힌다.
        # 그럴 땐 직전에 확정했던 코스 색을 그대로 쓴다.
        ego_color = measured_color or last_ego_course_color
        if split_index is None:
            probes.append(f"end@{stem[-1][0] if stem else '-'}")
            last_fork_debug = " | ".join(probes)
            return stem, [], ego_color

        raw = [
            follow_cell(
                cells_by_row,
                rows,
                split_index,
                successor,
                stop_on_split=False,
            )[0]
            for successor in successors
        ]
        followed = [
            (
                branch,
                vote_course_color(
                    white_line, yellow_line, branch[:BRANCH_COLOR_ROWS]
                ),
            )
            for branch in raw
            if is_branch_like(branch)
        ]

        # 다시 합쳐지는 가짜 갈래를 먼저 버리고, 남은 것 중 같은 색만 고른다.
        # 원래 followed는 '분기가 아닐 때 stem을 어디로 이을지' 고르는 데 쓴다.
        separate = drop_reconverging_branches(followed)
        kept = same_course_branches(separate, ego_color)
        probes.append(
            f"@{rows[split_index]} cand="
            + "/".join(describe_branch_candidate(branch) for branch in raw)
            + f" like={len(followed)}"
            + f"[{','.join(color or '?' for _, color in followed)}]"
            + f" sep={len(separate)} col={len(kept)}"
        )
        last_fork_debug = " | ".join(probes)

        if len(kept) >= 2:
            return stem, kept, ego_color

        if kept:
            continued = max(kept, key=len)
        elif separate:
            # 갈라져 나간 갈래가 있는데 전부 ego와 '다른 색으로 확인'됐다.
            # 내 코스는 여기서 끝이고 옆 코스만 보이는 것이므로, 아무거나
            # 이어붙이면 stem이 옆 도로(흰 도로 주행 중 노란 도로)로 넘어간다.
            return stem, [], ego_color
        elif followed:
            # 갈래들이 위에서 다시 합쳐진다(가짜 분기) → 가장 긴 쪽으로 잇는다.
            continued = max(followed, key=lambda item: len(item[0]))[0]
        else:
            return stem, [], ego_color

        # 진짜 분기가 아니다: 같은 색 갈래를 stem으로 계속 잇고 더 앞을 본다.
        # split_index는 항상 index보다 크므로 이 루프는 반드시 전진한다.
        cell = continued[0][1]
        index = split_index

    return stem, [], ego_color


def cell_path_to_center_track(path: list[tuple[int, tuple[int, int]]]) -> list[tuple[int, float, float]]:
    """(row, cell) 경로를 (row, center, width) 중심열로 바꾼다."""

    return [
        (row, segment_center(cell), float(cell[1] - cell[0] + 1))
        for row, cell in path
    ]


def build_road_branches_cells(
    road_clean: np.ndarray,
    white_line: np.ndarray,
    yellow_line: np.ndarray,
) -> tuple[list[RoadBranch], np.ndarray, str | None]:
    """차선으로 잘라낸 코스 셀을 연결 추적해 갈래를 만든다.

    반환: (갈래들, 셀 마스크(시각화용), 현재 주행 코스 색)
    갈림길이 아니면 갈래 1개, 같은 색 갈림길이면 2개 이상.
    """

    cells_mask = build_course_cells(road_clean, white_line, yellow_line)
    cells_by_row = [
        [
            (left, right)
            for left, right in segments
            if right - left + 1 >= MIN_CELL_WIDTH_PX
        ]
        for segments in find_line_segments_by_row(cells_mask)
    ]
    rows = [row for row in range(BEV_HEIGHT - 1, -1, -1) if cells_by_row[row]]
    if not rows:
        return [], cells_mask, None

    stem, branch_paths, ego_color = trace_ego_course(
        cells_by_row, rows, white_line, yellow_line
    )
    if branch_paths:
        paths = [stem + branch for branch in branch_paths]
    else:
        paths = [stem]

    tracks = [
        _smooth_center_track(cell_path_to_center_track(path)) for path in paths
    ]
    return _branch_paths_to_road_branches(tracks), cells_mask, ego_color


# 셀 덩어리를 '왼쪽부터' 순서대로 칠하는 팔레트(BGR). 채도를 낮춰, 위에 겹쳐
# 그리는 갈래 폴리라인(선명한 빨강/파랑)이 묻히지 않게 한다.
CELL_DEBUG_COLORS = (
    (70, 120, 70),
    (120, 120, 70),
    (70, 120, 120),
    (120, 90, 70),
    (90, 70, 120),
    (110, 110, 110),
)


def make_course_cell_preview(
    bev: np.ndarray,
    cells_mask: np.ndarray,
    branches: list["RoadBranch"],
    ego_color: str | None,
) -> np.ndarray:
    """차선으로 잘라낸 셀을 덩어리별로 칠하고 갈래를 겹쳐 그린다.

    셀 색이 갈리는 곳이 곧 '차선이 도로를 끊은 곳'이다. 갈래가 하나만
    나올 때 셀 자체가 안 갈렸는지, 갈렸는데 추적이 못 넘어간 건지 이
    그림만 보면 구별된다.

    색은 셀 중심의 좌우 위치 순서로 매긴다(맨 왼쪽 셀이 항상 첫 색). 연결
    성분 라벨 번호를 그대로 쓰면 셀이 하나 생겼다 사라질 때마다 번호가 밀려
    프레임마다 색이 통째로 바뀌고(반짝거림), 도로가 변한 것처럼 보인다.
    """

    count, labels, _, centroids = cv2.connectedComponentsWithStats(
        cells_mask, connectivity=8
    )
    order = sorted(range(1, count), key=lambda label: centroids[label][0])
    overlay = np.zeros_like(bev)
    for rank, label in enumerate(order):
        overlay[labels == label] = CELL_DEBUG_COLORS[
            rank % len(CELL_DEBUG_COLORS)
        ]
    preview = cv2.addWeighted(bev, 1.0, overlay, 0.5, 0.0)

    for branch in branches:
        color = BRANCH_COLORS[branch.lateral_rank % len(BRANCH_COLORS)]
        draw_vehicle_polyline(
            preview, branch.points[:, :2], color, f"B{branch.lateral_rank}"
        )
    cv2.putText(
        preview,
        f"BRANCHES: {len(branches)}  EGO: {ego_color or '?'}  CELLS: {count - 1}",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"min {MIN_BRANCH_LENGTH_ROWS}rx{MIN_BRANCH_WIDTH_PX}px | {last_fork_debug}",
        (4, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return preview


def vehicle_xy_to_bev_uv(x_m: float, y_m: float) -> tuple[float, float]:
    """base_link (x forward, y left) → BEV pixel (u=col, v=row)."""

    row = (X_MAX_M - float(x_m)) / METERS_PER_PIXEL
    col = (BEV_WIDTH - 1) / 2.0 - float(y_m) / METERS_PER_PIXEL
    return col, row


def bev_uv_to_vehicle_xy(u: float, v: float) -> tuple[float, float]:
    """BEV pixel (u=col, v=row) → base_link meters."""

    x_m = X_MAX_M - float(v) * METERS_PER_PIXEL
    y_m = ((BEV_WIDTH - 1) / 2.0 - float(u)) * METERS_PER_PIXEL
    return x_m, y_m


def _point_to_polyline_distance_m(
    point_xy: np.ndarray,
    polyline_xy: np.ndarray,
) -> float:
    """Min distance from a 2D point to a polyline (both in meters)."""

    if polyline_xy.shape[0] == 0:
        return float("inf")
    if polyline_xy.shape[0] == 1:
        return float(np.linalg.norm(point_xy - polyline_xy[0]))
    best = float("inf")
    for i in range(polyline_xy.shape[0] - 1):
        a = polyline_xy[i]
        b = polyline_xy[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            dist = float(np.linalg.norm(point_xy - a))
        else:
            t = float(np.clip(np.dot(point_xy - a, ab) / denom, 0.0, 1.0))
            proj = a + t * ab
            dist = float(np.linalg.norm(point_xy - proj))
        if dist < best:
            best = dist
    return best


def select_road_branch(
    branches: tuple | list,
    focus: str,
):
    """Pick branch for focus in {all, left, right}. all → None."""

    items = list(branches)
    if not items or focus == "all":
        return None
    ranks = [int(b.lateral_rank) for b in items]
    keep = min(ranks) if focus == "left" else max(ranks)
    for branch in items:
        if int(branch.lateral_rank) == keep:
            return branch
    return None


def filter_dash_mask_by_branch(
    dash_mask: np.ndarray,
    branch,
    *,
    max_lateral_m: float | None = None,
) -> np.ndarray:
    """Keep connected dash blobs whose centroid is near the branch centerline.

    Used so dash_left / dash_right only promote the dashed markings that belong
    to the chosen fork path (merge/split gore lines otherwise bleed into both).
    """

    if dash_mask.size == 0 or branch is None:
        return dash_mask
    pts = np.asarray(branch.points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros_like(dash_mask)
    poly = pts[:, :2]
    limit = float(DASH_BRANCH_ASSOC_M if max_lateral_m is None else max_lateral_m)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        dash_mask, connectivity=8
    )
    out = np.zeros_like(dash_mask)
    kept = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(3, DASH_MIN_COMPONENT_AREA_PX // 2):
            continue
        ys, xs = np.nonzero(labels == label)
        if ys.size == 0:
            continue
        cu = float(np.mean(xs))
        cv_ = float(np.mean(ys))
        xy = np.array(bev_uv_to_vehicle_xy(cu, cv_), dtype=np.float32)
        if _point_to_polyline_distance_m(xy, poly) <= limit:
            out[labels == label] = 255
            kept += 1
    return out


def make_dash_preview(
    debug: "LaneDebugFrame",
    *,
    focus: str = "all",
) -> np.ndarray:
    """Dash Phase-A view: make connect-parameter changes obvious.

    Previous preview painted dim yellow HSV under everything, so cyan raw and
    lime connect-fill were nearly invisible — trackbar moves looked like no-ops
    even when ``link`` pixel counts changed.

    Color key (mask-first on near-black):
      - gray   = road_clean silhouette (context only)
      - cyan   = raw dash/solid yellow components (extract)
      - lime   = pixels **added by connect** (gap/lat/head effect) — emphasized
    Branch polylines stay off in ``focus=all``.
    """

    if debug.bev.size == 0:
        return np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

    # Near-black canvas — do NOT underlay yellow HSV (it hides connect fill).
    preview = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    if debug.road_clean.size:
        preview[debug.road_clean > 0] = (28, 28, 28)

    yellow_pts = debug.yellow_dash_points_bev
    yellow_conn = debug.yellow_connected_bev
    white_pts = debug.white_dash_points_bev
    if yellow_pts.size == 0:
        yellow_pts = np.zeros(debug.bev.shape[:2], dtype=np.uint8)
    if yellow_conn.size == 0:
        yellow_conn = yellow_pts.copy()
    if white_pts.size == 0:
        white_pts = np.zeros(debug.bev.shape[:2], dtype=np.uint8)

    branch = select_road_branch(debug.road_branches, focus)
    if branch is not None:
        yellow_pts = filter_dash_mask_by_branch(yellow_pts, branch)
        yellow_conn = filter_dash_mask_by_branch(yellow_conn, branch)
        white_pts = filter_dash_mask_by_branch(white_pts, branch)

    # Connect fill only (what gap/lat/head change). Fat lime so it is obvious.
    link = ((yellow_conn > 0) & (yellow_pts == 0)).astype(np.uint8) * 255
    if np.any(link):
        link_fat = cv2.dilate(link, np.ones((3, 3), np.uint8), iterations=1)
        preview[link_fat > 0] = (0, 255, 60)

    # Raw extract — thinner cyan on top (do not drown the lime).
    if np.any(yellow_pts):
        pts_show = cv2.dilate(yellow_pts, np.ones((2, 2), np.uint8), iterations=1)
        preview[pts_show > 0] = (0, 220, 255)

    if np.any(white_pts):
        w_show = cv2.dilate(white_pts, np.ones((2, 2), np.uint8), iterations=1)
        preview[w_show > 0] = (220, 220, 220)

    if branch is not None:
        draw_vehicle_polyline(
            preview,
            branch.points[:, :2],
            (180, 180, 255),
            f"B{int(branch.lateral_rank)}",
        )

    y_raw = int(np.count_nonzero(yellow_pts))
    y_link = int(np.count_nonzero(link))
    y_conn = int(np.count_nonzero(yellow_conn))
    link_pct = (100.0 * y_link / max(1, y_conn)) if y_conn else 0.0
    cv2.putText(
        preview,
        (
            f"DASH[{focus}] raw={y_raw} link={y_link}({link_pct:.1f}%conn)  "
            f"gap={DASH_MAX_FORWARD_GAP_M:.2f}m lat={DASH_MAX_LATERAL_ERROR_M:.3f}m "
            f"head={DASH_MAX_HEADING_DIFF_DEG:.0f}"
        ),
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "WATCH LIME only when tuning gap/lat/head  |  gray=road  cyan=raw  (HSV hidden)",
        (4, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )
    return preview


def make_red_zone_preview(debug: LaneDebugFrame) -> np.ndarray:
    """Red obstacle-lane mask over BEV with coverage readout."""

    if debug.bev.size == 0:
        return np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    preview = debug.bev.copy()
    tint = np.zeros_like(preview)
    tint[debug.red_bev > 0] = (0, 0, 255)
    preview = cv2.addWeighted(preview, 1.0, tint, 0.55, 0.0)
    contours, _ = cv2.findContours(
        debug.red_bev, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 1)
    cv2.putText(
        preview,
        (
            f"RED ZONE  cov={100.0 * debug.red_coverage:.1f}%  "
            f"px={debug.red_pixel_count}  wrapH={RED_H_LOW_WRAP}"
        ),
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def make_crossing_preview(debug: LaneDebugFrame) -> np.ndarray:
    """Yellow/white crossing masks over road_raw."""

    base = debug.road_raw
    if base.size == 0:
        return np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    preview = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    preview[debug.crossing_mask > 0] = (0, 0, 255)
    preview[debug.white_crossing_mask > 0] = (0, 255, 255)
    cv2.putText(
        preview,
        (
            f"CROSSING  Y={debug.yellow_crossing_line}  "
            f"W={debug.white_crossing_line}  "
            f"cov>={CROSSING_COVERAGE_RATIO:.2f}  rows>={CROSSING_MIN_ROWS}"
        ),
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def _draw_boundary_columns(
    image: np.ndarray,
    columns_u: np.ndarray,
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
) -> None:
    pts: list[list[int]] = []
    for row in range(BEV_HEIGHT):
        u = columns_u[row]
        if np.isnan(u):
            continue
        pts.append([int(round(float(u))), int(row)])
    if len(pts) < 2:
        return
    cv2.polylines(
        image,
        [np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def make_fork_lane_pair_preview(
    debug: LaneDebugFrame,
    *,
    focus: str = "all",
) -> np.ndarray:
    """Show L/R outer+inner+center for marking-based fork split."""

    if debug.bev.size == 0:
        return np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

    preview = debug.bev.copy()
    road_overlay = np.zeros_like(preview)
    if debug.road_clean.size:
        road_overlay[debug.road_clean > 0] = (40, 40, 40)
    preview = cv2.addWeighted(preview, 1.0, road_overlay, 0.35, 0.0)

    # Dim raw yellow connected envelope for context.
    if debug.yellow_connected_bev.size:
        tint = preview.copy()
        tint[debug.yellow_connected_bev > 0] = (0, 180, 255)
        preview = cv2.addWeighted(preview, 0.75, tint, 0.25, 0.0)

    pairs = list(debug.fork_lane_pairs)
    if focus == "left":
        pairs = [p for p in pairs if int(p.lateral_rank) == 0]
    elif focus == "right":
        pairs = [p for p in pairs if int(p.lateral_rank) == 1]

    # BGR palette — must match the legend string below exactly.
    # L: red / orange / cyan    R: blue / sky / yellow
    palette = {
        (0, "outer"): (0, 0, 255),        # red
        (0, "inner"): (0, 140, 255),      # orange
        (0, "center"): (255, 255, 0),     # cyan (was wrongly yellow)
        (1, "outer"): (255, 64, 0),       # blue
        (1, "inner"): (255, 200, 80),     # sky (brighter light-blue)
        (1, "center"): (0, 255, 255),     # yellow (was wrongly cyan)
    }

    for pair in pairs:
        rank = int(pair.lateral_rank)
        _draw_boundary_columns(
            preview, pair.outer_u, palette[(rank, "outer")], thickness=2
        )
        _draw_boundary_columns(
            preview, pair.inner_u, palette[(rank, "inner")], thickness=2
        )
        _draw_boundary_columns(
            preview, pair.center_u, palette[(rank, "center")], thickness=3
        )
        # Label near the nearest valid center sample.
        pts = _boundary_u_to_vehicle_points(pair.center_u)
        if len(pts) >= 1:
            tag = f"L{rank}" if rank == 0 else f"R{rank}"
            miss = []
            if pair.outer_missing:
                miss.append("out?")
            if pair.inner_missing:
                miss.append("in?")
            if miss:
                tag = f"{tag}[{','.join(miss)}]"
            draw_vehicle_polyline(
                preview, pts[:2] if len(pts) >= 2 else pts, palette[(rank, "center")], tag
            )

    # Planner branches: use mute magenta/white so they are not confused with
    # pair center cyan/yellow (BRANCH_COLORS reused cyan before).
    branch_mute = ((180, 0, 180), (200, 200, 200), (160, 80, 160), (170, 170, 170))
    for branch in debug.road_branches:
        if focus == "left" and int(branch.lateral_rank) != 0:
            continue
        if focus == "right" and int(branch.lateral_rank) != 1:
            continue
        color = branch_mute[int(branch.lateral_rank) % len(branch_mute)]
        draw_vehicle_polyline(
            preview,
            branch.points[:, :2],
            color,
            f"B{int(branch.lateral_rank)}",
        )

    n_tracks = len(debug.fork_mark_tracks)
    n_pairs = len(debug.fork_lane_pairs)
    cv2.putText(
        preview,
        (
            f"FORK PAIRS focus={focus} tracks={n_tracks} pairs={n_pairs} "
            f"src={debug.fork_split_source or '-'} "
            f"assoc={FORK_TRACK_ASSOC_M:.2f}m minR={FORK_TRACK_MIN_ROWS}"
        ),
        (4, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "L out/in/ctr: red / orange / cyan    R out/in/ctr: blue / sky / yellow",
        (4, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    return preview


def make_fork_focus_preview(
    debug: LaneDebugFrame,
    *,
    focus: str = "all",
) -> np.ndarray:
    """Fork preview: prefer marking L/R pairs; else cell branches.

    When ``debug.active_branch_rank`` is set (planner lock), forced focus
    overlays only that layer so viz matches the path PP follows.
    """

    if getattr(debug, "active_branch_rank", None) is not None:
        focus = "left" if int(debug.active_branch_rank) == 0 else "right"

    if debug.fork_lane_pairs:
        return make_fork_lane_pair_preview(debug, focus=focus)

    if debug.bev.size == 0:
        return np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    branches = list(debug.road_branches)
    preview = make_course_cell_preview(
        debug.bev, debug.road_cells, branches, debug.ego_road_color
    )
    if focus == "all" or not branches:
        return preview
    ranks = [b.lateral_rank for b in branches]
    if focus == "left":
        keep = min(ranks)
    else:
        keep = max(ranks)
    dim = preview.copy()
    dim = (dim.astype(np.float32) * 0.35).astype(np.uint8)
    for branch in branches:
        if branch.lateral_rank != keep:
            continue
        color = BRANCH_COLORS[branch.lateral_rank % len(BRANCH_COLORS)]
        draw_vehicle_polyline(
            dim, branch.points[:, :2], color, f"B{branch.lateral_rank}"
        )
    cv2.putText(
        dim,
        f"FORK FOCUS={focus.upper()}  keep=B{keep}  n={len(branches)}",
        (4, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return dim


def render_mode_preview(mode: str, debug: LaneDebugFrame) -> np.ndarray:
    """Build a single BGR preview for tune_lane_detect modes."""

    mode = (mode or "white").strip().lower()
    if debug.bev.size == 0:
        return np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

    if mode == "white":
        return make_boundary_preview(
            debug.bev,
            debug.road_clean,
            debug.white_left,
            debug.white_right,
            "WHITE",
        )
    if mode == "yellow":
        base = make_boundary_preview(
            debug.bev,
            debug.road_clean,
            debug.yellow_left,
            debug.yellow_right,
            "YELLOW",
        )
        # Dash points (cyan) + connected (yellow tint) for quick check.
        overlay = base.copy()
        if debug.yellow_dash_points_bev is not None and debug.yellow_dash_points_bev.size:
            overlay[debug.yellow_dash_points_bev > 0] = (255, 255, 0)
        if debug.yellow_connected_bev.size:
            connected = cv2.cvtColor(
                debug.yellow_connected_bev, cv2.COLOR_GRAY2BGR
            )
            return cv2.addWeighted(overlay, 0.75, connected, 0.25, 0.0)
        return overlay
    if mode == "dash":
        return make_dash_preview(debug, focus="all")
    if mode == "dash_left":
        return make_dash_preview(debug, focus="left")
    if mode == "dash_right":
        return make_dash_preview(debug, focus="right")
    if mode == "fork":
        return make_fork_focus_preview(debug, focus="all")
    if mode == "fork_left":
        return make_fork_focus_preview(debug, focus="left")
    if mode == "fork_right":
        return make_fork_focus_preview(debug, focus="right")
    if mode == "red":
        return make_red_zone_preview(debug)
    if mode == "crossing":
        return make_crossing_preview(debug)
    return make_boundary_preview(
        debug.bev,
        debug.road_clean,
        debug.white_left,
        debug.white_right,
        mode.upper(),
    )
