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

# ┌──────────────────────────────────────────────────────────┐
# │  여기만 고치면 된다.                                      │
# │                                                          │
# │    "off"      창 없음                                    │
# │    "control"  주행 확인용 3개만 (아래 CONTROL_WINDOWS)    │
# │    "on"       디버그 창 전부                              │
# └──────────────────────────────────────────────────────────┘
VISUALIZE_MODE = "on"

# 보드/SSH/headless에서는 창을 띄우면 죽는다. 코드를 안 고치고 끄려면
# 환경변수로 덮어쓴다(있을 때만 우선).
#   LANE_VISUALIZE=off ros2 run inference inference_node
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
}


def resolve_visualize_mode(raw: str | None) -> str:
    """시각화 모드 문자열을 정규화한다. 모르는 값은 안전하게 OFF."""

    return _VISUALIZE_ALIASES.get((raw or "").strip().lower(), VISUALIZE_OFF)


VISUALIZE_MODE = resolve_visualize_mode(
    os.environ.get("LANE_VISUALIZE") or VISUALIZE_MODE
)
VISUALIZE = VISUALIZE_MODE != VISUALIZE_OFF

# CONTROL 모드에서 띄울 창. 판단제어로 나가는 결과(좌우 경계·갈래)만 본다.
CONTROL_WINDOWS = (
    "white_boundaries",
    "yellow_boundaries",
    "road_branches",
)


def window_enabled(name: str) -> bool:
    """현재 모드에서 이 창을 띄울지 결정한다."""

    if VISUALIZE_MODE == VISUALIZE_ON:
        return True
    if VISUALIZE_MODE == VISUALIZE_CONTROL:
        return name in CONTROL_WINDOWS
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
    # 갈림길 정보(판단제어용): 분기 발생 여부와 각 분기 경로.
    # 각 RoadBranch.lateral_rank = 분기 번호(0=가장 왼쪽), points = base_link
    # 센터라인(Nx3, x 전방/y 왼쪽/z=0). 갈림길이 없으면 branches는 단일 경로.
    fork_active: bool = False
    branches: tuple["RoadBranch", ...] = ()
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
        "white": ((0, 0, 174), (179, 29, 255)),
        "yellow": ((0, 32, 79), (55, 255, 255)),
        "black_road": ((0, 0, 0), (179, 255, 30)),
        "red_road": ((170, 125, 161), (179, 192, 229)),
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
RED_H_LOW_WRAP = 0


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
) -> None:
    """Live-tune detection scalars exposed by tune_lane_detect."""

    global CROSSING_COVERAGE_RATIO, CROSSING_MIN_ROWS
    global MIN_BRANCH_SEPARATION_M, MIN_BRANCH_SEPARATION_ROWS
    global DASH_MAX_LATERAL_ERROR_M, DASH_MAX_FORWARD_GAP_M
    global DASH_MAX_HEADING_DIFF_DEG, DASH_MIN_COMPONENT_AREA_PX
    global DASH_BRANCH_ASSOC_M, RED_H_LOW_WRAP
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

# 경계 사이가 실제 도로인지 확인하는 기준
MIN_CLEAN_ROAD_OVERLAP_RATIO = 0.60
MIN_RAW_ROAD_OVERLAP_RATIO = 0.30


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

# 점선 사이에서 허용할 최대 좌우 이동량
MAX_BOUNDARY_SHIFT_M = 0.12

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

BOUNDARY_SHIFT_PER_ROW_PX = 0.45

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

# '인코스 우선'은 아웃/인 두 코스가 나란히 있는 곳에서만 의미가 있다. 진입
# 직선처럼 도로가 한 코스 폭뿐이면 오른쪽 후보를 선호할 근거가 없는데, 그래도
# 강제하면 정상 차선이 PATH_WRONG_SIDE_PENALTY를 맞고 탈락하고, 대신 차선 안
# 노면 마킹(진입 화살표·점선)을 왼쪽 경계로 착각한 후보가 이겨 코스가 통째로
# 오른쪽으로 밀린다. 그래서 주행영역이 두 코스를 담을 만큼 넓은 행에서만 건다.
#
# 한 코스의 주행영역 폭 = 코스 폭 + 양쪽 선(road_clean은 선을 도로로 메운다).
# 두 코스면 그 두 배 가까이 된다. 사이에 넉넉한 여유가 있어 구분이 안전하다.
INNER_COURSE_MIN_ROAD_WIDTH_M = 2 * ROAD_WIDTH_M
INNER_COURSE_MIN_ROAD_WIDTH_PX = int(
    round(INNER_COURSE_MIN_ROAD_WIDTH_M / METERS_PER_PIXEL)
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
PATH_PAIR_BONUS = 1.0
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

# 도로 겹침은 카메라가 실제로 '본' 픽셀 중에서만 따진다. 시야 밖(검은 쐐기)을
# '도로 없음'으로 세면, 화각 바깥으로 뻗는 차로 가설이 전부 탈락해버린다.
MIN_OBSERVABLE_SPAN_M = 0.05
MIN_OBSERVABLE_SPAN_PX = max(
    1, int(round(MIN_OBSERVABLE_SPAN_M / METERS_PER_PIXEL))
)
# 차로가 사실상 통째로 화각 밖이면 도로임을 '확인'도 '반박'도 못 한다. 어차피
# 350 mm 가정 위에 세운 후보이므로 도로가 이어진다고 보되, 근거가 없으니 크게
# 감점한다. 여기서 버려버리면 노란 도로가 화각 밖인 갈림길에서 정답 가설이
# 사라지고, 반박된 오답만 남아 채택된다.
UNOBSERVED_LANE_PENALTY = 6.0

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


def calculate_overlap_ratio(
    mask_row: np.ndarray,
    left_u: int,
    right_u: int,
) -> float:
    """주어진 가로 구간에서 마스크가 차지하는 비율을 계산한다."""

    width = mask_row.shape[0]

    left_u = max(
        0,
        min(
            width - 1,
            int(left_u),
        ),
    )

    right_u = max(
        0,
        min(
            width - 1,
            int(right_u),
        ),
    )

    if right_u <= left_u:
        return 0.0

    region = mask_row[
        left_u:right_u + 1
    ]

    if region.size == 0:
        return 0.0

    return float(
        np.count_nonzero(region)
    ) / float(region.size)


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


def road_supports_inner_course(
    clean_road_row: np.ndarray,
    reference_center: float,
    road_segments: list[tuple[int, int]] | None = None,
) -> bool:
    """이 행의 주행영역이 아웃코스와 인코스를 나란히 담을 만큼 넓은가."""

    segments = (
        find_drivable_segments(clean_road_row)
        if road_segments is None
        else [
            segment
            for segment in road_segments
            if segment[1] - segment[0] + 1 >= MIN_BRANCH_WIDTH_PX
        ]
    )
    if not segments:
        return False
    segment = min(
        segments,
        key=lambda item: abs(segment_center(item) - reference_center),
    )
    width = segment[1] - segment[0] + 1
    return width >= INNER_COURSE_MIN_ROAD_WIDTH_PX


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


def _boundary_row_sums(
    raw_road_mask: np.ndarray,
    clean_road_mask: np.ndarray,
    observable_mask: np.ndarray | None,
    opposite_line_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """경계 후보 채점에 쓰는 행 누적합을 프레임당 한 번만 만든다.

    관측가능(FOV) 마스크가 없으면 기존 경로(슬라이스 계산)를 그대로 쓰도록
    None을 돌려준다.
    """

    if observable_mask is None:
        return None

    seen = observable_mask.astype(bool)
    return (
        _row_cumsum(seen),
        _row_cumsum(seen & (clean_road_mask > 0)),
        _row_cumsum(seen & (raw_road_mask > 0)),
        None
        if opposite_line_mask is None
        else _row_cumsum(opposite_line_mask > 0),
    )


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
    clean_overlap: float,
    raw_overlap: float,
    width_error: float,
    reference_center: float,
    previous_left: float | None,
    previous_right: float | None,
) -> float:
    """도로 겹침, 폭, 이전 경계 연속성을 이용해 후보 점수를 계산한다."""

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
        clean_overlap * 8.0
        + raw_overlap * 4.0
        - width_error_normalized * 3.0
        - center_error * 2.0
        - continuity_error * 5.0
    )


def enumerate_boundary_candidates(
    segments: list[tuple[int, int]],
    raw_road_row: np.ndarray,
    clean_road_row: np.ndarray,
    row_v: int,
    reference_centerline: np.ndarray | None,
    temporal_centerline: np.ndarray | None,
    temporal_left: np.ndarray | None,
    temporal_right: np.ndarray | None,
    required_side: str | None,
    row_sums: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray | None
    ] | None = None,
    segment_slopes: list[float] | None = None,
    is_ego_course: bool = False,
    opposite_segments: list[tuple[int, int]] | None = None,
    single_line_angle_deg: float | None = None,
    side_debug: dict[str, object] | None = None,
    road_segments: list[tuple[int, int]] | None = None,
) -> list[tuple[float, float, float, int]]:
    """한 행의 가능한 모든 (왼쪽, 오른쪽, 지역 점수) 후보를 반환한다.

    교차로에서는 한 행의 최적 후보가 잘못된 가지일 수 있으므로
    여기서 하나를 확정하지 않고 전체 경로 추적에 넘긴다.

    row_sums는 이 행의 누적합 묶음이다:
    (관측가능, 관측가능∧clean도로, 관측가능∧raw도로, 반대색 차선).
    후보마다 슬라이스를 세지 않고 O(1)로 구간 합을 얻는다.
    """

    if (
        reference_centerline is not None
        and not np.isnan(reference_centerline[row_v])
    ):
        reference_center = float(reference_centerline[row_v])
    else:
        reference_center = BEV_WIDTH / 2.0

    # 인코스가 존재할 수 있는 행에서만 '인코스 우선'을 건다.
    prefer_inner_course = required_side is not None and road_supports_inner_course(
        clean_road_row,
        reference_center,
        road_segments,
    )

    candidates: list[tuple[float, float, float, int]] = []

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
        side_name = (
            "left"
            if source == BOUNDARY_SOURCE_LEFT
            else "right"
            if source == BOUNDARY_SOURCE_RIGHT
            else None
        )

        def bump(reason: str) -> None:
            if side_debug is None or side_name is None:
                return
            key = f"{side_name}_{reason}"
            side_debug[key] = int(side_debug.get(key, 0)) + 1

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

        # 트랙의 모든 도로는 같은 색 선으로만 둘러싸인다(외곽 서킷=흰선,
        # 회전교차로·연결로=노란선). 그러니 반대편 경계를 350 mm로 '지어낸'
        # 한쪽선 후보의 차로 안에 다른 색 차선이 들어앉았다면, 그건 힌트가
        # 아니라 이 색 도로가 아니라는 '증거'다. 감점이 아니라 탈락시킨다.
        # 감점만 하면, 정답 가설이 화각 밖이라 약할 때 반박된 오답이 그대로
        # 유일 후보로 남아 채택된다.
        opposite_cum = None if row_sums is None else row_sums[3]
        opposite_line_conflict = (
            opposite_cum is not None
            and source in (BOUNDARY_SOURCE_LEFT, BOUNDARY_SOURCE_RIGHT)
            and opposite_line_inside_lane(
                opposite_cum,
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
        identity_matched = bool(identity_errors) and (
            identity_mean <= TEMPORAL_ID_MATCH_PX
        )

        matches_preferred_side = not prefer_inner_course or (
            candidate_matches_reference_side(
                center,
                reference_centerline,
                row_v,
                required_side,
            )
        )

        span_low = int(round(visible_left))
        span_high = int(round(visible_right))

        unobserved = False
        if row_sums is None:
            clean_overlap = calculate_overlap_ratio(
                clean_road_row, span_low, span_high
            )
            raw_overlap = calculate_overlap_ratio(
                raw_road_row, span_low, span_high
            )
        else:
            # 카메라가 본 픽셀만 분모로 삼는다. 시야 밖은 '도로 없음'이
            # 아니라 '모름'이므로 겹침 판정에서 아예 뺀다.
            # 후보마다 슬라이스를 세면 느리므로 행 누적합으로 O(1)로 구한다.
            seen_cum, clean_cum, raw_cum, _ = row_sums
            low, high = span_low, span_high + 1
            seen_count = int(seen_cum[high] - seen_cum[low])
            if seen_count < MIN_OBSERVABLE_SPAN_PX:
                # 확인도 반박도 불가 → 도로가 이어진다고 보되 뒤에서 감점한다.
                unobserved = True
                clean_overlap = 1.0
                raw_overlap = 1.0
            else:
                clean_overlap = (
                    float(clean_cum[high] - clean_cum[low]) / seen_count
                )
                raw_overlap = float(raw_cum[high] - raw_cum[low]) / seen_count

        if (
            clean_overlap < MIN_CLEAN_ROAD_OVERLAP_RATIO
            and not identity_matched
        ):
            bump("road")
            return
        if raw_overlap < MIN_RAW_ROAD_OVERLAP_RATIO and not identity_matched:
            bump("raw")
            return

        score = score_boundary_candidate(
            left_u,
            right_u,
            clean_overlap,
            raw_overlap,
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
        if not matches_preferred_side:
            # 차량이 회전교차로 한쪽으로 치우치면 정상 노란 코스가
            # 흰 중심선의 반대쪽에 보일 수 있다. 절대 탈락시키지 않고
            # 우선순위만 낮춘다.
            score -= PATH_WRONG_SIDE_PENALTY
        # 두 선을 실제로 관측한 PAIR 후보는 근거가 충분하므로 건드리지 않는다.
        # 반대편 경계를 350 mm로 '지어낸' 한쪽선 후보만 다른 색 차선으로
        # 검증한다. 지어낸 차로 안에 다른 색 차선이 들어앉았다면 그건 이 색
        # 도로가 아니다(흰 도로=흰선, 노란 도로=노란선으로만 둘러싸인다).
        if unobserved:
            score -= UNOBSERVED_LANE_PENALTY
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

            if measured_width <= 0.0 or width_error > ROAD_WIDTH_TOLERANCE_PX:
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
                if (
                    measured_width <= 0.0
                    or width_error > ROAD_WIDTH_TOLERANCE_PX
                ):
                    continue
                append_if_valid(
                    left_u,
                    right_u,
                    width_error,
                    PATH_PAIR_BONUS,
                    BOUNDARY_SOURCE_PAIR,
                )

    # 한쪽 노란선만 보이는 경우 350 mm 도로 폭(수직)을 수평으로 환산해 추정
    row_single_line_angle = (
        single_line_angle_deg if len(segments) == 1 else None
    )
    for index, (segment_start, segment_end) in enumerate(segments):
        lane_width_px = ROAD_WIDTH_PX * scales[index]

        detected_as_left = float(segment_end)
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

        detected_as_right = float(segment_start)
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


def track_boundary_path(
    boundary_mask: np.ndarray,
    raw_road_mask: np.ndarray,
    clean_road_mask: np.ndarray,
    reference_centerline: np.ndarray | None,
    temporal_centerline: np.ndarray | None,
    temporal_left: np.ndarray | None,
    temporal_right: np.ndarray | None,
    required_side: str | None,
    opposite_line_mask: np.ndarray | None = None,
    observable_mask: np.ndarray | None = None,
    is_ego_course: bool = False,
    use_single_line_angle_bias: bool = False,
    side_debug: dict[str, object] | None = None,
    boundary_segments_by_row: list[list[tuple[int, int]]] | None = None,
    opposite_segments_by_row: list[list[tuple[int, int]]] | None = None,
    road_segments_by_row: list[list[tuple[int, int]]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """모든 행의 후보를 연결해 위치/방향/곡률이 연속인 경로를 찾는다."""

    height = boundary_mask.shape[0]
    candidates_by_row: dict[int, list[tuple[float, float, float, int]]] = {}

    # 후보 하나마다 마스크를 슬라이스해 세면 프레임당 수천 번 numpy를 부른다.
    # 행 누적합을 한 번 만들어 두면 구간 합이 뺄셈 한 번으로 끝난다.
    all_sums = _boundary_row_sums(
        raw_road_mask,
        clean_road_mask,
        observable_mask,
        opposite_line_mask,
    )

    # 도로 폭 보정에는 관측선의 국소 기울기가 필요하므로 세그먼트를 먼저
    # 전부 모아 위/아래 행과 이어 본다.
    segments_by_row = (
        boundary_segments_by_row
        if boundary_segments_by_row is not None
        else find_line_segments_by_row(boundary_mask)
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
        """이 행에 선이 하나일 때 그 선이 속한 연결조각의 각도를 찾는다."""

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

    # 좌우 색이 다른 차로(합류부)를 만들려면 반대색 선의 세그먼트도 필요하다.
    opposite_by_row = opposite_segments_by_row
    if opposite_by_row is None and opposite_line_mask is not None:
        opposite_by_row = find_line_segments_by_row(opposite_line_mask)

    for v in range(height - 1, -1, -1):
        segments = segments_by_row[v]
        if not segments:
            continue
        row_angle_deg = row_single_line_angle_deg(v, segments)

        row_candidates = enumerate_boundary_candidates(
            segments,
            raw_road_mask[v],
            clean_road_mask[v],
            v,
            reference_centerline,
            temporal_centerline,
            temporal_left,
            temporal_right,
            required_side,
            None
            if all_sums is None
            else (
                all_sums[0][v],
                all_sums[1][v],
                all_sums[2][v],
                None if all_sums[3] is None else all_sums[3][v],
            ),
            slopes_by_row[v],
            is_ego_course,
            None if opposite_by_row is None else opposite_by_row[v],
            row_angle_deg,
            side_debug,
            None if road_segments_by_row is None else road_segments_by_row[v],
        )
        if row_candidates:
            candidates_by_row[v] = row_candidates

    raw_left = np.full(height, np.nan, dtype=np.float32)
    raw_right = np.full(height, np.nan, dtype=np.float32)
    left_observed = np.zeros(height, dtype=bool)
    right_observed = np.zeros(height, dtype=bool)
    if not candidates_by_row:
        if side_debug is not None:
            side_debug["selected_left"] = 0
            side_debug["selected_right"] = 0
        return raw_left, raw_right, left_observed, right_observed

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

    if side_debug is not None:
        side_debug["selected_left"] = int(np.count_nonzero(left_observed))
        side_debug["selected_right"] = int(np.count_nonzero(right_observed))

    return raw_left, raw_right, left_observed, right_observed


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
) -> np.ndarray:
    """좌우 경계의 중간을 중심선으로 계산한다."""

    centerline = np.full_like(
        left_boundary,
        np.nan,
    )

    valid = (
        ~np.isnan(
            left_boundary
        )
        & ~np.isnan(
            right_boundary
        )
    )

    centerline[
        valid
    ] = (
        left_boundary[valid]
        + right_boundary[valid]
    ) / 2.0

    return centerline


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
    raw_road_mask: np.ndarray,
    clean_road_mask: np.ndarray,
    reference_centerline: np.ndarray | None = None,
    temporal_centerline: np.ndarray | None = None,
    temporal_left: np.ndarray | None = None,
    temporal_right: np.ndarray | None = None,
    required_side: str | None = None,
    use_yellow_gap_limit: bool = True,
    smooth_course: bool = False,
    opposite_line_mask: np.ndarray | None = None,
    observable_mask: np.ndarray | None = None,
    is_ego_course: bool = False,
    use_single_line_angle_bias: bool = False,
    side_debug: dict[str, object] | None = None,
    boundary_segments_by_row: list[list[tuple[int, int]]] | None = None,
    opposite_segments_by_row: list[list[tuple[int, int]]] | None = None,
    road_segments_by_row: list[list[tuple[int, int]]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """행별 후보를 전체 경로로 연결해 교차로에서도 연속적인 경계를 만든다."""

    raw_left, raw_right, left_observed, right_observed = track_boundary_path(
        boundary_mask,
        raw_road_mask,
        clean_road_mask,
        reference_centerline,
        temporal_centerline,
        temporal_left,
        temporal_right,
        required_side,
        opposite_line_mask,
        observable_mask,
        is_ego_course,
        use_single_line_angle_bias,
        side_debug,
        boundary_segments_by_row,
        opposite_segments_by_row,
        road_segments_by_row,
    )
    if use_yellow_gap_limit:
        interpolated_left, interpolated_right = interpolate_yellow_boundary_pair(
            raw_left,
            raw_right,
        )
    else:
        interpolated_left, interpolated_right = interpolate_boundary_pair(
            raw_left,
            raw_right,
        )
    # 추적 허용 간격(0.35 m)보다 보간 간격(0.20 m)이 짧아서
    # 같은 DP 경로에 포함됐지만 화면에서 떨어진 덩어리가 남을
    # 수 있다. 차량에 가장 가까운 충분히 긴 연속 구간 하나만 유지한다.
    interpolated_left, interpolated_right = keep_nearest_continuous_run(
        interpolated_left,
        interpolated_right,
    )
    if smooth_course:
        interpolated_left, interpolated_right = smooth_boundary_pair(
            interpolated_left,
            interpolated_right,
        )
    return (
        left_observed,
        right_observed,
        interpolated_left,
        interpolated_right,
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
DASH_MIN_ROAD_SUPPORT_RATIO = 0.0005
DASH_MAX_LINE_THICKNESS_PX = 8
DASH_ENDPOINT_TANGENT_LENGTH_M = 0.08
DASH_DIRECTIONAL_EIGEN_RATIO = 2.0
# Max lateral distance (m) from a RoadBranch centerline to keep a dash blob
# for that fork path (tune_lane_detect dash_left / dash_right).
DASH_BRANCH_ASSOC_M = 0.22

# 끝점 링크로 안전성이 확인된 한 체인 안에서만 중심점을 2차 곡선으로 잇는다.
# 튜너의 Residual 25 px는 2배 확대 미리보기 기준이므로 Metric BEV에서는
# 약 12.5 px = 0.05 m다.
DASH_CHAIN_CURVE_DEGREE = 2
DASH_CHAIN_CURVE_MAX_RESIDUAL_M = 0.05
DASH_CHAIN_CURVE_FIT_ITERATIONS = 3
DASH_CHAIN_CURVE_MIN_COMPONENTS = 3


def find_crossing_lines(
    color_bev: np.ndarray,
    road_mask: np.ndarray,
    road_segments_by_row: list[list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """도로를 가로지르는 실선을 '행별 가로 커버리지'로 찾는다.

    가로 실선은 어느 행에서 도로 폭 대부분을 색으로 덮는다(가로지르니까).
    세로 차선은 가장자리만 덮어 커버리지가 낮다. 방향(PCA)에 의존하지
    않아 곡률에 강하고, warp 왜곡이 큰 원거리(상단)는 제외해 오탐을 줄인다.

    커버리지는 '연속된 도로 구간마다' 따로 잰다. 한 행의 최좌단~최우단을 도로
    폭으로 보면, 갈림길·회전교차로처럼 떨어진 옆 도로가 같이 보일 때 span이
    통째로 커져 커버리지가 희석된다. 그러면 내 도로를 100% 가로지르는 실선도
    검출에서 탈락하고, 그 선이 셀 커터에 남아 도로를 끊어 버린다.
    """

    result = np.zeros_like(color_bev)
    top_cut = int(BEV_HEIGHT * CROSSING_TOP_EXCLUDE_RATIO)
    min_span_px = int(round(CROSSING_MIN_SPAN_M / METERS_PER_PIXEL))
    for row in range(top_cut, BEV_HEIGHT):
        road_segments = (
            road_segments_by_row[row]
            if road_segments_by_row is not None
            else find_line_segments(road_mask[row])
        )
        for left, right in road_segments:
            span = right - left + 1
            if span < min_span_px:
                continue
            color_count = int(
                np.count_nonzero(color_bev[row, left:right + 1])
            )
            if color_count / span >= CROSSING_COVERAGE_RATIO:
                result[row, left:right + 1] = 255
    return result


def has_crossing_line(crossing_mask: np.ndarray) -> bool:
    """가로 실선 마스크가 충분한 행 수를 가지는지(단발 노이즈 배제)."""

    if crossing_mask.size == 0:
        return False
    return int(np.count_nonzero(crossing_mask.any(axis=1))) >= CROSSING_MIN_ROWS


def remove_crossing_from_boundary_mask(
    boundary_mask: np.ndarray,
    crossing_mask: np.ndarray,
) -> np.ndarray:
    """가로 실선과 주변 행을 세로 차선 추적용 마스크에서만 제거한다."""

    margin_rows = make_odd(
        int(round(CROSSING_REMOVAL_MARGIN_M / METERS_PER_PIXEL))
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, margin_rows))
    removal_mask = cv2.dilate(crossing_mask, kernel, iterations=1)
    return cv2.bitwise_and(boundary_mask, cv2.bitwise_not(removal_mask))


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
    road_clean: np.ndarray,
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
    road_support = cv2.dilate(
        road_clean,
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
                np.count_nonzero(road_support[sample_y[inside], sample_x[inside]])
            ) / float(np.count_nonzero(inside))
            if support_ratio < DASH_MIN_ROAD_SUPPORT_RATIO:
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


def make_interpolation_preview(
    boundary_mask: np.ndarray,
    interpolated_left: np.ndarray,
    interpolated_right: np.ndarray,
    label: str,
) -> np.ndarray:
    """추적에 실제 사용한 경계 마스크 위에 보간 결과를 표시한다."""

    preview = cv2.cvtColor(boundary_mask, cv2.COLOR_GRAY2BGR)
    draw_boundary(preview, interpolated_left, INTERPOLATED_LINE_COLOR)
    draw_boundary(preview, interpolated_right, INTERPOLATED_LINE_COLOR)

    cv2.putText(
        preview,
        f"{label}  BOUNDARY MASK + INTERPOLATED BOUNDARIES",
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def show_visualization(
    cropped_frame: np.ndarray,
    bev: np.ndarray,
    white_bev: np.ndarray,
    yellow_bev: np.ndarray,
    yellow_dash_points_bev: np.ndarray | None,
    yellow_connected_bev: np.ndarray,
    road_clean: np.ndarray,
    white_left: np.ndarray,
    white_right: np.ndarray,
    yellow_left: np.ndarray,
    yellow_right: np.ndarray,
    yellow_side_debug: dict[str, object] | None,
) -> None:
    """현재 LANE_VISUALIZE 모드에서 켜진 창만 표시한다.

    CONTROL 모드에서는 안 띄우는 창의 프리뷰를 아예 만들지 않는다(프리뷰 생성이
    프레임당 수 ms다).
    """

    def scaled(image: np.ndarray, nearest: bool = False) -> np.ndarray:
        interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(
            image,
            None,
            fx=VISUALIZATION_SCALE,
            fy=VISUALIZATION_SCALE,
            interpolation=interpolation,
        )

    def debug_count(name: str) -> int:
        if yellow_side_debug is None:
            return 0
        return int(yellow_side_debug.get(name, 0))

    yellow_debug_lines: tuple[str, ...] = ()
    if yellow_side_debug is not None:
        angle_value = yellow_side_debug.get("angle_deg")
        component_angle_values = tuple(
            float(value)
            for value in yellow_side_debug.get("component_angles", ())
        )
        if angle_value is None:
            angle_text = "OFF"
            bias_text = "NONE"
        else:
            angle = float(angle_value)
            angle_text = f"{angle:+.2f}med"
            if component_angle_values and all(
                value < 0.0 for value in component_angle_values
            ):
                bias_text = "LEFT"
            elif component_angle_values and all(
                value > 0.0 for value in component_angle_values
            ):
                bias_text = "RIGHT"
            elif any(value != 0.0 for value in component_angle_values):
                bias_text = "MIXED"
            else:
                bias_text = "NONE"
        yellow_debug_lines = (
            "SIDE "
            f"ANG={angle_text} BIAS={bias_text} "
            f"COMP={debug_count('components')}/"
            f"{debug_count('angled_components')}",
            "SEL "
            f"L={debug_count('selected_left')} "
            f"R={debug_count('selected_right')}  "
            f"VALID L={debug_count('left_valid')} "
            f"R={debug_count('right_valid')}",
            "ROW "
            f"ANGLE={debug_count('rows_angle')} "
            f"SINGLE={debug_count('rows_single')} "
            f"MULTI={debug_count('rows_multi')}",
            "L REJ "
            f"ROAD={debug_count('left_road')} "
            f"RAW={debug_count('left_raw')} "
            f"WHITE={debug_count('left_white')} "
            f"WP={debug_count('left_white_penalty')} "
            f"VIEW={debug_count('left_view')}",
            "R REJ "
            f"ROAD={debug_count('right_road')} "
            f"RAW={debug_count('right_raw')} "
            f"WHITE={debug_count('right_white')} "
            f"WP={debug_count('right_white_penalty')} "
            f"VIEW={debug_count('right_view')}",
        )

    if window_enabled("lane_origin"):
        cv2.imshow("lane_origin", cropped_frame)
    if window_enabled("white_hsv"):
        cv2.imshow("white_hsv", scaled(white_bev, nearest=True))
    if window_enabled("yellow_hsv"):
        cv2.imshow("yellow_hsv", scaled(yellow_bev, nearest=True))
    if (
        window_enabled("yellow_dash_points")
        and yellow_dash_points_bev is not None
    ):
        cv2.imshow(
            "yellow_dash_points",
            scaled(yellow_dash_points_bev, nearest=True),
        )
    if window_enabled("yellow_dash_connected"):
        cv2.imshow(
            "yellow_dash_connected",
            scaled(yellow_connected_bev, nearest=True),
        )
    if window_enabled("white_boundaries"):
        cv2.imshow(
            "white_boundaries",
            scaled(
                make_boundary_preview(
                    bev,
                    road_clean,
                    white_left,
                    white_right,
                    "WHITE",
                )
            ),
        )
    if window_enabled("yellow_boundaries"):
        cv2.imshow(
            "yellow_boundaries",
            scaled(
                make_boundary_preview(
                    bev,
                    road_clean,
                    yellow_left,
                    yellow_right,
                    "YELLOW",
                    yellow_debug_lines,
                )
            ),
        )
    if window_enabled("white_interpolation"):
        cv2.imshow(
            "white_interpolation",
            scaled(
                make_interpolation_preview(
                    white_bev,
                    white_left,
                    white_right,
                    "WHITE",
                )
            ),
        )
    if window_enabled("yellow_interpolation"):
        cv2.imshow(
            "yellow_interpolation",
            scaled(
                make_interpolation_preview(
                    yellow_connected_bev,
                    yellow_left,
                    yellow_right,
                    "YELLOW",
                )
            ),
        )
    if window_enabled("drivable_area"):
        cv2.imshow("drivable_area", scaled(road_clean, nearest=True))
    cv2.waitKey(1)


def detect(frame: np.ndarray) -> LaneDetections:
    """색상별 좌우 경계, 노란선 플래그와 road_clean을 반환한다."""

    detections, _debug = detect_with_debug(frame)
    return detections


def detect_with_debug(frame: np.ndarray) -> tuple[LaneDetections, LaneDebugFrame]:
    """Runtime detections plus intermediate masks for mode tuners."""

    global cached_shape
    global last_white_left
    global last_white_right
    global last_yellow_left
    global last_yellow_right

    if frame is None or frame.size == 0:
        return LaneDetections(), LaneDebugFrame()

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

    road_raw = cv2.bitwise_or(black_bev, red_bev)
    course_lines = cv2.bitwise_or(white_bev, yellow_bev)
    road_clean = fill_road_surface_holes(road_raw, course_lines)

    # 카메라 화각 밖(BEV 검은 쐐기)은 '도로 없음'이 아니라 '모름'이다.
    # 경계 후보의 도로 겹침을 여기서 본 픽셀만으로 따지게 한다.
    observable_bev = bev_observable_mask()

    # 가로 정지선/진입선은 이벤트 검출에는 남기되 세로 노란 경계
    # 추적에서는 제거해 경계 끝이 ㄴ자로 꺾이는 것을 막는다.
    crossing_road_segments_by_row = find_line_segments_by_row(road_clean)
    crossing_mask = find_crossing_lines(
        yellow_bev,
        road_clean,
        crossing_road_segments_by_row,
    )
    white_crossing_mask = find_crossing_lines(
        white_bev,
        road_clean,
        crossing_road_segments_by_row,
    )

    # 가로 마킹은 '도로 위에 그려진' 것이다. 그 자리도 당연히 주행가능 영역인데,
    # 마킹 픽셀은 흑색 도로로 안 잡히고 가장자리 안티에일리어싱은 어느 색으로도
    # 안 잡혀 도로에 가로 구멍이 뚫린다. fill_road_surface_holes의 line-support
    # 게이트로는 이 구멍이 안 메워져서, 셀이 가로로 끊기고 갈래 추적이 분기점에
    # 닿기도 전에 끝나 버린다(end@NN). 그래서 검출한 가로 마킹을 도로로 되돌린다.
    road_clean = cv2.bitwise_or(
        road_clean,
        cv2.dilate(
            cv2.bitwise_or(crossing_mask, white_crossing_mask),
            CROSSING_FILL_KERNEL,
        ),
    )

    yellow_boundary_raw_bev = remove_crossing_from_boundary_mask(
        yellow_bev,
        crossing_mask,
    )
    yellow_dash_points_bev = (
        extract_dash_point_mask(yellow_boundary_raw_bev)
        if window_enabled("yellow_dash_points")
        else None
    )
    yellow_boundary_bev = connect_dashed_components(
        yellow_boundary_raw_bev,
        road_clean,
    )

    # 흰 가로 마킹(갈림길 입구의 점선 진입선)도 코스 경계가 아니다. 셀 커터에
    # 남으면 도로를 가로로 끊어, 갈래 추적이 분기점에 닿기도 전에 끝나 버린다.
    white_cut_bev = remove_crossing_from_boundary_mask(
        white_bev,
        white_crossing_mask,
    )
    # 세로형 흰 점선(갈림/합류 가이드) — 가로 진입선은 extract에서 걸러진다.
    white_dash_points_bev = extract_dash_point_mask(white_cut_bev)
    white_dash_connected_bev = connect_dashed_components(
        white_cut_bev,
        road_clean,
    )

    # 같은 마스크를 흰/노란 추적, 반대색 검사와 도로 기준선 계산에서
    # 반복 분리하지 않도록 프레임당 한 번만 행 구간 목록을 만든다.
    white_segments_by_row = find_line_segments_by_row(white_bev)
    yellow_segments_by_row = find_line_segments_by_row(yellow_boundary_bev)
    road_segments_by_row = find_line_segments_by_row(road_clean)

    previous_white_center = None
    if last_white_left is not None and last_white_right is not None:
        previous_white_center = centerline_from_boundaries(
            last_white_left, last_white_right
        )

    # 직전 프레임에서 확정한 ego 코스 색. 그 코스의 차로만 차량을 품어야 한다.
    # (흰 도로를 달리는 중이면 노란 코스는 '옆 도로'이므로 품을 이유가 없다.)
    ego_course = last_ego_course_color

    (
        white_left_observed,
        white_right_observed,
        white_left,
        white_right,
    ) = build_global_boundary_course(
        boundary_mask=white_bev,
        raw_road_mask=road_raw,
        clean_road_mask=road_clean,
        temporal_centerline=previous_white_center,
        temporal_left=last_white_left,
        temporal_right=last_white_right,
        use_yellow_gap_limit=False,
        smooth_course=True,
        # 흰 차로 안에 노란 차선이 들어앉으면 그건 흰 도로가 아니다.
        # 가로선(정지선/진입선)은 코스 경계가 아니므로 제거한 마스크를 쓴다.
        opposite_line_mask=yellow_boundary_bev,
        observable_mask=observable_bev,
        is_ego_course=ego_course == "white",
        boundary_segments_by_row=white_segments_by_row,
        opposite_segments_by_row=yellow_segments_by_row,
        road_segments_by_row=road_segments_by_row,
    )
    white_centerline = centerline_from_boundaries(white_left, white_right)

    previous_yellow_center = None
    if last_yellow_left is not None and last_yellow_right is not None:
        previous_yellow_center = centerline_from_boundaries(
            last_yellow_left, last_yellow_right
        )

    # 인코스 판정 기준선: 흰 중심선이 있으면 그것, 없으면(노란선만 있는
    # 회전교차로 등) 주행가능영역 중심선으로 폴백한다. 기준이 NaN이면
    # required_side 판정이 통째로 무효화되어 인코스 우선이 안 걸린다.
    inner_course_reference = resolve_inner_course_reference(
        white_centerline,
        road_clean,
        road_segments_by_row,
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
    ) = build_global_boundary_course(
        boundary_mask=yellow_boundary_bev,
        raw_road_mask=road_raw,
        clean_road_mask=road_clean,
        # 이미 노란 코스 위를 달리고 있다면 '흰 도로 중심선'은 기준이 아니다.
        # 그걸 기준으로 두면 center_error가 노란 차로를 흰 도로 쪽으로 끌어당겨,
        # 노란선 하나만 보일 때 그 선을 반대쪽 경계로 읽게 만든다.
        # 기준을 None으로 두면 BEV 중앙(=차량)이 기준이 되고, 이건 참이다.
        # 인코스 우선(required_side)도 '진입 전에 어느 노란 코스냐'를 고르는
        # 규칙이지, 이미 그 위에 있을 때 쓰는 규칙이 아니다.
        reference_centerline=(
            None if ego_course == "yellow" else inner_course_reference
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
        opposite_line_mask=white_bev,
        observable_mask=observable_bev,
        is_ego_course=ego_course == "yellow",
        use_single_line_angle_bias=True,
        side_debug=yellow_side_debug,
        boundary_segments_by_row=yellow_segments_by_row,
        opposite_segments_by_row=white_segments_by_row,
        road_segments_by_row=road_segments_by_row,
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
    # 기존 left/right ID를 우선 유지한다. 후보의 도로 겹침 점수는
    # road_clean이 선의 어느 쪽인지 판단하는 보조 근거가 된다.
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
    road_branches, road_cells, ego_road_color = build_road_branches_cells(
        road_clean,
        white_cut_bev,
        yellow_boundary_bev,
    )
    fork_active = len(road_branches) >= 2

    # 흰/노란 차선 센터라인(좌우 경계 중점) → base_link 점열
    white_centerline_points = boundary_to_vehicle_points(
        centerline_from_boundaries(white_left, white_right)
    )
    yellow_centerline_points = boundary_to_vehicle_points(
        centerline_from_boundaries(yellow_left, yellow_right)
    )
    # 노란 가로 실선(정지선/진입선) 등장 여부
    yellow_crossing_line = has_crossing_line(crossing_mask)
    white_crossing_line = has_crossing_line(white_crossing_mask)

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

    if VISUALIZE:
        bev = bev_color
        if window_enabled("road_branches"):
            branch_preview = make_course_cell_preview(
                bev, road_cells, road_branches, ego_road_color
            )
            cv2.imshow(
                "road_branches",
                cv2.resize(
                    branch_preview,
                    None,
                    fx=VISUALIZATION_SCALE,
                    fy=VISUALIZATION_SCALE,
                    interpolation=cv2.INTER_NEAREST,
                ),
            )
        if window_enabled("line_fill"):
            # (가로선 시각화) 행별 가로 커버리지로 찾은 가로 실선을 빨강으로.
            fill_view = cv2.cvtColor(road_raw, cv2.COLOR_GRAY2BGR)
            fill_view[crossing_mask > 0] = (0, 0, 255)
            cv2.imshow(
                "line_fill",
                cv2.resize(
                    fill_view,
                    None,
                    fx=VISUALIZATION_SCALE,
                    fy=VISUALIZATION_SCALE,
                    interpolation=cv2.INTER_NEAREST,
                ),
            )
        show_visualization(
            cropped_frame=frame,
            bev=bev,
            white_bev=white_bev,
            yellow_bev=yellow_bev,
            yellow_dash_points_bev=yellow_dash_points_bev,
            yellow_connected_bev=yellow_boundary_bev,
            road_clean=road_clean,
            white_left=white_left,
            white_right=white_right,
            yellow_left=yellow_left,
            yellow_right=yellow_right,
            yellow_side_debug=yellow_side_debug,
        )

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
        crossing_mask=crossing_mask,
        white_crossing_mask=white_crossing_mask,
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
    )
    return detections, debug


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
    """road_clean의 공통 진입부와 분기 구간을 합친 경로 후보."""

    lateral_rank: int = 0
    confidence: float = 0.0
    width: float = 0.0
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )

def find_drivable_segments(row: np.ndarray) -> list[tuple[int, int]]:
    """한 BEV 행에서 최소 폭을 만족하는 주행 가능 구간을 찾는다."""

    return [
        (left, right)
        for left, right in find_line_segments(row)
        if right - left + 1 >= MIN_BRANCH_WIDTH_PX
    ]


def segment_center(segment: tuple[int, int]) -> float:
    return (segment[0] + segment[1]) / 2.0


def drivable_reference_centerline(
    road_clean: np.ndarray,
    road_segments_by_row: list[list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """차량이 달리는 도로의 행별 중심 열(없으면 NaN).

    인코스/아웃코스 판정의 기준축이다. 흰 중심선이 없는 구간(노란선만 있는
    회전교차로 등)에서도 항상 존재하므로 required_side 판정이 무효화되지
    않는다. 차량 정면축에서 시작해 근거리→원거리로 연속된 도로 구간을
    따라가므로 굽은 도로에서도 '도로 기준' 좌/우가 된다.
    """

    reference = np.full(BEV_HEIGHT, np.nan, dtype=np.float32)
    running = (BEV_WIDTH - 1) / 2.0
    for row in range(BEV_HEIGHT - 1, -1, -1):
        segments = (
            find_drivable_segments(road_clean[row])
            if road_segments_by_row is None
            else [
                segment
                for segment in road_segments_by_row[row]
                if segment[1] - segment[0] + 1 >= MIN_BRANCH_WIDTH_PX
            ]
        )
        if not segments:
            continue
        segment = min(
            segments,
            key=lambda item: abs(segment_center(item) - running),
        )
        running = segment_center(segment)
        reference[row] = running
    return reference


def resolve_inner_course_reference(
    white_centerline: np.ndarray,
    road_clean: np.ndarray,
    road_segments_by_row: list[list[tuple[int, int]]] | None = None,
) -> np.ndarray:
    """인코스 판정 기준선: 흰 중심선이 있으면 그것, 없으면 도로 중심선으로 폴백."""

    drivable = drivable_reference_centerline(
        road_clean,
        road_segments_by_row,
    )
    return np.where(np.isnan(white_centerline), drivable, white_centerline)












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
    """Dash-lane mode: raw points + connected, optionally filtered by fork path."""

    if debug.bev.size == 0:
        return np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

    preview = debug.bev.copy()
    road_overlay = np.zeros_like(preview)
    road_overlay[debug.road_clean > 0] = DRIVABLE_COLOR
    preview = cv2.addWeighted(preview, 1.0, road_overlay, 0.35, 0.0)

    yellow_pts = debug.yellow_dash_points_bev
    yellow_conn = debug.yellow_connected_bev
    white_pts = debug.white_dash_points_bev
    white_conn = debug.white_dash_connected_bev
    if yellow_pts.size == 0:
        yellow_pts = np.zeros(debug.bev.shape[:2], dtype=np.uint8)
    if yellow_conn.size == 0:
        yellow_conn = yellow_pts
    if white_pts.size == 0:
        white_pts = np.zeros(debug.bev.shape[:2], dtype=np.uint8)
    if white_conn.size == 0:
        white_conn = white_pts

    branch = select_road_branch(debug.road_branches, focus)
    if branch is not None:
        yellow_pts = filter_dash_mask_by_branch(yellow_pts, branch)
        yellow_conn = filter_dash_mask_by_branch(yellow_conn, branch)
        white_pts = filter_dash_mask_by_branch(white_pts, branch)
        white_conn = filter_dash_mask_by_branch(white_conn, branch)
        draw_vehicle_polyline(
            preview,
            branch.points[:, :2],
            BRANCH_COLORS[int(branch.lateral_rank) % len(BRANCH_COLORS)],
            f"B{int(branch.lateral_rank)}",
        )
    elif focus == "all":
        for b in debug.road_branches:
            draw_vehicle_polyline(
                preview,
                b.points[:, :2],
                BRANCH_COLORS[int(b.lateral_rank) % len(BRANCH_COLORS)],
                f"B{int(b.lateral_rank)}",
            )

    # Dim connected envelopes, bright raw dash points.
    preview[yellow_conn > 0] = (
        0.55 * preview[yellow_conn > 0] + 0.45 * np.array([0, 180, 255])
    ).astype(np.uint8)
    preview[white_conn > 0] = (
        0.55 * preview[white_conn > 0] + 0.45 * np.array([200, 200, 200])
    ).astype(np.uint8)
    preview[yellow_pts > 0] = (0, 255, 255)
    preview[white_pts > 0] = (255, 255, 255)

    y_n = int(np.count_nonzero(yellow_pts))
    w_n = int(np.count_nonzero(white_pts))
    cv2.putText(
        preview,
        (
            f"DASH focus={focus.upper()}  Ypx={y_n} Wpx={w_n}  "
            f"assoc<={DASH_BRANCH_ASSOC_M:.2f}m  "
            f"gap={DASH_MAX_FORWARD_GAP_M:.2f}m lat={DASH_MAX_LATERAL_ERROR_M:.3f}m"
        ),
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "cyan=yellow dash  white=white dash  tint=connected  branch=L/R path",
        (4, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (200, 255, 200),
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


def make_fork_focus_preview(
    debug: LaneDebugFrame,
    *,
    focus: str = "all",
) -> np.ndarray:
    """Fork preview; focus in {all, left, right} dims non-selected branches."""

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
        if debug.yellow_dash_points_bev.size:
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
