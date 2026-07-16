"""차선 인지 모듈 — Metric BEV + HSV.

입력  : BGR 프레임 (numpy, 320x180)
출력  : LaneResult (white_centerline 등, base_link 미터 좌표)

파이프라인
----------
1. MetricBev.warp : 카메라 프레임 → 등거리 BEV.
2. HSV 임계        : 흰색(주 차선) 마스크. config/lane_vision.yaml: hsv.white.
3. morphology      : open(스펙클 제거) → close(선 끊김 메움).
4. 근거리 시드      : 차는 두 선 사이 → 근거리 밴드 ego-중심 좌/우 분할로 식별.
5. 행별 좌우선 추적 : 근거리→원거리, 이전 위치 근처(근접)만으로 좇음.
                     ego-중심 재판정을 안 하므로 급커브에서 선이 중심을 넘어도 안 뒤집힘.
6. centerline      : 좌·우 모두 → 중점 / 한쪽만 → (track_width-line_width)/2 오프셋.
7. 미터 변환·평활  : bev_uv_to_xy (x 전방+, y 우측+) → 이동평균.

단측선 좌/우 오배정 방지(3중):
  1) 근거리 시드로 identity 확정,
  2) 도로영역(검정∪빨강 노면) 단서 — 단선의 어느 쪽이 주행가능면인지 보고 좌/우
     확정. 메모리가 없는 콜드스타트/reset() 직후에도 동작하며 시간로직보다 우선,
  3) 프레임 간 시간적 연속성(_reconcile_seed)으로 "우선을 좌선으로" 뒤집힘 방지.
reset() 로 시간 상태 초기화.

좌표 규약: white_centerline 은 [x 전방+, y 우측+] 미터. (LaneController 규약,
metric_bev.py 참조. 우회전 → y>0.)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from ..types import LaneMarking, LaneResult
from .metric_bev import MetricBev, load_bev_params


def _default_config_path() -> Optional[Path]:
    """설치 share / 소스 트리에서 lane_vision.yaml 탐색."""
    env = os.environ.get("LANE_VISION_CONFIG")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve()
    candidates = [
        # 소스 트리: modules/ → inference/ → src/inference/ → src/ → ws/config
        here.parents[4] / "config" / "lane_vision.yaml",
        here.parents[3] / "config" / "lane_vision.yaml",
    ]
    # 설치 share: 런타임에 ament 로 찾음
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("inference"))
        candidates.append(share / "config" / "lane_vision.yaml")
    except Exception:
        pass
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_config(config_file: Optional[str]) -> dict:
    path = Path(config_file) if config_file else _default_config_path()
    if path is None or yaml is None or not Path(path).is_file():
        return {}
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _hsv_bounds(block: dict) -> Tuple[tuple, tuple]:
    return (
        (int(block.get("h_min", 0)), int(block.get("s_min", 0)), int(block.get("v_min", 0))),
        (int(block.get("h_max", 179)), int(block.get("s_max", 255)), int(block.get("v_max", 255))),
    )


class LaneDetector:
    """Metric BEV 흰색 차선 → base_link centerline 검출기."""

    def __init__(self, params: dict | str | None = None):
        # params: dict(설정) | str(config 파일 경로) | None(자동 탐색)
        config_file = None
        overrides: dict = {}
        if isinstance(params, str):
            config_file = params
        elif isinstance(params, dict):
            config_file = params.get("config_file")
            overrides = params

        cfg = _load_config(config_file)

        # --- BEV ---
        self.bev = MetricBev(load_bev_params(config_file or _default_config_path()))
        self.p = self.bev.params
        self.mpp = self.p.meters_per_pixel

        # --- HSV ---
        hsv = cfg.get("hsv") or {}
        self.white_lo, self.white_hi = _hsv_bounds(hsv.get("white") or {
            "s_max": 20, "v_min": 210})
        yblock = hsv.get("yellow")
        self.yellow_lo, self.yellow_hi = (_hsv_bounds(yblock) if yblock else (None, None))
        # 도로 영역(주행가능면) — 단측선 좌/우 판정 보강용. 검정 노면 ∪ 빨강 노면.
        bblock = hsv.get("black_road")
        self.black_lo, self.black_hi = (_hsv_bounds(bblock) if bblock else (None, None))
        rblock = hsv.get("red_road")
        self.red_lo, self.red_hi = (_hsv_bounds(rblock) if rblock else (None, None))
        self.road_enabled = self.black_lo is not None or self.red_lo is not None

        # --- 검출 튜닝 ---
        ld = cfg.get("lane_detect") or {}
        self.color = str(overrides.get("color", ld.get("color", "white")))
        self.morph_open = int(overrides.get("morph_open", ld.get("morph_open", 3)))
        self.morph_close = int(overrides.get("morph_close", ld.get("morph_close", 7)))
        self.min_row_pixels = int(overrides.get("min_row_pixels", ld.get("min_row_pixels", 2)))
        self.smooth_window = int(overrides.get("smooth_window", ld.get("smooth_window", 5)))
        self.max_lane_gap_m = float(
            overrides.get("max_lane_gap_m", ld.get("max_lane_gap_m", 0.60)))
        self.line_width_m = float(overrides.get("line_width_m", ld.get("line_width_m", 0.03)))
        # 도로영역 기반 단측선 좌/우 판정 튜닝
        self.road_min_px = int(overrides.get("road_min_px", ld.get("road_min_px", 80)))
        self.road_side_margin = float(
            overrides.get("road_side_margin", ld.get("road_side_margin", 0.25)))

        # 단측선 오프셋: 검출은 선 중심(런 centroid) → 차로 중심까지 거리.
        #   = track_width/2 - line_width/2  (트랙 스펙 0.35, 0.03 → 0.16 m)
        self.lane_half_offset_px = max(
            1, int(round((self.p.track_width_m / 2.0 - self.line_width_m / 2.0) / self.mpp)))
        self.guide_half_px = self.lane_half_offset_px
        # 추적 파라미터 (BEV 픽셀)
        self.track_margin_px = max(8, int(round(0.12 / self.mpp)))   # 선 좇는 검색 반경
        self.reseed_gate_px = max(12, int(round(0.18 / self.mpp)))   # 단측선 시간적 식별 게이트
        self.min_line_sep_px = max(6, int(round(0.15 / self.mpp)))   # 좌·우가 같은 선 잡는 것 방지
        self.seed_min_peak = 3                                       # 시드 인정 최소 열합
        # centerline 출력 샘플 간격 (≈1 cm)
        self.row_step = max(1, int(round(0.012 / self.mpp)))

        # --- 프레임 간 시간적 상태 (좌/우 identity 연속성) ---
        # 단측선일 때 "우선을 좌선으로" 오인하지 않도록 직전 프레임의 근거리
        # 좌/우 선 column 을 기억해 식별에 사용. 색상별(흰/노랑) 독립 상태.
        self._prev_max_miss = 4      # 이 이상 연속 미검출이면 기억 폐기
        self._white_state = self._new_state()
        self._yellow_state = self._new_state()

    @staticmethod
    def _new_state() -> dict:
        return {"prev_left": None, "prev_right": None, "miss_left": 0, "miss_right": 0}

    def reset(self) -> None:
        """시간적 추적 상태 초기화 (새 시퀀스/테스트 시작 시)."""
        self._white_state = self._new_state()
        self._yellow_state = self._new_state()

    # ------------------------------------------------------------------ mask
    def _morph(self, mask: np.ndarray) -> np.ndarray:
        if self.morph_open >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_open, self.morph_open))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        if self.morph_close >= 3:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_close, self.morph_close))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _road_mask(self, hsv: np.ndarray):
        """검정 ∪ 빨강 노면 마스크(주행가능영역 근사). 채널 미설정 시 None.

        빨강 노면은 H=0 을 감싸므로, red_road.h_min≤10 이면 상단(H≥175)도 포함해
        In 코스 빨강 노면을 놓치지 않는다.
        """
        road = None
        if self.black_lo is not None:
            road = cv2.inRange(hsv, self.black_lo, self.black_hi)
        if self.red_lo is not None:
            red = cv2.inRange(hsv, self.red_lo, self.red_hi)
            if int(self.red_lo[0]) <= 10:
                wrap_lo = (175, int(self.red_lo[1]), int(self.red_lo[2]))
                wrap_hi = (179, int(self.red_hi[1]), int(self.red_hi[2]))
                red = cv2.bitwise_or(red, cv2.inRange(hsv, wrap_lo, wrap_hi))
            road = red if road is None else cv2.bitwise_or(road, red)
        if road is None:
            return None
        return self._morph(road)

    def _masks(self, bev_bgr: np.ndarray):
        """BEV → (white, yellow, boundary, road). boundary=추종용 경계, road=주행면."""
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, self.white_lo, self.white_hi)
        yellow = (cv2.inRange(hsv, self.yellow_lo, self.yellow_hi)
                  if self.yellow_lo is not None else np.zeros_like(white))
        if self.color == "white":
            boundary = white
        elif self.color == "yellow":
            boundary = yellow
        else:  # both
            boundary = cv2.bitwise_or(white, yellow)
        road = self._road_mask(hsv)
        return self._morph(white), self._morph(yellow), self._morph(boundary), road

    # -------------------------------------------------------- run helpers
    @staticmethod
    def _runs(on_cols: np.ndarray) -> List[Tuple[int, int, float]]:
        """정렬된 on-column 인덱스 → (시작, 끝, 중심) 연속 런 목록."""
        if on_cols.size == 0:
            return []
        splits = np.where(np.diff(on_cols) > 1)[0]
        segments = np.split(on_cols, splits + 1)
        return [(int(s[0]), int(s[-1]), float(s.mean())) for s in segments if s.size]

    def _seed_sides(self, mask: np.ndarray, center: float):
        """근거리 밴드(차가 두 선 사이)에서 좌/우 시드 column.

        가장 가까운 행들에서는 차가 반드시 좌·우 경계선 사이에 있으므로
        'ego 중심 좌/우 분할'이 신뢰 가능하다. 급커브의 좌우 뒤바뀜은 far
        영역에서만 생기며, 그건 시드가 아니라 근접-추적으로 처리한다.
        """
        H, W = mask.shape
        band = mask[int(H * 0.72):, :]                 # 하단 ~28% = 근거리
        colsum = (band > 0).sum(axis=0).astype(np.float32)
        if colsum.sum() == 0:
            return None, None
        if colsum.size >= 5:
            colsum = np.convolve(colsum, np.ones(5) / 5.0, mode="same")
        c = int(round(center))
        left = right = None
        if c > 0:
            lp = int(np.argmax(colsum[:c]))
            if colsum[lp] >= self.seed_min_peak:
                left = float(lp)
        if c < W:
            rp = int(np.argmax(colsum[c:]) + c)
            if colsum[rp] >= self.seed_min_peak:
                right = float(rp)
        return left, right

    def _road_side_of_line(self, road, col) -> int:
        """근거리 밴드에서 단선(col) 기준 도로(주행가능면)가 좌/우 어디에 더 많은가.

        반환: +1 오른쪽에 도로 → 단선은 '좌측 경계'
              -1 왼쪽에 도로   → 단선은 '우측 경계'
               0 판정 보류(도로 근거 부족 / 양쪽 다 도로 = 회전교차로 등)
        메모리 불필요 → 콜드스타트/reset 직후 단선에서도 좌/우 확정 가능.
        """
        if road is None:
            return 0
        H, W = road.shape
        band = road[int(H * 0.72):, :]                  # _seed_sides 와 동일 근거리 밴드
        c = int(round(col))
        lpx = int((band[:, :c] > 0).sum()) if c > 0 else 0
        rpx = int((band[:, c:] > 0).sum()) if c < W else 0
        tot = lpx + rpx
        if tot < self.road_min_px:
            return 0
        r = (rpx - lpx) / float(tot)
        if r > self.road_side_margin:
            return 1
        if r < -self.road_side_margin:
            return -1
        return 0

    def _reconcile_seed(self, left, right, center, state, road=None):
        """단측선 시드의 좌/우 identity 를 도로영역 + 직전 프레임으로 확정.

        차가 선 사이면(둘 다 보임) center-split 을 신뢰. 한쪽만 보이면:
          (a) 도로영역 단서가 확실하면 그쪽으로 확정(메모리 불필요, 최우선),
          (b) 아니면 직전 프레임의 좌/우 column 중 더 가까운 쪽으로 배정.
        → 차가 선을 살짝 넘어 단선이 중심 반대편에 나타나도 좌/우가 뒤집히지 않는다.
        """
        if left is not None and right is not None:
            return left, right                          # 선 사이 = 신뢰
        single = left if right is None else right
        if single is None:
            return None, None
        # (a) 도로영역 단서 — 메모리 없이도 좌/우 확정. 시간로직보다 우선.
        side = self._road_side_of_line(road, single)
        if side > 0:
            return single, None                         # 도로가 오른쪽 → 좌측 경계
        if side < 0:
            return None, single                         # 도로가 왼쪽  → 우측 경계
        # (b) 도로 단서 보류 → 프레임 간 시간적 연속성으로 배정
        pl, pr = state["prev_left"], state["prev_right"]
        has_l, has_r = pl is not None, pr is not None
        dl = abs(single - pl) if has_l else None
        dr = abs(single - pr) if has_r else None
        tight = self.reseed_gate_px                     # 양쪽 경쟁 시 (~0.18 m)
        # 한쪽만 추적 중이었으면 경쟁자가 없으므로 관대하게(반대편 선 차로폭
        # 거리보다는 작게) 그 트랙의 연속으로 본다.
        loose = max(1, int(round((self.p.track_width_m * 0.8) / self.mpp)))  # ~0.28 m
        if has_l and has_r:
            # 둘 다 기억됨 → 더 가까운 쪽, 단 tight 이내일 때만 확정
            if dl <= dr:
                return (single, None) if dl <= tight else (left, right)
            return (None, single) if dr <= tight else (left, right)
        if has_r and not has_l:                         # 우측만 추적 중이었음
            if dr <= loose:
                return None, single                     # 우선 연속 → 뒤집힘 방지
        if has_l and not has_r:                         # 좌측만 추적 중이었음
            if dl <= loose:
                return single, None
        # 기억 없음/멀음 → center-split 유지
        return left, right

    @staticmethod
    def _track_run(runs, expected: Optional[float], gate: float):
        """expected column 근처(±gate)의 런 중심. 없으면 None. (좌/우 무판정)"""
        if expected is None or not runs:
            return None
        cens = np.array([r[2] for r in runs], dtype=np.float32)
        d = np.abs(cens - expected)
        j = int(np.argmin(d))
        return float(cens[j]) if d[j] <= gate else None

    # ---------------------------------------------------- 추적 코어 (테스트 대상)
    def _track_centerline(self, mask: np.ndarray, state: dict, road=None):
        """boundary mask(BEV) → (pts_uv, stats). 시드→행별 근접추적→시간기억.

        pts_uv: (center_u, v, mode) 리스트. mode 0=양선,1=좌선만,2=우선만.
        좌/우 identity 는 근거리 시드에서 확정하고 이후 행에서는 근접으로만
        추적한다(ego 중심 재판정 없음) → 급커브에서 선이 중심을 넘어도 안 뒤집힘.
        state: 색상별 프레임 간 시간 상태(dict) — 흰/노랑 독립.
        road: 도로영역 마스크(선택). 단측선 좌/우 확정 보강용.
        """
        H, W = mask.shape
        center = self.bev.u_center

        left_seed, right_seed = self._seed_sides(mask, center)
        left_seed, right_seed = self._reconcile_seed(
            left_seed, right_seed, center, state, road)

        left_col, right_col = left_seed, right_seed
        pts_uv: List[Tuple[float, float, int]] = []
        left_hits = right_hits = both_hits = valid_rows = 0

        for v in range(H - 1, -1, -1):
            on = np.where(mask[v] > 0)[0]
            if on.size < self.min_row_pixels:
                continue
            runs = self._runs(on)
            L = self._track_run(runs, left_col, self.track_margin_px)
            R = self._track_run(runs, right_col, self.track_margin_px)
            # 좌·우가 같은 런을 잡으면(원거리 합류) 각자 expected 에 가까운 쪽만 유지
            if L is not None and R is not None and abs(R - L) < self.min_line_sep_px:
                if left_col is not None and right_col is not None:
                    if abs(L - left_col) <= abs(R - right_col):
                        R = None
                    else:
                        L = None
            if L is not None:
                left_col = L if left_col is None else 0.6 * L + 0.4 * left_col
                left_hits += 1
            if R is not None:
                right_col = R if right_col is None else 0.6 * R + 0.4 * right_col
                right_hits += 1

            cu = None
            mode = 0
            if L is not None and R is not None and (R - L) * self.mpp <= self.max_lane_gap_m:
                cu = 0.5 * (L + R)
                both_hits += 1
            elif L is not None:                     # 좌측 경계선만 → 우로 오프셋
                cu = L + self.guide_half_px
                mode = 1
            elif R is not None:                     # 우측 경계선만 → 좌로 오프셋
                cu = R - self.guide_half_px
                mode = 2

            if cu is not None and 0 <= cu < W:
                valid_rows += 1
                if (H - 1 - v) % self.row_step == 0:
                    pts_uv.append((cu, float(v), mode))

        # --- 시간적 기억 갱신 (근거리 시드 = identity 확정본) ---
        if left_seed is not None:
            state["prev_left"], state["miss_left"] = left_seed, 0
        else:
            state["miss_left"] += 1
            if state["miss_left"] > self._prev_max_miss:
                state["prev_left"] = None
        if right_seed is not None:
            state["prev_right"], state["miss_right"] = right_seed, 0
        else:
            state["miss_right"] += 1
            if state["miss_right"] > self._prev_max_miss:
                state["prev_right"] = None

        return pts_uv, {
            "left_hits": left_hits, "right_hits": right_hits,
            "both_hits": both_hits, "valid_rows": valid_rows,
            "left_seed": left_seed, "right_seed": right_seed,
        }

    # ------------------------------------------------------------------ detect
    def detect(self, frame) -> LaneResult:
        result = LaneResult()
        result.meters_per_pixel = self.mpp
        result.x_forward_max = self.p.x_max_m
        if frame is None or cv2 is None:
            return result

        bev = self.bev.warp(frame)
        white_m, yellow_m, mask, road = self._masks(bev)
        H, W = mask.shape

        # 주 경로: 주행 차로 중심선 (경계선 색 무관, boundary=흰∪노랑).
        # road 단서로 단측선 좌/우 오배정 방지(콜드스타트 포함).
        white_pts, st = self._track_centerline(mask, self._white_state, road)
        left_hits = st["left_hits"]
        right_hits = st["right_hits"]
        valid_rows = st["valid_rows"]
        centerline = self._pts_to_centerline(white_pts)

        # 노랑 전용 경로: In 모드 진입/회전교차로 추종용. 항상 호출해 상태 aging.
        yellow_pts, _yst = self._track_centerline(yellow_m, self._yellow_state)
        yellow_centerline = self._pts_to_centerline(yellow_pts)

        # --- 색상별 가시성 (근거리 밴드 픽셀 수 기준) ---
        band = slice(int(H * 0.55), H)
        white_px = int((white_m[band] > 0).sum())
        yellow_px = int((yellow_m[band] > 0).sum())
        px_thresh = max(20, int(0.02 / self.mpp))   # ≈2cm 상당 선 길이

        # --- LaneResult 채우기 ---
        total = max(valid_rows, 1)
        result.white_centerline = centerline
        result.yellow_centerline = yellow_centerline if yellow_px >= px_thresh else []
        result.white_visible = white_px >= px_thresh
        result.yellow_visible = yellow_px >= px_thresh
        result.white_confidence = float(min(1.0, white_px / (px_thresh * 4)))
        result.yellow_confidence = float(min(1.0, yellow_px / (px_thresh * 4)))
        result.left_visible = left_hits >= max(3, valid_rows * 0.3)
        result.right_visible = right_hits >= max(3, valid_rows * 0.3)
        result.left_confidence = float(min(1.0, left_hits / total))
        result.right_confidence = float(min(1.0, right_hits / total))
        # 노란 가로 실선(정지/교차 진입) 근사: 근거리에서 넓은 가로 폭 점유.
        if yellow_px >= px_thresh:
            yb = yellow_m[band]
            cols_hit = int((yb.sum(axis=0) > 0).sum())
            result.yellow_crossing_line = cols_hit >= int(W * 0.5)

        if centerline:
            xs = [p[0] for p in centerline]
            ys = [p[1] for p in centerline]
            heading = float(np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])) if len(xs) > 1 else 0.0
            result.lanes.append(LaneMarking(
                color=1, side_hint=3, confidence=result.white_confidence,
                length=float(xs[-1] - xs[0]), heading=heading, points=list(centerline)))
        return result

    def _pts_to_centerline(self, pts_uv) -> List[Tuple[float, float]]:
        """추적 pts_uv(center_u, v, mode) → base_link (x전방+, y우측+) 평활 polyline."""
        pts_uv = sorted(pts_uv, key=lambda p: -p[1])   # near(v큰)→far
        line = []
        for cu, v, _mode in pts_uv:
            x, y = self.bev.bev_uv_to_xy(cu, v)
            line.append((float(x), float(y)))
        return self._smooth(line, self.smooth_window)

    @staticmethod
    def _smooth(pts: List[Tuple[float, float]], win: int) -> List[Tuple[float, float]]:
        if win < 3 or len(pts) < win:
            return pts
        ys = np.array([p[1] for p in pts])
        k = np.ones(win) / win
        ys_s = np.convolve(ys, k, mode="same")
        # 경계는 원본 유지
        half = win // 2
        ys_s[:half] = ys[:half]
        ys_s[-half:] = ys[-half:]
        return [(pts[i][0], float(ys_s[i])) for i in range(len(pts))]

    # ---------------------------------------------------------- debug view
    def debug_view(self, frame) -> np.ndarray:
        """BEV + 마스크(흰=빨강, 노랑=초록) + centerline(초록점) 오버레이."""
        bev = self.bev.warp(frame)
        white_m, yellow_m, _, _road = self._masks(bev)
        vis = bev.copy()
        vis[white_m > 0] = (0, 0, 255)
        vis[yellow_m > 0] = (0, 255, 0)
        r = self.detect(frame)
        for x, y in r.yellow_centerline:                # 노랑 경로 = 주황
            u, v = self.bev.xy_to_bev_uv(x, y)
            if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                cv2.circle(vis, (int(u), int(v)), 2, (0, 165, 255), -1)
        for x, y in r.white_centerline:                 # 주 중심선 = 마젠타(마스크와 구분)
            u, v = self.bev.xy_to_bev_uv(x, y)
            if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                cv2.circle(vis, (int(u), int(v)), 2, (255, 0, 255), -1)
        cu = int(round(self.bev.u_center))
        cv2.line(vis, (cu, 0), (cu, vis.shape[0] - 1), (255, 255, 0), 1)
        return vis
