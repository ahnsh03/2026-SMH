"""LaneDetector 단위 테스트 — metric BEV 좌표 규약 + 단측선 좌/우 식별.

핵심: 한쪽 차선만 보일 때 우선을 좌선으로(또는 그 반대) 오인하지 않아야 한다.
합성 BEV boundary mask 로 추적 코어(_track_centerline)를 직접 검증한다.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_WS = Path(__file__).resolve().parents[3]        # 2026-SMH-team-new
sys.path.insert(0, str(_WS / "src" / "inference"))
CFG = str(_WS / "config" / "lane_vision.yaml")

from inference.modules.lane_detection import LaneDetector  # noqa: E402


def _det():
    d = LaneDetector(CFG)
    d.reset()
    return d


def _blank(det):
    return np.zeros((det.bev.bev_height, det.bev.bev_width), np.uint8)


def _vline(mask, col, thickness=6):
    c = int(round(col))
    h = thickness // 2
    mask[:, max(0, c - h):c + h + 1] = 255


def _mean_center_xy(det, pts_uv):
    xs, ys = [], []
    for cu, v, _m in pts_uv:
        x, y = det.bev.bev_uv_to_xy(cu, v)
        xs.append(float(x))
        ys.append(float(y))
    return np.mean(xs), np.mean(ys)


def test_metric_sign_right_positive():
    """오른쪽 column 은 y>0 (우측+), 왼쪽 column 은 y<0 이어야 한다."""
    det = _det()
    c = det.bev.u_center
    _, y_right = det.bev.bev_uv_to_xy(c + 50, 100)
    _, y_left = det.bev.bev_uv_to_xy(c - 50, 100)
    assert y_right > 0 > y_left


def test_both_lines_centered():
    """좌·우 선 대칭 → 중심선 y≈0."""
    det = _det()
    c = det.bev.u_center
    off = int(round(0.16 / det.mpp))       # 선 중심 ~±0.16 m
    m = _blank(det)
    _vline(m, c - off)
    _vline(m, c + off)
    pts, st = det._track_centerline(m, det._white_state)
    assert st["both_hits"] > 0
    _, ymean = _mean_center_xy(det, pts)
    assert abs(ymean) < 0.03               # 거의 중앙


def test_left_only_offsets_right():
    """좌측 선만 → 중심선은 그 선의 '오른쪽'(우로 오프셋)."""
    det = _det()
    c = det.bev.u_center
    line_col = c - int(round(0.16 / det.mpp))
    m = _blank(det)
    _vline(m, line_col)
    pts, st = det._track_centerline(m, det._white_state)
    assert st["left_hits"] > 0 and st["right_hits"] == 0
    # 모든 중심점의 column 이 선보다 오른쪽
    assert all(cu > line_col for cu, _v, _m in pts)
    _, line_y = det.bev.bev_uv_to_xy(line_col, 100)
    _, center_y = _mean_center_xy(det, pts)
    assert center_y > line_y               # 중심선이 선의 우측


def test_right_only_offsets_left():
    """우측 선만 → 중심선은 그 선의 '왼쪽'(좌로 오프셋)."""
    det = _det()
    c = det.bev.u_center
    line_col = c + int(round(0.16 / det.mpp))
    m = _blank(det)
    _vline(m, line_col)
    pts, st = det._track_centerline(m, det._white_state)
    assert st["right_hits"] > 0 and st["left_hits"] == 0
    assert all(cu < line_col for cu, _v, _m in pts)
    _, center_y = _mean_center_xy(det, pts)
    _, line_y = det.bev.bev_uv_to_xy(line_col, 100)
    assert center_y < line_y               # 중심선이 선의 좌측


def test_single_line_no_side_flip_temporal():
    """급주행: 우측 선만 계속 보이며 중심 근처로 이동해도 '좌선'으로 안 뒤집힘.

    프레임1: 우측 선이 ego 오른쪽. 프레임2: 같은 우측 선이 중심 근처(살짝 왼쪽)로.
    시간적 연속성으로 프레임2도 '우측 선'으로 식별 → 중심선은 계속 그 왼쪽.
    """
    det = _det()
    c = det.bev.u_center
    # 프레임1: 명확히 우측 (ego + 0.16 m)
    m1 = _blank(det)
    col1 = c + int(round(0.16 / det.mpp))
    _vline(m1, col1)
    _, st1 = det._track_centerline(m1, det._white_state)
    assert st1["right_seed"] is not None and st1["left_seed"] is None

    # 프레임2: 같은 선이 중심보다 약간 '왼쪽'으로 이동 (차가 우측 선을 살짝 넘음)
    m2 = _blank(det)
    col2 = c - int(round(0.03 / det.mpp))       # 중심 좌측 3cm
    _vline(m2, col2)
    pts2, st2 = det._track_centerline(m2, det._white_state)
    # center-split 이라면 '좌선'으로 볼 상황이지만, 시간적 식별로 '우선' 유지
    assert st2["right_seed"] is not None, "우측 선이 좌측으로 오인됨(뒤집힘)"
    assert st2["left_seed"] is None
    # 따라서 중심선은 그 선의 '왼쪽'
    assert all(cu < col2 for cu, _v, _m in pts2)


def test_curve_lines_cross_center_no_flip():
    """far 에서 두 선이 한쪽으로 휘어 중심을 넘어도 identity 유지."""
    det = _det()
    c = det.bev.u_center
    H = det.bev.bev_height
    off = int(round(0.16 / det.mpp))
    m = _blank(det)
    # near(v 큰쪽)=대칭, far(v 작은쪽)로 갈수록 둘 다 우측 이동 (우커브)
    for v in range(H):
        far = 1.0 - v / (H - 1)                  # v=0(far)→1, v=H-1(near)→0
        s = int(far * 2.2 * off)                 # far 로 갈수록 크게 우이동
        _pt(m, c - off + s, v)
        _pt(m, c + off + s, v)
    pts, st = det._track_centerline(m, det._white_state)
    assert st["both_hits"] > H * 0.3             # 대부분 양선 추적 성공
    # near 는 중앙, far 는 우측 → 중심선 y 가 near<far (우로 증가)
    ys = [det.bev.bev_uv_to_xy(cu, v)[1] for cu, v, _m in pts]
    xs = [det.bev.bev_uv_to_xy(cu, v)[0] for cu, v, _m in pts]
    order = np.argsort(xs)
    ys_sorted = np.array(ys)[order]
    assert ys_sorted[-1] > ys_sorted[0]         # far 로 갈수록 우측(우커브)


def _pt(mask, col, row, thickness=6):
    c = int(round(col))
    h = thickness // 2
    if 0 <= c < mask.shape[1]:
        mask[row, max(0, c - h):c + h + 1] = 255


def _road_band(det, c0, c1):
    """근거리 밴드([H*0.72:])의 [c0:c1] 열을 도로로 채운 마스크."""
    r = _blank(det)
    band0 = int(det.bev.bev_height * 0.72)
    lo = max(0, int(c0))
    hi = min(det.bev.bev_width, int(c1))
    r[band0:, lo:hi] = 255
    return r


def test_road_cue_left_boundary_cold_start():
    """콜드스타트(기억 없음): 단선이 중심 우측이라 center-split 이면 '우선'이지만,
    도로가 선의 오른쪽에 있으면 도로 단서로 '좌측 경계'로 확정한다."""
    det = _det()                                    # reset 됨 → prev 기억 없음
    c = det.bev.u_center
    line_col = c + int(round(0.05 / det.mpp))        # 중심 살짝 우측
    m = _blank(det)
    _vline(m, line_col)
    road = _road_band(det, line_col + 10, det.bev.bev_width)   # 도로=선의 오른쪽
    pts, st = det._track_centerline(m, det._white_state, road)
    assert st["left_seed"] is not None and st["right_seed"] is None, \
        "도로 단서로 좌측 경계 식별 실패(center-split 오배정)"
    assert all(cu > line_col for cu, _v, _m in pts)  # 중심선은 선의 오른쪽으로 오프셋


def test_road_cue_right_boundary_cold_start():
    """대칭: 단선이 중심 좌측이라도 도로가 선의 왼쪽에 있으면 '우측 경계'로 확정."""
    det = _det()
    c = det.bev.u_center
    line_col = c - int(round(0.05 / det.mpp))        # 중심 살짝 좌측
    m = _blank(det)
    _vline(m, line_col)
    road = _road_band(det, 0, line_col - 10)          # 도로=선의 왼쪽
    pts, st = det._track_centerline(m, det._white_state, road)
    assert st["right_seed"] is not None and st["left_seed"] is None, \
        "도로 단서로 우측 경계 식별 실패"
    assert all(cu < line_col for cu, _v, _m in pts)   # 중심선은 선의 왼쪽으로 오프셋


def test_road_both_sides_falls_back_to_center_split():
    """회전교차로 근사(선 양쪽 모두 도로) → 도로 단서 보류 → center-split 유지."""
    det = _det()
    c = det.bev.u_center
    line_col = c - int(round(0.16 / det.mpp))         # 좌측 선
    m = _blank(det)
    _vline(m, line_col)
    half = int(round(0.25 / det.mpp))
    road = _road_band(det, line_col - half, line_col + half)  # 선 좌우 대칭 도로
    _, st = det._track_centerline(m, det._white_state, road)
    # 도로 불균형이 margin 이하 → 보류 → 기존 center-split(좌선) 유지
    assert st["left_seed"] is not None and st["right_seed"] is None


def test_road_cue_overrides_wrong_temporal_memory():
    """도로 단서는 시간 기억보다 우선: 직전이 '좌선'이었어도 지금 도로가
    선의 왼쪽이면 '우선'으로 확정(회전/차로 변경 후 정합)."""
    det = _det()
    c = det.bev.u_center
    # 좌선 기억 심기
    det._white_state["prev_left"] = float(c - int(round(0.16 / det.mpp)))
    det._white_state["prev_right"] = None
    line_col = c - int(round(0.05 / det.mpp))         # 좌선 기억 근처
    m = _blank(det)
    _vline(m, line_col)
    road = _road_band(det, 0, line_col - 10)           # 도로=선의 왼쪽 → 우측 경계
    _, st = det._track_centerline(m, det._white_state, road)
    assert st["right_seed"] is not None and st["left_seed"] is None, \
        "도로 단서가 시간 기억을 못 덮음"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
