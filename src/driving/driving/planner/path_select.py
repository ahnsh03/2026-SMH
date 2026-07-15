"""주행 경로 선택 (스켈레톤).

LaneDetections 로부터 이번 프레임에 추종할 단일 센터라인 polyline 을 고른다.
  - 일반 주행: 흰색/노란색 센터라인 우선순위
  - 갈림길(fork_active): 방향 표지/분기 신뢰도로 branch 하나 선택
  - 인지 소실: 직전 유효 경로를 잠깐 캐시 (control_node 에서 처리)
"""
from __future__ import annotations

from typing import List, Tuple

Point = Tuple[float, float]


def select_path(lane_msg) -> List[Point]:
    """추종 대상 센터라인을 (x, y) 리스트로 반환. 없으면 빈 리스트."""
    # 갈림길: 분기 중 신뢰도 최고를 선택 (방향 표지 연동은 TODO)
    if lane_msg.fork_active and lane_msg.branches:
        best = max(lane_msg.branches, key=lambda b: b.confidence)
        return [(p.x, p.y) for p in best.centerline]

    # 일반 주행: 흰색 우선, 없으면 노란색
    if lane_msg.white_centerline:
        return [(p.x, p.y) for p in lane_msg.white_centerline]
    if lane_msg.yellow_centerline:
        return [(p.x, p.y) for p in lane_msg.yellow_centerline]
    return []
