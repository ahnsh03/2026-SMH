"""선택 갈래 정책 (Out 갈림 / In 탈출 잠금 후).

Roles
-----
* **EXPLORE** — 갈래 후보 2개를 유지해 플래너가 LEFT→rank0 / RIGHT→rank1 잠금.
* **LOCKED** — 선택 ``ForkLanePair`` / ``RoadBranch``(갈래)만 유지. 반대 갈래 제거.
* **EGO_ONLY** — locked와 동일 + far-only 합류 spur 무시.

용어: docs/lane-occlusion-fork-strategy.md §0. 플래너가 미션 선택, 이 모듈은
인지 출력만 재구성.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from inference.modules.lane_detection import (
    ForkLanePair,
    LaneDebugFrame,
    LaneDetections,
    RoadBranch,
    fork_lane_pairs_to_road_branches,
)

# base_link: smaller x = closer. Merge starts often appear only beyond this.
MERGE_SPUR_NEAR_X_M = 0.55
MERGE_SPUR_MIN_NEAR_POINTS = 3


def suppress_merge_spur_branches(
    branches: tuple[RoadBranch, ...] | list[RoadBranch],
    *,
    near_x_m: float = MERGE_SPUR_NEAR_X_M,
    min_near_points: int = MERGE_SPUR_MIN_NEAR_POINTS,
) -> tuple[RoadBranch, ...]:
    """Keep corridors that reach the near field; drop far-only merge starts.

    When another lane begins to join ours, its first visible paint often sits
    only in the far half of BEV. Ego lane still has points near the bumper —
    keep those, drop the far-only spur.
    """

    items = list(branches or ())
    if len(items) < 2:
        return tuple(items)

    def near_count(branch: RoadBranch) -> int:
        points = np.asarray(getattr(branch, 'points', ()), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
            return 0
        xy = points[:, :2]
        finite = np.isfinite(xy).all(axis=1)
        return int(np.sum(finite & (xy[:, 0] <= float(near_x_m))))

    reaching = [b for b in items if near_count(b) >= int(min_near_points)]
    if not reaching:
        return tuple(items)
    if len(reaching) < len(items):
        return tuple(reaching)
    return tuple(items)


def select_fork_pairs_for_rank(
    pairs: tuple[ForkLanePair, ...] | list[ForkLanePair],
    rank: int,
) -> tuple[ForkLanePair, ...]:
    """Return only the locked fork layer (LEFT=0 / RIGHT=1)."""

    return tuple(
        p for p in (pairs or ()) if int(getattr(p, 'lateral_rank', -1)) == int(rank)
    )


def select_branches_for_rank(
    branches: tuple[RoadBranch, ...] | list[RoadBranch],
    rank: int,
) -> tuple[RoadBranch, ...]:
    matched = tuple(
        b
        for b in (branches or ())
        if int(getattr(b, 'lateral_rank', -1)) == int(rank)
    )
    if matched:
        return matched
    # Fallback: list index only when lateral_rank was never set and we still
    # have exactly the classic L/R pair (explore→lock transition frame).
    items = list(branches or ())
    if len(items) == 2 and 0 <= int(rank) < 2:
        return (items[int(rank)],)
    return ()


def _pair_centerline_xy(pair: ForkLanePair) -> np.ndarray:
    branches = fork_lane_pairs_to_road_branches([pair])
    if not branches:
        return np.empty((0, 2), dtype=np.float32)
    points = np.asarray(branches[0].points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    return points[:, :2].copy()


def _prefer_yellow(detections: LaneDetections, debug: LaneDebugFrame) -> bool:
    """Which color channel receives the locked fork centerline.

    Out (``prefer_yellow=False``): always white — never follow yellow paint.
    In (``True``): yellow. Legacy None: split source / ego / confidence.
    """

    pref = getattr(debug, 'prefer_yellow', None)
    if pref is False:
        return False
    if pref is True:
        return True
    source = str(getattr(debug, 'fork_split_source', '') or '')
    if source.startswith('yellow'):
        return True
    ego = getattr(debug, 'ego_road_color', None)
    if ego == 'yellow':
        return True
    if ego == 'white':
        return False
    return float(detections.yellow_confidence) > float(detections.white_confidence)


def collapse_to_selected_lane(
    detections: LaneDetections,
    debug: LaneDebugFrame,
    rank: int,
) -> tuple[LaneDetections, LaneDebugFrame]:
    """Drop the opposite fork and project the locked layer onto ego centerline.

    ``fork_active`` becomes False so downstream treats this as normal single-
    lane follow (selected path only). Width-prior / one-sided stitch already
    live inside the selected ``ForkLanePair``.
    """

    pairs = select_fork_pairs_for_rank(debug.fork_lane_pairs, rank)
    if not pairs and debug.fork_lane_pairs:
        # Rank missing in marking pairs — still try cell branches.
        pairs = ()

    selected_branches = select_branches_for_rank(detections.branches, rank)
    if pairs:
        selected_branches = tuple(fork_lane_pairs_to_road_branches(list(pairs)))
        # Keep the locked lateral_rank even as a lone branch.
        selected_branches = tuple(
            replace(b, lateral_rank=int(rank)) if hasattr(b, 'lateral_rank') else b
            for b in selected_branches
        )

    if not selected_branches and not pairs:
        # Nothing to collapse onto — only scrub the opposite published branch.
        scrubbed = select_branches_for_rank(detections.branches, rank)
        scrubbed = suppress_merge_spur_branches(scrubbed)
        return (
            replace(
                detections,
                fork_active=False,
                branches=scrubbed,
                active_branch_rank=int(rank),
                lane_policy='locked',
            ),
            replace(
                debug,
                fork_active=False,
                fork_lane_pairs=pairs,
                road_branches=scrubbed,
                active_branch_rank=int(rank),
                lane_policy='locked',
            ),
        )

    center_xy = (
        _pair_centerline_xy(pairs[0])
        if pairs
        else np.asarray(selected_branches[0].points, dtype=np.float32)[:, :2]
    )
    conf = float(selected_branches[0].confidence) if selected_branches else 0.7
    if conf <= 0.0:
        conf = 0.7

    use_yellow = _prefer_yellow(detections, debug)
    if use_yellow:
        white_cl = np.asarray(detections.white_centerline, dtype=np.float32)
        yellow_cl = center_xy
        yellow_conf = max(float(detections.yellow_confidence), conf)
        white_conf = float(detections.white_confidence)
        yellow_vis = True
        white_vis = bool(detections.white_visible)
    else:
        white_cl = center_xy
        yellow_cl = np.asarray(detections.yellow_centerline, dtype=np.float32)
        white_conf = max(float(detections.white_confidence), conf)
        yellow_conf = float(detections.yellow_confidence)
        white_vis = center_xy.shape[0] >= 2
        yellow_vis = bool(detections.yellow_visible)

    # Single ego branch kept with lateral_rank for strict planner matching.
    if not selected_branches and center_xy.shape[0] >= 2:
        xyz = np.column_stack(
            (
                center_xy[:, 0],
                center_xy[:, 1],
                np.zeros(len(center_xy), dtype=np.float32),
            )
        ).astype(np.float32)
        selected_branches = (
            RoadBranch(
                lateral_rank=int(rank),
                confidence=conf,
                points=xyz,
            ),
        )
    selected_branches = suppress_merge_spur_branches(selected_branches)

    new_det = replace(
        detections,
        fork_active=False,
        branches=selected_branches,
        white_centerline=white_cl,
        yellow_centerline=yellow_cl,
        white_confidence=white_conf,
        yellow_confidence=yellow_conf,
        white_visible=white_vis,
        yellow_visible=yellow_vis,
        active_branch_rank=int(rank),
        lane_policy='locked',
    )
    new_dbg = replace(
        debug,
        fork_active=False,
        fork_lane_pairs=pairs,
        road_branches=selected_branches,
        active_branch_rank=int(rank),
        lane_policy='locked',
    )
    return new_det, new_dbg


def apply_active_lane_policy(
    detections: LaneDetections,
    debug: LaneDebugFrame,
    active_branch_rank: int | None,
) -> tuple[LaneDetections, LaneDebugFrame]:
    """Apply explore → locked/ego-only reshape.

    ``active_branch_rank`` is the planner lock (None while exploring).
    """

    if active_branch_rank is None:
        pruned = suppress_merge_spur_branches(detections.branches)
        if pruned == tuple(detections.branches or ()):
            debug = replace(
                debug,
                active_branch_rank=None,
                lane_policy='explore',
            )
            return (
                replace(
                    detections,
                    active_branch_rank=None,
                    lane_policy='explore',
                ),
                debug,
            )
        # Explore can still drop far-only merge spurs so fork_active stays honest.
        fork_active = len(pruned) >= 2
        return (
            replace(
                detections,
                branches=pruned,
                fork_active=fork_active,
                active_branch_rank=None,
                lane_policy='ego_only',
            ),
            replace(
                debug,
                road_branches=pruned,
                fork_active=fork_active,
                active_branch_rank=None,
                lane_policy='ego_only',
            ),
        )

    return collapse_to_selected_lane(detections, debug, int(active_branch_rank))


def focus_name_for_rank(rank: int | None) -> str:
    """Map locked rank to fork preview focus label."""

    if rank is None:
        return 'all'
    return 'left' if int(rank) == 0 else 'right'


def parse_selected_rank_from_planner_debug(text: str) -> int | None:
    """Extract ``rank=N`` from ``/debug/planner`` payload (``rank=-`` → None)."""

    import re

    match = re.search(r'\brank=(\d+|-)\b', text or '')
    if not match or match.group(1) == '-':
        return None
    return int(match.group(1))


def parse_fork_perception_from_planner_debug(text: str) -> bool | None:
    """Extract ``fork_on=0|1`` from planner debug; None if absent."""

    import re

    match = re.search(r'\bfork_on=([01])\b', text or '')
    if not match:
        return None
    return match.group(1) == '1'
