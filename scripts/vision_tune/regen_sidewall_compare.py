#!/usr/bin/env python3
"""Regenerate before/after side-wall compare for in_exit."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "inference"))

from inference.modules import lane_detection as ld  # noqa: E402


def main() -> int:
    frame = cv2.imread(
        str(
            _ROOT
            / "data/captures/lane_tune_logs/auto_fork/in_roundabout_exit/runs/20260713_152921/source_frame.png"
        )
    )
    out = _ROOT / "data/captures/fork_rail_sweeps/20260714_110005_heading/in_exit"
    out.mkdir(parents=True, exist_ok=True)

    # --- OLD: disable clips + skate-fill along near-wall tip ---
    _ext = ld.extend_boundary_pair_far_along_marks
    ld.clip_boundary_u_at_side_wall = lambda c, **k: np.asarray(c, np.float32).copy()
    ld.clip_fork_lane_pairs_at_side_wall = lambda pairs, **k: list(pairs)

    def fill(left, right, mask, **kw):
        lo, ro = _ext(left, right, mask, **kw)
        for arr in (lo, ro):
            v = np.flatnonzero(~np.isnan(arr))
            if not v.size:
                continue
            tip = int(v[0])
            u = float(arr[tip])
            if u <= 20 or u >= ld.BEV_WIDTH - 21:
                arr[:tip] = u
        return lo, ro

    ld.extend_boundary_pair_far_along_marks = fill
    _, dbg_old = ld.detect_with_debug(frame, prefer_yellow=True)

    # --- NEW: reload production module ---
    importlib.reload(ld)
    _, dbg_new = ld.detect_with_debug(frame, prefer_yellow=True)

    def tip(pairs, tag: str) -> None:
        for p in pairs:
            o = np.asarray(p.outer_u)
            v = np.flatnonzero(~np.isnan(o))
            print(
                f"{tag} rank{p.lateral_rank}: tip_row={v[0]} tip_u={o[v[0]]:.1f} "
                f"x={ld.X_MAX_M - v[0] * ld.METERS_PER_PIXEL:.2f}"
            )

    tip(dbg_old.fork_lane_pairs, "OLD")
    tip(dbg_new.fork_lane_pairs, "NEW")
    print("SIDE_WALL_MARGIN_PX", ld.SIDE_WALL_MARGIN_PX)

    def annot(img, title, pairs):
        cv2.putText(
            img, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
        )
        parts = []
        for lab, p in zip(("L", "R"), pairs):
            o = np.asarray(p.outer_u)
            v = np.flatnonzero(~np.isnan(o))
            parts.append(
                f"{lab}: r={v[0]} u={o[v[0]]:.0f} "
                f"x={ld.X_MAX_M - v[0] * ld.METERS_PER_PIXEL:.2f}"
            )
        cv2.putText(
            img,
            "  ".join(parts),
            (4, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        return img

    img_old = annot(
        ld.make_fork_focus_preview(dbg_old, focus="all"),
        "BEFORE wall-skate to TOP",
        list(dbg_old.fork_lane_pairs),
    )
    img_new = annot(
        ld.make_fork_focus_preview(dbg_new, focus="all"),
        "AFTER side margin=16px stop",
        list(dbg_new.fork_lane_pairs),
    )
    dbg_c = replace(dbg_new, road_branches=())
    img_clean = annot(
        ld.make_fork_focus_preview(dbg_c, focus="all"),
        "AFTER pairs-only",
        list(dbg_new.fork_lane_pairs),
    )
    sheet = np.hstack([img_old, img_new, img_clean])
    cv2.imwrite(str(out / "COMPARE_before_after_sheet.png"), sheet)
    cv2.imwrite(str(out / "COMPARE_before_skate.png"), img_old)
    cv2.imwrite(str(out / "COMPARE_after_sidewall.png"), img_new)
    print("wrote", out / "COMPARE_before_after_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
