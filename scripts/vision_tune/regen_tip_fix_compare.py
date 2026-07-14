#!/usr/bin/env python3
"""Compare A0-style baseline tip vs tip finalize on in_exit / out_fork."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "inference"))

from inference.modules import lane_detection as ld  # noqa: E402


SCENES = {
    "in_exit": (
        _ROOT
        / "data/captures/lane_tune_logs/auto_fork/in_roundabout_exit/runs/20260713_152921/source_frame.png",
        True,
    ),
    "out_fork": (
        _ROOT
        / "data/captures/lane_tune_logs/auto_fork/out_fork/runs/20260713_152900/source_frame.png",
        False,
    ),
}


def annot(img: np.ndarray, title: str, pairs) -> np.ndarray:
    cv2.putText(img, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    parts = []
    for lab, p in zip(("L", "R"), pairs):
        o = np.asarray(p.outer_u)
        i = np.asarray(p.inner_u)
        vo = np.flatnonzero(~np.isnan(o))
        vi = np.flatnonzero(~np.isnan(i))
        if vo.size == 0:
            continue
        ou, orow = float(o[vo[0]]), int(vo[0])
        iu = float(i[vi[0]]) if vi.size else float("nan")
        irow = int(vi[0]) if vi.size else -1
        parts.append(f"{lab}: out r={orow} u={ou:.0f}  in r={irow} u={iu:.0f}")
    cv2.putText(
        img,
        "  ".join(parts),
        (4, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return img


def main() -> int:
    out = _ROOT / "data/captures/fork_rail_sweeps/_tip_fix_compare"
    out.mkdir(parents=True, exist_ok=True)

    for scene, (path, prefer_yellow) in SCENES.items():
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"missing {path}")
        _, dbg = ld.detect_with_debug(frame, prefer_yellow=prefer_yellow)
        pairs = list(dbg.fork_lane_pairs)
        print(scene, "pairs", len(pairs))
        for p in pairs:
            o = np.asarray(p.outer_u)
            v = np.flatnonzero(~np.isnan(o))
            print(
                f"  rank{p.lateral_rank}: tip_row={v[0]} tip_u={o[v[0]]:.1f} "
                f"x={ld.X_MAX_M - v[0] * ld.METERS_PER_PIXEL:.2f} n={v.size}"
            )
        img = annot(
            ld.make_fork_focus_preview(replace(dbg, road_branches=()), focus="all"),
            f"{scene}: tip paint+heading fix",
            pairs,
        )
        # side-by-side with stored A0 baseline if available
        ref = (
            _ROOT
            / "data/captures/fork_rail_sweeps/20260714_103245"
            / scene
            / "A0_baseline.png"
        )
        tiles = [img]
        if ref.is_file():
            ref_img = cv2.imread(str(ref))
            if ref_img is not None:
                cv2.putText(
                    ref_img,
                    "REF A0_baseline (old)",
                    (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                tiles.insert(0, cv2.resize(ref_img, (img.shape[1], img.shape[0])))
        sheet = np.hstack(tiles)
        cv2.imwrite(str(out / f"{scene}_compare.png"), sheet)
        cv2.imwrite(str(out / f"{scene}_current.png"), img)
        print("wrote", out / f"{scene}_compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
