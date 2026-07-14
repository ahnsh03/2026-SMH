#!/usr/bin/env python3
"""Verify course-separated tip finalize: in_curve vs out_forward."""

from __future__ import annotations

import json
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
        "in_curve",
    ),
    "out_fork": (
        _ROOT
        / "data/captures/lane_tune_logs/auto_fork/out_fork/runs/20260713_152900/source_frame.png",
        False,
        "out_forward",
    ),
}


def _tip(series: np.ndarray) -> dict | None:
    valid = np.flatnonzero(~np.isnan(series))
    if valid.size == 0:
        return None
    r0 = int(valid[0])
    return {"row": r0, "u": float(series[r0]), "n": int(valid.size)}


def annot(img: np.ndarray, title: str, pairs) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2, cv2.LINE_AA
    )
    parts = []
    for lab, p in zip(("L", "R"), pairs):
        o = np.asarray(p.outer_u)
        i = np.asarray(p.inner_u)
        vo = np.flatnonzero(~np.isnan(o))
        vi = np.flatnonzero(~np.isnan(i))
        if vo.size == 0:
            continue
        parts.append(
            f"{lab}: out r={int(vo[0])} u={float(o[vo[0]]):.0f}  "
            f"in r={int(vi[0]) if vi.size else -1} "
            f"u={float(i[vi[0]]) if vi.size else float('nan'):.0f}"
        )
        cv2.circle(out, (int(round(float(o[vo[0]]))), int(vo[0])), 4, (0, 255, 255), 1)
    cv2.putText(
        out,
        "  ".join(parts),
        (4, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> int:
    out = _ROOT / "data/captures/fork_rail_sweeps/_course_split_compare"
    out.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"scenes": []}
    contact: list[np.ndarray] = []

    for scene, (path, prefer_yellow, expect_mode) in SCENES.items():
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"missing {path}")
        _, dbg = ld.detect_with_debug(frame, prefer_yellow=prefer_yellow)
        pairs = list(dbg.fork_lane_pairs)
        src = str(dbg.fork_split_source)
        mode = ld._fork_tip_mode_for_mark_color(src.replace("_marks", "") if src.endswith("_marks") else src)
        # road_split_marks → road_split
        color = src.replace("_marks", "") if src.endswith("_marks") else src
        mode = ld._fork_tip_mode_for_mark_color(color)
        print(scene, "src", src, "tip_mode", mode, "expect", expect_mode)
        assert mode == expect_mode, (mode, expect_mode)

        row = {"scene": scene, "source": src, "tip_mode": mode, "pairs": []}
        for p in pairs:
            tip_o = _tip(np.asarray(p.outer_u))
            tip_i = _tip(np.asarray(p.inner_u))
            tip_c = _tip(np.asarray(p.center_u))
            print(f"  rank{p.lateral_rank} out={tip_o} in={tip_i} ctr={tip_c}")
            row["pairs"].append({"rank": int(p.lateral_rank), "outer": tip_o, "inner": tip_i, "center": tip_c})
        metrics["scenes"].append(row)

        img = annot(
            ld.make_fork_focus_preview(replace(dbg, road_branches=()), focus="all"),
            f"{scene}: tip_mode={mode}",
            pairs,
        )
        cv2.imwrite(str(out / f"{scene}_current.png"), img)

        tiles = [img]
        # Prefer A0 / baseline refs when available.
        if scene == "in_exit":
            ref = (
                _ROOT
                / "data/captures/fork_rail_sweeps/_tip_form_compare"
                / "in_exit_current.png"
            )
            label = "REF prior in_exit tip+heading"
        else:
            ref = (
                _ROOT
                / "data/captures/fork_rail_sweeps/20260714_110005_heading"
                / "out_fork"
                / "A0_old_wall_skate.png"
            )
            label = "REF A0_old_wall_skate (preferred out)"
        if ref.is_file():
            ref_img = cv2.imread(str(ref))
            if ref_img is not None:
                cv2.putText(
                    ref_img,
                    label,
                    (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                tiles.insert(0, cv2.resize(ref_img, (img.shape[1], img.shape[0])))
        # Also stack P0/H0 if out_fork
        if scene == "out_fork":
            for name in ("P0_production_now", "H0_side_wall_clip", "H3_heading_gate", "H4_heading_45"):
                p = (
                    _ROOT
                    / "data/captures/fork_rail_sweeps/20260714_110005_heading/out_fork"
                    / f"{name}.png"
                )
                if p.is_file():
                    rimg = cv2.imread(str(p))
                    if rimg is not None:
                        cv2.putText(
                            rimg,
                            name,
                            (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        tiles.append(cv2.resize(rimg, (img.shape[1], img.shape[0])))

        sheet = np.hstack(tiles)
        cv2.imwrite(str(out / f"{scene}_compare.png"), sheet)
        contact.append(img)
        print("wrote", out / f"{scene}_compare.png")

    if contact:
        cv2.imwrite(str(out / "contact_sheet.png"), np.hstack(contact))
        print("wrote", out / "contact_sheet.png")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", out / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
