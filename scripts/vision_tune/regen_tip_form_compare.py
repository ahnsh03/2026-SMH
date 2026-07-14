#!/usr/bin/env python3
"""Verify tip finalize on in_exit / out_fork; write compare sheets + metrics."""

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
    ),
    "out_fork": (
        _ROOT
        / "data/captures/lane_tune_logs/auto_fork/out_fork/runs/20260713_152900/source_frame.png",
        False,
    ),
}


def _roughness(series: np.ndarray) -> float | None:
    valid = np.flatnonzero(~np.isnan(series))
    if valid.size < 5:
        return None
    d2 = np.diff(series[valid].astype(np.float64), n=2)
    return float(np.mean(np.abs(d2)))


def _tip(series: np.ndarray) -> dict | None:
    valid = np.flatnonzero(~np.isnan(series))
    if valid.size == 0:
        return None
    r0 = int(valid[0])
    return {
        "row": r0,
        "u": float(series[r0]),
        "n": int(valid.size),
        "x_m": float(ld.X_MAX_M - r0 * ld.METERS_PER_PIXEL),
    }


def _mark_for_scene(dbg) -> np.ndarray | None:
    for name in (
        "yellow_connected_bev",
        "yellow_boundary_bev",
        "white_dash_connected_bev",
        "white_bev",
    ):
        m = getattr(dbg, name, None)
        if isinstance(m, np.ndarray) and np.count_nonzero(m) > 0:
            return m
    return None


def annot(img: np.ndarray, title: str, pairs, paint_tips: dict) -> np.ndarray:
    out = img.copy()
    cv2.putText(
        out, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
    )
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
        # Rail tip marker.
        cv2.circle(out, (int(round(ou)), orow), 4, (0, 255, 255), 1, cv2.LINE_AA)
        pt = paint_tips.get(lab)
        if pt is not None:
            pr, pu = int(pt[0]), float(pt[1])
            cv2.drawMarker(
                out,
                (int(round(pu)), pr),
                (0, 165, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=10,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
    cv2.putText(
        out,
        "  ".join(parts),
        (4, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "o=rail tip  x=paint tip",
        (4, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def scene_metrics(scene: str, dbg, mark: np.ndarray | None) -> dict:
    entry: dict = {
        "scene": scene,
        "source": str(getattr(dbg, "fork_split_source", "")),
        "pairs": [],
    }
    paint_tips = {}
    for lab, rank, side in (("L", 0, "left"), ("R", 1, "right")):
        p = dbg.fork_lane_pairs[rank]
        o = np.asarray(p.outer_u, dtype=np.float32)
        seed = None
        v = np.flatnonzero(~np.isnan(o))
        if v.size:
            seed = float(o[int(v[min(len(v) - 1, v.size // 2)])])
        paint = (
            ld._track_outer_paint_tip(mark, side=side, seed_u=seed)
            if mark is not None
            else None
        )
        paint_tips[lab] = paint
        tip_o = _tip(o)
        tip_i = _tip(np.asarray(p.inner_u))
        tip_c = _tip(np.asarray(p.center_u))
        row = {
            "side": lab,
            "outer": tip_o,
            "inner": tip_i,
            "center": tip_c,
            "rough_outer": _roughness(o),
            "rough_inner": _roughness(np.asarray(p.inner_u)),
            "rough_center": _roughness(np.asarray(p.center_u)),
            "paint_tip": (
                {"row": int(paint[0]), "u": float(paint[1])} if paint else None
            ),
        }
        if paint and tip_o:
            row["tip_vs_paint"] = {
                "drow": int(tip_o["row"] - paint[0]),
                "du": float(tip_o["u"] - paint[1]),
            }
        entry["pairs"].append(row)
    entry["_paint_tips"] = paint_tips
    return entry


def main() -> int:
    out = _ROOT / "data/captures/fork_rail_sweeps/_tip_form_compare"
    out.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"scenes": []}
    contact_tiles: list[np.ndarray] = []

    for scene, (path, prefer_yellow) in SCENES.items():
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"missing {path}")
        _, dbg = ld.detect_with_debug(frame, prefer_yellow=prefer_yellow)
        pairs = list(dbg.fork_lane_pairs)
        mark = _mark_for_scene(dbg)
        m = scene_metrics(scene, dbg, mark)
        paint_tips = m.pop("_paint_tips")
        metrics["scenes"].append(m)

        print(scene, "src", m["source"], "pairs", len(pairs))
        for row in m["pairs"]:
            o = row["outer"]
            paint = row.get("paint_tip")
            tvp = row.get("tip_vs_paint")
            print(
                f"  {row['side']} out r={o['row'] if o else '-'} "
                f"u={(o['u'] if o else float('nan')):.1f} "
                f"rough={row['rough_outer']} paint={paint} tip_vs_paint={tvp}"
            )

        img = annot(
            ld.make_fork_focus_preview(replace(dbg, road_branches=()), focus="all"),
            f"{scene}: tip paint+heading",
            pairs,
            paint_tips,
        )
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
        contact_tiles.append(img)
        print("wrote", out / f"{scene}_compare.png")

    if contact_tiles:
        # 2x1 contact: in_exit | out_fork
        contact = np.hstack(contact_tiles)
        cv2.imwrite(str(out / "contact_sheet.png"), contact)
        print("wrote", out / "contact_sheet.png")

    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
