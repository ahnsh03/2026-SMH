#!/usr/bin/env python3
"""Decision sweep: (a) FOV clip / (b) stem policy / combos + reference baselines.

Writes annotated previews + metrics under::

  data/captures/fork_rail_sweeps/<stamp>_decision/

Variants (look at contact_sheet + REPORT.md)::

  A0_baseline              current production rails
  A_fov_paint_clip         (a) keep stem stitch, clip to outer paint
  B_stem_shared_mid        (b) stem=shared mid only; after apex=independent ±w
  AB_fov_and_stem          (a)+(b)
  G1_indep_everywhere      independent ±w every row (no stem special-case)
  C_no_far_extend          disable far-extend only
  D_both_only_extend       far-extend only when both marks hit
  E_clip_curvature         paint-clip + curvature-parallel inners
  F_viz_pairs_only         baseline geometry, hide road_branches overlay
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(_ROOT / "src" / "inference"))
sys.path.insert(0, str(_ROOT / "scripts" / "vision_tune"))

from inference.modules import lane_detection as ld  # noqa: E402
import sweep_fork_rail_variants as sw  # noqa: E402

SCENES = {
    "in_exit": {
        "frame": _ROOT
        / "data/captures/lane_tune_logs/auto_fork/in_roundabout_exit/runs/20260713_152921/source_frame.png",
        "prefer_yellow": True,
    },
    "out_fork": {
        "frame": _ROOT
        / "data/captures/lane_tune_logs/auto_fork/out_fork/runs/20260713_152900/source_frame.png",
        "prefer_yellow": False,
    },
}


@dataclass
class Row:
    scene: str
    variant: str
    intent: str
    n_pairs: int
    outer_rows_L: int
    outer_rows_R: int
    outer_delta: int
    far_x_L: float
    far_x_R: float
    crossed_inners: int
    crossed_denom: int
    overextend_fraction: float
    paint_err_px: float
    width_m: float


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def mark_mask(dbg: ld.LaneDebugFrame, prefer_yellow: bool) -> np.ndarray:
    if prefer_yellow:
        m = dbg.yellow_connected_bev
        if m is None or getattr(m, "size", 0) == 0:
            m = dbg.yellow_bev
    else:
        m = dbg.white_bev
    if m is None or getattr(m, "size", 0) == 0:
        return np.zeros((ld.BEV_HEIGHT, ld.BEV_WIDTH), dtype=np.uint8)
    return m


def fov_paint_clip(
    pairs: list[ld.ForkLanePair],
    mask: np.ndarray,
    *,
    assoc_px: float = 8.0,
) -> list[ld.ForkLanePair]:
    out: list[ld.ForkLanePair] = []
    for p in sw._pair_as_mutable(pairs):
        o = p.outer_u
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                p.inner_u[row] = np.nan
                p.center_u[row] = np.nan
                continue
            cols = np.flatnonzero(mask[row] > 0)
            bad = cols.size == 0 or float(
                np.min(np.abs(cols.astype(np.float32) - float(o[row])))
            ) > assoc_px
            if bad:
                o[row] = np.nan
                p.inner_u[row] = np.nan
                p.center_u[row] = np.nan
        conf = float(np.clip(np.count_nonzero(~np.isnan(p.center_u)) / ld.BEV_HEIGHT, 0, 1))
        out.append(replace(p, outer_u=o, confidence=conf))
    return out


def policy_b_stem_shared_mid(pairs: list[ld.ForkLanePair]) -> list[ld.ForkLanePair]:
    """(b) Stem: shared mid (+ optional shared outer as corridor). Fork: ±w.

    - Stem rows: both centers = mid; inners NaN (no crossed orange/sky).
    - Fork rows: each outer ± full_w / ± half_w independently.
    - Single-outer FOV rows: that side only ±w.
    """

    if len(pairs) < 2:
        return pairs
    by = {int(p.lateral_rank): p for p in pairs}
    if 0 not in by or 1 not in by:
        return pairs

    lo = ld._nan_moving_average(
        ld._interpolate_nans_1d(by[0].outer_u.astype(np.float32, copy=True)), window=7
    )
    ro = ld._nan_moving_average(
        ld._interpolate_nans_1d(by[1].outer_u.astype(np.float32, copy=True)), window=7
    )
    full_w = ld.FORK_PAIR_WIDTH_M / ld.METERS_PER_PIXEL
    half_w = 0.5 * full_w

    sep_row = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
    for row in range(ld.BEV_HEIGHT):
        if np.isnan(lo[row]) or np.isnan(ro[row]) or ro[row] <= lo[row]:
            continue
        sep_row[row] = float(ro[row] - lo[row])

    stem_end = None
    for row in range(ld.BEV_HEIGHT - 1, -1, -1):
        s = sep_row[row]
        if np.isnan(s):
            continue
        if s <= full_w * 1.25:
            stem_end = row
            break
    fork_start = None
    if stem_end is not None:
        for row in range(stem_end, -1, -1):
            s = sep_row[row]
            if not np.isnan(s) and s >= full_w * 1.55:
                fork_start = row
                break

    li = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
    ri = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
    c0 = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
    c1 = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)

    for row in range(ld.BEV_HEIGHT):
        o_l, o_r = lo[row], ro[row]
        if np.isnan(o_l) and np.isnan(o_r):
            continue
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

        mid = 0.5 * (float(o_l) + float(o_r))
        if stem_end is not None and row >= stem_end:
            # Stem: shared mid only — no inner lines (avoids orange/sky swap look).
            c0[row] = mid
            c1[row] = mid
            continue
        if fork_start is not None and row <= fork_start:
            t = 1.0
        elif fork_start is not None and stem_end is not None and stem_end > fork_start:
            t = float(
                np.clip((stem_end - row) / max(1.0, float(stem_end - fork_start)), 0.0, 1.0)
            )
        else:
            sep = float(o_r - o_l)
            t = float(np.clip((sep - full_w * 1.05) / max(1.0, 1.15 * full_w), 0.0, 1.0))

        if t < 0.5:
            c0[row] = mid
            c1[row] = mid
        else:
            li[row] = float(o_l) + full_w
            ri[row] = float(o_r) - full_w
            c0[row] = float(o_l) + half_w
            c1[row] = float(o_r) - half_w

    rebuilt: list[ld.ForkLanePair] = []
    for rank, outer, inner, center in (
        (0, lo, li, c0),
        (1, ro, ri, c1),
    ):
        conf = float(np.clip(np.count_nonzero(~np.isnan(center)) / ld.BEV_HEIGHT, 0, 1))
        rebuilt.append(
            ld.ForkLanePair(
                lateral_rank=rank,
                outer_u=outer.astype(np.float32, copy=True),
                inner_u=inner.astype(np.float32, copy=True),
                center_u=center.astype(np.float32, copy=True),
                outer_missing=False,
                inner_missing=True,
                confidence=conf,
            )
        )
    return rebuilt


def indep_everywhere(pairs: list[ld.ForkLanePair]) -> list[ld.ForkLanePair]:
    full_w = ld.FORK_PAIR_WIDTH_M / ld.METERS_PER_PIXEL
    half = 0.5 * full_w
    out: list[ld.ForkLanePair] = []
    for p in sw._pair_as_mutable(pairs):
        side = "left" if int(p.lateral_rank) == 0 else "right"
        o = p.outer_u
        i = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
        c = np.full(ld.BEV_HEIGHT, np.nan, dtype=np.float32)
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                continue
            if side == "left":
                i[row] = float(o[row]) + full_w
                c[row] = float(o[row]) + half
            else:
                i[row] = float(o[row]) - full_w
                c[row] = float(o[row]) - half
        conf = float(np.clip(np.count_nonzero(~np.isnan(c)) / ld.BEV_HEIGHT, 0, 1))
        out.append(replace(p, inner_u=i, center_u=c, confidence=conf, inner_missing=True))
    return out


def crossed_inners(pairs: list[ld.ForkLanePair]) -> tuple[int, int]:
    if len(pairs) < 2:
        return 0, 0
    li = np.asarray(pairs[0].inner_u)
    ri = np.asarray(pairs[1].inner_u)
    both = np.flatnonzero(~np.isnan(li) & ~np.isnan(ri))
    if both.size == 0:
        return 0, 0
    return int(np.sum(li[both] > ri[both])), int(both.size)


def far_x(outer: np.ndarray) -> float:
    ov = np.flatnonzero(~np.isnan(outer))
    if ov.size == 0:
        return float("nan")
    return float(ld.X_MAX_M - ov[0] * ld.METERS_PER_PIXEL)


def make_preview(
    dbg: ld.LaneDebugFrame,
    pairs: list[ld.ForkLanePair],
    *,
    hide_branches: bool = False,
) -> np.ndarray:
    d2 = sw.make_debug_with_pairs(dbg, pairs)
    if hide_branches:
        d2 = replace(d2, road_branches=())
    return ld.make_fork_focus_preview(d2, focus="all")


def annotate(img: np.ndarray, pairs: list[ld.ForkLanePair], title: str, intent: str) -> np.ndarray:
    cv2.putText(img, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, intent[:90], (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 255, 200), 1, cv2.LINE_AA)
    if len(pairs) >= 2:
        lo = int(np.count_nonzero(~np.isnan(pairs[0].outer_u)))
        ro = int(np.count_nonzero(~np.isnan(pairs[1].outer_u)))
        cx, n = crossed_inners(pairs)
        cv2.putText(
            img,
            f"outer_rows L={lo} R={ro} d={lo-ro}  crossed_in={cx}/{n}  "
            f"farL={far_x(pairs[0].outer_u):.2f} farR={far_x(pairs[1].outer_u):.2f}",
            (4, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def detect(frame: np.ndarray, prefer_yellow: bool, *, far_mode: str = "default"):
    """far_mode: default | off | both_only"""
    orig = ld.extend_boundary_pair_far_along_marks
    if far_mode == "off":
        ld.extend_boundary_pair_far_along_marks = (
            lambda l, r, m, **k: (np.asarray(l, np.float32).copy(), np.asarray(r, np.float32).copy())
        )
    elif far_mode == "both_only":
        ld.extend_boundary_pair_far_along_marks = sw.extend_both_only
    try:
        return ld.detect_with_debug(frame, prefer_yellow=prefer_yellow)
    finally:
        ld.extend_boundary_pair_far_along_marks = orig


def build_variants(frame, prefer_yellow: bool):
    _, dbg = detect(frame, prefer_yellow, far_mode="default")
    base = list(dbg.fork_lane_pairs)
    mask = mark_mask(dbg, prefer_yellow)

    _, dbg_off = detect(frame, prefer_yellow, far_mode="off")
    _, dbg_both = detect(frame, prefer_yellow, far_mode="both_only")

    a_clip = fov_paint_clip(base, mask)
    b_stem = policy_b_stem_shared_mid(base)
    ab = fov_paint_clip(policy_b_stem_shared_mid(base), mask)
    g1 = indep_everywhere(base)
    e_curv = sw.apply_curvature_parallel_rails(fov_paint_clip(base, mask))

    return dbg, {
        "A0_baseline": (base, "현재: stem 교차 inner + far-extend", False, dbg),
        "A_fov_paint_clip": (
            a_clip,
            "(a) stem 유지, outer 페인트 없으면 그 행 절단",
            False,
            dbg,
        ),
        "B_stem_shared_mid": (
            b_stem,
            "(b) stem=공유 mid만(inner 없음), apex 후 ±w",
            False,
            dbg,
        ),
        "AB_fov_and_stem": (
            ab,
            "(a)+(b) FOV clip + stem shared mid",
            False,
            dbg,
        ),
        "G1_indep_everywhere": (
            g1,
            "참고: 전 구간 독립 ±w (stem 특수처리 없음)",
            False,
            dbg,
        ),
        "C_no_far_extend": (
            list(dbg_off.fork_lane_pairs),
            "참고: far-extend OFF",
            False,
            dbg_off,
        ),
        "D_both_only_extend": (
            list(dbg_both.fork_lane_pairs),
            "참고: far-extend 양쪽 마킹일 때만",
            False,
            dbg_both,
        ),
        "E_clip_curvature": (
            e_curv,
            "참고: paint-clip + κ 평행곡선 inner/center",
            False,
            dbg,
        ),
        "F_viz_pairs_only": (
            base,
            "참고: 기하=A0, road_branches 오버레이 숨김",
            True,
            dbg,
        ),
    }


def contact_sheet(items: list[tuple[str, np.ndarray]], path: Path, cols: int = 3) -> None:
    if not items:
        return
    tiles = []
    h = w = None
    for name, im in items:
        canvas = im.copy()
        cv2.putText(
            canvas, name, (4, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
        )
        if h is None:
            h, w = canvas.shape[:2]
        tiles.append(cv2.resize(canvas, (w, h)))
    rows = int(np.ceil(len(tiles) / cols))
    pad = np.zeros((h, w, 3), dtype=np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(pad.copy())
    grid = [np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)]
    cv2.imwrite(str(path), np.vstack(grid))


def main() -> int:
    stamp = _stamp()
    out_root = _ROOT / "data" / "captures" / "fork_rail_sweeps" / f"{stamp}_decision"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    md = [
        f"# Decision sweep `{stamp}_decision`",
        "",
        "직접 보고 고를 것: **A**(FOV) / **B**(stem) / **AB**(둘 다).",
        "",
        "| id | 의도 |",
        "|----|------|",
        "| A0 | 현재 |",
        "| **A_fov_paint_clip** | (a) 페인트 밖 절단, stem 교차 유지 |",
        "| **B_stem_shared_mid** | (b) 줄기는 mid만, 갈림 후 독립 ±w |",
        "| **AB_fov_and_stem** | (a)+(b) |",
        "| G1 / C / D / E / F | 참고 변형 |",
        "",
    ]

    t0 = time.time()
    for scene, spec in SCENES.items():
        frame = cv2.imread(str(spec["frame"]))
        if frame is None:
            raise SystemExit(f"missing {spec['frame']}")
        scene_dir = out_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        dbg0, variants = build_variants(frame, bool(spec["prefer_yellow"]))
        mask = mark_mask(dbg0, bool(spec["prefer_yellow"]))
        previews: list[tuple[str, np.ndarray]] = []
        md.append(f"## {scene}")
        md.append("")
        md.append(
            "| variant | farL | farR | d_rows | crossed | overext | paint_err | width |"
        )
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        for name, (pairs, intent, hide_br, dbg) in variants.items():
            if len(pairs) < 2:
                print(f"[{scene}] {name}: <2 pairs — skip")
                continue
            metrics = sw.score_pairs(pairs, mark_mask(dbg, bool(spec["prefer_yellow"])))
            cx, cn = crossed_inners(pairs)
            lo = int(np.count_nonzero(~np.isnan(pairs[0].outer_u)))
            ro = int(np.count_nonzero(~np.isnan(pairs[1].outer_u)))
            img = make_preview(dbg, pairs, hide_branches=hide_br)
            img = annotate(img, pairs, name, intent)
            cv2.imwrite(str(scene_dir / f"{name}.png"), img)
            previews.append((name, img))
            meta = {
                "scene": scene,
                "variant": name,
                "intent": intent,
                "outer_rows_L": lo,
                "outer_rows_R": ro,
                "crossed_inners": cx,
                "crossed_denom": cn,
                **metrics,
            }
            (scene_dir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            row = Row(
                scene=scene,
                variant=name,
                intent=intent,
                n_pairs=len(pairs),
                outer_rows_L=lo,
                outer_rows_R=ro,
                outer_delta=lo - ro,
                far_x_L=far_x(pairs[0].outer_u),
                far_x_R=far_x(pairs[1].outer_u),
                crossed_inners=cx,
                crossed_denom=cn,
                overextend_fraction=float(metrics["overextend_fraction"]),
                paint_err_px=float(metrics["mean_outer_paint_err_px"]),
                width_m=float(metrics["mean_same_row_width_m"]),
            )
            rows.append(row)
            md.append(
                f"| `{name}` | {row.far_x_L:.2f} | {row.far_x_R:.2f} | {row.outer_delta} | "
                f"{cx}/{cn} | {row.overextend_fraction:.3f} | {row.paint_err_px:.1f} | {row.width_m:.3f} |"
            )
            print(
                f"[{scene}] {name}: farL={row.far_x_L:.2f} farR={row.far_x_R:.2f} "
                f"cross={cx}/{cn} over={row.overextend_fraction:.3f}"
            )

        contact_sheet(previews, scene_dir / "contact_sheet.png", cols=3)
        md.append("")
        md.append(f"![contact]({scene}/contact_sheet.png)")
        md.append("")

    csv_path = out_root / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))

    md.append(f"elapsed_s={time.time() - t0:.1f}")
    (out_root / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_root / "results.json").write_text(
        json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out_root}")
    print(f"Open: {out_root / 'REPORT.md'}")
    print(f"Contact: {out_root / 'in_exit' / 'contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
