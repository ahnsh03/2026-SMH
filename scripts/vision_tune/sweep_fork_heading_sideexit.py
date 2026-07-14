#!/usr/bin/env python3
"""Heading / side-exit sweep for curved forks (in_roundabout_exit).

Root cause found:
  Left/right fork outers hit BEV side walls (~row 34–36) with heading ~±50–60°,
  then keep filling farther rows while pinned at u=0 / u=W-1 — "skating"
  vertically to the top of the image. That looks like an 11자 tip to X_MAX
  instead of lanes exiting left/right sides.

Variants:
  A0_baseline              current
  H0_side_wall_clip        drop rows beyond first FOV-side contact
  H1_wall_then_frenet      H0 + path-normal ±w (XY offset, dense raster)
  H2_wall_then_curvature   H0 + osculating parallel
  H3_heading_gate          stop when |heading| > thresh for N tipward steps
  H4_wall_and_heading      H0 + H3
  H5_extend_no_wall_skate  redetect with patched extend (no wall continue)
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
    tip_row_L: float
    tip_row_R: float
    tip_u_L: float
    tip_u_R: float
    tip_heading_L_deg: float
    tip_heading_R_deg: float
    side_exit_L: int
    side_exit_R: int
    wall_skate_rows_L: int
    wall_skate_rows_R: int
    far_x_L: float
    far_x_R: float
    overextend_fraction: float
    paint_err_px: float


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


def first_side_wall_row(columns_u: np.ndarray, *, margin_px: float = 1.5) -> int | None:
    """Near→far: first row where u sits on FOV left/right margin."""

    o = np.asarray(columns_u, dtype=np.float32)
    valid = np.flatnonzero(~np.isnan(o))
    lo = float(margin_px)
    hi = float(ld.BEV_WIDTH - 1) - float(margin_px)
    for row in valid[::-1]:
        u = float(o[row])
        if u <= lo or u >= hi:
            return int(row)
    return None


def clip_column_at_side_wall(
    columns_u: np.ndarray,
    *,
    margin_px: float = 1.5,
    keep_wall_tip: bool = True,
) -> np.ndarray:
    out = np.asarray(columns_u, dtype=np.float32).copy()
    wall = first_side_wall_row(out, margin_px=margin_px)
    if wall is None:
        return out
    # Clear farther than the side-exit tip (smaller row = farther X).
    cut = wall if keep_wall_tip else wall + 1
    out[:cut] = np.nan
    return out


def tip_heading_deg(columns_u: np.ndarray, *, span: int = 12) -> float:
    o = np.asarray(columns_u, dtype=np.float32)
    valid = np.flatnonzero(~np.isnan(o))
    if valid.size < 3:
        return float("nan")
    rows = valid[: min(span, valid.size)]
    r0, r1 = int(rows[0]), int(rows[-1])
    x0 = ld.X_MAX_M - r0 * ld.METERS_PER_PIXEL
    x1 = ld.X_MAX_M - r1 * ld.METERS_PER_PIXEL
    y0 = ((ld.BEV_WIDTH - 1) / 2.0 - float(o[r0])) * ld.METERS_PER_PIXEL
    y1 = ((ld.BEV_WIDTH - 1) / 2.0 - float(o[r1])) * ld.METERS_PER_PIXEL
    # tipward: from nearer sample (r1) toward tip (r0)
    return float(np.degrees(np.arctan2(y0 - y1, x0 - x1)))


def wall_skate_rows(columns_u: np.ndarray, *, margin_px: float = 1.5) -> int:
    wall = first_side_wall_row(columns_u, margin_px=margin_px)
    if wall is None:
        return 0
    o = np.asarray(columns_u, dtype=np.float32)
    valid = np.flatnonzero(~np.isnan(o))
    return int(np.sum(valid < wall))


def clip_pairs_side_wall(pairs: list[ld.ForkLanePair], *, margin_px: float = 1.5) -> list[ld.ForkLanePair]:
    out: list[ld.ForkLanePair] = []
    for p in sw._pair_as_mutable(pairs):
        o = clip_column_at_side_wall(p.outer_u, margin_px=margin_px)
        i = np.asarray(p.inner_u, dtype=np.float32).copy()
        c = np.asarray(p.center_u, dtype=np.float32).copy()
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                i[row] = np.nan
                c[row] = np.nan
        conf = float(np.clip(np.count_nonzero(~np.isnan(c)) / ld.BEV_HEIGHT, 0, 1))
        out.append(replace(p, outer_u=o, inner_u=i, center_u=c, confidence=conf))
    return out


def clip_column_by_heading_gate(
    columns_u: np.ndarray,
    *,
    max_abs_heading_deg: float = 50.0,
    persist_rows: int = 8,
    min_progress_m: float = 0.04,
) -> np.ndarray:
    """Near→far: if tipward heading stays steeply lateral, cut beyond gate."""

    out = np.asarray(columns_u, dtype=np.float32).copy()
    valid = np.flatnonzero(~np.isnan(out))
    if valid.size < persist_rows + 2:
        return out

    streak = 0
    gate = None
    # walk near→far along valid descending? valid is ascending row = far→near
    # walk from near (end) toward far (start)
    for k in range(len(valid) - 1, 0, -1):
        r_near = int(valid[k])
        r_far = int(valid[k - 1])
        x_n = ld.X_MAX_M - r_near * ld.METERS_PER_PIXEL
        x_f = ld.X_MAX_M - r_far * ld.METERS_PER_PIXEL
        y_n = ((ld.BEV_WIDTH - 1) / 2.0 - float(out[r_near])) * ld.METERS_PER_PIXEL
        y_f = ((ld.BEV_WIDTH - 1) / 2.0 - float(out[r_far])) * ld.METERS_PER_PIXEL
        dx = x_f - x_n
        dy = y_f - y_n
        ds = float(np.hypot(dx, dy))
        if ds < 1e-6:
            continue
        ang = abs(float(np.degrees(np.arctan2(dy, dx))))
        # lateral-dominant when |heading| from +X is large
        if ang >= max_abs_heading_deg and abs(dy) >= min_progress_m * 0.5:
            streak += 1
            if streak >= persist_rows:
                gate = r_far
                break
        else:
            streak = 0
    if gate is not None:
        out[:gate] = np.nan
    return out


def clip_pairs_heading_gate(
    pairs: list[ld.ForkLanePair],
    *,
    max_abs_heading_deg: float = 50.0,
) -> list[ld.ForkLanePair]:
    out: list[ld.ForkLanePair] = []
    for p in sw._pair_as_mutable(pairs):
        o = clip_column_by_heading_gate(p.outer_u, max_abs_heading_deg=max_abs_heading_deg)
        i = np.asarray(p.inner_u, dtype=np.float32).copy()
        c = np.asarray(p.center_u, dtype=np.float32).copy()
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                i[row] = np.nan
                c[row] = np.nan
        conf = float(np.clip(np.count_nonzero(~np.isnan(c)) / ld.BEV_HEIGHT, 0, 1))
        out.append(replace(p, outer_u=o, inner_u=i, center_u=c, confidence=conf))
    return out


def extend_no_wall_skate(
    left: np.ndarray,
    right: np.ndarray,
    boundary_mask: np.ndarray,
    *,
    assoc_m: float = ld.FAR_COURSE_ASSOC_M,
    max_miss_rows: int = ld.FAR_COURSE_MAX_MISS_ROWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy helper; production extend now stops at FOV sides."""

    del assoc_m, max_miss_rows
    return ld.extend_boundary_pair_far_along_marks(left, right, boundary_mask)


def build_variants(frame: np.ndarray, prefer_yellow: bool):
    # Snapshot "before production wall-clip" via post-hoc reconstruction is hard
    # once lane_detection clips. For A0 we still show current production.
    _, dbg = ld.detect_with_debug(frame, prefer_yellow=prefer_yellow)
    base = list(dbg.fork_lane_pairs)
    mask = mark_mask(dbg, prefer_yellow)

    # Synthetic "old skate" for visual A0_old reference: re-extend tip upward
    # along constant wall u (only for comparison sheet).
    def simulate_wall_skate(pairs: list[ld.ForkLanePair]) -> list[ld.ForkLanePair]:
        out: list[ld.ForkLanePair] = []
        for p in sw._pair_as_mutable(pairs):
            o = np.asarray(p.outer_u, dtype=np.float32).copy()
            valid = np.flatnonzero(~np.isnan(o))
            if valid.size == 0:
                out.append(p)
                continue
            tip = int(valid[0])
            u_tip = float(o[tip])
            # If tip already mid-side, fill upward along wall for demo contrast.
            if tip > 2 and (u_tip <= 2.0 or u_tip >= ld.BEV_WIDTH - 3):
                o[:tip] = u_tip
            i = np.asarray(p.inner_u, dtype=np.float32).copy()
            c = np.asarray(p.center_u, dtype=np.float32).copy()
            # crude same-row fill for demo
            full_w = ld.FORK_PAIR_WIDTH_M / ld.METERS_PER_PIXEL
            half = 0.5 * full_w
            side = "left" if int(p.lateral_rank) == 0 else "right"
            for row in range(ld.BEV_HEIGHT):
                if np.isnan(o[row]):
                    continue
                if side == "left":
                    i[row] = float(o[row]) + full_w
                    c[row] = float(o[row]) + half
                else:
                    i[row] = float(o[row]) - full_w
                    c[row] = float(o[row]) - half
            out.append(replace(p, outer_u=o, inner_u=i, center_u=c))
        return out

    a0_old = simulate_wall_skate(base)
    h0 = clip_pairs_side_wall(base)  # idempotent if production already clipped
    h1 = sw.apply_frenet_normal_width(list(base))
    h2 = sw.apply_curvature_parallel_rails(list(base))
    h3 = clip_pairs_heading_gate(base, max_abs_heading_deg=50.0)
    h4 = clip_pairs_heading_gate(list(base), max_abs_heading_deg=45.0)
    # H5: production base after wall clip + Frenet
    h5 = sw.apply_frenet_normal_width(clip_pairs_side_wall(base))

    return dbg, mask, {
        "A0_old_wall_skate": (a0_old, "참고: 예전 측벽 스키팅(인공 재현)"),
        "P0_production_now": (base, "생산 코드: side-wall clip 적용 후"),
        "H0_side_wall_clip": (h0, "명시적 측벽 클립(멱등)"),
        "H1_frenet_on_prod": (h1, "생산 tip + path-normal ±w"),
        "H2_curvature_on_prod": (h2, "생산 tip + κ 평행"),
        "H3_heading_gate": (h3, "|heading|>50° tip 절단"),
        "H4_heading_45": (h4, "|heading|>45° tip 절단"),
        "H5_wall_frenet": (h5, "wall clip + Frenet"),
    }


def tip_u_row(columns_u: np.ndarray) -> tuple[float, float]:
    o = np.asarray(columns_u, dtype=np.float32)
    valid = np.flatnonzero(~np.isnan(o))
    if valid.size == 0:
        return float("nan"), float("nan")
    tip = int(valid[0])
    return float(o[tip]), float(tip)


def side_exit_flag(columns_u: np.ndarray, *, margin_px: float = 3.0) -> int:
    u, row = tip_u_row(columns_u)
    if np.isnan(u) or np.isnan(row):
        return 0
    on_side = u <= margin_px or u >= (ld.BEV_WIDTH - 1) - margin_px
    not_top = row >= 8
    return int(on_side and not_top)


def annotate(img: np.ndarray, pairs: list[ld.ForkLanePair], title: str, note: str) -> np.ndarray:
    cv2.putText(img, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, note[:92], (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 255, 180), 1, cv2.LINE_AA)
    if len(pairs) >= 2:
        parts = []
        for lab, p in (("L", pairs[0]), ("R", pairs[1])):
            u, r = tip_u_row(p.outer_u)
            h = tip_heading_deg(p.outer_u)
            sk = wall_skate_rows(p.outer_u)
            parts.append(f"{lab}: tip(r={r:.0f},u={u:.0f}) h={h:.0f} skate={sk}")
        cv2.putText(
            img,
            "  ".join(parts),
            (4, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def contact_sheet(items: list[tuple[str, np.ndarray]], path: Path, cols: int = 3) -> None:
    if not items:
        return
    tiles = []
    h = w = None
    for name, im in items:
        canvas = im.copy()
        cv2.putText(
            canvas,
            name,
            (4, canvas.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
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


def metrics_row(scene: str, variant: str, pairs: list[ld.ForkLanePair], mask: np.ndarray) -> Row:
    sc = sw.score_pairs(pairs, mask)
    L, R = pairs[0], pairs[1]
    uL, rL = tip_u_row(L.outer_u)
    uR, rR = tip_u_row(R.outer_u)
    return Row(
        scene=scene,
        variant=variant,
        tip_row_L=rL,
        tip_row_R=rR,
        tip_u_L=uL,
        tip_u_R=uR,
        tip_heading_L_deg=tip_heading_deg(L.outer_u),
        tip_heading_R_deg=tip_heading_deg(R.outer_u),
        side_exit_L=side_exit_flag(L.outer_u),
        side_exit_R=side_exit_flag(R.outer_u),
        wall_skate_rows_L=wall_skate_rows(L.outer_u),
        wall_skate_rows_R=wall_skate_rows(R.outer_u),
        far_x_L=float(ld.X_MAX_M - rL * ld.METERS_PER_PIXEL) if not np.isnan(rL) else float("nan"),
        far_x_R=float(ld.X_MAX_M - rR * ld.METERS_PER_PIXEL) if not np.isnan(rR) else float("nan"),
        overextend_fraction=float(sc["overextend_fraction"]),
        paint_err_px=float(sc["mean_outer_paint_err_px"]),
    )


def main() -> int:
    stamp = _stamp()
    out_root = _ROOT / "data" / "captures" / "fork_rail_sweeps" / f"{stamp}_heading"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    md = [
        f"# Heading / side-exit sweep `{stamp}_heading`",
        "",
        "문제: 곡선 분기에서 outer가 FOV **옆면**에 닿은 뒤 u=0/W-1에 붙어 **위로 스키팅**.",
        "목표: tip이 상단(X_MAX)이 아니라 **좌/우 측면 출구**에 끝나게.",
        "",
    ]
    t0 = time.time()

    for scene, spec in SCENES.items():
        frame = cv2.imread(str(spec["frame"]))
        if frame is None:
            raise SystemExit(f"missing {spec['frame']}")
        scene_dir = out_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        dbg, mask, variants = build_variants(frame, bool(spec["prefer_yellow"]))
        previews: list[tuple[str, np.ndarray]] = []
        md.append(f"## {scene}")
        md.append("")
        md.append(
            "| variant | tip_row L/R | tip_u L/R | head L/R | side_exit | skate | far_x | overext | paint |"
        )
        md.append("|---|---|---|---|---|---|---|---:|---:|")

        for name, (pairs, note) in variants.items():
            if len(pairs) < 2:
                print(f"[{scene}] {name}: <2 pairs")
                continue
            img = ld.make_fork_focus_preview(sw.make_debug_with_pairs(dbg, pairs), focus="all")
            img = annotate(img, pairs, name, note)
            cv2.imwrite(str(scene_dir / f"{name}.png"), img)
            previews.append((name, img))
            row = metrics_row(scene, name, pairs, mask)
            rows.append(row)
            (scene_dir / f"{name}.json").write_text(json.dumps(asdict(row), indent=2), encoding="utf-8")
            md.append(
                f"| `{name}` | {row.tip_row_L:.0f}/{row.tip_row_R:.0f} | "
                f"{row.tip_u_L:.0f}/{row.tip_u_R:.0f} | "
                f"{row.tip_heading_L_deg:.0f}/{row.tip_heading_R_deg:.0f} | "
                f"{row.side_exit_L}/{row.side_exit_R} | "
                f"{row.wall_skate_rows_L}/{row.wall_skate_rows_R} | "
                f"{row.far_x_L:.2f}/{row.far_x_R:.2f} | "
                f"{row.overextend_fraction:.3f} | {row.paint_err_px:.1f} |"
            )
            print(
                f"[{scene}] {name}: tip_row={row.tip_row_L:.0f}/{row.tip_row_R:.0f} "
                f"u={row.tip_u_L:.0f}/{row.tip_u_R:.0f} skate={row.wall_skate_rows_L}/{row.wall_skate_rows_R} "
                f"side={row.side_exit_L}/{row.side_exit_R} head={row.tip_heading_L_deg:.0f}/{row.tip_heading_R_deg:.0f}"
            )

        contact_sheet(previews, scene_dir / "contact_sheet.png", cols=3)
        md.append("")
        md.append(f"![contact]({scene}/contact_sheet.png)")
        md.append("")

    with (out_root / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))
    md.append(f"elapsed_s={time.time() - t0:.1f}")
    (out_root / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_root / "results.json").write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print(f"\nWrote {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
