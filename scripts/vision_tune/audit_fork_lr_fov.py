#!/usr/bin/env python3
"""Audit fork pair L/R ordering vs stem-cross, and FOV-asymmetric outer lengths."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(_ROOT / "src" / "inference"))
sys.path.insert(0, str(_ROOT / "scripts" / "vision_tune"))

from inference.modules import lane_detection as ld  # noqa: E402
import sweep_fork_rail_variants as sw  # noqa: E402

FRAME = (
    _ROOT
    / "data/captures/lane_tune_logs/auto_fork/in_roundabout_exit/runs/20260713_152921/source_frame.png"
)
OUT = _ROOT / "data/captures/fork_rail_sweeps/20260714_lr_fov_audit"


def strip_stem_cross(pairs: list[ld.ForkLanePair]) -> list[ld.ForkLanePair]:
    """Force independent parallel rails (no opposite-outer stem inner)."""

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


def fov_paint_clip(
    pairs: list[ld.ForkLanePair],
    mask: np.ndarray,
    *,
    assoc_px: float = 8.0,
) -> list[ld.ForkLanePair]:
    """Drop rail rows whose outer is not near paint (asymmetric FOV aware)."""

    out: list[ld.ForkLanePair] = []
    for p in sw._pair_as_mutable(pairs):
        o = p.outer_u
        for row in range(ld.BEV_HEIGHT):
            if np.isnan(o[row]):
                p.inner_u[row] = np.nan
                p.center_u[row] = np.nan
                continue
            cols = np.flatnonzero(mask[row] > 0)
            if cols.size == 0 or float(np.min(np.abs(cols.astype(np.float32) - float(o[row])))) > assoc_px:
                o[row] = np.nan
                p.inner_u[row] = np.nan
                p.center_u[row] = np.nan
        conf = float(np.clip(np.count_nonzero(~np.isnan(p.center_u)) / ld.BEV_HEIGHT, 0, 1))
        out.append(replace(p, outer_u=o, confidence=conf))
    return out


def crossed_count(pairs: list[ld.ForkLanePair]) -> tuple[int, int]:
    if len(pairs) < 2:
        return 0, 0
    li = np.asarray(pairs[0].inner_u)
    ri = np.asarray(pairs[1].inner_u)
    both = np.flatnonzero(~np.isnan(li) & ~np.isnan(ri))
    if both.size == 0:
        return 0, 0
    return int(np.sum(li[both] > ri[both])), int(both.size)


def annotate(img: np.ndarray, pairs: list[ld.ForkLanePair], title: str) -> np.ndarray:
    y = 36
    cv2.putText(img, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    for p in pairs:
        o = np.asarray(p.outer_u)
        i = np.asarray(p.inner_u)
        both = np.flatnonzero(~np.isnan(o) & ~np.isnan(i))
        if both.size == 0:
            continue
        mo, mi = float(np.mean(o[both])), float(np.mean(i[both]))
        ok = (mo < mi) if int(p.lateral_rank) == 0 else (mo > mi)
        status = "OK" if ok else "SWAP"
        color = (0, 255, 0) if ok else (0, 0, 255)
        n = int(np.count_nonzero(~np.isnan(o)))
        far = float(ld.X_MAX_M - np.flatnonzero(~np.isnan(o))[0] * ld.METERS_PER_PIXEL)
        cv2.putText(
            img,
            f"r{p.lateral_rank} o/i={mo:.0f}/{mi:.0f} {status} rows={n} far={far:.2f}m",
            (4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 15
    if len(pairs) >= 2:
        lo = int(np.count_nonzero(~np.isnan(pairs[0].outer_u)))
        ro = int(np.count_nonzero(~np.isnan(pairs[1].outer_u)))
        cx, n = crossed_count(pairs)
        cv2.putText(
            img,
            f"outer_rows L={lo} R={ro} d={lo-ro}  crossed_inners={cx}/{n}",
            (4, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return img


def main() -> int:
    frame = cv2.imread(str(FRAME))
    assert frame is not None
    _, dbg = ld.detect_with_debug(frame, prefer_yellow=True)
    OUT.mkdir(parents=True, exist_ok=True)

    a0 = list(dbg.fork_lane_pairs)
    left_obs, right_obs, yl, yr, alt_l, alt_r = ld.build_global_boundary_course(
        dbg.yellow_bev, dbg.road_raw, dbg.road_clean, find_alternate=True
    )
    del left_obs, right_obs

    report = {
        "findings": [],
        "zones": {},
        "variants": {},
    }

    for zone, a, b in (("far0-80", 0, 80), ("mid80-160", 80, 160), ("near200-320", 200, 320)):

        def mid(L, R):
            both = ~np.isnan(L[a:b]) & ~np.isnan(R[a:b])
            if not np.any(both):
                return float("nan")
            return float(np.nanmedian(0.5 * (L[a:b][both] + R[a:b][both])))

        report["zones"][zone] = {
            "primary_mid_u": mid(yl, yr),
            "alt_mid_u": mid(alt_l, alt_r),
        }

    cx, n = crossed_count(a0)
    report["findings"].append(
        f"A0 crossed_inners={cx}/{n} (stem uses opposite outer as inner near ego)"
    )
    report["findings"].append(
        f"FOV outer rows L={np.count_nonzero(~np.isnan(a0[0].outer_u))} "
        f"R={np.count_nonzero(~np.isnan(a0[1].outer_u))} — asymmetry exists"
    )

    # Order check
    for p in a0:
        both = np.flatnonzero(~np.isnan(p.outer_u) & ~np.isnan(p.inner_u))
        mo, mi = float(np.mean(p.outer_u[both])), float(np.mean(p.inner_u[both]))
        ok = (mo < mi) if int(p.lateral_rank) == 0 else (mo > mi)
        report["findings"].append(
            f"A0 rank{p.lateral_rank} mean_u o/i={mo:.1f}/{mi:.1f} order_ok={ok}"
        )

    g_stem_off = strip_stem_cross(a0)
    g_fov = fov_paint_clip(a0, dbg.yellow_connected_bev)
    g_stem_fov = fov_paint_clip(strip_stem_cross(a0), dbg.yellow_connected_bev)
    f0 = sw.apply_curvature_parallel_rails(a0)
    f0_flip = []
    for p in sw._pair_as_mutable(a0):
        tmp = replace(p, lateral_rank=1 - int(p.lateral_rank))
        curved = sw.apply_curvature_parallel_rails([tmp])[0]
        f0_flip.append(
            replace(
                curved,
                lateral_rank=int(p.lateral_rank),
                outer_u=np.asarray(p.outer_u, dtype=np.float32).copy(),
            )
        )

    variants = {
        "A0_baseline": a0,
        "G1_no_stem_cross": g_stem_off,
        "G0_fov_paint_clip": g_fov,
        "G2_nostem_fov": g_stem_fov,
        "F0_curvature": f0,
        "F0_signflip": f0_flip,
    }

    for name, pairs in variants.items():
        cx, n = crossed_count(pairs)
        lo = int(np.count_nonzero(~np.isnan(pairs[0].outer_u)))
        ro = int(np.count_nonzero(~np.isnan(pairs[1].outer_u)))
        orders = []
        for p in pairs:
            both = np.flatnonzero(~np.isnan(p.outer_u) & ~np.isnan(p.inner_u))
            if both.size == 0:
                orders.append(None)
                continue
            mo, mi = float(np.mean(p.outer_u[both])), float(np.mean(p.inner_u[both]))
            orders.append(bool((mo < mi) if int(p.lateral_rank) == 0 else (mo > mi)))
        report["variants"][name] = {
            "crossed_inners": cx,
            "crossed_denom": n,
            "outer_rows_L": lo,
            "outer_rows_R": ro,
            "outer_delta_L_minus_R": lo - ro,
            "order_ok_by_rank": orders,
        }
        img = ld.make_fork_focus_preview(sw.make_debug_with_pairs(dbg, pairs), focus="all")
        img = annotate(img, pairs, name)
        cv2.imwrite(str(OUT / f"{name}.png"), img)

    # Contact of key variants
    keys = ["A0_baseline", "G1_no_stem_cross", "G0_fov_paint_clip", "G2_nostem_fov", "F0_signflip"]
    tiles = []
    for k in keys:
        im = cv2.imread(str(OUT / f"{k}.png"))
        if im is None:
            continue
        tiles.append(im)
    if tiles:
        h = min(t.shape[0] for t in tiles)
        w = min(t.shape[1] for t in tiles)
        tiles = [cv2.resize(t, (w, h)) for t in tiles]
        sheet = np.hstack(tiles)
        cv2.imwrite(str(OUT / "contact_lr_fov.png"), sheet)

    (OUT / "AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
