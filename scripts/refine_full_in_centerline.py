#!/usr/bin/env python3
"""Refine FULL IN waypoints onto yellow(+dash) DT centerline and smooth kinks.

Fixes:
- fork approach / north spur (≈24–36): stay on yellow center
- circle entry tangent (≈48–54)
- SW dashed zone (~114): pull to DT peak (less south bias)
- north merge (≈150–162): follow yellow/dash curve then OUT top (no left spike)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

WIDTH_M = 12.0
HEIGHT_M = 8.9975


def world_to_px_f(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    u = (x + WIDTH_M / 2.0) / WIDTH_M * w - 0.5
    v = (HEIGHT_M / 2.0 - y) / HEIGHT_M * h - 0.5
    return float(u), float(v)


def world_to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    u, v = world_to_px_f(x, y, w, h)
    return int(round(u)), int(round(v))


def px_to_world(u: float, v: float, w: int, h: int) -> tuple[float, float]:
    x = (u + 0.5) / w * WIDTH_M - WIDTH_M / 2.0
    y = HEIGHT_M / 2.0 - (v + 0.5) / h * HEIGHT_M
    return float(x), float(y)


def mask_yellow_with_dashes(hsv: np.ndarray) -> np.ndarray:
    """Solid yellow + dimmer dashed paint."""
    solid = cv2.inRange(hsv, (12, 70, 80), (42, 255, 255))
    dash = cv2.inRange(hsv, (8, 35, 55), (48, 255, 255))
    m = cv2.bitwise_or(solid, dash)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k2, iterations=1)
    return m


def corridor_mask(
    base: np.ndarray,
    w: int,
    h: int,
    poly_world: list[tuple[float, float]],
    dilate_px: int = 55,
) -> np.ndarray:
    pts = np.array([world_to_px(x, y, w, h) for x, y in poly_world], dtype=np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k)
    return cv2.bitwise_and(base, mask)


def dt_snap(
    x: float,
    y: float,
    dt: np.ndarray,
    ribbon: np.ndarray,
    w: int,
    h: int,
    radius_px: int = 28,
) -> tuple[float, float]:
    u0, v0 = world_to_px(x, y, w, h)
    best = (u0, v0)
    best_d = -1.0
    for dv in range(-radius_px, radius_px + 1):
        for du in range(-radius_px, radius_px + 1):
            u, v = u0 + du, v0 + dv
            if u < 0 or v < 0 or u >= w or v >= h:
                continue
            if ribbon[v, u] == 0:
                continue
            d = float(dt[v, u])
            if d > best_d:
                best_d = d
                best = (u, v)
    if best_d < 0:
        return x, y
    return px_to_world(best[0], best[1], w, h)


def resample_xy(xy: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(xy) < 2:
        return xy
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    targets = np.arange(0.0, total, spacing_m)
    if total - targets[-1] > spacing_m * 0.3:
        targets = np.append(targets, total)
    out = []
    j = 0
    for t in targets:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        if j + 1 >= len(s):
            out.append(xy[-1])
            break
        a = (t - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out.append((1 - a) * xy[j] + a * xy[j + 1])
    return np.asarray(out, dtype=np.float64)


def smooth_open(xy: np.ndarray, win: int = 7) -> np.ndarray:
    if len(xy) < win:
        return xy
    pad = win // 2
    ext = np.vstack([np.repeat(xy[:1], pad, 0), xy, np.repeat(xy[-1:], pad, 0)])
    k = np.ones(win) / win
    return np.column_stack(
        [np.convolve(ext[:, 0], k, mode="valid"), np.convolve(ext[:, 1], k, mode="valid")]
    )


def replace_segment_bezier(
    xy: np.ndarray,
    i0: int,
    i1: int,
    controls: list[tuple[float, float]],
    n: int | None = None,
) -> np.ndarray:
    """Replace xy[i0:i1+1] with cubic/quad Bezier through controls (incl. ends)."""
    pts = np.array([xy[i0], *controls, xy[i1]], dtype=np.float64)
    if n is None:
        n = max(2, i1 - i0 + 1)
    # De Casteljau sample
    m = len(pts) - 1
    ts = np.linspace(0.0, 1.0, n)
    curve = []
    for t in ts:
        layer = pts.copy()
        for _ in range(m):
            layer = (1 - t) * layer[:-1] + t * layer[1:]
        curve.append(layer[0])
    curve = np.asarray(curve, dtype=np.float64)
    return np.vstack([xy[:i0], curve, xy[i1 + 1 :]])


def fit_circle_cw_arc(
    center: tuple[float, float],
    radius: float,
    ang0: float,
    ang1: float,
    n: int,
) -> np.ndarray:
    """CW arc from ang0 to ang1 (world atan2), angles decreasing."""
    # unwrap CW: go from ang0 down to ang1
    a0 = ang0
    a1 = ang1
    while a1 > a0:
        a1 -= 2 * math.pi
    # ensure at least a bit of CW travel
    if a0 - a1 < 1e-3:
        a1 -= 0.15
    angs = np.linspace(a0, a1, n)
    cx, cy = center
    return np.column_stack([cx + radius * np.cos(angs), cy + radius * np.sin(angs)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full-json",
        type=Path,
        default=Path("/workspace/data/captures/full_in_course/full_in_waypoints.json"),
    )
    ap.add_argument(
        "--in-json",
        type=Path,
        default=Path(
            "/workspace/data/captures/in_roundabout_route/in_roundabout_waypoints.json"
        ),
    )
    ap.add_argument(
        "--out-best",
        type=Path,
        default=Path("/workspace/data/captures/out_best_route/out_best_waypoints.json"),
    )
    ap.add_argument(
        "--image",
        type=Path,
        default=Path(
            "/workspace/src/dracer_sim/models/track_plane/materials/textures/track_cw_real.png"
        ),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/data/captures/full_in_course"),
    )
    ap.add_argument("--spacing-m", type=float, default=0.15)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"failed to read {args.image}")
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = mask_yellow_with_dashes(hsv)

    # Keep IN corridor components (fork, circle, merge)
    poly = [
        (-2.6, -4.4),
        (0.5, -4.4),
        (1.0, -2.0),
        (2.2, 2.0),
        (2.4, 4.2),
        (-0.5, 4.2),
        (-2.8, 2.5),
        (-2.8, -1.0),
    ]
    ribbon = corridor_mask(yellow, w, h, poly, dilate_px=40)
    # also keep annulus around roundabout center from IN meta
    in_meta = json.loads(args.in_json.read_text(encoding="utf-8"))["meta"]
    cx, cy = in_meta["roundabout"]["fit_center_xy_m"]
    r = float(in_meta["roundabout"]["fit_radius_m"])
    ring = np.zeros_like(ribbon)
    uc, vc = world_to_px(cx, cy, w, h)
    cv2.circle(ring, (uc, vc), int((r + 0.45) / (WIDTH_M / w)), 255, -1)
    hole = np.zeros_like(ribbon)
    cv2.circle(hole, (uc, vc), int(max(0.2, r - 0.45) / (WIDTH_M / w)), 255, -1)
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(hole))
    ribbon = cv2.bitwise_or(ribbon, cv2.bitwise_and(yellow, ring))
    dt = cv2.distanceTransform(ribbon, cv2.DIST_L2, 5)

    data = json.loads(args.full_json.read_text(encoding="utf-8"))
    xy = np.array([[p["x_m"], p["y_m"]] for p in data["waypoints"]], dtype=np.float64)
    out_xy = np.array(
        [[p["x_m"], p["y_m"]] for p in json.loads(args.out_best.read_text())["waypoints"]],
        dtype=np.float64,
    )

    # --- Pass 1: snap yellow-ish segments onto DT (not pure OUT east/bottom) ---
    refined = xy.copy()
    for i, (x, y) in enumerate(xy):
        # skip pure OUT bottom start and OUT after merge east
        if i <= 18:
            continue
        if x > 1.85 and y > 3.4:  # outer top east after merge
            continue
        if y < -3.7 and x > -0.3:  # outer bottom east of fork
            continue
        if abs(math.hypot(x - cx, y - cy) - r) < 0.55 or y > -3.85:
            sx, sy = dt_snap(x, y, dt, ribbon, w, h, radius_px=32)
            # blend: keep some path direction, avoid huge jumps
            if math.hypot(sx - x, sy - y) < 0.35:
                refined[i] = (0.35 * x + 0.65 * sx, 0.35 * y + 0.65 * sy)

    # --- Pass 2: geometric fixes for known kinks ---
    # 24–34: fork along yellow dashed center (west then north), not cutting corner
    def idx_near(target, lo, hi):
        seg = refined[lo : hi + 1]
        j = int(np.argmin(np.linalg.norm(seg - np.array(target), axis=1)))
        return lo + j

    i_fork = idx_near((-0.45, -3.93), 15, 30)
    i_spur = idx_near((-1.97, -2.85), 28, 48)
    ctrl = []
    for pt in [
        (-0.70, -3.93),
        (-1.00, -3.95),
        (-1.30, -4.02),
        (-1.50, -4.05),
        (-1.62, -3.92),
        (-1.75, -3.70),
        (-1.88, -3.40),
        (-1.96, -3.10),
    ]:
        sx, sy = dt_snap(pt[0], pt[1], dt, ribbon, w, h, radius_px=42)
        # keep fork on the south/west dash (avoid cutting inside north)
        if sy > pt[1] + 0.08:
            sy = 0.4 * sy + 0.6 * pt[1]
        ctrl.append((sx, sy))
    n_fork = max(12, i_spur - i_fork)
    fork_curve = []
    anchors = [refined[i_fork], *ctrl, refined[i_spur]]
    A = np.array(anchors, dtype=np.float64)
    for t in np.linspace(0, 1, n_fork):
        layer = A.copy()
        for _ in range(len(A) - 1):
            layer = (1 - t) * layer[:-1] + t * layer[1:]
        sx, sy = dt_snap(float(layer[0, 0]), float(layer[0, 1]), dt, ribbon, w, h, 38)
        fork_curve.append((sx, sy))
    refined = np.vstack([refined[:i_fork], fork_curve, refined[i_spur + 1 :]])

    # 48–54: tangent CW entry onto ring
    # re-find after length change — use geometry
    def find_range_by_y(y0, y1, x_max=-1.4):
        idxs = [
            i
            for i, p in enumerate(refined)
            if y0 <= p[1] <= y1 and p[0] < x_max and abs(math.hypot(p[0] - cx, p[1] - cy) - r) < 1.2
        ]
        return idxs

    # entry: approach on x≈-1.97, then join circle CW
    i_ap = None
    for i, p in enumerate(refined):
        if abs(p[0] + 1.97) < 0.12 and -1.35 < p[1] < -1.05:
            i_ap = i
    i_on = None
    for i, p in enumerate(refined):
        if i_ap is not None and i > i_ap and abs(math.hypot(p[0] - cx, p[1] - cy) - r) < 0.12:
            if p[1] > -0.9:
                i_on = i
                break
    if i_ap is not None:
        # end of replacement: first solid on-ring CW point west of entry
        if i_on is None:
            i_on = min(len(refined) - 1, i_ap + 8)
        ang_ap = math.atan2(refined[i_ap, 1] - cy, refined[i_ap, 0] - cx)
        # desired join angle slightly CW from south-west approach
        join_ang = math.atan2(-1.05 - cy, -1.95 - cx)  # near SW
        # better: project DT peak near west ring
        jx, jy = dt_snap(-2.05, -0.55, dt, ribbon, w, h, 40)
        join_ang = math.atan2(jy - cy, jx - cx)
        # approach stays on spur center then arc CW onto ring
        # Stay on entry spur center longer, then CW-tangent onto ring (no inward cut)
        spur_pts = []
        for y in np.linspace(float(refined[i_ap, 1]), -1.05, 6):
            sx, sy = dt_snap(-1.97, float(y), dt, ribbon, w, h, 30)
            spur_pts.append((sx, sy))
        # Join circle near due-west of center for clean CW tangent
        jx, jy = cx - r, cy - 0.15
        jx, jy = dt_snap(jx, jy, dt, ribbon, w, h, 22)
        join_ang = math.atan2(jy - cy, jx - cx)
        n_arc = 8
        i_end = i_ap + 1
        while i_end < len(refined) - 1:
            if refined[i_end, 1] > -0.35 and abs(
                math.hypot(refined[i_end, 0] - cx, refined[i_end, 1] - cy) - r
            ) < 0.22:
                break
            i_end += 1
            if i_end - i_ap > 16:
                break
        end_ang = math.atan2(refined[i_end, 1] - cy, refined[i_end, 0] - cx)
        arc = fit_circle_cw_arc((cx, cy), r, join_ang, end_ang, n_arc)
        # snap arc lightly to yellow
        arc_s = np.array(
            [dt_snap(float(a[0]), float(a[1]), dt, ribbon, w, h, 18) for a in arc],
            dtype=np.float64,
        )
        blend = np.vstack([spur_pts[:-1], arc_s])
        refined = np.vstack([refined[:i_ap], blend, refined[i_end + 1 :]])

    # ~114 SW dashed: force snap with larger radius and prefer higher y (north)
    for i, p in enumerate(refined):
        rr = math.hypot(p[0] - cx, p[1] - cy)
        ang = math.atan2(p[1] - cy, p[0] - cx)
        # SW sector during second pass (ang around -2.0..-0.8)
        if 1.2 < rr < 1.85 and -2.4 < ang < -0.6:
            # search DT with north preference
            u0, v0 = world_to_px(float(p[0]), float(p[1]), w, h)
            best = (u0, v0)
            best_score = -1e9
            for dv in range(-40, 41):
                for du in range(-40, 41):
                    u, v = u0 + du, v0 + dv
                    if u < 0 or v < 0 or u >= w or v >= h or ribbon[v, u] == 0:
                        continue
                    wx, wy = px_to_world(u, v, w, h)
                    # prefer larger DT and slightly larger y (less "내려옴")
                    score = float(dt[v, u]) + 8.0 * (wy - p[1])
                    if score > best_score:
                        best_score = score
                        best = (u, v)
            refined[i] = px_to_world(best[0], best[1], w, h)

    # Circle → north exit spur: leave on tangent, stay on yellow dash center
    i_ex0 = None
    for i, p in enumerate(refined):
        rr = math.hypot(p[0] - cx, p[1] - cy)
        if abs(rr - r) < 0.20 and p[1] > 1.15 and -1.2 < p[0] < -0.55:
            i_ex0 = i
    i_ex1 = None
    for i, p in enumerate(refined):
        if 0.30 < p[0] < 0.65 and 2.35 < p[1] < 2.70:
            i_ex1 = i
            break
    if i_ex0 is not None and i_ex1 is not None and i_ex1 > i_ex0:
        guides = [
            (float(refined[i_ex0, 0]), float(refined[i_ex0, 1])),
            (-0.70, 1.50),
            (-0.35, 1.55),
            (0.05, 1.70),
            (0.30, 2.05),
            (0.42, 2.40),
            (float(refined[i_ex1, 0]), float(refined[i_ex1, 1])),
        ]
        guided = []
        for gx, gy in guides:
            sx, sy = dt_snap(gx, gy, dt, ribbon, w, h, 34)
            guided.append((sx, sy))
        G = np.asarray(guided, dtype=np.float64)
        n_ex = max(10, i_ex1 - i_ex0)
        curve = []
        for t in np.linspace(0.0, 1.0, n_ex):
            layer = G.copy()
            for _ in range(len(G) - 1):
                layer = (1 - t) * layer[:-1] + t * layer[1:]
            curve.append(layer[0])
        exit_arr = smooth_open(np.asarray(curve, dtype=np.float64), win=5)
        refined = np.vstack([refined[:i_ex0], exit_arr, refined[i_ex1 + 1 :]])

    # 150–162 merge: Bezier along yellow/dash then OUT top — x non-decreasing
    i_m0 = None
    for i, p in enumerate(refined):
        if 0.25 < p[0] < 0.75 and 2.45 < p[1] < 2.85:
            i_m0 = i
    i_m1 = None
    for i, p in enumerate(refined):
        if p[0] > 1.65 and 3.55 < p[1] < 3.85:
            i_m1 = i
            break
    if i_m0 is not None and i_m1 is not None and i_m1 > i_m0:
        # Guides: yellow exit center → dashed NE curve → OUT top dash → wp124
        guides = [
            (float(refined[i_m0, 0]), float(refined[i_m0, 1])),
            (0.50, 2.95),
            (0.55, 3.25),
            (0.72, 3.55),
            (0.95, 3.78),
            (1.20, 3.80),
            (1.45, 3.74),
            (float(out_xy[124, 0]), float(out_xy[124, 1])),
        ]
        guided = []
        for gx, gy in guides:
            sx, sy = dt_snap(gx, gy, dt, ribbon, w, h, 36)
            # don't let DT pull west of the guide (out_in_merge attractor)
            if sx < gx - 0.04:
                sx = 0.7 * gx + 0.3 * sx
            # keep near OUT lane once high enough
            if gy >= 3.65:
                sy = 0.35 * sy + 0.65 * gy
                if gy >= 3.72:
                    sy = min(sy, 3.90)
            guided.append((sx, sy))
        G = np.asarray(guided, dtype=np.float64)
        n_merge = max(14, i_m1 - i_m0 + 2)
        curve = []
        for t in np.linspace(0.0, 1.0, n_merge):
            layer = G.copy()
            for _ in range(len(G) - 1):
                layer = (1 - t) * layer[:-1] + t * layer[1:]
            curve.append(layer[0])
        merge_arr = np.asarray(curve, dtype=np.float64)
        # enforce non-decreasing x after climbing past y=3.2
        for k in range(1, len(merge_arr)):
            if merge_arr[k, 1] > 3.2 and merge_arr[k, 0] < merge_arr[k - 1, 0]:
                merge_arr[k, 0] = merge_arr[k - 1, 0]
        merge_arr = smooth_open(merge_arr, win=5)
        for k in range(1, len(merge_arr)):
            if merge_arr[k, 1] > 3.2 and merge_arr[k, 0] < merge_arr[k - 1, 0]:
                merge_arr[k, 0] = merge_arr[k - 1, 0]
        i_after = i_m1
        while i_after < len(refined) and refined[i_after, 0] < out_xy[124, 0] + 0.02:
            i_after += 1
        refined = np.vstack([refined[:i_m0], merge_arr, refined[i_after:]])

    # global light smooth + snap again on IN corridor + resample
    refined = smooth_open(refined, win=5)
    for i, (x, y) in enumerate(refined):
        if i < 12:
            continue
        if x > 2.0 and y > 3.4:
            continue
        if y < -3.75 and x > -0.2:
            continue
        if y > -4.2:
            sx, sy = dt_snap(float(x), float(y), dt, ribbon, w, h, 22)
            if math.hypot(sx - x, sy - y) < 0.22:
                refined[i] = (0.5 * x + 0.5 * sx, 0.5 * y + 0.5 * sy)

    sampled = resample_xy(refined, args.spacing_m)
    sampled = smooth_open(sampled, win=3)

    waypoints = []
    cum = 0.0
    prev = None
    for i, (x, y) in enumerate(sampled):
        if prev is not None:
            cum += math.hypot(x - prev[0], y - prev[1])
        prev = (float(x), float(y))
        u, v = world_to_px_f(float(x), float(y), w, h)
        waypoints.append(
            {
                "i": i,
                "s_m": round(cum, 4),
                "x_m": round(float(x), 4),
                "y_m": round(float(y), 4),
                "u_px": round(u, 2),
                "v_px": round(v, 2),
            }
        )

    meta = dict(data.get("meta", {}))
    meta.update(
        {
            "route": "FULL IN course (DT-refined yellow+dash centerline)",
            "refine": "yellow+dash DT snap; fork/entry/merge Bezier; SW lift; OUT stitch",
            "num_waypoints": len(waypoints),
            "path_length_m": round(cum, 3),
            "spacing_m": args.spacing_m,
            "roundabout_center_xy_m": [cx, cy],
            "roundabout_radius_m": r,
        }
    )

    (args.out_dir / "full_in_waypoints.json").write_text(
        json.dumps({"meta": meta, "waypoints": waypoints}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.out_dir / "full_in_waypoints.csv").open("w", encoding="utf-8") as f:
        f.write("i,s_m,x_m,y_m,u_px,v_px\n")
        for wp in waypoints:
            f.write(
                f"{wp['i']},{wp['s_m']},{wp['x_m']},{wp['y_m']},{wp['u_px']},{wp['v_px']}\n"
            )

    # overlays
    overlay = img.copy()
    tint = overlay.copy()
    tint[ribbon > 0] = (tint[ribbon > 0] * 0.55 + np.array([0, 90, 140])).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.6, tint, 0.4, 0)
    for i in range(len(sampled) - 1):
        p0 = world_to_px(*sampled[i], w, h)
        p1 = world_to_px(*sampled[i + 1], w, h)
        cv2.line(overlay, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)
    step = max(1, len(waypoints) // 40)
    for wp in waypoints[::step]:
        p = (int(round(wp["u_px"])), int(round(wp["v_px"])))
        cv2.circle(overlay, p, 3, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            str(wp["i"]),
            (p[0] + 2, p[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"FULL IN refined | n={len(waypoints)} | L={cum:.2f}m | ds={args.spacing_m}m",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "full_in_overlay.png"), overlay)
    cv2.imwrite(str(args.out_dir / "yellow_dash_ribbon.png"), ribbon)

    plot_w, plot_h = 1200, 920
    plot = np.full((plot_h, plot_w, 3), 28, dtype=np.uint8)
    margin = 55

    def w2plot(x, y):
        px = margin + (x + WIDTH_M / 2) / WIDTH_M * (plot_w - 2 * margin)
        py = margin + (HEIGHT_M / 2 - y) / HEIGHT_M * (plot_h - 2 * margin)
        return int(round(px)), int(round(py))

    for i in range(len(waypoints) - 1):
        cv2.line(
            plot,
            w2plot(waypoints[i]["x_m"], waypoints[i]["y_m"]),
            w2plot(waypoints[i + 1]["x_m"], waypoints[i + 1]["y_m"]),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        plot,
        f"Gazebo [m] FULL IN refined  L={cum:.2f}m  n={len(waypoints)}",
        (margin, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "full_in_gazebo_m.png"), plot)

    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# FULL IN 코스 웨이포인트 (점선 포함 중앙선 보정)",
                "",
                "- yellow solid + dashed paint DT centerline snap",
                "- fork / circle tangent entry / SW dashed / north merge 보정",
                f"- **n={len(waypoints)}**, **L≈{cum:.2f} m**, ds={args.spacing_m} m",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
