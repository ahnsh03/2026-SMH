#!/usr/bin/env python3
"""Extract IN-course yellow centerline: inout fork → roundabout → top merge.

Endpoints anchored near OUT best-route wp20 (start) and wp124 (end), with
LIMO spawn_poses (in_roundabout_entry / exit / in_out_merge) as guides.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

WIDTH_M = 12.0
HEIGHT_M = 8.9975

# LIMO spawn_poses.yaml (gazebo world)
SPAWNS = {
    "inout_fork": (-0.3, -3.92),
    "in_roundabout_entry": (-1.97, -2.15),
    "in_roundabout_exit": (-0.9, 1.39),
    "in_out_merge": (0.45, 2.45),
    "out_in_merge": (0.0, 3.71),
}

# From prior OUT extract (approx anchors requested by user)
OUT_WP20 = (-1.365, -4.097)
OUT_WP124 = (1.732, 3.721)


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


def mask_yellow(hsv: np.ndarray) -> np.ndarray:
    """Solid yellow + dimmer dashed paint (fork / merge / island)."""
    solid = cv2.inRange(hsv, (12, 70, 80), (42, 255, 255))
    dash = cv2.inRange(hsv, (8, 35, 55), (48, 255, 255))
    m = cv2.bitwise_or(solid, dash)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)


def build_yellow_ribbon(yellow: np.ndarray, close_ks: int = 21) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    closed = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, k, iterations=2)
    # Keep components that intersect the IN corridor (spawn bbox)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    h, w = yellow.shape
    seeds = [
        world_to_px(*SPAWNS["inout_fork"], w, h),
        world_to_px(*SPAWNS["in_roundabout_entry"], w, h),
        world_to_px(*SPAWNS["in_roundabout_exit"], w, h),
        world_to_px(*SPAWNS["in_out_merge"], w, h),
        world_to_px(*OUT_WP20, w, h),
        world_to_px(*OUT_WP124, w, h),
    ]
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        for u, v in seeds:
            if 0 <= u < w and 0 <= v < h and labels[v, u] == i:
                keep[i] = True
                break
        # also keep large components near corridor x in [-3, 2], y in [-4.2, 4]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = x + bw / 2, y + bh / 2
        wx, wy = px_to_world(cx, cy, w, h)
        if -3.5 <= wx <= 2.5 and -4.3 <= wy <= 4.2 and stats[i, cv2.CC_STAT_AREA] > 80:
            keep[i] = True
    out = np.zeros_like(closed)
    for i in range(1, n):
        if keep[i]:
            out[labels == i] = 255
    return out


def extract_centerline(ribbon: np.ndarray) -> np.ndarray:
    """Outer-contour inward DT peak (same idea as OUT extract)."""
    cnts, hier = cv2.findContours(ribbon, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not cnts or hier is None:
        raise RuntimeError("no yellow contours")
    # Prefer contour whose bbox covers roundabout area
    h, w = ribbon.shape
    best = None
    best_score = -1.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = px_to_world(x + bw / 2, y + bh / 2, w, h)
        # prefer components near roundabout center ~ (-0.5, 0)
        dist = math.hypot(cx + 0.5, cy - 0.0)
        score = area / (1.0 + dist)
        if score > best_score:
            best_score = score
            best = c
    if best is None:
        best = max(cnts, key=cv2.contourArea)

    # Fill ribbon may be multi-piece tree; better: DT ridge on full ribbon
    dist = cv2.distanceTransform(ribbon, cv2.DIST_L2, 5)
    dil = cv2.dilate(dist, np.ones((3, 3), np.uint8))
    ridge = ((dist >= dil - 0.2) & (ribbon > 0) & (dist >= 1.5)).astype(np.uint8) * 255
    if hasattr(cv2, "ximgproc"):
        skel = cv2.ximgproc.thinning(ribbon)
    else:
        skel = _morph_skel(ridge)
    return skel


def _morph_skel(mask: np.ndarray) -> np.ndarray:
    sk = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = mask.copy()
    while cv2.countNonZero(img):
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        sk = cv2.bitwise_or(sk, cv2.subtract(img, opened))
        img = eroded
    return sk


def graph(skel: np.ndarray):
    ys, xs = np.where(skel > 0)
    pts = set(zip(xs.tolist(), ys.tolist()))
    nbrs: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for u, v in pts:
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                if du == 0 and dv == 0:
                    continue
                n = (u + du, v + dv)
                if n in pts:
                    nbrs[(u, v)].append(n)
    return pts, nbrs


def nearest(pts, u, v):
    return min(pts, key=lambda p: (p[0] - u) ** 2 + (p[1] - v) ** 2)


def prune_spurs(skel: np.ndarray, passes: int = 40) -> np.ndarray:
    pts, nbrs = graph(skel)
    for _ in range(passes):
        ends = [p for p, ns in nbrs.items() if len(ns) == 1]
        if not ends:
            break
        for p in ends:
            for n in list(nbrs.get(p, [])):
                if n in nbrs:
                    nbrs[n] = [x for x in nbrs[n] if x != p]
            nbrs.pop(p, None)
            pts.discard(p)
    out = np.zeros_like(skel)
    for u, v in pts:
        out[v, u] = 255
    return out


def astar_path(
    pts: set,
    nbrs: dict,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    import heapq

    def h(p):
        return math.hypot(p[0] - goal[0], p[1] - goal[1])

    open_h = [(h(start), 0.0, start, None)]
    came: dict[tuple[int, int], tuple[int, int] | None] = {}
    gscore = {start: 0.0}
    closed = set()
    while open_h:
        _, g, cur, parent = heapq.heappop(open_h)
        if cur in closed:
            continue
        came[cur] = parent
        closed.add(cur)
        if cur == goal or (cur[0] - goal[0]) ** 2 + (cur[1] - goal[1]) ** 2 <= 4:
            # reconstruct
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            path.reverse()
            if path[-1] != goal:
                path.append(goal)
            return path
        for nxt in nbrs.get(cur, []):
            ng = g + math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
            if nxt in closed and ng >= gscore.get(nxt, 1e18):
                continue
            if ng < gscore.get(nxt, 1e18):
                gscore[nxt] = ng
                heapq.heappush(open_h, (ng + h(nxt), ng, nxt, cur))
    return []


def fit_roundabout_ring(
    ribbon: np.ndarray, w: int, h: int
) -> tuple[float, float, float, np.ndarray]:
    """Return (cx_px, cy_px, r_px, closed_loop_px Nx2) for yellow circle centerline."""
    # Seed center near spawn geometry (entry SW, exit N).
    seed = world_to_px(-0.7, 0.0, w, h)
    mpp = WIDTH_M / w
    r0 = 1.456 / mpp  # mid of R_in=1.278, R_out=1.634
    mask = np.zeros_like(ribbon)
    cv2.circle(mask, seed, int(r0 * 1.4), 255, -1)
    ring = cv2.bitwise_and(ribbon, mask)
    ys, xs = np.where(ring > 0)
    if len(xs) < 50:
        raise RuntimeError("roundabout ring pixels not found")
    d = np.hypot(xs - seed[0], ys - seed[1])
    ann = (d > r0 * 0.55) & (d < r0 * 1.25)
    if int(ann.sum()) < 40:
        ann = d > 0
    pts = np.column_stack([xs[ann], ys[ann]]).astype(np.float32)
    (fcx, fcy), fr = cv2.minEnclosingCircle(pts)
    # Refine with radial median in annulus around enclosing circle
    d2 = np.hypot(pts[:, 0] - fcx, pts[:, 1] - fcy)
    keep = (d2 > fr * 0.55) & (d2 < fr * 1.05)
    if int(keep.sum()) >= 40:
        pts = pts[keep]
        (fcx, fcy), fr = cv2.minEnclosingCircle(pts)
        d2 = np.hypot(pts[:, 0] - fcx, pts[:, 1] - fcy)
        fr = float(np.median(d2))

    # Sample centerline by angle — median radius per bin
    ang = np.arctan2(pts[:, 1] - fcy, pts[:, 0] - fcx)
    n_bins = 120
    bins = np.linspace(-math.pi, math.pi, n_bins + 1)
    loop = []
    for i in range(n_bins):
        sel = (ang >= bins[i]) & (ang < bins[i + 1])
        if not np.any(sel):
            a = 0.5 * (bins[i] + bins[i + 1])
            loop.append([fcx + fr * math.cos(a), fcy + fr * math.sin(a)])
            continue
        r_med = float(np.median(np.hypot(pts[sel, 0] - fcx, pts[sel, 1] - fcy)))
        a = float(np.median(ang[sel]))
        loop.append([fcx + r_med * math.cos(a), fcy + r_med * math.sin(a)])
    loop_arr = np.asarray(loop, dtype=np.float64)
    # Close
    loop_arr = np.vstack([loop_arr, loop_arr[:1]])
    # Orient loop clockwise in Gazebo world (negative shoelace about center).
    world = np.array([px_to_world(u, v, w, h) for u, v in loop_arr[:-1]], dtype=np.float64)
    c = world.mean(axis=0)
    area2 = 0.0
    for i in range(len(world)):
        x1, y1 = world[i] - c
        x2, y2 = world[(i + 1) % len(world)] - c
        area2 += x1 * y2 - x2 * y1
    if area2 > 0:
        loop_arr = loop_arr[::-1]
    return float(fcx), float(fcy), float(fr), loop_arr


def rotate_closed_loop(loop: np.ndarray, start_idx: int) -> np.ndarray:
    """Rotate closed loop (last==first) so it begins at start_idx."""
    body = loop[:-1]
    rot = np.concatenate([body[start_idx:], body[:start_idx]], axis=0)
    return np.vstack([rot, rot[:1]])


def build_entry_circle_exit(
    skel_pts: set,
    skel_nbrs: dict,
    ribbon: np.ndarray,
    w: int,
    h: int,
    start_px: tuple[int, int],
    goal_px: tuple[int, int],
    circle_loop: np.ndarray,
) -> tuple[list[tuple[float, float]], dict]:
    """south approach → full CW lap on circle → north exit."""
    entry_w = SPAWNS["in_roundabout_entry"]
    exit_w = SPAWNS["in_roundabout_exit"]
    eu, ev = world_to_px(*entry_w, w, h)
    xu, xv = world_to_px(*exit_w, w, h)

    def nearest_loop_idx(u, v):
        d2 = (circle_loop[:-1, 0] - u) ** 2 + (circle_loop[:-1, 1] - v) ** 2
        return int(np.argmin(d2))

    i_entry = nearest_loop_idx(eu, ev)
    i_exit = nearest_loop_idx(xu, xv)
    body = circle_loop[:-1]
    i_join = i_entry

    loop_from_join = rotate_closed_loop(circle_loop, i_join)
    # Full CW lap, then continue CW from join to exit.
    lap = loop_from_join[:-1].copy()
    n = len(body)
    exit_in_rot = (i_exit - i_join) % n
    if exit_in_rot == 0:
        exit_in_rot = n
    to_exit_after_lap = lap[1 : exit_in_rot + 1] if exit_in_rot > 0 else lap[1:]
    circle_run = np.vstack([lap, lap[:1], to_exit_after_lap])

    join_px = (int(round(lap[0, 0])), int(round(lap[0, 1])))
    exit_px = (
        int(round(circle_run[-1, 0])),
        int(round(circle_run[-1, 1])),
    )
    join_s = nearest(skel_pts, *join_px)
    exit_s = nearest(skel_pts, *exit_px)

    entry_path = astar_path(skel_pts, skel_nbrs, start_px, join_s)
    exit_path = astar_path(skel_pts, skel_nbrs, exit_s, goal_px)
    if len(entry_path) < 2:
        entry_path = [start_px, join_s]
    if len(exit_path) < 2:
        exit_path = [exit_s, goal_px]

    path = (
        [(float(u), float(v)) for u, v in entry_path[:-1]]
        + [(float(u), float(v)) for u, v in circle_run]
        + [(float(u), float(v)) for u, v in exit_path[1:]]
    )
    info = {
        "circle_center_px": [
            float(circle_loop[:-1, 0].mean()),
            float(circle_loop[:-1, 1].mean()),
        ],
        "join_idx": i_join,
        "exit_idx": i_exit,
        "exit_in_rot": int(exit_in_rot),
        "lap_points": int(len(lap)),
        "direction": "CW",
        "policy": "entry → full CW lap → continue to exit → merge",
    }
    return path, info


def sample_polyline(pts_px: np.ndarray, spacing_m: float, w: int, h: int) -> np.ndarray:
    if len(pts_px) < 2:
        return pts_px.astype(np.float64)
    world = np.array([px_to_world(u, v, w, h) for u, v in pts_px], dtype=np.float64)
    seg = np.linalg.norm(np.diff(world, axis=0), axis=1)
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
            out.append(pts_px[-1])
            break
        a = (t - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out.append((1 - a) * pts_px[j] + a * pts_px[j + 1])
    return np.asarray(out, dtype=np.float64)


def smooth(pts: np.ndarray, win: int = 9) -> np.ndarray:
    if len(pts) < win:
        return pts
    pad = win // 2
    # open path: edge pad
    ext = np.vstack([np.repeat(pts[:1], pad, 0), pts, np.repeat(pts[-1:], pad, 0)])
    k = np.ones(win) / win
    return np.column_stack(
        [np.convolve(ext[:, 0], k, mode="valid"), np.convolve(ext[:, 1], k, mode="valid")]
    )


def load_out_xy(out_json: Path) -> np.ndarray:
    data = json.loads(out_json.read_text(encoding="utf-8"))
    wps = data["waypoints"]
    return np.array([[w["x_m"], w["y_m"]] for w in wps], dtype=np.float64)


def stitch_exit_to_out_dash(
    path_xy: np.ndarray,
    out_xy: np.ndarray,
    *,
    out_end_i: int = 124,
    out_search0: int = 112,
    min_in_y: float = 2.35,
) -> tuple[np.ndarray, dict]:
    """Cut yellow exit if it climbs above the white top dash; append OUT → wp124.

    Yellow paint ends early / drifts north of the dashed merge. Hand off to the
    trusted OUT centerline for the final merge into the outer lane.
    """
    if len(path_xy) < 5 or out_end_i >= len(out_xy):
        return path_xy, {"stitched": False}

    best = (1e18, None, None)
    for i, (x, y) in enumerate(path_xy):
        if y < min_in_y:
            continue
        # exit spur is east of circle center (~-0.7)
        if x < -1.2:
            continue
        for j in range(out_search0, out_end_i + 1):
            ox, oy = out_xy[j]
            d = math.hypot(x - ox, y - oy)
            # heavy penalty for sitting above the outer lane centerline
            pen = 5.0 * max(0.0, y - oy - 0.02)
            score = d + pen
            if score < best[0]:
                best = (score, i, j)

    if best[1] is None:
        # fallback: nearest overall to OUT end among late IN points
        late = path_xy[path_xy[:, 1] >= min_in_y]
        if len(late) == 0:
            return path_xy, {"stitched": False}
        d = np.linalg.norm(late - out_xy[out_end_i], axis=1)
        i_rel = int(np.argmin(d))
        # map back
        idxs = np.where(path_xy[:, 1] >= min_in_y)[0]
        best = (float(d[i_rel]), int(idxs[i_rel]), out_end_i)

    i_cut, j_join = int(best[1]), int(best[2])
    # keep a short run-in on yellow, then OUT dash to end
    stitched = np.vstack([path_xy[: i_cut + 1], out_xy[j_join : out_end_i + 1]])
    # drop near-duplicate at joint
    if len(stitched) >= 2 and np.linalg.norm(stitched[i_cut + 1] - stitched[i_cut]) < 0.05:
        stitched = np.vstack([stitched[: i_cut + 1], stitched[i_cut + 2 :]])
    info = {
        "stitched": True,
        "in_cut_i": i_cut,
        "out_join_i": j_join,
        "out_end_i": out_end_i,
        "joint_dist_m": round(float(best[0]), 3),
        "joint_xy": [round(float(path_xy[i_cut, 0]), 4), round(float(path_xy[i_cut, 1]), 4)],
    }
    return stitched, info


def main() -> int:
    ap = argparse.ArgumentParser()
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
        default=Path("/workspace/data/captures/in_roundabout_route"),
    )
    ap.add_argument("--spacing-m", type=float, default=0.15)
    ap.add_argument("--close-ks", type=int, default=19)
    ap.add_argument(
        "--out-best-json",
        type=Path,
        default=Path("/workspace/data/captures/out_best_route/out_best_waypoints.json"),
        help="OUT centerline used to finish the top dashed merge → wp124",
    )
    ap.add_argument("--out-merge-end-i", type=int, default=124)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"failed to read {args.image}")
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow = mask_yellow(hsv)
    ribbon = build_yellow_ribbon(yellow, args.close_ks)
    skel = extract_centerline(ribbon)
    skel = prune_spurs(skel, passes=50)

    pts, nbrs = graph(skel)
    if not pts:
        raise SystemExit("empty yellow skeleton")

    start_opts = [
        nearest(pts, *world_to_px(*OUT_WP20, w, h)),
        nearest(pts, *world_to_px(*SPAWNS["inout_fork"], w, h)),
        nearest(pts, *world_to_px(*SPAWNS["in_roundabout_entry"], w, h)),
    ]
    # Follow yellow exit toward the top dash, but not past/above OUT lane —
    # final meters are stitched onto OUT best centerline → wp124.
    goal_opts = [
        nearest(pts, *world_to_px(0.55, 3.45, w, h)),
        nearest(pts, *world_to_px(*SPAWNS["in_out_merge"], w, h)),
        nearest(pts, *world_to_px(*SPAWNS["out_in_merge"], w, h)),
    ]
    goal = min(
        goal_opts,
        key=lambda p: abs(px_to_world(p[0], p[1], w, h)[1] - 3.45)
        + 0.3 * abs(px_to_world(p[0], p[1], w, h)[0] - 0.55),
    )
    start = min(start_opts, key=lambda p: px_to_world(p[0], p[1], w, h)[1])

    fcx, fcy, fr, circle_loop = fit_roundabout_ring(ribbon, w, h)
    path_f, circle_info = build_entry_circle_exit(
        pts, nbrs, ribbon, w, h, start, goal, circle_loop
    )
    if len(path_f) < 30:
        raise SystemExit(f"path too short after full-lap build: {len(path_f)}")

    path_arr = np.asarray(path_f, dtype=np.float64)
    path_arr = smooth(path_arr, win=9)
    world_path = np.array(
        [px_to_world(float(u), float(v), w, h) for u, v in path_arr], dtype=np.float64
    )
    stitch_info = {"stitched": False}
    if args.out_best_json.is_file():
        out_xy = load_out_xy(args.out_best_json)
        world_path, stitch_info = stitch_exit_to_out_dash(
            world_path, out_xy, out_end_i=args.out_merge_end_i
        )
        path_arr = np.array(
            [world_to_px_f(float(x), float(y), w, h) for x, y in world_path],
            dtype=np.float64,
        )
    sampled = sample_polyline(path_arr, args.spacing_m, w, h)
    circle_info.update(
        {
            "fit_center_px": [fcx, fcy],
            "fit_radius_px": fr,
            "fit_radius_m": round(fr * (WIDTH_M / w), 3),
            "fit_center_xy_m": list(px_to_world(fcx, fcy, w, h)),
            "exit_stitch": stitch_info,
        }
    )

    waypoints = []
    cum = 0.0
    prev = None
    for i, (u, v) in enumerate(sampled):
        x, y = px_to_world(float(u), float(v), w, h)
        if prev is not None:
            cum += math.hypot(x - prev[0], y - prev[1])
        prev = (x, y)
        waypoints.append(
            {
                "i": i,
                "s_m": round(cum, 4),
                "x_m": round(x, 4),
                "y_m": round(y, 4),
                "u_px": round(float(u), 2),
                "v_px": round(float(v), 2),
            }
        )

    # Anchors (exit/merge: prefer last visit — path passes exit area twice)
    world_pts = np.array([[wp["x_m"], wp["y_m"]] for wp in waypoints])
    anchors = {}
    prefer_last = {"in_roundabout_exit", "in_out_merge", "out_in_merge", "out_wp124_ref"}
    for name, xy in {
        **SPAWNS,
        "out_wp20_ref": OUT_WP20,
        "out_wp124_ref": OUT_WP124,
    }.items():
        d = np.linalg.norm(world_pts - np.array(xy), axis=1)
        near = np.where(d <= max(0.25, float(d.min()) + 1e-6))[0]
        j = int(near[-1] if name in prefer_last and len(near) else np.argmin(d))
        anchors[name] = {
            "xy": list(xy),
            "nearest_wp": j,
            "nearest_xy": [waypoints[j]["x_m"], waypoints[j]["y_m"]],
            "dist_m": round(float(d[j]), 3),
        }

    # Lap metric: polar unwrap about fitted center while on-ring (not path heading)
    cx_m, cy_m = circle_info["fit_center_xy_m"]
    r_m = float(circle_info["fit_radius_m"])
    rad = np.hypot(world_pts[:, 0] - cx_m, world_pts[:, 1] - cy_m)
    on_ring = np.where(np.abs(rad - r_m) < 0.35)[0]
    polar_turn = 0.0
    if len(on_ring) >= 2:
        prev = math.atan2(world_pts[on_ring[0], 1] - cy_m, world_pts[on_ring[0], 0] - cx_m)
        for i in on_ring[1:]:
            ang = math.atan2(world_pts[i, 1] - cy_m, world_pts[i, 0] - cx_m)
            dth = ang - prev
            while dth <= -math.pi:
                dth += 2 * math.pi
            while dth > math.pi:
                dth -= 2 * math.pi
            polar_turn += dth
            prev = ang

    meta = {
        "route": "IN yellow roundabout (full CW lap then exit)",
        "color": "yellow_only",
        "direction": "south(entry) → CW full lap → north(exit/merge)",
        "roundabout_policy": "one_full_cw_lap_before_exit",
        "start_anchor": "OUT wp20 / inout_fork vicinity",
        "end_anchor": "OUT best wp124 (top dash stitch after yellow exit)",
        "spacing_m": args.spacing_m,
        "num_waypoints": len(waypoints),
        "path_length_m": round(cum, 3),
        "circle_turn_rad": round(polar_turn, 3),
        "circle_turn_laps": round(abs(polar_turn) / (2 * math.pi), 2),
        "roundabout": circle_info,
        "gazebo_track_plane_m": {"width_m": WIDTH_M, "height_m": HEIGHT_M},
        "meters_per_pixel": {"x": WIDTH_M / w, "y": HEIGHT_M / h},
        "spawn_anchors": anchors,
        "image_size_px": [w, h],
    }

    (args.out_dir / "in_roundabout_waypoints.json").write_text(
        json.dumps({"meta": meta, "waypoints": waypoints}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (args.out_dir / "in_roundabout_waypoints.csv").open("w", encoding="utf-8") as f:
        f.write("i,s_m,x_m,y_m,u_px,v_px\n")
        for wp in waypoints:
            f.write(
                f"{wp['i']},{wp['s_m']},{wp['x_m']},{wp['y_m']},{wp['u_px']},{wp['v_px']}\n"
            )

    # Overlay
    overlay = img.copy()
    tint = overlay.copy()
    tint[ribbon > 0] = (tint[ribbon > 0] * 0.5 + np.array([0, 90, 140])).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0)
    for i in range(len(sampled) - 1):
        cv2.line(
            overlay,
            (int(round(sampled[i, 0])), int(round(sampled[i, 1]))),
            (int(round(sampled[i + 1, 0])), int(round(sampled[i + 1, 1]))),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    step = max(1, len(waypoints) // 30)
    for wp in waypoints[::step]:
        p = (int(round(wp["u_px"])), int(round(wp["v_px"])))
        cv2.circle(overlay, p, 3, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            str(wp["i"]),
            (p[0] + 3, p[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    for name, (x, y) in SPAWNS.items():
        u, v = world_to_px(x, y, w, h)
        cv2.drawMarker(overlay, (u, v), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(
            overlay,
            name,
            (u + 6, v - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    for label, xy in [("OUT_wp20", OUT_WP20), ("OUT_wp124", OUT_WP124)]:
        u, v = world_to_px(*xy, w, h)
        cv2.drawMarker(overlay, (u, v), (0, 165, 255), cv2.MARKER_DIAMOND, 14, 2)
        cv2.putText(
            overlay,
            label,
            (u + 6, v + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
    # Draw fitted circle for QA
    cv2.circle(
        overlay,
        (int(round(fcx)), int(round(fcy))),
        int(round(fr)),
        (255, 128, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        f"IN yellow FULL CW LAP | n={len(waypoints)} | L={cum:.2f}m | "
        f"laps≈{meta['circle_turn_laps']:.2f} | ds={args.spacing_m}m",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "in_roundabout_overlay.png"), overlay)
    cv2.imwrite(str(args.out_dir / "yellow_mask.png"), yellow)
    cv2.imwrite(str(args.out_dir / "yellow_ribbon.png"), ribbon)
    cv2.imwrite(str(args.out_dir / "yellow_skeleton.png"), skel)

    # Gazebo plot
    plot_w, plot_h = 1200, 920
    plot = np.full((plot_h, plot_w, 3), 28, dtype=np.uint8)
    margin = 55

    def w2plot(x, y):
        px = margin + (x + WIDTH_M / 2) / WIDTH_M * (plot_w - 2 * margin)
        py = margin + (HEIGHT_M / 2 - y) / HEIGHT_M * (plot_h - 2 * margin)
        return int(round(px)), int(round(py))

    for gx in range(-6, 7):
        cv2.line(plot, w2plot(gx, -HEIGHT_M / 2), w2plot(gx, HEIGHT_M / 2), (55, 55, 55), 1)
    for gy in range(-4, 5):
        cv2.line(plot, w2plot(-WIDTH_M / 2, gy), w2plot(WIDTH_M / 2, gy), (55, 55, 55), 1)
    for i in range(len(waypoints) - 1):
        cv2.line(
            plot,
            w2plot(waypoints[i]["x_m"], waypoints[i]["y_m"]),
            w2plot(waypoints[i + 1]["x_m"], waypoints[i + 1]["y_m"]),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if waypoints:
        cv2.circle(plot, w2plot(waypoints[0]["x_m"], waypoints[0]["y_m"]), 6, (0, 0, 255), -1)
        cv2.circle(
            plot,
            w2plot(waypoints[-1]["x_m"], waypoints[-1]["y_m"]),
            6,
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            plot,
            "start",
            w2plot(waypoints[0]["x_m"], waypoints[0]["y_m"] + 0.15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            plot,
            "end",
            w2plot(waypoints[-1]["x_m"], waypoints[-1]["y_m"] + 0.15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
    for name, (x, y) in SPAWNS.items():
        p = w2plot(x, y)
        cv2.drawMarker(plot, p, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
        cv2.putText(plot, name, (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
    cv2.putText(
        plot,
        f"Gazebo [m] IN yellow roundabout  L={cum:.2f}m  n={len(waypoints)}",
        (margin, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "in_roundabout_gazebo_m.png"), plot)

    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# IN 코스 회전교차로 웨이포인트 (노란 차선)",
                "",
                "- **시작:** OUT best wp20 부근 / `inout_fork` → `in_roundabout_entry`",
                "- **끝:** 노란 탈출 후 **OUT best 상단 점선**에 스티치 → wp124",
                "- **색:** yellow + OUT merge stitch",
                "- **정책:** 남쪽 진입 → **시계방향(CW) 한 바퀴(풀 랩)** → 북쪽 진출/합류 "
                "(첫 통과 때 진출로로 나가지 않음)",
                f"- **n={len(waypoints)}**, **L≈{cum:.2f} m**, "
                f"ring_laps≈{meta['circle_turn_laps']:.2f}, ds={args.spacing_m} m",
                f"- exit_stitch: `{meta['roundabout'].get('exit_stitch')}`",
                "",
                "## Files",
                "- `in_roundabout_waypoints.json` / `.csv`",
                "- `in_roundabout_overlay.png`",
                "- `in_roundabout_gazebo_m.png`",
                "",
                "## Spawn poses (참고)",
                "```",
                "inout_fork, in_roundabout_entry, in_roundabout_exit, in_out_merge, out_in_merge",
                "```",
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
