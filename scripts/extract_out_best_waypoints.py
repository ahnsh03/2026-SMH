#!/usr/bin/env python3
"""Extract OUT-course best-route waypoints from track_cw_real.png (Gazebo scale).

Method
------
1. Mask white rail paint (ignore yellow IN paint / logo blocks).
2. Morph-close rails into a filled OUT lane ribbon (span-selected component).
3. Take the ribbon outer contour; walk inward along normals to the local
   distance-transform peak → approximate lane centerline.
4. Order CCW from spawn ``start`` (heading west / -X).
5. Resample at fixed world spacing.

The outer-contour midline naturally takes the outer (left) branch of the OUT
parallel fork. Coordinates match ``track_plane.yaml`` / Gazebo world frame
(plane centered at origin, image u→+x, image v→−y).
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

SPAWNS = {
    "start": (2.6, -3.92),
    "inout_fork": (-0.3, -3.92),
    "out_fork": (-4.4, 3.72),
    "out_fork_merge_left": (-1.14, 4.05),
    "out_fork_merge_right": (-1.14, 3.38),
    "obstacle": (4.7, 2.97),
}


def world_to_px(x: float, y: float, w: int, h: int) -> tuple[int, int]:
    u = (x + WIDTH_M / 2.0) / WIDTH_M * w - 0.5
    v = (HEIGHT_M / 2.0 - y) / HEIGHT_M * h - 0.5
    return int(round(u)), int(round(v))


def px_to_world(u: float, v: float, w: int, h: int) -> tuple[float, float]:
    x = (u + 0.5) / w * WIDTH_M - WIDTH_M / 2.0
    y = HEIGHT_M / 2.0 - (v + 0.5) / h * HEIGHT_M
    return float(x), float(y)


def filter_white(hsv: np.ndarray, w: int) -> np.ndarray:
    white = cv2.inRange(hsv, (0, 0, 185), (179, 55, 255))
    white[:, int(w * 0.92) :] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    out = np.zeros_like(white)
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        fill = a / (bw * bh + 1.0)
        if a < 25:
            continue
        if a > 2500 and fill > 0.3 and bw < 100:
            continue
        out[labels == i] = 255
    return out


def build_ribbon(white: np.ndarray, close_ks: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
    closed = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    best_i, best_span = None, -1
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if x > 700 and bw < 300:
            continue
        span = bw * bh
        if span > best_span:
            best_span = span
            best_i = i
    if best_i is None:
        raise RuntimeError("no track ribbon component")
    return np.where(labels == best_i, 255, 0).astype(np.uint8)


def extract_centerline(ribbon: np.ndarray) -> np.ndarray:
    cnts, hier = cv2.findContours(ribbon, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        raise RuntimeError("no contours on ribbon")
    outer = max(
        (c for c, hi in zip(cnts, hier[0]) if hi[3] < 0),
        key=cv2.contourArea,
    )
    pts = outer.reshape(-1, 2).astype(np.float64)
    cxy = pts.mean(axis=0)
    dist = cv2.distanceTransform(ribbon, cv2.DIST_L2, 5)
    h, w = ribbon.shape
    mids = []
    n = len(pts)
    for i in range(n):
        tang = pts[(i + 1) % n] - pts[(i - 1) % n]
        tang = tang / (np.linalg.norm(tang) + 1e-9)
        nrm = np.array([tang[1], -tang[0]], dtype=np.float64)
        if np.dot(nrm, cxy - pts[i]) < 0:
            nrm = -nrm
        best = None
        best_d = -1.0
        for t in np.linspace(0.0, 45.0, 46):
            q = pts[i] + nrm * t
            uu, vv = int(round(q[0])), int(round(q[1]))
            if 0 <= uu < w and 0 <= vv < h and ribbon[vv, uu] > 0:
                d = float(dist[vv, uu])
                if d > best_d:
                    best_d = d
                    best = q
        if best is not None and best_d >= 2.0:
            mids.append(best)
    return np.asarray(mids, dtype=np.float64)


def order_ccw_from_start(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """Order closed loop CCW, starting near spawn heading west (−X, yaw≈−π)."""
    world = np.array([px_to_world(u, v, w, h) for u, v in pts], dtype=np.float64)
    sx, sy = SPAWNS["start"]
    i0 = int(np.argmin((world[:, 0] - sx) ** 2 + (world[:, 1] - sy) ** 2))
    ordered = np.concatenate([pts[i0:], pts[:i0]], axis=0)
    ow = np.array([px_to_world(u, v, w, h) for u, v in ordered], dtype=np.float64)
    look = min(40, max(5, len(ow) // 40))
    # On bottom straight, CCW = westbound (−X), matching spawn yaw −π.
    if ow[look, 0] > ow[0, 0]:
        ordered = ordered[::-1]
        ow = np.array([px_to_world(u, v, w, h) for u, v in ordered], dtype=np.float64)
        i0 = int(np.argmin((ow[:, 0] - sx) ** 2 + (ow[:, 1] - sy) ** 2))
        ordered = np.concatenate([ordered[i0:], ordered[:i0]], axis=0)
    return ordered


def smooth_circular(pts: np.ndarray, win: int = 15) -> np.ndarray:
    if len(pts) < win:
        return pts
    pad = win // 2
    ext = np.vstack([pts[-pad:], pts, pts[:pad]])
    k = np.ones(win) / win
    return np.column_stack(
        [np.convolve(ext[:, 0], k, mode="valid"), np.convolve(ext[:, 1], k, mode="valid")]
    )


def sample_polyline(pts: np.ndarray, spacing_m: float, w: int, h: int) -> np.ndarray:
    work = np.vstack([pts, pts[0]])
    world = np.array([px_to_world(u, v, w, h) for u, v in work], dtype=np.float64)
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
            out.append(work[-1])
            break
        a = (t - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out.append((1 - a) * work[j] + a * work[j + 1])
    arr = np.asarray(out, dtype=np.float64)
    if len(arr) > 2:
        w0 = np.array(px_to_world(*arr[0], w, h))
        w1 = np.array(px_to_world(*arr[-1], w, h))
        if np.linalg.norm(w0 - w1) < spacing_m * 0.5:
            arr = arr[:-1]
    return arr


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
        default=Path("/workspace/data/captures/out_best_route"),
    )
    ap.add_argument("--spacing-m", type=float, default=0.20)
    ap.add_argument("--close-ks", type=int, default=39)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"failed to read {args.image}")
    h, w = img.shape[:2]
    if (w, h) != (1211, 908):
        raise SystemExit(f"unexpected size {w}x{h}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = filter_white(hsv, w)
    ribbon = build_ribbon(white, args.close_ks)
    mids = extract_centerline(ribbon)
    ordered = order_ccw_from_start(mids, w, h)
    smooth = smooth_circular(ordered, win=15)
    sampled = sample_polyline(smooth, args.spacing_m, w, h)

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
    close = math.hypot(
        waypoints[0]["x_m"] - waypoints[-1]["x_m"],
        waypoints[0]["y_m"] - waypoints[-1]["y_m"],
    )
    path_len = cum + close

    world_pts = np.array([[wp["x_m"], wp["y_m"]] for wp in waypoints], dtype=np.float64)
    anchors = {}
    for name, (x, y) in SPAWNS.items():
        d = np.linalg.norm(world_pts - np.array([x, y]), axis=1)
        j = int(np.argmin(d))
        anchors[name] = {
            "spawn_xy": [x, y],
            "nearest_wp": j,
            "nearest_xy": [waypoints[j]["x_m"], waypoints[j]["y_m"]],
            "dist_m": round(float(d[j]), 3),
        }

    fork_note = {
        "kept_branch": "left/outer",
        "reason": "outer-contour midline follows northern branch of OUT parallel fork",
        "merge_anchor_dist_m": {
            "left": anchors["out_fork_merge_left"]["dist_m"],
            "right": anchors["out_fork_merge_right"]["dist_m"],
        },
    }

    meta = {
        "source_image": str(args.image),
        "image_size_px": [w, h],
        "gazebo_track_plane_m": {"width_m": WIDTH_M, "height_m": HEIGHT_M},
        "meters_per_pixel": {"x": WIDTH_M / w, "y": HEIGHT_M / h},
        "frame": "gazebo_world (track_plane centered at origin)",
        "mapping": {
            "formula": "x=(u+0.5)/W*width - width/2; y=height/2 - (v+0.5)/H*height",
            "u_right": "+x",
            "v_down": "-y",
        },
        "route": "OUT best (white lane center, CCW)",
        "out_fork": fork_note,
        "spacing_m": args.spacing_m,
        "close_ks": args.close_ks,
        "num_waypoints": len(waypoints),
        "path_length_m": round(path_len, 3),
        "spawn_anchors": anchors,
        "spawns_projected": {
            k: {
                "x_m": v[0],
                "y_m": v[1],
                "u_px": world_to_px(*v, w, h)[0],
                "v_px": world_to_px(*v, w, h)[1],
            }
            for k, v in SPAWNS.items()
        },
        "forks_handled": [
            {
                "id": "inout_fork",
                "policy": "stay on white OUT (ignore yellow IN branch)",
            },
            {
                "id": "out_fork",
                "policy": "take left/outer parallel white branch",
            },
        ],
    }

    (args.out_dir / "out_best_waypoints.json").write_text(
        json.dumps({"meta": meta, "waypoints": waypoints}, indent=2),
        encoding="utf-8",
    )
    with (args.out_dir / "out_best_waypoints.csv").open("w", encoding="utf-8") as f:
        f.write("i,s_m,x_m,y_m,u_px,v_px\n")
        for wp in waypoints:
            f.write(
                f"{wp['i']},{wp['s_m']},{wp['x_m']},{wp['y_m']},{wp['u_px']},{wp['v_px']}\n"
            )

    overlay = img.copy()
    tint = overlay.copy()
    tint[ribbon > 0] = (tint[ribbon > 0] * 0.55 + np.array([0, 90, 0])).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.55, tint, 0.45, 0)
    for i in range(len(sampled) - 1):
        cv2.line(
            overlay,
            (int(round(sampled[i, 0])), int(round(sampled[i, 1]))),
            (int(round(sampled[i + 1, 0])), int(round(sampled[i + 1, 1]))),
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.line(
        overlay,
        (int(round(sampled[-1, 0])), int(round(sampled[-1, 1]))),
        (int(round(sampled[0, 0])), int(round(sampled[0, 1]))),
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    step = max(1, len(waypoints) // 40)
    for wp in waypoints[::step]:
        p = (int(round(wp["u_px"])), int(round(wp["v_px"])))
        cv2.circle(overlay, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
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
            0.32,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"OUT best CCW | n={len(waypoints)} | L={path_len:.2f}m | "
        f"fork={fork_note['kept_branch']} | ds={args.spacing_m}m",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "out_best_overlay.png"), overlay)
    cv2.imwrite(str(args.out_dir / "white_mask.png"), white)
    cv2.imwrite(str(args.out_dir / "center_ribbon.png"), ribbon)

    # Gazebo meter plot
    plot_w, plot_h = 1200, 920
    plot = np.full((plot_h, plot_w, 3), 28, dtype=np.uint8)
    margin = 55

    def w2plot(x, y):
        px = margin + (x + WIDTH_M / 2) / WIDTH_M * (plot_w - 2 * margin)
        py = margin + (HEIGHT_M / 2 - y) / HEIGHT_M * (plot_h - 2 * margin)
        return int(round(px)), int(round(py))

    for gx in range(-6, 7):
        cv2.line(plot, w2plot(gx, -HEIGHT_M / 2), w2plot(gx, HEIGHT_M / 2), (55, 55, 55), 1)
        cv2.putText(
            plot,
            f"{gx}",
            (w2plot(gx, -HEIGHT_M / 2)[0] - 8, plot_h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (170, 170, 170),
            1,
        )
    for gy in range(-4, 5):
        cv2.line(plot, w2plot(-WIDTH_M / 2, gy), w2plot(WIDTH_M / 2, gy), (55, 55, 55), 1)
        cv2.putText(
            plot,
            f"{gy}",
            (10, w2plot(0, gy)[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (170, 170, 170),
            1,
        )
    corners = [
        (-WIDTH_M / 2, -HEIGHT_M / 2),
        (WIDTH_M / 2, -HEIGHT_M / 2),
        (WIDTH_M / 2, HEIGHT_M / 2),
        (-WIDTH_M / 2, HEIGHT_M / 2),
    ]
    for i in range(4):
        cv2.line(plot, w2plot(*corners[i]), w2plot(*corners[(i + 1) % 4]), (110, 110, 110), 2)
    for i in range(len(waypoints) - 1):
        cv2.line(
            plot,
            w2plot(waypoints[i]["x_m"], waypoints[i]["y_m"]),
            w2plot(waypoints[i + 1]["x_m"], waypoints[i + 1]["y_m"]),
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.line(
        plot,
        w2plot(waypoints[-1]["x_m"], waypoints[-1]["y_m"]),
        w2plot(waypoints[0]["x_m"], waypoints[0]["y_m"]),
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(plot, w2plot(waypoints[0]["x_m"], waypoints[0]["y_m"]), 6, (0, 0, 255), -1)
    for name, (x, y) in SPAWNS.items():
        p = w2plot(x, y)
        cv2.drawMarker(plot, p, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(
            plot, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1
        )
    cv2.putText(
        plot,
        f"Gazebo world [m]  track_plane {WIDTH_M}x{HEIGHT_M}  "
        f"OUT best CCW  L={path_len:.2f}m  n={len(waypoints)}",
        (margin, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "out_best_gazebo_m.png"), plot)

    print(json.dumps(meta, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
