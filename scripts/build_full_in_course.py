#!/usr/bin/env python3
"""Stitch OUT start → IN roundabout → OUT remainder into one full IN course.

Uses trusted OUT best-route outside the yellow detour, and the IN roundabout
extract (already exit-stitched onto OUT wp124) for the fork→merge segment.
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


def load_wps(path: Path) -> tuple[dict, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    xy = np.array([[w["x_m"], w["y_m"]] for w in data["waypoints"]], dtype=np.float64)
    return data, xy


def nearest_i(xy: np.ndarray, target: np.ndarray, lo: int = 0, hi: int | None = None) -> int:
    hi = len(xy) if hi is None else hi
    seg = xy[lo:hi]
    return lo + int(np.argmin(np.linalg.norm(seg - target, axis=1)))


def resample_xy(xy: np.ndarray, spacing_m: float) -> np.ndarray:
    if len(xy) < 2:
        return xy
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-6:
        return xy[:1]
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-best",
        type=Path,
        default=Path("/workspace/data/captures/out_best_route/out_best_waypoints.json"),
    )
    ap.add_argument(
        "--in-route",
        type=Path,
        default=Path("/workspace/data/captures/in_roundabout_route/in_roundabout_waypoints.json"),
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
    ap.add_argument("--fork-hint-i", type=int, default=20)
    ap.add_argument("--merge-hint-i", type=int, default=124)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_data, out_xy = load_wps(args.out_best)
    in_data, in_xy = load_wps(args.in_route)

    # OUT → IN handoff near OUT wp20 / IN start
    i_fork = nearest_i(
        out_xy,
        in_xy[0],
        lo=max(0, args.fork_hint_i - 8),
        hi=min(len(out_xy), args.fork_hint_i + 12),
    )
    # Drop IN points still on OUT approach (keep from first meaningful diverge)
    j_in0 = 0
    for j in range(min(12, len(in_xy))):
        if np.linalg.norm(in_xy[j] - out_xy[i_fork]) < 0.12:
            j_in0 = j
    # IN end should already be OUT wp124; snap merge index on OUT
    i_merge = nearest_i(
        out_xy,
        in_xy[-1],
        lo=max(0, args.merge_hint_i - 10),
        hi=min(len(out_xy), args.merge_hint_i + 8),
    )

    joined = np.vstack(
        [
            out_xy[: i_fork + 1],
            in_xy[j_in0 + 1 :],  # skip near-duplicate at fork
            out_xy[i_merge + 1 :],  # continue outer after merge
        ]
    )
    # scrub tiny jumps
    keep = [0]
    for i in range(1, len(joined)):
        if np.linalg.norm(joined[i] - joined[keep[-1]]) >= 0.02:
            keep.append(i)
    joined = joined[keep]
    sampled = resample_xy(joined, args.spacing_m)

    waypoints = []
    cum = 0.0
    prev = None
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"failed to read {args.image}")
    h, w = img.shape[:2]
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

    meta = {
        "route": "FULL IN course (OUT start → yellow roundabout CW → OUT finish)",
        "composition": "OUT[0:fork] + IN[roundabout] + OUT[merge:end]",
        "spacing_m": args.spacing_m,
        "num_waypoints": len(waypoints),
        "path_length_m": round(cum, 3),
        "splice": {
            "out_fork_i": i_fork,
            "out_merge_i": i_merge,
            "in_start_i": j_in0,
            "in_end_xy": [float(in_xy[-1, 0]), float(in_xy[-1, 1])],
            "out_fork_xy": [float(out_xy[i_fork, 0]), float(out_xy[i_fork, 1])],
            "out_merge_xy": [float(out_xy[i_merge, 0]), float(out_xy[i_merge, 1])],
            "fork_gap_m": round(float(np.linalg.norm(in_xy[j_in0] - out_xy[i_fork])), 3),
            "merge_gap_m": round(float(np.linalg.norm(in_xy[-1] - out_xy[i_merge])), 3),
        },
        "sources": {
            "out_best": str(args.out_best),
            "in_roundabout": str(args.in_route),
        },
        "gazebo_track_plane_m": {"width_m": WIDTH_M, "height_m": HEIGHT_M},
    }

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

    # Overlay: OUT faint + full course emphasized
    overlay = img.copy()
    for i in range(len(out_xy) - 1):
        p0 = world_to_px_f(*out_xy[i], w, h)
        p1 = world_to_px_f(*out_xy[i + 1], w, h)
        cv2.line(
            overlay,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
    for i in range(len(sampled) - 1):
        p0 = world_to_px_f(*sampled[i], w, h)
        p1 = world_to_px_f(*sampled[i + 1], w, h)
        cv2.line(
            overlay,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
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
    # mark splice
    for label, xy, color in [
        ("fork", out_xy[i_fork], (255, 0, 255)),
        ("merge", out_xy[i_merge], (0, 165, 255)),
    ]:
        u, v = world_to_px_f(float(xy[0]), float(xy[1]), w, h)
        cv2.drawMarker(overlay, (int(u), int(v)), color, cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(
            overlay,
            label,
            (int(u) + 6, int(v) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        overlay,
        f"FULL IN | n={len(waypoints)} | L={cum:.2f}m | "
        f"OUT0..{i_fork}+IN+OUT{i_merge}..end | ds={args.spacing_m}m",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.out_dir / "full_in_overlay.png"), overlay)

    # Gazebo plot
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
        f"Gazebo [m] FULL IN  L={cum:.2f}m  n={len(waypoints)}",
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
                "# FULL IN 코스 웨이포인트 (OUT + 회전교차로)",
                "",
                f"- **구성:** OUT[0..{i_fork}] + IN(노란 CW 원형) + OUT[{i_merge}..end]",
                f"- **n={len(waypoints)}**, **L≈{cum:.2f} m**, ds={args.spacing_m} m",
                "- 상단 합류는 OUT best 점선 중앙선(wp124 부근)으로 종료 스티치",
                "",
                "## Files",
                "- `full_in_waypoints.json` / `.csv`",
                "- `full_in_overlay.png`",
                "- `full_in_gazebo_m.png`",
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
