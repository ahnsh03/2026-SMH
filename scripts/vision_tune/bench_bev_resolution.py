#!/usr/bin/env python3
"""Score BEV resolutions (meters_per_pixel) against a recorded bag.

The board is CPU-bound: lane_detection.detect costs ~255 ms at the current
meters_per_pixel=0.004, so perception runs at ~2.2 Hz against a 30 Hz camera and
control steers on ~450 ms stale lane data. Coarsening the BEV grid is the
cheapest lever, but it trades detection quality for speed — this script measures
BOTH on real recorded frames so the trade is a number, not a guess.

Usage:
    python3 scripts/vision_tune/bench_bev_resolution.py <bag_dir> [--frames 40]
    python3 scripts/vision_tune/bench_bev_resolution.py <bag_dir> --mpp 0.004,0.006

Reads /camera/image/compressed straight from the bag's .db3 (no ROS graph, so it
runs headless on the board). Each meters_per_pixel value runs in a fresh
subprocess because lane_detection reads the YAML at import time.

Pick the coarsest value whose detect-rate still matches 0.004 on YOUR track.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import shutil
import sqlite3
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CFG = os.path.join(REPO, "config", "lane_vision.yaml")
TOPIC = "/camera/image/compressed"

CHILD = r'''
import pickle, sys, time
sys.path.insert(0, "{build}")
import numpy as np, cv2
from inference.modules import lane_detection as L

frames = pickle.load(open("{frames}", "rb"))
L.detect(frames[0])  # warm-up

times, n_left, n_right, n_both = [], 0, 0, 0
for f in frames:
    t0 = time.perf_counter()
    det = L.detect(f)
    times.append((time.perf_counter() - t0) * 1000.0)
    has_l = getattr(det, "left", None) is not None and len(det.left.points) > 0
    has_r = getattr(det, "right", None) is not None and len(det.right.points) > 0
    n_left += bool(has_l)
    n_right += bool(has_r)
    n_both += bool(has_l and has_r)

print("RESULT %d %d %.1f %.1f %d %d %d %d" % (
    L.BEV_WIDTH, L.BEV_HEIGHT,
    sum(times) / len(times), min(times),
    n_left, n_right, n_both, len(frames),
))
'''


def load_frames(bag_dir: str, limit: int):
    """Decode CompressedImage payloads from a rosbag2 sqlite3 file."""
    dbs = glob.glob(os.path.join(bag_dir, "*.db3"))
    if not dbs:
        sys.exit(f"no .db3 found in {bag_dir}")
    import numpy as np
    import cv2

    con = sqlite3.connect(dbs[0])
    row = con.execute("SELECT id FROM topics WHERE name=?", (TOPIC,)).fetchone()
    if row is None:
        sys.exit(f"{TOPIC} not in bag {bag_dir}")
    blobs = con.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT ?",
        (row[0], limit),
    ).fetchall()
    con.close()

    frames = []
    for (blob,) in blobs:
        # CDR payload: find the JPEG SOI and decode from there. Robust enough for
        # CompressedImage, whose only variable-length tail is the jpeg buffer.
        start = bytes(blob).find(b"\xff\xd8\xff")
        if start < 0:
            continue
        img = cv2.imdecode(np.frombuffer(blob[start:], np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            frames.append(img)
    if not frames:
        sys.exit("no decodable frames in bag")
    return frames


def set_mpp(backup: str, mpp: float) -> None:
    """Rewrite only metric_ipm.meters_per_pixel; the rest is derived in code."""
    out, in_metric = [], False
    for ln in open(backup, encoding="utf-8").read().splitlines():
        if not ln.startswith((" ", "\t")) and ln.strip():
            in_metric = ln.startswith("metric_ipm:")
        if in_metric and ln.strip().startswith("meters_per_pixel:"):
            out.append(f"  meters_per_pixel: {mpp}")
        else:
            out.append(ln)
    open(CFG, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--mpp", default="0.004,0.005,0.006,0.008")
    args = ap.parse_args()

    frames = load_frames(args.bag, args.frames)
    print(f"bag: {args.bag}\nframes: {len(frames)}  shape: {frames[0].shape}\n")

    tmp = tempfile.mkdtemp(prefix="bevbench_")
    frames_pkl = os.path.join(tmp, "frames.pkl")
    pickle.dump(frames, open(frames_pkl, "wb"))
    backup = os.path.join(tmp, "lane_vision.yaml.orig")
    shutil.copy(CFG, backup)

    build = os.path.join(REPO, "build", "inference")
    hdr = f"{'mpp':>7s} {'BEV':>10s} {'px':>8s} {'mean':>9s} {'min':>9s}   {'left':>6s} {'right':>6s} {'both':>6s}"
    print(hdr)
    print("-" * len(hdr))
    try:
        for raw in args.mpp.split(","):
            mpp = float(raw)
            set_mpp(backup, mpp)
            r = subprocess.run(
                [sys.executable, "-c", CHILD.format(build=build, frames=frames_pkl)],
                capture_output=True,
                text=True,
            )
            hit = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
            if not hit:
                err = (r.stderr.strip().splitlines() or ["?"])[-1]
                print(f"{mpp:7.3f}   FAILED: {err[:58]}")
                continue
            _, w, h, mean, mn, nl, nr, nb, tot = hit[0].split()
            w, h, tot = int(w), int(h), int(tot)
            pct = lambda n: f"{100.0 * int(n) / tot:5.0f}%"
            print(
                f"{mpp:7.3f} {w}x{h:<6d} {w * h:8d} {float(mean):8.1f}ms {float(mn):8.1f}ms"
                f"   {pct(nl)} {pct(nr)} {pct(nb)}"
            )
    finally:
        shutil.copy(backup, CFG)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\nconfig restored -> {CFG}")

    print(
        "\nleft/right/both = share of frames where that boundary was found.\n"
        "Take the coarsest mpp whose 'both' still matches 0.004, then set\n"
        "metric_ipm.meters_per_pixel in config/lane_vision.yaml to it."
    )


if __name__ == "__main__":
    main()
