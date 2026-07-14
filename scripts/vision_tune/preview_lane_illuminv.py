#!/usr/bin/env python3
"""Eyeball the illumination-invariant lane / drivable-area pipeline on one bag.

The board is headless (no DISPLAY), so this renders to files instead of a live
window: an MP4 you can scrub, plus a contact-sheet PNG of sampled frames.

Why this pipeline instead of HSV inRange: fixed HSV thresholds break here because
scene brightness drifts frame to frame (V-mean 82-132 across bag_20260711_144948)
and in dim frames the grey road and blue mat collapse to the same brightness.
This detector uses only quantities that survive an illumination scale:
  lane    = local brightness ridge (top-hat on L), not absolute brightness
  offtrack= blue *direction* (opponent B-(R+G)/2 over intensity), not saturation
            -- so a red road stays drivable while a blue mat does not
  drivable= flood-fill from the ego footprint, blocked by lane|offtrack, not by
            per-pixel colour -- shadowed road stays included if lanes enclose it

It reads the .db3 directly (no `ros2 bag play` / no running ROS graph needed).

Examples (from repo root, after `source /opt/ros/humble/setup.bash`):

  python3 scripts/vision_tune/preview_lane_illuminv.py \
      /home/topst/D-Racer-Kit/bagfile/bag_20260711_144948

  # faster pass: every 3rd frame, contact sheet only
  python3 scripts/vision_tune/preview_lane_illuminv.py <bag> --stride 3 --no-video

Output (default ./lane_preview_out/):
  overlay.mp4        per-frame BEV with lane (red) + drivable (green) + offtrack (blue)
  contact_sheet.png  a grid of sampled frames, raw | BEV overlay
  panels.mp4         optional wide diagnostic strip (--panels): raw|BEV|lane|masks|drivable|overlay
"""

from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

# metric_ipm lives next to this file; import by path so we don't depend on the
# `inference` package being on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metric_ipm  # noqa: E402


# ---- colours (BGR) for the overlay ----------------------------------------
C_LANE = (0, 0, 255)      # red    : detected lane line
C_DRIVE = (0, 170, 0)     # green  : drivable area
C_OFF = (200, 60, 0)      # blue   : off-track (blue mat)


def load_compressed_frames(bag_dir: str, topic: str):
    """Yield BGR frames from a rosbag2 sqlite3 bag, in timestamp order.

    Decodes sensor_msgs/CompressedImage without spinning a ROS node.
    """
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage

    db3s = sorted(glob.glob(os.path.join(bag_dir, "*.db3")))
    if not db3s:
        raise FileNotFoundError(f"no .db3 under {bag_dir}")

    for db3 in db3s:
        con = sqlite3.connect(db3)
        try:
            row = con.execute(
                "SELECT id FROM topics WHERE name=?", (topic,)
            ).fetchone()
            if row is None:
                names = [r[0] for r in con.execute("SELECT name FROM topics")]
                raise KeyError(
                    f"topic {topic!r} not in {os.path.basename(db3)}; "
                    f"available: {names}"
                )
            tid = row[0]
            cur = con.execute(
                "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp",
                (tid,),
            )
            for (blob,) in cur:
                msg = deserialize_message(bytes(blob), CompressedImage)
                img = cv2.imdecode(
                    np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR
                )
                if img is not None:
                    yield img
        finally:
            con.close()


class LaneIlluminInv:
    """Illumination-invariant lane + drivable-area masks in the BEV grid."""

    def __init__(self, img_w=320, img_h=180, meters_per_pixel=0.006,
                 line_thr_frac=0.45, line_thr_floor=12.0,
                 blue_thr=0.06, red_thr=0.12):
        p = metric_ipm.MetricIpmParams(
            meters_per_pixel=meters_per_pixel
        ).clamp()
        self.crop = metric_ipm.resolve_crop_top_px(img_w, img_h, p)
        self.map_x, self.map_y, valid = metric_ipm.build_ipm_maps(img_w, img_h, p)
        self.valid = valid.astype(bool)
        self.bev_h, self.bev_w = self.map_x.shape
        # Top-hat kernel ~3x the painted-line width. In the BEV a line has a
        # constant physical width, so one kernel size holds for near and far --
        # the thing that is impossible in the raw image (near line 15px, far 2px).
        line_w_m = 0.02
        k = max(3, (int(round(line_w_m / p.meters_per_pixel)) * 3) | 1)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        self.line_thr_frac = line_thr_frac
        self.line_thr_floor = line_thr_floor
        self.blue_thr = blue_thr
        self.red_thr = red_thr
        self.kernel_px = k

    def warp(self, img):
        return cv2.remap(
            img[self.crop:], self.map_x, self.map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )

    def detect(self, img):
        bev = self.warp(img)
        bevf = bev.astype(np.float32)
        b, g, r = bevf[:, :, 0], bevf[:, :, 1], bevf[:, :, 2]
        s = b + g + r + 1e-6
        blueness = (b - (r + g) / 2) / s
        redness = (r - (g + b) / 2) / s
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel)
        # Per-frame adaptive: a line is a strong local ridge relative to this
        # frame's own top-hat distribution -- no absolute brightness anywhere.
        peak = float(np.percentile(tophat[self.valid], 99.0))
        thr = max(self.line_thr_floor, peak * self.line_thr_frac)
        line = ((tophat > thr) & self.valid).astype(np.uint8)

        off = ((blueness > self.blue_thr) & self.valid).astype(np.uint8)
        red = ((redness > self.red_thr) & self.valid).astype(np.uint8)

        block = cv2.dilate(cv2.bitwise_or(line, off), np.ones((3, 3), np.uint8))
        free = ((block == 0) & self.valid).astype(np.uint8)
        n, labels = cv2.connectedComponents(free)
        # Ego footprint: bottom-centre of the BEV is under the car => on road.
        seed = labels[self.bev_h - 6:self.bev_h - 1,
                      self.bev_w // 2 - 4:self.bev_w // 2 + 4]
        seed = seed[seed > 0]
        if seed.size:
            drivable = np.isin(labels, np.unique(seed)).astype(np.uint8)
        else:
            drivable = np.zeros_like(free)
        return {
            "bev": bev, "line": line, "off": off, "red": red,
            "drivable": drivable,
        }

    def overlay(self, det):
        ov = det["bev"].copy()
        dr, off, line = det["drivable"], det["off"], det["line"]
        ov[dr > 0] = (0.45 * ov[dr > 0]
                      + 0.55 * np.array(C_DRIVE)).astype(np.uint8)
        ov[off > 0] = (0.6 * ov[off > 0]
                       + 0.4 * np.array(C_OFF)).astype(np.uint8)
        ov[line > 0] = C_LANE
        cov = 100.0 * dr.sum() / max(1, self.valid.sum())
        cv2.putText(ov, f"drivable {cov:4.0f}%", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        return ov, cov


def _mask_bgr(mask, w, h):
    return cv2.resize(cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), (w, h),
                      interpolation=cv2.INTER_NEAREST)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Preview illumination-invariant lane detection on one bag.")
    ap.add_argument("bag", help="rosbag2 directory (contains *.db3)")
    ap.add_argument("--topic", default="/camera/image/compressed")
    ap.add_argument("--out", default="lane_preview_out",
                    help="output directory (default ./lane_preview_out)")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (default 1)")
    ap.add_argument("--fps", type=float, default=15.0, help="output video fps")
    ap.add_argument("--mpp", type=float, default=0.006,
                    help="BEV meters/pixel (0.004 sharp/slow .. 0.008 coarse)")
    ap.add_argument("--blue-thr", type=float, default=0.06)
    ap.add_argument("--red-thr", type=float, default=0.12)
    ap.add_argument("--line-frac", type=float, default=0.45)
    ap.add_argument("--sheet-cols", type=int, default=4)
    ap.add_argument("--sheet-rows", type=int, default=4)
    ap.add_argument("--no-video", action="store_true",
                    help="skip overlay.mp4, contact sheet only")
    ap.add_argument("--panels", action="store_true",
                    help="also write panels.mp4 (wide diagnostic strip)")
    ap.add_argument("--show", action="store_true",
                    help="live cv2.imshow playback (needs a display; use ssh -X "
                         "on this headless board). keys: q/ESC quit, "
                         "SPACE pause, .=step, s=save current overlay png")
    ap.add_argument("--scale", type=int, default=3,
                    help="--show window magnification (default 3x)")
    args = ap.parse_args(argv)

    bag = args.bag.rstrip("/")
    if not os.path.isdir(bag):
        ap.error(f"not a directory: {bag}")
    os.makedirs(args.out, exist_ok=True)

    win = None
    if args.show:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            ap.error(
                "--show needs a display but DISPLAY/WAYLAND_DISPLAY are unset. "
                "This board is headless: reconnect with `ssh -X` (X server on "
                "your laptop), or drop --show and watch overlay.mp4 instead.")
        win = "lane preview (illumination-invariant)"
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            ap.error(f"cannot open a GUI window: {str(exc).splitlines()[0]}\n"
                     "Drop --show and watch overlay.mp4 instead.")

    det = LaneIlluminInv(meters_per_pixel=args.mpp, blue_thr=args.blue_thr,
                         red_thr=args.red_thr, line_thr_frac=args.line_frac)
    bw, bh = det.bev_w, det.bev_h
    print(f"BEV {bw}x{bh} @ {args.mpp} m/px  tophat kernel={det.kernel_px}px  "
          f"stride={args.stride}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid = None if args.no_video else cv2.VideoWriter(
        os.path.join(args.out, "overlay.mp4"), fourcc, args.fps, (bw, bh))
    panel_w = bw * 6
    pvid = cv2.VideoWriter(
        os.path.join(args.out, "panels.mp4"), fourcc, args.fps,
        (panel_w, bh)) if args.panels else None

    thumbs = []                       # (raw_small, overlay) for the contact sheet
    n_target = args.sheet_cols * args.sheet_rows
    covs = []
    n_used = 0

    frames = load_compressed_frames(bag, args.topic)
    delay = max(1, int(1000 / args.fps))   # --show playback pacing
    paused = False
    quit_early = False
    for i, img in enumerate(frames):
        if i % args.stride:
            continue
        d = det.detect(img)
        ov, cov = det.overlay(d)
        covs.append(cov)
        n_used += 1

        if win is not None:
            raw = cv2.resize(img, (bw, bh))
            view = np.hstack([raw, ov])          # raw | overlay, side by side
            if args.scale != 1:
                view = cv2.resize(
                    view, None, fx=args.scale, fy=args.scale,
                    interpolation=cv2.INTER_NEAREST)
            cv2.imshow(win, view)
            while True:
                k = cv2.waitKey(0 if paused else delay) & 0xFF
                if k in (ord("q"), 27):          # q / ESC
                    quit_early = True
                    break
                if k == ord(" "):                # toggle pause
                    paused = not paused
                    if paused:
                        continue                 # stay on this frame
                    break
                if k == ord("."):                # step one frame (stay paused)
                    paused = True
                    break
                if k == ord("s"):                # save current overlay
                    p = os.path.join(args.out, f"show_frame_{n_used:05d}.png")
                    cv2.imwrite(p, ov)
                    print(f"saved {p}")
                    continue
                if not paused:
                    break                        # normal auto-advance
            if quit_early:
                break
        if vid is not None:
            vid.write(ov)
        if pvid is not None:
            raw = cv2.resize(img, (bw, bh))
            strip = np.hstack([
                raw, d["bev"],
                _mask_bgr(d["line"], bw, bh),
                _mask_bgr(cv2.bitwise_or(d["off"], d["red"] * 2), bw, bh),
                _mask_bgr(d["drivable"], bw, bh),
                ov,
            ])
            pvid.write(strip)
        thumbs.append((cv2.resize(img, (bw, bh)), ov))

    if vid is not None:
        vid.release()
    if pvid is not None:
        pvid.release()
    if win is not None:
        cv2.destroyAllWindows()

    if n_used == 0:
        print("!! no frames processed", file=sys.stderr)
        return 1

    # Contact sheet: evenly sample n_target processed frames.
    pick = [int(round(j * (len(thumbs) - 1) / max(1, n_target - 1)))
            for j in range(min(n_target, len(thumbs)))]
    cells = []
    for idx in pick:
        raw, ov = thumbs[idx]
        cells.append(np.vstack([raw, ov]))
    # pad to a full grid
    while len(cells) < n_target:
        cells.append(np.zeros_like(cells[0]))
    rows = []
    for r in range(args.sheet_rows):
        rows.append(np.hstack(cells[r * args.sheet_cols:(r + 1) * args.sheet_cols]))
    sheet = np.vstack(rows)
    sheet_path = os.path.join(args.out, "contact_sheet.png")
    cv2.imwrite(sheet_path, sheet)

    covs = np.array(covs)
    print(f"processed {n_used} frames")
    print(f"drivable coverage: mean {covs.mean():.1f}%  "
          f"min {covs.min():.1f}%  max {covs.max():.1f}%")
    print(f"wrote {sheet_path}")
    if vid is not None:
        print(f"wrote {os.path.join(args.out, 'overlay.mp4')}")
    if pvid is not None:
        print(f"wrote {os.path.join(args.out, 'panels.mp4')}")
    print("contact sheet layout: each cell = raw (top) / BEV overlay (bottom); "
          "red=lane  green=drivable  blue=off-track")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
