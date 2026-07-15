#!/usr/bin/env python3
"""Per-frame BEV / mask panels (binary + BEV).

Also writes:
  6_ego_blob.png — largest white CC touching BEV bottom (robot); other blobs dropped
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT = Path(__file__).resolve().parent
_ROOT = _SCRIPT.parents[1]
_INFERENCE = _ROOT / 'src' / 'inference'
for p in (_SCRIPT, _INFERENCE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from hsv import default_config_path, load_hsv_ranges, make_mask  # noqa: E402
from metric_ipm import load_metric_ipm, warp_metric_ipm  # noqa: E402

FROM_BAG = {
    'in': _ROOT / 'data' / 'captures' / 'from_bag' / 'in',
    'out': _ROOT / 'data' / 'captures' / 'from_bag' / 'out',
    'out_glare': _ROOT / 'data' / 'captures' / 'from_bag' / 'out_glare',
}
OUT_DIR = _ROOT / 'data' / 'captures' / 'raw_hsv_masks'
PANEL_H = 220
PANEL_W = 260
GAP = 4

def _odd(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 else k + 1


def _bin(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, np.uint8(255), np.uint8(0))


def _or2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _bin(cv2.bitwise_or(_bin(a), _bin(b)))


def _morph_open(mask: np.ndarray, k: int = 5, iterations: int = 1) -> np.ndarray:
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(k), _odd(k)))
    return cv2.morphologyEx(
        _bin(mask), cv2.MORPH_OPEN, ker, iterations=max(1, int(iterations))
    )


def _morph_close(mask: np.ndarray, k: int = 15, iterations: int = 2) -> np.ndarray:
    """Close with ellipse + tall rect (fills elongated road holes in BEV)."""
    binary = _bin(mask)
    k = _odd(k)
    iters = max(1, int(iterations))
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ellipse, iterations=iters)
    # Vertical bias: holes often stretch along the road (row direction in BEV).
    tall_h = _odd(max(k + 4, int(round(k * 1.4))))
    tall_w = _odd(max(3, k // 2))
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (tall_w, tall_h))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, rect, iterations=1)
    return out


def _fill_enclosed_holes(mask: np.ndarray, max_hole_px: int = 5000) -> np.ndarray:
    """Fill enclosed black holes (area ≤ max_hole_px; use huge limit for all)."""
    binary = _bin(mask)
    if binary.size == 0:
        return binary
    h, w = binary.shape
    flood = binary.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_and(cv2.bitwise_not(flood), cv2.bitwise_not(binary))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (holes > 0).astype(np.uint8), connectivity=8
    )
    out = binary.copy()
    limit = int(max_hole_px)
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if 0 < area <= limit:
            out[labels == lab] = 255
    return out


def _drop_top_edge_only_blobs(
    mask: np.ndarray,
    *,
    min_area: int = 350,
    top_band_ratio: float = 0.025,
    near_band_ratio: float = 0.18,
) -> np.ndarray:
    """Drop BEV-top-only large CCs before morph (trial #2). See morph_blob."""

    binary = _bin(mask)
    if binary.size == 0 or not np.any(binary):
        return binary
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary

    top_h = max(2, int(round(h * float(top_band_ratio))))
    near_h = max(2, int(round(h * float(near_band_ratio))))
    top_labs = {int(lab) for lab in np.unique(labels[:top_h, :]) if int(lab) > 0}
    bot_labs = {
        int(lab) for lab in np.unique(labels[h - near_h :, :]) if int(lab) > 0
    }

    out = binary.copy()
    for lab in top_labs:
        if lab in bot_labs:
            continue
        if int(stats[lab, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == lab] = 0
    return out


def _keep_near_floor_blob(
    mask: np.ndarray,
    *,
    near_band_ratio: float = 0.35,
    min_near_area: int = 80,
    centroid_lower_frac: float = 0.55,
) -> np.ndarray:
    """Near-robot CC before morph (black / cyan). Score = near-band pixels."""

    binary = _bin(mask)
    if binary.size == 0 or not np.any(binary):
        return binary
    h, w = binary.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)

    near_h = max(2, int(round(h * float(near_band_ratio))))
    near_slice = labels[h - near_h :, :]
    near_labs = {int(lab) for lab in np.unique(near_slice) if int(lab) > 0}

    near_ok: list[tuple[int, int]] = []
    for lab in near_labs:
        near_area = int(np.count_nonzero(near_slice == lab))
        if near_area >= int(min_near_area):
            near_ok.append((lab, near_area))

    if near_ok:
        best = max(near_ok, key=lambda t: t[1])[0]
    else:
        v_cut = float(h) * float(centroid_lower_frac)
        lower: list[tuple[int, int]] = []
        for lab in range(1, n):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            cy = float(cents[lab][1])
            if cy >= v_cut and area >= int(min_near_area):
                lower_area = int(np.count_nonzero(labels[h // 2 :, :] == lab))
                lower.append((lab, lower_area if lower_area > 0 else area))
        if lower:
            best = max(lower, key=lambda t: t[1])[0]
        else:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    out = np.zeros_like(binary)
    if best > 0:
        out[labels == best] = 255
    return out


# Alias used by older call sites / cyan comments.
_keep_near_cyan_blob = _keep_near_floor_blob


def _keep_bottom_ego_blob(
    mask: np.ndarray,
    *,
    near_band_ratio: float = 0.18,
) -> np.ndarray:
    """After morph: keep CC with max pixels in the BEV bottom band.

    Same scoring as ``_keep_near_floor_blob`` (band mass, not total area).
    Trial #2 uses this after morph; trial #1 also uses it for final ego.
    """
    binary = _bin(mask)
    if binary.size == 0 or not np.any(binary):
        return binary
    h, w = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary)

    near_h = max(2, int(round(h * float(near_band_ratio))))
    near_slice = labels[h - near_h :, :]
    near_labs = {int(lab) for lab in np.unique(near_slice) if int(lab) > 0}
    if not near_labs:
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(binary)
        out[labels == best] = 255
        return out

    best = 0
    best_near = -1
    for lab in near_labs:
        near_area = int(np.count_nonzero(near_slice == lab))
        if near_area > best_near:
            best_near = near_area
            best = lab
    out = np.zeros_like(binary)
    if best > 0:
        out[labels == best] = 255
    return out


def _clean_lane_mask(
    mask: np.ndarray,
    *,
    open_k: int,
    close_k: int,
    open_iters: int = 1,
    close_iters: int = 2,
    max_hole_px: int = 5000,
) -> np.ndarray:
    """open (noise) → close (gaps) → fill enclosed holes."""
    opened = _morph_open(mask, open_k, iterations=open_iters)
    closed = _morph_close(opened, close_k, iterations=close_iters)
    return _fill_enclosed_holes(closed, max_hole_px=max_hole_px)


def _fit_bgr(img: np.ndarray, w: int = PANEL_W, h: int = PANEL_H) -> np.ndarray:
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None or img.size == 0:
        return out
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    # BEV color: area; binary: nearest
    interp = cv2.INTER_AREA if img.ndim == 3 else cv2.INTER_NEAREST
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = r
    return out


def _fit_bin(mask: np.ndarray, w: int = PANEL_W, h: int = PANEL_H) -> np.ndarray:
    gray = np.zeros((h, w), dtype=np.uint8)
    m = _bin(mask)
    if m.size == 0:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ih, iw = m.shape[:2]
    s = min(w / max(iw, 1), h / max(ih, 1))
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    r = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    gray[y0 : y0 + nh, x0 : x0 + nw] = r
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _title(bgr: np.ndarray, text: str) -> None:
    cv2.putText(
        bgr, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 0), 1, cv2.LINE_AA
    )


def _cov(mask: np.ndarray) -> float:
    m = _bin(mask)
    if m.size == 0:
        return 0.0
    return 100.0 * float(np.count_nonzero(m)) / float(m.size)


def _panel_bin(mask: np.ndarray, title: str) -> np.ndarray:
    p = _fit_bin(mask)
    _title(p, f'{title} {_cov(mask):.1f}%')
    return p


def extract_five_from_bev(
    bev: np.ndarray,
    config_path: Path,
    *,
    open_k: int = 5,
    close_k: int = 17,
    open_iters: int = 1,
    close_iters: int = 2,
    max_hole_px: int = 5000,
    course: str | None = None,
    prefer_yellow: bool | None = None,
    black_mode: str = 'near',
) -> dict[str, np.ndarray]:
    """Photo/bag SSOT panels from an **already-warped** Metric IPM BEV.

    Paint|road for morph/ego (never OR white∧yellow):
      IN  (prefer_yellow): yellow present → yellow|road ; else white|road
      OUT (!prefer_yellow): always white|road

    ``black_mode``:
      - ``near`` (default, trial #1): near-band-mass CC before morph
      - ``top_drop`` (trial #2): drop BEV-top-only large CCs, then morph → bottom

    Morph defaults: open 5 / close 17 / 2 iters (restored one step vs soft 3/13/1).

    Do **not** pass camera frames here (would double-warp). Use ``extract_five``.
    """

    ranges = load_hsv_ranges(config_path)
    white = _bin(make_mask(bev, ranges['white']))
    yellow = _bin(make_mask(bev, ranges['yellow']))
    black_raw = _bin(make_mask(bev, ranges['black_road'], morph=False))
    mode = (black_mode or 'near').strip().lower()
    if mode in ('top_drop', 'top-drop', 'trial2', '2'):
        black = _drop_top_edge_only_blobs(black_raw)
    else:
        # default / trial #1
        black = _keep_near_floor_blob(black_raw)
    red = _bin(make_mask(bev, ranges['red_road']))
    cyan1 = (
        _bin(make_mask(bev, ranges['black_cyan'], morph=False))
        if 'black_cyan' in ranges
        else np.zeros_like(black)
    )
    cyan2 = (
        _bin(make_mask(bev, ranges['black_cyan_2'], morph=False))
        if 'black_cyan_2' in ranges
        else np.zeros_like(black)
    )
    cyan_raw = _or2(cyan1, cyan2)
    cyan = _keep_near_floor_blob(cyan_raw)
    road = _or2(_or2(black, red), cyan)

    # Course paint: exclusive white *or* yellow (SSOT · resolve_course_lane_mask).
    if prefer_yellow is None:
        key = (course or '').strip().lower()
        if key in ('out', 'out_glare', 'out_course'):
            prefer_yellow = False
        else:
            # default / in / unknown → IN rule (yellow if present else white)
            prefer_yellow = True
    try:
        from inference.modules.perception.blob.rail_corridor import (
            resolve_course_lane_mask,
        )
    except ModuleNotFoundError:
        from inference.inference.modules.perception.blob.rail_corridor import (
            resolve_course_lane_mask,
        )

    paint, used_yellow = resolve_course_lane_mask(
        white, yellow, prefer_yellow=bool(prefer_yellow)
    )
    if paint is None or paint.size == 0:
        paint = np.zeros_like(road)
    else:
        paint = _bin(paint)

    white_road = _or2(white, road)
    yellow_road = _or2(yellow, road)
    lane_road = _or2(paint, road)  # IN/OUT exclusive paint | road

    # 4) stronger open + light close on road (noise then small holes)
    road_open = _clean_lane_mask(
        road,
        open_k=open_k,
        close_k=max(close_k - 4, open_k + 2),
        open_iters=open_iters,
        close_iters=max(1, close_iters - 1),
        max_hole_px=max_hole_px,
    )

    # 5) open+strong close+hole-fill on (course_paint | road)
    cleaned = _clean_lane_mask(
        lane_road,
        open_k=open_k,
        close_k=close_k,
        open_iters=open_iters,
        close_iters=close_iters,
        max_hole_px=max_hole_px,
    )

    # 6) only the blob attached to BEV bottom (robot / bumper)
    ego_blob = _keep_bottom_ego_blob(cleaned)

    return {
        'bev': bev,
        'white_road': white_road,
        'yellow_road': yellow_road,
        'cyan_road': cyan,
        'cyan_raw': cyan_raw,
        'cyan1': cyan1,
        'cyan2': cyan2,
        'road_open': road_open,
        'morph_fill': cleaned,
        'ego_blob': ego_blob,
        'white': white,
        'yellow': yellow,
        'black': black,
        'black_raw': black_raw,
        'road': road,
        'cyan': cyan,
        'paint': paint,
        'lane_road': lane_road,
        'used_yellow': used_yellow,
        'prefer_yellow': bool(prefer_yellow),
        'black_mode': mode if mode in ('top_drop', 'top-drop', 'trial2', '2') else 'near',
    }


def extract_five(
    frame: np.ndarray,
    config_path: Path,
    *,
    open_k: int = 5,
    close_k: int = 17,
    open_iters: int = 1,
    close_iters: int = 2,
    max_hole_px: int = 5000,
    course: str | None = None,
    prefer_yellow: bool | None = None,
    black_mode: str = 'near',
) -> dict[str, np.ndarray]:
    """Camera frame → Metric IPM BEV once → ``extract_five_from_bev``."""

    ipm = load_metric_ipm(config_path)
    bev = warp_metric_ipm(frame, ipm)
    return extract_five_from_bev(
        bev,
        config_path,
        open_k=open_k,
        close_k=close_k,
        open_iters=open_iters,
        close_iters=close_iters,
        max_hole_px=max_hole_px,
        course=course,
        prefer_yellow=prefer_yellow,
        black_mode=black_mode,
    )


def build_mosaic(
    five: dict[str, np.ndarray],
    label: str,
    *,
    open_k: int,
    close_k: int,
) -> np.ndarray:
    """Focus mosaic: BEV | cleaned | ego-bottom blob only."""
    p0 = _fit_bgr(five['bev'])
    _title(p0, '1 BEV')
    p1 = _panel_bin(five['morph_fill'], '2 morph fill')
    p2 = _panel_bin(five['ego_blob'], '3 ego blob')

    gap = np.full((PANEL_H, GAP, 3), 28, dtype=np.uint8)
    row = np.hstack([p0, gap, p1, gap, p2])

    footer = np.full((40, row.shape[1], 3), 16, dtype=np.uint8)
    cv2.putText(
        footer,
        f'{label}  open_k={open_k} close_k={close_k}',
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    paint = 'Y' if five.get('used_yellow') else 'W'
    cv2.putText(
        footer,
        f'3=ego bottom CC AFTER morph; cyan=pre-ego BEFORE morph; '
        f'road=black|red|cyan_near',
        (8, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([row, footer])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--from-bag', choices=('in', 'out', 'out_glare'), default='out')
    ap.add_argument('--folder', type=Path, default=None)
    ap.add_argument('--config', type=Path, default=default_config_path())
    ap.add_argument('--index', type=int, default=1)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--open-k', type=int, default=5, help='open kernel (noise)')
    ap.add_argument('--close-k', type=int, default=17, help='close kernel (fill)')
    ap.add_argument('--open-iters', type=int, default=1)
    ap.add_argument('--close-iters', type=int, default=2)
    ap.add_argument('--max-hole-px', type=int, default=5000)
    args = ap.parse_args(argv)

    open_k = int(args.open_k)
    close_k = int(args.close_k)
    open_iters = int(args.open_iters)
    close_iters = int(args.close_iters)
    max_hole_px = int(args.max_hole_px)

    folder = args.folder.expanduser().resolve() if args.folder else FROM_BAG[args.from_bag]
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    if not paths:
        raise SystemExit(f'No images in {folder}')

    bag = args.from_bag if args.folder is None else folder.name
    out_root = OUT_DIR / bag
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    indices = (
        list(range(0, len(paths), max(1, args.stride)))
        if args.all
        else [max(0, min(len(paths) - 1, args.index - 1))]
    )

    for i in indices:
        frame = cv2.imread(str(paths[i]))
        if frame is None:
            continue
        stem = paths[i].stem
        five = extract_five(
            frame,
            args.config,
            open_k=open_k,
            close_k=close_k,
            open_iters=open_iters,
            close_iters=close_iters,
            max_hole_px=max_hole_px,
            course=bag,
        )
        frame_dir = out_root / f'{i + 1:04d}_{stem}'
        frame_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(frame_dir / '1_bev.png'), five['bev'])
        cv2.imwrite(str(frame_dir / '2_white_road.png'), five['white_road'])
        cv2.imwrite(str(frame_dir / '3_yellow_road.png'), five['yellow_road'])
        cv2.imwrite(str(frame_dir / '3b_black_cyan.png'), five['cyan'])
        cv2.imwrite(str(frame_dir / '4_road_open.png'), five['road_open'])
        cv2.imwrite(str(frame_dir / '5_morph_fill.png'), five['morph_fill'])
        cv2.imwrite(str(frame_dir / '6_ego_blob.png'), five['ego_blob'])

        mosaic = build_mosaic(
            five,
            label=f'[{i + 1}/{len(paths)}] {paths[i].name}',
            open_k=open_k,
            close_k=close_k,
        )
        mosaic_path = out_root / f'{i + 1:04d}_{stem}_mosaic.png'
        cv2.imwrite(str(mosaic_path), mosaic)
        print(f'Wrote {frame_dir.name}/ + {mosaic_path.name}', flush=True)

    print(f'Dir: {out_root}  (n={len(indices)})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
