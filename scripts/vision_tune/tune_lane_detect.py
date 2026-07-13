#!/usr/bin/env python3
"""Mode-by-mode lane perception tuner (Gazebo-free when bringup is already up).

Does **not** launch Gazebo. Assumes `/camera/image/compressed` is already
publishing (sim-bringup or real car), or use ``--image`` / ``--folder``.

Staged workflow (recommended):
  Phase A — keys 2–3: yellow HSV, then dash connect until **4 clear strands**
  Phase B — keys 4–5: optional dash_left / dash_right filter check
  Phase C — keys 6–8: L/R fork pair split (only after Phase A looks good)

Hotkeys:
  c / SPACE  save review bundle under data/lane_tune_logs/<stamp>/
  s          write hsv + detect_tune into config/lane_vision.yaml
  1–9 / 0    switch mode
  n / p      next/prev (folder mode)
  q / ESC    quit

Examples (inside 2026-smh-sim):

  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 scripts/vision_tune/tune_lane_detect.py              # starts on dash
  python3 scripts/vision_tune/tune_lane_detect.py --mode fork
  python3 scripts/vision_tune/tune_lane_detect.py --folder data/captures/lane_tune_logs

Do **not** re-run ``sim_auto_driving`` just for visualization (starts a second Gazebo).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_INFER_SRC = _REPO_ROOT / 'src' / 'inference'
if _INFER_SRC.is_dir() and str(_INFER_SRC) not in sys.path:
    sys.path.insert(0, str(_INFER_SRC))

from hsv import (  # noqa: E402
    CHANNEL_NAMES,
    HsvRange,
    default_config_path,
    load_hsv_ranges,
    save_hsv_ranges,
)
from window_layout import place_window, visible_work_area  # noqa: E402

MODES = (
    'white',
    'yellow',
    'dash',
    'dash_left',
    'dash_right',
    'fork',
    'fork_left',
    'fork_right',
    'red',
    'crossing',
)

# Staged tuning phases shown in the HUD (mode name → phase tag).
_MODE_PHASE = {
    'white': 'prep',
    'yellow': 'A-hsv',
    'dash': 'A-dash',
    'dash_left': 'B-filter',
    'dash_right': 'B-filter',
    'fork': 'C-split',
    'fork_left': 'C-split',
    'fork_right': 'C-split',
    'red': 'other',
    'crossing': 'other',
}

WIN = 'lane_detect_tune'
PREVIEW_SCALE = 2.0
DEFAULT_LOG_ROOT = _REPO_ROOT / 'data' / 'captures' / 'lane_tune_logs'

_DASH_TRACKBARS = (
    ('dash_lat_mm', 200, 50),
    ('dash_gap_cm', 60, 30),
    ('dash_head_deg', 90, 30),
    ('dash_area', 80, 12),
    ('dash_assoc_cm', 50, 22),
)

# Trackbar names depend on mode; rebuilt on mode switch.
_MODE_TRACKBARS: dict[str, tuple[tuple[str, int, int], ...]] = {
    'white': (
        ('h_min', 179, 0),
        ('h_max', 179, 179),
        ('s_min', 255, 0),
        ('s_max', 255, 29),
        ('v_min', 255, 174),
        ('v_max', 255, 255),
    ),
    'yellow': (
        ('h_min', 179, 0),
        ('h_max', 179, 55),
        ('s_min', 255, 32),
        ('s_max', 255, 255),
        ('v_min', 255, 79),
        ('v_max', 255, 255),
        ('dash_lat_mm', 200, 50),
    ),
    'dash': _DASH_TRACKBARS,
    'dash_left': _DASH_TRACKBARS,
    'dash_right': _DASH_TRACKBARS,
    'fork': (
        ('branch_sep_cm', 50, 15),
        ('fork_assoc_cm', 40, 8),
        ('fork_min_rows', 80, 18),
        ('fork_width_cm', 60, 35),
    ),
    'fork_left': (
        ('branch_sep_cm', 50, 15),
        ('fork_assoc_cm', 40, 8),
        ('fork_min_rows', 80, 18),
        ('fork_width_cm', 60, 35),
    ),
    'fork_right': (
        ('branch_sep_cm', 50, 15),
        ('fork_assoc_cm', 40, 8),
        ('fork_min_rows', 80, 18),
        ('fork_width_cm', 60, 35),
    ),
    'red': (
        ('h_min', 179, 170),
        ('h_max', 179, 179),
        ('s_min', 255, 125),
        ('s_max', 255, 192),
        ('v_min', 255, 161),
        ('v_max', 255, 229),
        ('h_low_wrap', 30, 0),
    ),
    'crossing': (
        ('cov_%', 100, 40),
        ('min_rows', 20, 3),
    ),
}

_HSV_MODE_CHANNEL = {
    'white': 'white',
    'yellow': 'yellow',
    'red': 'red_road',
}


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f'No such folder: {folder}')
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    return sorted(
        p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()
    )


def _decode_compressed(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _load_detect_tune(path: Path) -> dict[str, float | int]:
    defaults: dict[str, float | int] = {
        'crossing_coverage_ratio': 0.40,
        'crossing_min_rows': 3,
        'min_branch_separation_m': 0.15,
        'dash_max_lateral_error_m': 0.05,
        'dash_max_forward_gap_m': 0.30,
        'dash_max_heading_diff_deg': 30,
        'dash_min_component_area_px': 12,
        'dash_branch_assoc_m': 0.22,
        'red_h_low_wrap': 0,
        'fork_track_assoc_m': 0.08,
        'fork_track_min_rows': 18,
        'fork_pair_width_m': 0.35,
    }
    if not path.is_file():
        return defaults
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    block = data.get('detect_tune') or {}
    if not isinstance(block, dict):
        return defaults
    out = dict(defaults)
    for key in defaults:
        if key in block:
            out[key] = type(defaults[key])(block[key])
    return out


def _save_detect_tune(tune: dict[str, float | int], path: Path) -> Path:
    existing: dict[str, Any] = {}
    if path.is_file():
        with path.open('r', encoding='utf-8') as f:
            existing = yaml.safe_load(f) or {}
    existing['detect_tune'] = {
        'crossing_coverage_ratio': float(tune['crossing_coverage_ratio']),
        'crossing_min_rows': int(tune['crossing_min_rows']),
        'min_branch_separation_m': float(tune['min_branch_separation_m']),
        'dash_max_lateral_error_m': float(tune['dash_max_lateral_error_m']),
        'dash_max_forward_gap_m': float(tune['dash_max_forward_gap_m']),
        'dash_max_heading_diff_deg': float(tune['dash_max_heading_diff_deg']),
        'dash_min_component_area_px': int(tune['dash_min_component_area_px']),
        'dash_branch_assoc_m': float(tune['dash_branch_assoc_m']),
        'red_h_low_wrap': int(tune['red_h_low_wrap']),
        'fork_track_assoc_m': float(tune['fork_track_assoc_m']),
        'fork_track_min_rows': int(tune['fork_track_min_rows']),
        'fork_pair_width_m': float(tune['fork_pair_width_m']),
        'note': 'Tuned with scripts/vision_tune/tune_lane_detect.py',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(existing, f, sort_keys=False, allow_unicode=True)
    return path


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')


def _chown_to_host(path: Path) -> None:
    """Avoid root-owned logs that the host user cannot edit/delete."""

    if os.geteuid() != 0:
        return
    uid = os.environ.get('HOST_UID') or os.environ.get('SUDO_UID')
    gid = os.environ.get('HOST_GID') or os.environ.get('SUDO_GID')
    try:
        if uid is not None:
            os.chown(path, int(uid), int(gid if gid is not None else uid))
        else:
            os.chown(path, 1000, 1000)
    except OSError:
        pass


def _mask_bgr(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if mask.ndim == 2:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def _save_capture_bundle(state: 'DetectTuneState', log_root: Path) -> Path | None:
    """Write frame + dash/fork debug artifacts for offline review."""

    if state.last_frame is None or state.last_preview is None:
        print('[capture] no frame yet — wait for camera/image')
        return None

    stamp = _stamp()
    out_dir = log_root / f'{stamp}_{state.mode}'
    out_dir.mkdir(parents=True, exist_ok=True)
    _chown_to_host(out_dir)

    frame = state.last_frame
    preview = state.last_preview
    debug = state.last_debug

    cv2.imwrite(str(out_dir / 'frame.png'), frame)
    cv2.imwrite(str(out_dir / 'preview.png'), preview)
    if debug is not None:
        if getattr(debug, 'bev', None) is not None and debug.bev.size:
            cv2.imwrite(str(out_dir / 'bev.png'), debug.bev)
        if getattr(debug, 'yellow_bev', None) is not None and debug.yellow_bev.size:
            cv2.imwrite(str(out_dir / 'yellow_hsv.png'), _mask_bgr(debug.yellow_bev))
        if (
            getattr(debug, 'yellow_dash_points_bev', None) is not None
            and debug.yellow_dash_points_bev.size
        ):
            cv2.imwrite(
                str(out_dir / 'yellow_dash_points.png'),
                _mask_bgr(debug.yellow_dash_points_bev),
            )
        if (
            getattr(debug, 'yellow_connected_bev', None) is not None
            and debug.yellow_connected_bev.size
        ):
            cv2.imwrite(
                str(out_dir / 'yellow_connected.png'),
                _mask_bgr(debug.yellow_connected_bev),
            )
        if getattr(debug, 'road_clean', None) is not None and debug.road_clean.size:
            cv2.imwrite(str(out_dir / 'road_clean.png'), _mask_bgr(debug.road_clean))

    phase = _MODE_PHASE.get(state.mode, '?')
    meta: dict[str, Any] = {
        'stamp_utc': stamp,
        'mode': state.mode,
        'phase': phase,
        'phase_guide': {
            'A-hsv/A-dash': 'Tune until 4 yellow strands (2 outer solid + 2 dash) are clear',
            'B-filter': 'Optional: dash_left/right keep only one gore side',
            'C-split': 'Only after Phase A: L/R outer+inner+center pairs',
        },
        'detect_tune': dict(state.tune),
        'hsv_yellow': {
            'h_min': int(state.ranges['yellow'].h_min),
            'h_max': int(state.ranges['yellow'].h_max),
            's_min': int(state.ranges['yellow'].s_min),
            's_max': int(state.ranges['yellow'].s_max),
            'v_min': int(state.ranges['yellow'].v_min),
            'v_max': int(state.ranges['yellow'].v_max),
        },
        'user_note': state.capture_note or '',
        'files': [
            'frame.png',
            'preview.png',
            'bev.png',
            'yellow_hsv.png',
            'yellow_dash_points.png',
            'yellow_connected.png',
            'road_clean.png',
            'meta.yaml',
        ],
    }
    if debug is not None:
        meta['debug'] = {
            'fork_active': bool(getattr(debug, 'fork_active', False)),
            'n_road_branches': len(getattr(debug, 'road_branches', ()) or ()),
            'n_fork_lane_pairs': len(getattr(debug, 'fork_lane_pairs', ()) or ()),
            'n_fork_mark_tracks': len(getattr(debug, 'fork_mark_tracks', ()) or ()),
            'fork_split_source': str(getattr(debug, 'fork_split_source', '') or ''),
            'ego_road_color': getattr(debug, 'ego_road_color', None),
            'yellow_crossing_line': bool(
                getattr(debug, 'yellow_crossing_line', False)
            ),
        }

    meta_path = out_dir / 'meta.yaml'
    with meta_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    for path in out_dir.iterdir():
        _chown_to_host(path)

    latest = log_root / 'LATEST.txt'
    latest.write_text(f'{out_dir.relative_to(_REPO_ROOT)}\n', encoding='utf-8')
    _chown_to_host(latest)

    # Append to a simple index the agent can skim.
    index_path = log_root / 'INDEX.md'
    line = (
        f'- `{out_dir.name}` · mode={state.mode} · phase={phase} · '
        f'note={state.capture_note or "-"}\n'
    )
    header = (
        '# lane_tune_logs\n\n'
        'Press `c` / SPACE in `tune_lane_detect.py` to append entries.\n\n'
    )
    if index_path.is_file():
        index_path.write_text(
            index_path.read_text(encoding='utf-8') + line,
            encoding='utf-8',
        )
    else:
        index_path.write_text(header + line, encoding='utf-8')
    _chown_to_host(index_path)

    # Clear one-shot note after save.
    state.capture_note = ''
    print(f'[capture] saved → {out_dir}')
    print(f'[capture] LATEST → {latest}')
    return out_dir


def _import_lane_detection():
    try:
        from inference.modules import lane_detection as ld  # type: ignore
    except ModuleNotFoundError:
        try:
            from inference.inference.modules import lane_detection as ld  # type: ignore
        except ModuleNotFoundError as exc:
            raise SystemExit(
                'Cannot import lane_detection. Inside 2026-smh-sim:\n'
                '  source /opt/ros/humble/setup.bash && source install/setup.bash\n'
                f'Original: {exc}'
            ) from exc
    return ld


class DetectTuneState:
    def __init__(
        self,
        ranges: dict[str, HsvRange],
        tune: dict[str, float | int],
        mode: str,
        log_root: Path | None = None,
    ):
        self.ranges = {k: v.clamp() for k, v in ranges.items()}
        self.tune = dict(tune)
        self.mode_idx = MODES.index(mode) if mode in MODES else 0
        self._suppress = False
        self._trackbar_keys: tuple[str, ...] = ()
        self.log_root = Path(log_root) if log_root else DEFAULT_LOG_ROOT
        self.last_frame: np.ndarray | None = None
        self.last_debug: Any = None
        self.last_preview: np.ndarray | None = None
        self.capture_note: str = ''

    @property
    def mode(self) -> str:
        return MODES[self.mode_idx]

    def set_mode(self, idx: int) -> None:
        self.mode_idx = int(np.clip(idx, 0, len(MODES) - 1))
        self._rebuild_trackbars()
        phase = _MODE_PHASE.get(self.mode, '?')
        print(f'Mode → {self.mode}  [{phase}]')

    def _rebuild_trackbars(self) -> None:
        # Destroy/recreate window so trackbars match the mode.
        try:
            cv2.destroyWindow(WIN)
        except cv2.error:
            pass
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        specs = _MODE_TRACKBARS[self.mode]
        self._trackbar_keys = tuple(name for name, _vmax, _init in specs)
        self._suppress = True
        for name, vmax, _init in specs:
            cv2.createTrackbar(name, WIN, 0, vmax, lambda *_: None)
        self._push_trackbars()
        self._suppress = False
        ox, oy, _, _ = visible_work_area()
        place_window(WIN, ox, oy)

    def _push_trackbars(self) -> None:
        self._suppress = True
        mode = self.mode
        ch = _HSV_MODE_CHANNEL.get(mode)
        if ch is not None:
            rng = self.ranges[ch]
            mapping = {
                'h_min': rng.h_min,
                'h_max': rng.h_max,
                's_min': rng.s_min,
                's_max': rng.s_max,
                'v_min': rng.v_min,
                'v_max': rng.v_max,
            }
            for key, val in mapping.items():
                if key in self._trackbar_keys:
                    cv2.setTrackbarPos(key, WIN, int(val))
        if mode == 'yellow' and 'dash_lat_mm' in self._trackbar_keys:
            mm = int(round(float(self.tune['dash_max_lateral_error_m']) * 1000))
            cv2.setTrackbarPos('dash_lat_mm', WIN, int(np.clip(mm, 5, 200)))
        if mode.startswith('dash'):
            if 'dash_lat_mm' in self._trackbar_keys:
                mm = int(round(float(self.tune['dash_max_lateral_error_m']) * 1000))
                cv2.setTrackbarPos('dash_lat_mm', WIN, int(np.clip(mm, 5, 200)))
            if 'dash_gap_cm' in self._trackbar_keys:
                cm = int(round(float(self.tune['dash_max_forward_gap_m']) * 100))
                cv2.setTrackbarPos('dash_gap_cm', WIN, int(np.clip(cm, 5, 60)))
            if 'dash_head_deg' in self._trackbar_keys:
                cv2.setTrackbarPos(
                    'dash_head_deg',
                    WIN,
                    int(np.clip(int(self.tune['dash_max_heading_diff_deg']), 5, 90)),
                )
            if 'dash_area' in self._trackbar_keys:
                cv2.setTrackbarPos(
                    'dash_area',
                    WIN,
                    int(np.clip(int(self.tune['dash_min_component_area_px']), 3, 80)),
                )
            if 'dash_assoc_cm' in self._trackbar_keys:
                cm = int(round(float(self.tune['dash_branch_assoc_m']) * 100))
                cv2.setTrackbarPos('dash_assoc_cm', WIN, int(np.clip(cm, 5, 50)))
        if mode.startswith('fork') and 'branch_sep_cm' in self._trackbar_keys:
            cm = int(round(float(self.tune['min_branch_separation_m']) * 100))
            cv2.setTrackbarPos('branch_sep_cm', WIN, int(np.clip(cm, 2, 50)))
            if 'fork_assoc_cm' in self._trackbar_keys:
                cm_a = int(round(float(self.tune['fork_track_assoc_m']) * 100))
                cv2.setTrackbarPos('fork_assoc_cm', WIN, int(np.clip(cm_a, 2, 40)))
            if 'fork_min_rows' in self._trackbar_keys:
                cv2.setTrackbarPos(
                    'fork_min_rows',
                    WIN,
                    int(np.clip(int(self.tune['fork_track_min_rows']), 5, 80)),
                )
            if 'fork_width_cm' in self._trackbar_keys:
                cm_w = int(round(float(self.tune['fork_pair_width_m']) * 100))
                cv2.setTrackbarPos('fork_width_cm', WIN, int(np.clip(cm_w, 15, 60)))
        if mode == 'red' and 'h_low_wrap' in self._trackbar_keys:
            cv2.setTrackbarPos(
                'h_low_wrap', WIN, int(self.tune['red_h_low_wrap'])
            )
        if mode == 'crossing':
            if 'cov_%' in self._trackbar_keys:
                pct = int(round(float(self.tune['crossing_coverage_ratio']) * 100))
                cv2.setTrackbarPos('cov_%', WIN, int(np.clip(pct, 5, 100)))
            if 'min_rows' in self._trackbar_keys:
                cv2.setTrackbarPos(
                    'min_rows',
                    WIN,
                    int(np.clip(int(self.tune['crossing_min_rows']), 1, 20)),
                )
        self._suppress = False

    def sync_from_trackbars(self) -> None:
        if self._suppress or not self._trackbar_keys:
            return
        mode = self.mode
        ch = _HSV_MODE_CHANNEL.get(mode)
        if ch is not None:
            keys = {k: cv2.getTrackbarPos(k, WIN) for k in self._trackbar_keys}
            self.ranges[ch] = HsvRange(
                h_min=keys.get('h_min', 0),
                h_max=keys.get('h_max', 179),
                s_min=keys.get('s_min', 0),
                s_max=keys.get('s_max', 255),
                v_min=keys.get('v_min', 0),
                v_max=keys.get('v_max', 255),
            ).clamp()
        if mode == 'yellow' and 'dash_lat_mm' in self._trackbar_keys:
            self.tune['dash_max_lateral_error_m'] = (
                max(5, cv2.getTrackbarPos('dash_lat_mm', WIN)) / 1000.0
            )
        if mode.startswith('dash'):
            if 'dash_lat_mm' in self._trackbar_keys:
                self.tune['dash_max_lateral_error_m'] = (
                    max(5, cv2.getTrackbarPos('dash_lat_mm', WIN)) / 1000.0
                )
            if 'dash_gap_cm' in self._trackbar_keys:
                self.tune['dash_max_forward_gap_m'] = (
                    max(5, cv2.getTrackbarPos('dash_gap_cm', WIN)) / 100.0
                )
            if 'dash_head_deg' in self._trackbar_keys:
                self.tune['dash_max_heading_diff_deg'] = max(
                    5, cv2.getTrackbarPos('dash_head_deg', WIN)
                )
            if 'dash_area' in self._trackbar_keys:
                self.tune['dash_min_component_area_px'] = max(
                    3, cv2.getTrackbarPos('dash_area', WIN)
                )
            if 'dash_assoc_cm' in self._trackbar_keys:
                self.tune['dash_branch_assoc_m'] = (
                    max(5, cv2.getTrackbarPos('dash_assoc_cm', WIN)) / 100.0
                )
        if mode.startswith('fork') and 'branch_sep_cm' in self._trackbar_keys:
            self.tune['min_branch_separation_m'] = (
                max(2, cv2.getTrackbarPos('branch_sep_cm', WIN)) / 100.0
            )
            if 'fork_assoc_cm' in self._trackbar_keys:
                self.tune['fork_track_assoc_m'] = (
                    max(2, cv2.getTrackbarPos('fork_assoc_cm', WIN)) / 100.0
                )
            if 'fork_min_rows' in self._trackbar_keys:
                self.tune['fork_track_min_rows'] = max(
                    5, cv2.getTrackbarPos('fork_min_rows', WIN)
                )
            if 'fork_width_cm' in self._trackbar_keys:
                self.tune['fork_pair_width_m'] = (
                    max(15, cv2.getTrackbarPos('fork_width_cm', WIN)) / 100.0
                )
        if mode == 'red' and 'h_low_wrap' in self._trackbar_keys:
            self.tune['red_h_low_wrap'] = int(
                cv2.getTrackbarPos('h_low_wrap', WIN)
            )
        if mode == 'crossing':
            if 'cov_%' in self._trackbar_keys:
                self.tune['crossing_coverage_ratio'] = (
                    max(5, cv2.getTrackbarPos('cov_%', WIN)) / 100.0
                )
            if 'min_rows' in self._trackbar_keys:
                self.tune['crossing_min_rows'] = max(
                    1, cv2.getTrackbarPos('min_rows', WIN)
                )

    def apply_to_module(self, ld: Any) -> None:
        self.sync_from_trackbars()
        packed = {
            name: (self.ranges[name].lower(), self.ranges[name].upper())
            for name in CHANNEL_NAMES
        }
        ld.apply_hsv_thresholds(packed)
        ld.apply_detect_tune(
            crossing_coverage_ratio=float(self.tune['crossing_coverage_ratio']),
            crossing_min_rows=int(self.tune['crossing_min_rows']),
            min_branch_separation_m=float(self.tune['min_branch_separation_m']),
            dash_max_lateral_error_m=float(self.tune['dash_max_lateral_error_m']),
            dash_max_forward_gap_m=float(self.tune['dash_max_forward_gap_m']),
            dash_max_heading_diff_deg=float(self.tune['dash_max_heading_diff_deg']),
            dash_min_component_area_px=int(self.tune['dash_min_component_area_px']),
            dash_branch_assoc_m=float(self.tune['dash_branch_assoc_m']),
            red_h_low_wrap=int(self.tune['red_h_low_wrap']),
            fork_track_assoc_m=float(self.tune['fork_track_assoc_m']),
            fork_track_min_rows=int(self.tune['fork_track_min_rows']),
            fork_pair_width_m=float(self.tune['fork_pair_width_m']),
        )


def _show_frame(frame: np.ndarray, state: DetectTuneState, ld: Any) -> None:
    state.apply_to_module(ld)
    _dets, debug = ld.detect_with_debug(frame)
    preview = ld.render_mode_preview(state.mode, debug)
    if preview.size == 0:
        return
    state.last_frame = frame.copy()
    state.last_debug = debug
    state.last_preview = preview.copy()

    h, w = preview.shape[:2]
    scaled = cv2.resize(
        preview,
        (int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)),
        interpolation=cv2.INTER_NEAREST,
    )
    phase = _MODE_PHASE.get(state.mode, '?')
    label = (
        f'[{state.mode_idx + 1}/{len(MODES)}] {state.mode} [{phase}]  '
        f'tracks={len(getattr(debug, "fork_mark_tracks", ()) or ())}  '
        f'pairs={len(getattr(debug, "fork_lane_pairs", ()) or ())}  '
        f'src={getattr(debug, "fork_split_source", "") or "-"}'
    )
    cv2.putText(
        scaled,
        label,
        (8, scaled.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    hud = (
        'Phase A:2-3 yellow/dash → B:4-5 → C:6-8 fork | '
        'c/SPACE=log bundle  s=yaml  q=quit'
    )
    cv2.putText(
        scaled,
        hud,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(WIN, scaled)
    cv2.resizeWindow(WIN, scaled.shape[1], min(720, scaled.shape[0] + 160))


def _handle_key(
    key: int,
    state: DetectTuneState,
    config_path: Path,
) -> str | None:
    if key in (ord('q'), 27):
        return 'quit'
    if key in (ord('c'), ord(' ')):
        state.sync_from_trackbars()
        _save_capture_bundle(state, state.log_root)
        return None
    if key == ord('s'):
        state.sync_from_trackbars()
        save_hsv_ranges(state.ranges, config_path)
        _save_detect_tune(state.tune, config_path)
        print(f'Saved hsv + detect_tune → {config_path}')
        return None
    if key == ord('0'):
        state.set_mode(9)  # crossing
        return None
    if key in (
        ord('1'),
        ord('2'),
        ord('3'),
        ord('4'),
        ord('5'),
        ord('6'),
        ord('7'),
        ord('8'),
        ord('9'),
    ):
        state.set_mode(key - ord('1'))
        return None
    if key == ord('n'):
        return 'next'
    if key == ord('p'):
        return 'prev'
    return None


def run_folder(
    folder: Path,
    state: DetectTuneState,
    config_path: Path,
    ld: Any,
) -> int:
    paths = _list_images(folder)
    if not paths:
        raise SystemExit(f'No images in {folder}')
    frames = [cv2.imread(str(p)) for p in paths]
    frames = [f for f in frames if f is not None]
    if not frames:
        raise SystemExit(f'Failed to load images from {folder}')
    state._rebuild_trackbars()
    idx = 0
    print(
        f'Folder mode: {len(frames)} images. '
        f'Keys: 2-3 Phase A dash | c=log | s=yaml | n/p | q'
    )
    while True:
        _show_frame(frames[idx], state, ld)
        key = cv2.waitKey(30) & 0xFF
        action = _handle_key(key, state, config_path)
        if action == 'quit':
            break
        if action == 'next' and len(frames) > 1:
            idx = (idx + 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {paths[idx].name}')
        if action == 'prev' and len(frames) > 1:
            idx = (idx - 1) % len(frames)
            print(f'[{idx + 1}/{len(frames)}] {paths[idx].name}')
    cv2.destroyAllWindows()
    return 0


def run_image(
    image: Path,
    state: DetectTuneState,
    config_path: Path,
    ld: Any,
) -> int:
    frame = cv2.imread(str(image))
    if frame is None:
        raise SystemExit(f'Failed to read image: {image}')
    state._rebuild_trackbars()
    print(f'Image mode: {image}. Keys: 2-3 Phase A | c=log | s | q')
    while True:
        _show_frame(frame, state, ld)
        key = cv2.waitKey(30) & 0xFF
        if _handle_key(key, state, config_path) == 'quit':
            break
    cv2.destroyAllWindows()
    return 0


def run_topic(
    topic: str,
    state: DetectTuneState,
    config_path: Path,
    ld: Any,
) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'rclpy not found. Inside 2026-smh-sim:\n'
            '  source /opt/ros/humble/setup.bash\n'
            f'Original error: {exc}'
        ) from exc

    image_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    class DetectTuneNode(Node):
        def __init__(self) -> None:
            super().__init__('lane_detect_tune')
            self.frame: np.ndarray | None = None
            self.create_subscription(
                CompressedImage, topic, self._on_compressed, image_qos
            )
            self.get_logger().info(
                f'Lane detect tune on {topic} (Gazebo not launched by this tool)'
            )

        def _on_compressed(self, msg: CompressedImage) -> None:
            frame = _decode_compressed(bytes(msg.data))
            if frame is not None:
                self.frame = frame

    state._rebuild_trackbars()
    rclpy.init()
    node = DetectTuneNode()
    print('Live topic mode.')
    print('  Phase A (now): key 3=dash — tune dash_* until 4 strands look solid')
    print(f'  c / SPACE → save bundle under {state.log_root}')
    print('  s → yaml   q → quit')
    print('Ensure sim-bringup (or car) is already publishing the camera topic.')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.frame is not None:
                _show_frame(node.frame, state, ld)
            key = cv2.waitKey(1) & 0xFF
            if _handle_key(key, state, config_path) == 'quit':
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Lane perception tuner — Phase A dash connect first, then L/R split. '
            'Hotkey c saves a review bundle.'
        )
    )
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--folder', type=Path, default=None)
    parser.add_argument('--image', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=default_config_path())
    parser.add_argument(
        '--mode',
        choices=list(MODES),
        default='dash',
        help='Initial mode (default: dash = Phase A connect)',
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help=f'Capture root (default: {DEFAULT_LOG_ROOT})',
    )
    args = parser.parse_args(argv)

    ld = _import_lane_detection()
    ld.VISUALIZE_MODE = ld.VISUALIZE_OFF
    ld.VISUALIZE = False

    ranges = load_hsv_ranges(args.config)
    tune = _load_detect_tune(args.config)
    live = ld.get_detect_tune()
    for key, val in live.items():
        tune.setdefault(key, val)

    log_root = args.log_dir
    if not log_root.is_absolute():
        log_root = (_REPO_ROOT / log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    _chown_to_host(log_root)

    state = DetectTuneState(ranges, tune, args.mode, log_root=log_root)
    state.apply_to_module(ld)
    print(
        f'Starting mode={state.mode} [{_MODE_PHASE.get(state.mode, "?")}]  '
        f'log_dir={log_root}'
    )

    if args.image is not None:
        return run_image(args.image, state, args.config, ld)
    if args.folder is not None:
        return run_folder(args.folder, state, args.config, ld)
    return run_topic(args.topic, state, args.config, ld)


if __name__ == '__main__':
    raise SystemExit(main())
