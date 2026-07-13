#!/usr/bin/env python3
"""Mode-by-mode lane perception tuner (Gazebo-free when bringup is already up).

Does **not** launch Gazebo. Assumes `/camera/image/compressed` is already
publishing (sim-bringup or real car), or use ``--image`` / ``--folder``.

Modes (keys 1–9, 0):
  white | yellow | dash | dash_left | dash_right |
  fork | fork_left | fork_right | red | crossing

Dash modes isolate dashed markings at forks/merges. ``dash_left`` /
``dash_right`` keep only blobs near that RoadBranch centerline so the chosen
path's dashed lane can be promoted without the other gore line.
One preview window + trackbars on the same window. Saves HSV + detect_tune
into ``config/lane_vision.yaml``.

Examples (inside 2026-smh-sim, Gazebo already running via sim-bringup):

  source /opt/ros/humble/setup.bash
  source install/setup.bash
  python3 scripts/vision_tune/tune_lane_detect.py --mode white
  python3 scripts/vision_tune/tune_lane_detect.py --folder data/captures/sim
  python3 scripts/vision_tune/tune_lane_detect.py --image /path/to/frame.png

Do **not** re-run ``sim_auto_driving`` just for visualization (starts a second Gazebo).
"""

from __future__ import annotations

import argparse
import sys
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

WIN = 'lane_detect_tune'
PREVIEW_SCALE = 2.0

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
    ),
    'fork_left': (
        ('branch_sep_cm', 50, 15),
    ),
    'fork_right': (
        ('branch_sep_cm', 50, 15),
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
        'note': 'Tuned with scripts/vision_tune/tune_lane_detect.py',
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(existing, f, sort_keys=False, allow_unicode=True)
    return path


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
    ):
        self.ranges = {k: v.clamp() for k, v in ranges.items()}
        self.tune = dict(tune)
        self.mode_idx = MODES.index(mode) if mode in MODES else 0
        self._suppress = False
        self._trackbar_keys: tuple[str, ...] = ()

    @property
    def mode(self) -> str:
        return MODES[self.mode_idx]

    def set_mode(self, idx: int) -> None:
        self.mode_idx = int(np.clip(idx, 0, len(MODES) - 1))
        self._rebuild_trackbars()

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
        )


def _show_frame(frame: np.ndarray, state: DetectTuneState, ld: Any) -> None:
    state.apply_to_module(ld)
    _dets, debug = ld.detect_with_debug(frame)
    preview = ld.render_mode_preview(state.mode, debug)
    if preview.size == 0:
        return
    h, w = preview.shape[:2]
    scaled = cv2.resize(
        preview,
        (int(w * PREVIEW_SCALE), int(h * PREVIEW_SCALE)),
        interpolation=cv2.INTER_NEAREST,
    )
    label = (
        f'[{state.mode_idx + 1}/{len(MODES)}] {state.mode}  '
        f'fork={debug.fork_active} nB={len(debug.road_branches)}  '
        f'red={100.0 * debug.red_coverage:.1f}%  '
        f'crossY={debug.yellow_crossing_line}'
    )
    cv2.putText(
        scaled,
        label,
        (8, scaled.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        scaled,
        '1-9/0 mode  s=save  n/p folder  q=quit  (no Gazebo launch)',
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
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
    if key == ord('s'):
        state.sync_from_trackbars()
        save_hsv_ranges(state.ranges, config_path)
        _save_detect_tune(state.tune, config_path)
        print(f'Saved hsv + detect_tune → {config_path}')
        return None
    if key == ord('0'):
        state.set_mode(9)  # crossing
        print(f'Mode → {state.mode}')
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
        print(f'Mode → {state.mode}')
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
    print(f'Folder mode: {len(frames)} images. Keys: 1-9/0 mode  s  n/p  q')
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
    print(f'Image mode: {image}. Keys: 1-9/0 mode  s  q')
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
    print('Live topic mode. Keys: 1-9/0 mode  s=save  q=quit')
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
        description='Mode-by-mode lane perception tuner (no Gazebo launch)'
    )
    parser.add_argument('--topic', default='/camera/image/compressed')
    parser.add_argument('--folder', type=Path, default=None)
    parser.add_argument('--image', type=Path, default=None)
    parser.add_argument('--config', type=Path, default=default_config_path())
    parser.add_argument(
        '--mode',
        choices=list(MODES),
        default='white',
        help='Initial review mode',
    )
    args = parser.parse_args(argv)

    ld = _import_lane_detection()
    # Keep OpenCV imshow from detect() off; this tool owns the window.
    ld.VISUALIZE_MODE = ld.VISUALIZE_OFF
    ld.VISUALIZE = False

    ranges = load_hsv_ranges(args.config)
    tune = _load_detect_tune(args.config)
    # Prefer live module defaults when yaml section missing keys already applied.
    live = ld.get_detect_tune()
    for key, val in live.items():
        tune.setdefault(key, val)

    state = DetectTuneState(ranges, tune, args.mode)
    state.apply_to_module(ld)

    if args.image is not None:
        return run_image(args.image, state, args.config, ld)
    if args.folder is not None:
        return run_folder(args.folder, state, args.config, ld)
    return run_topic(args.topic, state, args.config, ld)


if __name__ == '__main__':
    raise SystemExit(main())
