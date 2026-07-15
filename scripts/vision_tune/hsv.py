"""HSV ranges for lane / road masks — shared by tune_hsv and runtime stub.

Schema: ``config/lane_vision.yaml`` → ``hsv``

  active: sim | real_car
  profiles.<name>.<channel>: h_min … v_max
  <channel>: flattened active profile (runtime reads these)

Docs: docs/hsv-profiles.md
Tool owner: 안승현. Sim seed: 장원태 ``feature/wontae-lane``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

CHANNEL_NAMES = (
    'white',
    'yellow',
    'black_road',
    'red_road',
    'black_cyan',
    'black_cyan_2',
)
PROFILE_NAMES = ('sim', 'real_car')

# OUT scoreboard glare → cyan wash on asphalt (tuned out_glare 2026-07-15).
_BLACK_CYAN_DEFAULT: dict[str, int] = {
    'h_min': 90,
    'h_max': 100,
    's_min': 190,
    's_max': 220,
    'v_min': 200,
    'v_max': 230,
}

# Secondary cyan / teal asphalt patch (IN bag ~930, 2026-07-15 tune).
_BLACK_CYAN_2_DEFAULT: dict[str, int] = {
    'h_min': 97,
    'h_max': 105,
    's_min': 240,
    's_max': 255,
    'v_min': 105,
    'v_max': 180,
}

# Gazebo / Won Tae seed (OpenCV HSV).
_SIM_DEFAULTS: dict[str, dict[str, int]] = {
    'white': {
        'h_min': 0,
        'h_max': 179,
        's_min': 0,
        's_max': 29,
        'v_min': 174,
        'v_max': 255,
    },
    'yellow': {
        'h_min': 0,
        'h_max': 55,
        's_min': 32,
        's_max': 255,
        'v_min': 79,
        'v_max': 255,
    },
    'black_road': {
        'h_min': 0,
        'h_max': 179,
        's_min': 0,
        's_max': 255,
        'v_min': 0,
        'v_max': 30,
    },
    'red_road': {
        'h_min': 170,
        'h_max': 179,
        's_min': 125,
        's_max': 192,
        'v_min': 161,
        'v_max': 229,
    },
    'black_cyan': dict(_BLACK_CYAN_DEFAULT),
    'black_cyan_2': dict(_BLACK_CYAN_2_DEFAULT),
}

# origin/board field tune (bag_20260711_144948, D3-G 2026-07-14).
_BOARD_DEFAULTS: dict[str, dict[str, int]] = {
    'white': {
        'h_min': 0,
        'h_max': 179,
        's_min': 0,
        's_max': 29,
        'v_min': 180,
        'v_max': 255,
    },
    'yellow': {
        'h_min': 15,
        'h_max': 45,
        's_min': 110,
        's_max': 255,
        'v_min': 120,
        'v_max': 255,
    },
    'black_road': {
        'h_min': 0,
        'h_max': 89,
        's_min': 0,
        's_max': 255,
        'v_min': 10,
        'v_max': 200,
    },
    'red_road': {
        'h_min': 170,
        'h_max': 179,
        's_min': 130,
        's_max': 255,
        'v_min': 120,
        'v_max': 250,
    },
    'black_cyan': dict(_BLACK_CYAN_DEFAULT),
    'black_cyan_2': dict(_BLACK_CYAN_2_DEFAULT),
}

# Real-car field tune from bag replay captures (2026-07-15, commits 0191811 + 35ba99e).
_REAL_CAR_DEFAULTS: dict[str, dict[str, int]] = {
    'white': {
        'h_min': 0,
        'h_max': 179,
        's_min': 0,
        's_max': 20,
        'v_min': 210,
        'v_max': 255,
    },
    'yellow': {
        'h_min': 15,
        'h_max': 50,
        's_min': 50,
        's_max': 150,
        'v_min': 160,
        'v_max': 255,
    },
    'black_road': {
        'h_min': 17,
        'h_max': 70,
        's_min': 0,
        's_max': 255,
        'v_min': 15,
        'v_max': 140,
    },
    'red_road': {
        'h_min': 0,
        'h_max': 9,
        's_min': 155,
        's_max': 255,
        'v_min': 120,
        'v_max': 255,
    },
    # OUT LED billboard floor wash (from_bag/out_glare, 2026-07-15).
    'black_cyan': dict(_BLACK_CYAN_DEFAULT),
    'black_cyan_2': dict(_BLACK_CYAN_2_DEFAULT),
}

_DEFAULTS = _SIM_DEFAULTS

_PROFILE_META: dict[str, dict[str, Any]] = {
    'sim': {
        'environment': 'Gazebo LIMO (C920e 320x180 sim camera)',
        'source': 'feature/wontae-lane seed',
        'tuned': '2026-07-12',
        'tool': 'scripts/vision_tune/tune_hsv.py (key d)',
    },
    'real_car': {
        'environment': 'D3-G field (C920e 320x180)',
        'source': 'bag replay → data/captures/from_bag/{in,out}',
        'bags': 'bag_20260711_150234 (IN), bag_20260711_144948 (OUT)',
        'tuned': '2026-07-15',
        'tool': 'scripts/vision_tune/tune_hsv.py --from-bag',
        'commits': '0191811, 35ba99e',
    },
}


@dataclass(frozen=True)
class HsvRange:
    h_min: int = 0
    h_max: int = 179
    s_min: int = 0
    s_max: int = 255
    v_min: int = 0
    v_max: int = 255

    def clamp(self) -> HsvRange:
        h0 = int(np.clip(self.h_min, 0, 179))
        h1 = int(np.clip(self.h_max, 0, 179))
        if h1 < h0:
            h0, h1 = h1, h0
        s0 = int(np.clip(self.s_min, 0, 255))
        s1 = int(np.clip(self.s_max, 0, 255))
        if s1 < s0:
            s0, s1 = s1, s0
        v0 = int(np.clip(self.v_min, 0, 255))
        v1 = int(np.clip(self.v_max, 0, 255))
        if v1 < v0:
            v0, v1 = v1, v0
        return HsvRange(h0, h1, s0, s1, v0, v1)

    def lower(self) -> np.ndarray:
        p = self.clamp()
        return np.array([p.h_min, p.s_min, p.v_min], dtype=np.uint8)

    def upper(self) -> np.ndarray:
        p = self.clamp()
        return np.array([p.h_max, p.s_max, p.v_max], dtype=np.uint8)

    def to_dict(self) -> dict[str, int]:
        return asdict(self.clamp())

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, fallback: dict[str, int]) -> HsvRange:
        src = fallback if not isinstance(data, dict) else {**fallback, **data}
        return cls(
            h_min=int(src['h_min']),
            h_max=int(src['h_max']),
            s_min=int(src['s_min']),
            s_max=int(src['s_max']),
            v_min=int(src['v_min']),
            v_max=int(src['v_max']),
        ).clamp()


def default_config_path() -> Path:
    here = Path(__file__).resolve()
    for base in here.parents:
        cand = base / 'config' / 'lane_vision.yaml'
        if cand.is_file():
            return cand
    return here.parents[2] / 'config' / 'lane_vision.yaml'


def default_range(channel: str) -> HsvRange:
    """Gazebo / Won Tae sim seed."""
    if channel not in _SIM_DEFAULTS:
        raise KeyError(f'Unknown HSV channel: {channel}')
    return HsvRange.from_dict(_SIM_DEFAULTS[channel], _SIM_DEFAULTS[channel])


def sim_range(channel: str) -> HsvRange:
    return default_range(channel)


def real_car_range(channel: str) -> HsvRange:
    if channel not in _REAL_CAR_DEFAULTS:
        raise KeyError(f'Unknown HSV channel: {channel}')
    return HsvRange.from_dict(_REAL_CAR_DEFAULTS[channel], _REAL_CAR_DEFAULTS[channel])


def profile_seed_defaults(name: str) -> dict[str, dict[str, int]]:
    if name == 'sim':
        return _SIM_DEFAULTS
    if name == 'real_car':
        return _REAL_CAR_DEFAULTS
    raise KeyError(f'Unknown HSV profile: {name}')


def profile_ranges(name: str) -> dict[str, HsvRange]:
    seeds = profile_seed_defaults(name)
    return {ch: HsvRange.from_dict(seeds[ch], seeds[ch]) for ch in CHANNEL_NAMES}


def board_range(channel: str) -> HsvRange:
    """origin/board field baseline (real-car bag tune)."""
    if channel not in _BOARD_DEFAULTS:
        raise KeyError(f'Unknown HSV channel: {channel}')
    return HsvRange.from_dict(_BOARD_DEFAULTS[channel], _BOARD_DEFAULTS[channel])


def board_ranges() -> dict[str, HsvRange]:
    return {name: board_range(name) for name in CHANNEL_NAMES}


def _profile_block_from_seeds(
    name: str,
    seeds: dict[str, dict[str, int]],
) -> dict[str, Any]:
    block: dict[str, Any] = {'meta': dict(_PROFILE_META.get(name, {}))}
    for ch in CHANNEL_NAMES:
        block[ch] = dict(seeds[ch])
    return block


def _default_profiles() -> dict[str, Any]:
    return {
        'sim': _profile_block_from_seeds('sim', _SIM_DEFAULTS),
        'real_car': _profile_block_from_seeds('real_car', _REAL_CAR_DEFAULTS),
    }


def _flatten_profile(profiles: dict[str, Any], active: str) -> dict[str, dict[str, int]]:
    prof = profiles.get(active) or {}
    out: dict[str, dict[str, int]] = {}
    seeds = profile_seed_defaults(active)
    for ch in CHANNEL_NAMES:
        raw = prof.get(ch) if isinstance(prof, dict) else None
        if isinstance(raw, dict):
            out[ch] = {k: int(raw[k]) for k in ('h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max')}
        else:
            out[ch] = dict(seeds[ch])
    return out


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def active_profile_name(path: Path | None = None) -> str:
    cfg_path = path or default_config_path()
    data = _read_config(cfg_path)
    raw = (data.get('hsv') or {}).get('active', 'real_car')
    return raw if raw in PROFILE_NAMES else 'real_car'


def load_hsv_ranges(path: Path | None = None) -> dict[str, HsvRange]:
    cfg_path = path or default_config_path()
    block: dict[str, Any] = {}
    if cfg_path.is_file():
        data = _read_config(cfg_path)
        raw = data.get('hsv') or {}
        if isinstance(raw, dict):
            block = raw
    active = block.get('active', 'real_car')
    if active not in PROFILE_NAMES:
        active = 'real_car'
    seeds = profile_seed_defaults(active)
    out: dict[str, HsvRange] = {}
    for name in CHANNEL_NAMES:
        out[name] = HsvRange.from_dict(block.get(name), seeds[name])
    return out


def save_hsv_ranges(
    ranges: dict[str, HsvRange],
    path: Path | None = None,
    *,
    profile: str | None = None,
) -> Path:
    cfg_path = path or default_config_path()
    existing = _read_config(cfg_path)
    prev_hsv = existing.get('hsv') if isinstance(existing.get('hsv'), dict) else {}
    active = profile or prev_hsv.get('active') or active_profile_name(cfg_path)
    if active not in PROFILE_NAMES:
        active = 'real_car'

    profiles = prev_hsv.get('profiles') if isinstance(prev_hsv.get('profiles'), dict) else {}
    if not profiles:
        profiles = _default_profiles()
    else:
        for pname in PROFILE_NAMES:
            if pname not in profiles:
                profiles[pname] = _profile_block_from_seeds(
                    pname,
                    profile_seed_defaults(pname),
                )

    prof_entry = dict(profiles.get(active) or {})
    prof_entry['meta'] = dict(_PROFILE_META.get(active, prof_entry.get('meta') or {}))
    for name in CHANNEL_NAMES:
        rng = ranges.get(name) or real_car_range(name)
        prof_entry[name] = rng.to_dict()
    profiles[active] = prof_entry

    hsv_block: dict[str, Any] = {
        'active': active,
        'profiles': profiles,
    }
    for name in CHANNEL_NAMES:
        hsv_block[name] = prof_entry[name]
    hsv_block['note'] = (
        'OpenCV HSV (H 0-179). Runtime reads flattened channels (= profiles[active]). '
        'See docs/hsv-profiles.md. Tuned with scripts/vision_tune/tune_hsv.py.'
    )
    existing['hsv'] = hsv_block
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(existing, f, sort_keys=False, allow_unicode=True)
    return cfg_path


def apply_hsv_profile(name: str, path: Path | None = None) -> Path:
    """Set ``hsv.active`` and flatten ``profiles[name]`` into runtime channels."""
    if name not in PROFILE_NAMES:
        raise KeyError(f'Unknown HSV profile: {name}')
    cfg_path = path or default_config_path()
    existing = _read_config(cfg_path)
    prev_hsv = existing.get('hsv') if isinstance(existing.get('hsv'), dict) else {}
    profiles = prev_hsv.get('profiles') if isinstance(prev_hsv.get('profiles'), dict) else {}
    if not profiles:
        profiles = _default_profiles()
    else:
        for pname in PROFILE_NAMES:
            if pname not in profiles:
                profiles[pname] = _profile_block_from_seeds(
                    pname,
                    profile_seed_defaults(pname),
                )
    flat = _flatten_profile(profiles, name)
    ranges = {
        ch: HsvRange.from_dict(flat[ch], profile_seed_defaults(name)[ch])
        for ch in CHANNEL_NAMES
    }
    return save_hsv_ranges(ranges, cfg_path, profile=name)


def make_mask(bgr: np.ndarray, rng: HsvRange, *, morph: bool = True) -> np.ndarray:
    """Binary mask for one HSV range (single segment; no red wrap).

    When ``morph`` is True, apply a single light open only (no close) so
    channel masks do not bridge toward off-track or fork arms.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    p = rng.clamp()
    mask = cv2.inRange(hsv, p.lower(), p.upper())
    if morph:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def overlay_mask(
    bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    out = bgr.copy()
    tint = np.zeros_like(out)
    tint[:] = color
    selected = mask > 0
    out[selected] = (
        (1.0 - alpha) * out[selected].astype(np.float32)
        + alpha * tint[selected].astype(np.float32)
    ).astype(np.uint8)
    return out


def expand_range_with_sample(
    rng: HsvRange,
    hsv_pixel: np.ndarray,
    *,
    h_pad: int = 5,
    s_pad: int = 20,
    v_pad: int = 20,
) -> HsvRange:
    """Expand current range so it includes a clicked HSV sample (±pad)."""
    h, s, v = (int(x) for x in hsv_pixel.reshape(3))
    p = rng.clamp()
    return HsvRange(
        h_min=min(p.h_min, max(0, h - h_pad)),
        h_max=max(p.h_max, min(179, h + h_pad)),
        s_min=min(p.s_min, max(0, s - s_pad)),
        s_max=max(p.s_max, min(255, s + s_pad)),
        v_min=min(p.v_min, max(0, v - v_pad)),
        v_max=max(p.v_max, min(255, v + v_pad)),
    ).clamp()


def _main_cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='HSV profile utilities')
    parser.add_argument(
        '--apply-profile',
        choices=PROFILE_NAMES,
        help='Set hsv.active and flatten profiles/<name> into runtime channels',
    )
    parser.add_argument('--config', type=Path, default=default_config_path())
    args = parser.parse_args()
    if args.apply_profile:
        out = apply_hsv_profile(args.apply_profile, args.config)
        print(f'Applied HSV profile {args.apply_profile!r} → {out}')
        return 0
    parser.print_help()
    return 1


if __name__ == '__main__':
    raise SystemExit(_main_cli())
