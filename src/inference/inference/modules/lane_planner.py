"""Single-color path select + Pure Pursuit steering — 담당: 안승현.

Consumes perception ``LaneDetections`` for **one** follow color at a time
(white default, or yellow). No HSV / IPM / auto color switching here.

Sim defaults use **LIMO Gazebo** bicycle geometry (``wheelbase_m=0.24``,
``max_steer_angle_rad=0.5236``). D-Racer real-car values differ — see
``docs/vehicle-geometry.md`` and TODO comments in ``config/lane_control.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from inference.types import LaneDetections, LaneMarking, LaneResult

# LIMO Gazebo Ackermann (sim SSOT). Do NOT copy blindly to D-Racer.
_SIM_WHEELBASE_M = 0.24
_SIM_MAX_STEER_RAD = 0.5574  # ±atan(0.24/0.385) — team R_min on LIMO Gazebo


def _normalize_follow_color(color: str | None) -> str:
    c = (color or 'white').strip().lower()
    if c in ('yellow', 'y', 'yel'):
        return 'yellow'
    return 'white'


def _follow_color_id(color: str) -> int:
    if _normalize_follow_color(color) == 'yellow':
        return LaneMarking.COLOR_YELLOW
    return LaneMarking.COLOR_WHITE


def _repo_root() -> Path:
    for base in Path(__file__).resolve().parents:
        if (base / 'config' / 'lane_control.yaml').exists():
            return base
        if (base / 'src' / 'inference').is_dir() and (base / 'config').is_dir():
            return base
    return Path(__file__).resolve().parents[4]


def default_control_config_path() -> Path:
    return _repo_root() / 'config' / 'lane_control.yaml'


@dataclass
class LaneControlParams:
    """Pure Pursuit + output smoothing.

    ``lookahead_x_m`` is the geometric look-ahead distance Ld [m].
    ``wheelbase_m`` / ``max_steer_angle_rad`` are bicycle-model geometry.
    """

    lookahead_x_m: float = 0.80
    wheelbase_m: float = _SIM_WHEELBASE_M
    max_steer_angle_rad: float = _SIM_MAX_STEER_RAD
    # If >0 and speed_mps known later: L_d = clip(gain * v, min, max). 0 = fixed.
    lookahead_gain: float = 0.0
    ema_alpha: float = 0.40
    steer_rate_limit: float = 0.15
    max_steer: float = 1.0
    track_half_width_m: float = 0.175
    steer_slowdown_thresh: float = 0.50
    steer_slowdown_scale: float = 0.80
    min_confidence: float = 0.15
    hold_decay: float = 0.92
    follow_color: str = 'white'

    def clamp(self) -> LaneControlParams:
        return LaneControlParams(
            lookahead_x_m=float(np.clip(self.lookahead_x_m, 0.3, 1.5)),
            wheelbase_m=float(np.clip(self.wheelbase_m, 0.10, 0.40)),
            max_steer_angle_rad=float(np.clip(self.max_steer_angle_rad, 0.15, 0.80)),
            lookahead_gain=float(np.clip(self.lookahead_gain, 0.0, 5.0)),
            ema_alpha=float(np.clip(self.ema_alpha, 0.05, 1.0)),
            steer_rate_limit=float(np.clip(self.steer_rate_limit, 0.01, 1.0)),
            max_steer=float(np.clip(self.max_steer, 0.1, 1.0)),
            track_half_width_m=float(np.clip(self.track_half_width_m, 0.05, 0.4)),
            steer_slowdown_thresh=float(np.clip(self.steer_slowdown_thresh, 0.1, 1.0)),
            steer_slowdown_scale=float(np.clip(self.steer_slowdown_scale, 0.2, 1.0)),
            min_confidence=float(np.clip(self.min_confidence, 0.0, 1.0)),
            hold_decay=float(np.clip(self.hold_decay, 0.5, 0.99)),
            follow_color=_normalize_follow_color(self.follow_color),
        )


def load_control_params(path: Path | None = None) -> LaneControlParams:
    cfg_path = path or default_control_config_path()
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open('r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            data = loaded
    return LaneControlParams(
        lookahead_x_m=float(data.get('lookahead_x_m', 0.80)),
        wheelbase_m=float(data.get('wheelbase_m', _SIM_WHEELBASE_M)),
        max_steer_angle_rad=float(
            data.get('max_steer_angle_rad', _SIM_MAX_STEER_RAD)
        ),
        lookahead_gain=float(data.get('lookahead_gain', 0.0)),
        ema_alpha=float(data.get('ema_alpha', 0.40)),
        steer_rate_limit=float(data.get('steer_rate_limit', 0.15)),
        max_steer=float(data.get('max_steer', 1.0)),
        track_half_width_m=float(data.get('track_half_width_m', 0.175)),
        steer_slowdown_thresh=float(data.get('steer_slowdown_thresh', 0.50)),
        steer_slowdown_scale=float(data.get('steer_slowdown_scale', 0.80)),
        min_confidence=float(data.get('min_confidence', 0.15)),
        hold_decay=float(data.get('hold_decay', 0.92)),
        follow_color=str(data.get('follow_color', 'white')),
    ).clamp()


def save_control_params(params: LaneControlParams, path: Path | None = None) -> Path:
    cfg_path = path or default_control_config_path()
    p = params.clamp()
    payload = {
        'follow_color': p.follow_color,
        'lookahead_x_m': p.lookahead_x_m,
        'wheelbase_m': p.wheelbase_m,
        'max_steer_angle_rad': p.max_steer_angle_rad,
        'lookahead_gain': p.lookahead_gain,
        'ema_alpha': p.ema_alpha,
        'steer_rate_limit': p.steer_rate_limit,
        'max_steer': p.max_steer,
        'track_half_width_m': p.track_half_width_m,
        'steer_slowdown_thresh': p.steer_slowdown_thresh,
        'steer_slowdown_scale': p.steer_slowdown_scale,
        'min_confidence': p.min_confidence,
        'hold_decay': p.hold_decay,
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open('w', encoding='utf-8') as f:
        f.write('# Lane planner / Pure Pursuit gains (Phase 2+)\n')
        f.write('# Tuned with: scripts/vision_tune/tune_lane_control.py\n')
        f.write('# Sim defaults = LIMO Gazebo (wheelbase 0.24 m, δ_max ±30°).\n')
        f.write('# D-Racer: measure L / δ_max (or R_min) before retuning — see vehicle-geometry.md\n')
        f.write('# follow_color: white | yellow (single mode; no auto switch)\n')
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    return cfg_path


def _y_at_x(xy: np.ndarray, x_target: float) -> float | None:
    """Interpolate lateral y at forward x along a polyline (x,y)."""
    if xy is None or xy.shape[0] == 0:
        return None
    xs = xy[:, 0]
    ys = xy[:, 1]
    if xy.shape[0] == 1:
        return float(ys[0])
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    if x_target <= xs[0]:
        return float(ys[0])
    if x_target >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(x_target, xs, ys))


def centerline_y_at_lookahead(
    detections: LaneDetections,
    lookahead_x_m: float,
    track_half_width_m: float,
    follow_color: str = 'white',
) -> tuple[float | None, float]:
    """Return (y_center, confidence) at look-ahead x for one follow color."""
    color_id = _follow_color_id(follow_color)
    left, right = detections.pair_for_color(color_id)
    y_l = _y_at_x(left.xy(), lookahead_x_m) if left is not None else None
    y_r = _y_at_x(right.xy(), lookahead_x_m) if right is not None else None

    if y_l is not None and y_r is not None:
        conf = 0.5 * (
            (left.confidence if left else 0.0) + (right.confidence if right else 0.0)
        )
        conf = max(conf, 0.5)
        return 0.5 * (y_l + y_r), float(np.clip(conf, 0.0, 1.0))

    if y_l is not None:
        conf = max(left.confidence if left else 0.3, 0.3)
        return y_l - track_half_width_m, float(np.clip(conf, 0.0, 1.0))

    if y_r is not None:
        conf = max(right.confidence if right else 0.3, 0.3)
        return y_r + track_half_width_m, float(np.clip(conf, 0.0, 1.0))

    return None, 0.0


def pure_pursuit_steer(
    x_t: float,
    y_t: float,
    *,
    wheelbase_m: float,
    lookahead_m: float,
    max_steer_angle_rad: float,
    max_steer: float = 1.0,
) -> tuple[float, float, float]:
    """Bicycle Pure Pursuit → normalized steering.

    Returns ``(steering_offset, alpha_rad, delta_rad)``.

    ``α = atan2(y_t, x_t)``, ``δ = atan(2 L sinα / L_d)``.
    base_link +y = left; D-Racer +steering = right → ``steering = -δ/δ_max``.
    """
    ld = max(float(lookahead_m), 1e-3)
    x = float(x_t)
    y = float(y_t)
    alpha = float(np.arctan2(y, max(x, 1e-6)))
    delta = float(np.arctan2(2.0 * wheelbase_m * np.sin(alpha), ld))
    # Clamp physical angle then normalize; negate for D-Racer sign.
    delta_c = float(np.clip(delta, -max_steer_angle_rad, max_steer_angle_rad))
    raw = float(
        np.clip(-delta_c / max(max_steer_angle_rad, 1e-6), -max_steer, max_steer)
    )
    return raw, alpha, delta_c


class LanePlanner:
    """Stateful Pure Pursuit + EMA + rate-limit for single-color lane follow."""

    def __init__(self, params: LaneControlParams | None = None):
        self.params = (params or load_control_params()).clamp()
        self._ema_steer = 0.0
        self._steer = 0.0
        self._last_conf = 0.0
        self.last_debug: dict[str, float] = {
            'raw': 0.0,
            'ema': 0.0,
            'steer': 0.0,
            'y_c': 0.0,
            'x_t': 0.0,
            'alpha': 0.0,
            'delta': 0.0,
            'conf': 0.0,
        }

    def reset(self) -> None:
        self._ema_steer = 0.0
        self._steer = 0.0
        self._last_conf = 0.0
        self.last_debug = {
            'raw': 0.0,
            'ema': 0.0,
            'steer': 0.0,
            'y_c': 0.0,
            'x_t': 0.0,
            'alpha': 0.0,
            'delta': 0.0,
            'conf': 0.0,
        }

    def set_params(self, params: LaneControlParams) -> None:
        self.params = params.clamp()

    def set_follow_color(self, color: str) -> None:
        self.params = replace(self.params, follow_color=color).clamp()
        self.reset()

    def effective_lookahead(self, speed_mps: float | None = None) -> float:
        p = self.params
        if p.lookahead_gain > 0.0 and speed_mps is not None and speed_mps > 0.0:
            return float(np.clip(p.lookahead_gain * speed_mps, 0.3, 1.5))
        return p.lookahead_x_m

    def step(
        self,
        detections: LaneDetections,
        *,
        speed_mps: float | None = None,
    ) -> LaneResult:
        p = self.params
        ld = self.effective_lookahead(speed_mps)
        y_c, conf = centerline_y_at_lookahead(
            detections,
            ld,
            p.track_half_width_m,
            follow_color=p.follow_color,
        )

        if y_c is None or conf < p.min_confidence:
            self._ema_steer *= p.hold_decay
            self._steer *= p.hold_decay
            self._last_conf *= p.hold_decay
            self.last_debug = {
                'raw': 0.0,
                'ema': float(self._ema_steer),
                'steer': float(self._steer),
                'y_c': float('nan'),
                'x_t': float(ld),
                'alpha': 0.0,
                'delta': 0.0,
                'conf': float(self._last_conf),
            }
            return LaneResult(
                steering_offset=float(np.clip(self._steer, -p.max_steer, p.max_steer)),
                confidence=float(self._last_conf),
                throttle_scale=1.0 if self._last_conf > p.min_confidence else 0.0,
            )

        raw, alpha, delta = pure_pursuit_steer(
            ld,
            y_c,
            wheelbase_m=p.wheelbase_m,
            lookahead_m=ld,
            max_steer_angle_rad=p.max_steer_angle_rad,
            max_steer=p.max_steer,
        )
        a = p.ema_alpha
        self._ema_steer = (1.0 - a) * self._ema_steer + a * raw
        step = float(
            np.clip(self._ema_steer - self._steer, -p.steer_rate_limit, p.steer_rate_limit)
        )
        self._steer = float(np.clip(self._steer + step, -p.max_steer, p.max_steer))
        self._last_conf = conf
        self.last_debug = {
            'raw': raw,
            'ema': float(self._ema_steer),
            'steer': float(self._steer),
            'y_c': float(y_c),
            'x_t': float(ld),
            'alpha': float(alpha),
            'delta': float(delta),
            'conf': float(conf),
        }

        scale = 1.0
        if abs(self._steer) > p.steer_slowdown_thresh:
            scale = p.steer_slowdown_scale

        return LaneResult(
            steering_offset=self._steer,
            confidence=conf,
            throttle_scale=scale,
        )


_PLANNER = LanePlanner()


def plan(detections: LaneDetections, planner: LanePlanner | None = None) -> LaneResult:
    """Module entry used by lane_detection / tests / tuner."""
    return (planner or _PLANNER).step(detections)


def get_shared_planner() -> LanePlanner:
    return _PLANNER


def mock_lane(
    y_left: float,
    y_right: float,
    *,
    color: int = LaneMarking.COLOR_WHITE,
    x0: float = 0.3,
    x1: float = 1.4,
    n: int = 8,
    confidence: float = 0.9,
) -> LaneDetections:
    """Synthetic straight L/R for unit tests."""
    xs = np.linspace(x0, x1, n, dtype=np.float32)
    left_pts = np.stack([xs, np.full(n, y_left, dtype=np.float32)], axis=1)
    right_pts = np.stack([xs, np.full(n, y_right, dtype=np.float32)], axis=1)
    is_yellow = color == LaneMarking.COLOR_YELLOW
    return LaneDetections(
        lanes=(
            LaneMarking(
                id=1,
                color=color,
                side_hint=LaneMarking.SIDE_LEFT,
                confidence=confidence,
                length=float(x1 - x0),
                points=left_pts,
            ),
            LaneMarking(
                id=2,
                color=color,
                side_hint=LaneMarking.SIDE_RIGHT,
                confidence=confidence,
                length=float(x1 - x0),
                points=right_pts,
            ),
        ),
        white_visible=not is_yellow,
        yellow_visible=is_yellow,
    )


def mock_white_lane(
    y_left: float,
    y_right: float,
    *,
    x0: float = 0.3,
    x1: float = 1.4,
    n: int = 8,
    confidence: float = 0.9,
) -> LaneDetections:
    """Synthetic straight white L/R for unit tests."""
    return mock_lane(
        y_left,
        y_right,
        color=LaneMarking.COLOR_WHITE,
        x0=x0,
        x1=x1,
        n=n,
        confidence=confidence,
    )
