"""White-lane path select + P/EMA steering — 담당: 안승현.

Consumes perception ``LaneDetections`` (white L/R only in Phase 2) and
returns pipeline-compatible ``LaneResult``. No HSV / IPM here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from inference.types import LaneDetections, LaneMarking, LaneResult


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
    lookahead_x_m: float = 0.80
    kp: float = 2.0
    ema_alpha: float = 0.40
    steer_rate_limit: float = 0.15
    max_steer: float = 1.0
    track_half_width_m: float = 0.175
    steer_slowdown_thresh: float = 0.50
    steer_slowdown_scale: float = 0.80
    min_confidence: float = 0.15
    hold_decay: float = 0.92

    def clamp(self) -> LaneControlParams:
        return LaneControlParams(
            lookahead_x_m=float(np.clip(self.lookahead_x_m, 0.3, 1.5)),
            kp=float(np.clip(self.kp, 0.1, 8.0)),
            ema_alpha=float(np.clip(self.ema_alpha, 0.05, 1.0)),
            steer_rate_limit=float(np.clip(self.steer_rate_limit, 0.01, 1.0)),
            max_steer=float(np.clip(self.max_steer, 0.1, 1.0)),
            track_half_width_m=float(np.clip(self.track_half_width_m, 0.05, 0.4)),
            steer_slowdown_thresh=float(np.clip(self.steer_slowdown_thresh, 0.1, 1.0)),
            steer_slowdown_scale=float(np.clip(self.steer_slowdown_scale, 0.2, 1.0)),
            min_confidence=float(np.clip(self.min_confidence, 0.0, 1.0)),
            hold_decay=float(np.clip(self.hold_decay, 0.5, 0.99)),
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
        kp=float(data.get('kp', 2.0)),
        ema_alpha=float(data.get('ema_alpha', 0.40)),
        steer_rate_limit=float(data.get('steer_rate_limit', 0.15)),
        max_steer=float(data.get('max_steer', 1.0)),
        track_half_width_m=float(data.get('track_half_width_m', 0.175)),
        steer_slowdown_thresh=float(data.get('steer_slowdown_thresh', 0.50)),
        steer_slowdown_scale=float(data.get('steer_slowdown_scale', 0.80)),
        min_confidence=float(data.get('min_confidence', 0.15)),
        hold_decay=float(data.get('hold_decay', 0.92)),
    ).clamp()


def save_control_params(params: LaneControlParams, path: Path | None = None) -> Path:
    cfg_path = path or default_control_config_path()
    p = params.clamp()
    payload = {
        'lookahead_x_m': p.lookahead_x_m,
        'kp': p.kp,
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
        f.write('# Lane planner / control gains (Phase 2+)\n')
        f.write('# Tuned with: scripts/vision_tune/tune_lane_control.py\n')
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
) -> tuple[float | None, float]:
    """Return (y_center, confidence) at look-ahead x using white L/R only."""
    left = detections.white_left()
    right = detections.white_right()
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
        # Left marking: center is to the right (−y in base_link? left marking has +y)
        # Lane left boundary at +y_l → center ≈ y_l - half_width
        return y_l - track_half_width_m, float(np.clip(conf, 0.0, 1.0))

    if y_r is not None:
        conf = max(right.confidence if right else 0.3, 0.3)
        # Right marking at y_r (negative) → center ≈ y_r + half_width
        return y_r + track_half_width_m, float(np.clip(conf, 0.0, 1.0))

    return None, 0.0


class LanePlanner:
    """Stateful P + EMA + rate-limit steering for white-lane follow."""

    def __init__(self, params: LaneControlParams | None = None):
        self.params = (params or load_control_params()).clamp()
        self._ema_steer = 0.0
        self._steer = 0.0
        self._last_conf = 0.0

    def reset(self) -> None:
        self._ema_steer = 0.0
        self._steer = 0.0
        self._last_conf = 0.0

    def set_params(self, params: LaneControlParams) -> None:
        self.params = params.clamp()

    def step(self, detections: LaneDetections) -> LaneResult:
        p = self.params
        y_c, conf = centerline_y_at_lookahead(
            detections,
            p.lookahead_x_m,
            p.track_half_width_m,
        )

        if y_c is None or conf < p.min_confidence:
            # Hold / decay toward zero when lost.
            self._ema_steer *= p.hold_decay
            self._steer *= p.hold_decay
            self._last_conf *= p.hold_decay
            return LaneResult(
                steering_offset=float(np.clip(self._steer, -p.max_steer, p.max_steer)),
                confidence=float(self._last_conf),
                throttle_scale=1.0 if self._last_conf > p.min_confidence else 0.0,
            )

        # base_link: +y = left. D-Racer: +steering = right.
        # Centerline left of vehicle (y_c > 0) → steer left (−).
        raw = float(np.clip(-p.kp * y_c, -p.max_steer, p.max_steer))
        a = p.ema_alpha
        self._ema_steer = (1.0 - a) * self._ema_steer + a * raw
        delta = float(np.clip(self._ema_steer - self._steer, -p.steer_rate_limit, p.steer_rate_limit))
        self._steer = float(np.clip(self._steer + delta, -p.max_steer, p.max_steer))
        self._last_conf = conf

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
    xs = np.linspace(x0, x1, n, dtype=np.float32)
    left_pts = np.stack([xs, np.full(n, y_left, dtype=np.float32)], axis=1)
    right_pts = np.stack([xs, np.full(n, y_right, dtype=np.float32)], axis=1)
    return LaneDetections(
        lanes=(
            LaneMarking(
                id=1,
                color=LaneMarking.COLOR_WHITE,
                side_hint=LaneMarking.SIDE_LEFT,
                confidence=confidence,
                length=float(x1 - x0),
                points=left_pts,
            ),
            LaneMarking(
                id=2,
                color=LaneMarking.COLOR_WHITE,
                side_hint=LaneMarking.SIDE_RIGHT,
                confidence=confidence,
                length=float(x1 - x0),
                points=right_pts,
            ),
        ),
        white_visible=True,
        yellow_visible=False,
    )
