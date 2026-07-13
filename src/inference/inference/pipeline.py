"""Synchronized perception and mission-level driving planner.

Perception modules return Python objects directly to :class:`MainPlanner`.
ROS topics are published by ``inference_node`` only for control and debugging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from inference.modules import aruco_detection, lane_detection, traffic_sign
from inference.types import (
    ControlCommand,
    DrivingState,
    PathSource,
    PipelineContext,
    PlannerOutput,
    RouteMode,
    TrafficSignal,
    TurnSign,
)


def default_planner_config_path() -> Path:
    """Find the source-workspace planner config used on PC and D3-G."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'config' / 'main_planner.yaml'
        if candidate.is_file():
            return candidate
    return Path('/home/topst/2026-SMH/config/main_planner.yaml')


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class PlannerConfig:
    route_mode: RouteMode = RouteMode.OUT
    prefer_yellow: bool = False
    default_out_branch_rank: int = 0
    sign_confirm_frames: int = 3
    fork_path_hold_frames: int = 3
    fork_reentry_cooldown_sec: float = 2.0
    perception_to_rear_axle_x_m: float = 0.0
    lookahead_m: float = 0.80
    curve_lookahead_m: float = 0.58
    curvature_full_scale: float = 1.20
    lookahead_shrink_rate_m: float = 0.08
    lookahead_grow_rate_m: float = 0.07
    curvature_near_m: float = 0.30
    curvature_mid_m: float = 0.60
    curvature_far_m: float = 0.90
    near_path_fit_span_m: float = 0.30
    near_path_extrapolation_max_m: float = 0.35
    wheelbase_m: float = 0.24
    max_steer_angle_rad: float = 0.5236
    max_steering_command: float = 0.75
    steering_rate_limit_per_sec: float = 4.5
    path_lost_steering_return_rate_per_sec: float = 2.5
    nominal_control_dt_sec: float = 0.10
    cte_gain: float = 0.12
    cte_softening_m: float = 0.20
    cte_deadband_m: float = 0.02
    max_cte_steering: float = 0.18
    heading_gain: float = 0.25
    heading_preview_m: float = 0.30
    heading_sample_span_m: float = 0.15
    max_heading_steering: float = 0.20
    cruise_throttle: float = 0.13
    curve_throttle: float = 0.07
    curve_steering_threshold: float = 0.50
    default_throttle: float = 0.0
    min_points: int = 5
    white_min_confidence: float = 0.15
    yellow_min_confidence: float = 0.15
    yellow_valid_on_frames: int = 3
    path_lost_hold_frames: int = 2
    path_lost_stop_frames: int = 10
    fork_exit_off_frames: int = 8
    roundabout_entry_on_yellow: bool = True
    min_lap_time_sec: float = 5.0
    exit_branch_rank: int = 0
    branch_required_events: int = 2
    crossing_required_events: int = 2
    branch_on_frames: int = 3
    branch_off_frames: int = 8
    crossing_on_frames: int = 2
    crossing_off_frames: int = 5
    roundabout_lookahead_m: float = 0.32
    roundabout_throttle: float = 0.06
    require_green_to_start: bool = False
    stop_on_red: bool = False
    command_watchdog_sec: float = 0.5
    log_state_changes: bool = True
    log_decision_changes: bool = True
    debug_publish_hz: float = 2.0


def load_planner_config(
    path: str | Path | None = None,
    *,
    route_mode: str | None = None,
) -> PlannerConfig:
    cfg_path = Path(path).expanduser() if path else default_planner_config_path()
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open('r', encoding='utf-8') as stream:
            loaded = yaml.safe_load(stream) or {}
        if isinstance(loaded, dict):
            data = loaded

    route = _section(data, 'route')
    pp = _section(data, 'pure_pursuit')
    speed = _section(data, 'speed')
    path_cfg = _section(data, 'path')
    rb = _section(data, 'roundabout')
    signals = _section(data, 'signals')
    safety = _section(data, 'safety')
    debug = _section(data, 'debug')
    selected_route = route_mode or route.get('mode', RouteMode.OUT.value)
    mode = RouteMode(str(selected_route).lower())
    # IN → yellow priority by default; OUT → white (prefer_yellow false).
    if mode is RouteMode.IN:
        prefer_yellow = bool(route.get('prefer_yellow', True))
    else:
        prefer_yellow = bool(route.get('prefer_yellow', False))
    return PlannerConfig(
        route_mode=mode,
        prefer_yellow=prefer_yellow,
        default_out_branch_rank=int(route.get('default_out_branch_rank', 0)),
        sign_confirm_frames=max(1, int(route.get('sign_confirm_frames', 3))),
        fork_path_hold_frames=max(0, int(route.get('fork_path_hold_frames', 3))),
        fork_reentry_cooldown_sec=max(
            0.0, float(route.get('fork_reentry_cooldown_sec', 2.0))
        ),
        perception_to_rear_axle_x_m=float(
            path_cfg.get('perception_to_rear_axle_x_m', 0.0)
        ),
        lookahead_m=max(0.05, float(pp.get('lookahead_m', 0.80))),
        curve_lookahead_m=max(0.05, float(pp.get('curve_lookahead_m', 0.58))),
        curvature_full_scale=max(
            0.01, float(pp.get('curvature_full_scale', 1.20))
        ),
        lookahead_shrink_rate_m=max(
            0.001, float(pp.get('lookahead_shrink_rate_m', 0.08))
        ),
        lookahead_grow_rate_m=max(
            0.001, float(pp.get('lookahead_grow_rate_m', 0.07))
        ),
        curvature_near_m=max(0.05, float(pp.get('curvature_near_m', 0.30))),
        curvature_mid_m=max(0.10, float(pp.get('curvature_mid_m', 0.60))),
        curvature_far_m=max(0.15, float(pp.get('curvature_far_m', 0.90))),
        near_path_fit_span_m=max(
            0.05, float(pp.get('near_path_fit_span_m', 0.30))
        ),
        near_path_extrapolation_max_m=max(
            0.0, float(pp.get('near_path_extrapolation_max_m', 0.35))
        ),
        wheelbase_m=max(0.01, float(pp.get('wheelbase_m', 0.24))),
        max_steer_angle_rad=max(0.01, float(pp.get('max_steer_angle_rad', 0.5236))),
        max_steering_command=float(
            np.clip(pp.get('max_steering_command', 0.75), 0.1, 1.0)
        ),
        steering_rate_limit_per_sec=max(
            0.01, float(pp.get('steering_rate_limit_per_sec', 4.5))
        ),
        path_lost_steering_return_rate_per_sec=max(
            0.01,
            float(pp.get('path_lost_steering_return_rate_per_sec', 2.5)),
        ),
        nominal_control_dt_sec=float(
            np.clip(pp.get('nominal_control_dt_sec', 0.10), 0.01, 0.25)
        ),
        cte_gain=max(0.0, float(pp.get('cte_gain', 0.12))),
        cte_softening_m=max(0.01, float(pp.get('cte_softening_m', 0.20))),
        cte_deadband_m=max(0.0, float(pp.get('cte_deadband_m', 0.02))),
        max_cte_steering=float(
            np.clip(pp.get('max_cte_steering', 0.18), 0.0, 1.0)
        ),
        heading_gain=max(0.0, float(pp.get('heading_gain', 0.25))),
        heading_preview_m=max(0.05, float(pp.get('heading_preview_m', 0.30))),
        heading_sample_span_m=max(
            0.02, float(pp.get('heading_sample_span_m', 0.15))
        ),
        max_heading_steering=float(
            np.clip(pp.get('max_heading_steering', 0.20), 0.0, 1.0)
        ),
        cruise_throttle=float(np.clip(speed.get('cruise_throttle', 0.13), -1.0, 1.0)),
        curve_throttle=float(np.clip(speed.get('curve_throttle', 0.07), -1.0, 1.0)),
        curve_steering_threshold=float(
            np.clip(speed.get('curve_steering_threshold', 0.50), 0.0, 1.0)
        ),
        default_throttle=float(np.clip(speed.get('default_throttle', 0.0), -1.0, 1.0)),
        min_points=max(2, int(path_cfg.get('min_points', 5))),
        white_min_confidence=float(np.clip(path_cfg.get('white_min_confidence', 0.15), 0.0, 1.0)),
        yellow_min_confidence=float(
            np.clip(path_cfg.get('yellow_min_confidence', 0.15), 0.0, 1.0)
        ),
        yellow_valid_on_frames=max(1, int(path_cfg.get('yellow_valid_on_frames', 3))),
        path_lost_hold_frames=max(
            0, int(path_cfg.get('path_lost_hold_frames', 2))
        ),
        path_lost_stop_frames=max(1, int(path_cfg.get('path_lost_stop_frames', 10))),
        fork_exit_off_frames=max(1, int(path_cfg.get('fork_exit_off_frames', 8))),
        roundabout_entry_on_yellow=bool(rb.get('entry_on_yellow', True)),
        min_lap_time_sec=max(0.0, float(rb.get('min_lap_time_sec', 5.0))),
        exit_branch_rank=int(rb.get('exit_branch_rank', 0)),
        branch_required_events=max(1, int(rb.get('branch_required_events', 2))),
        crossing_required_events=max(1, int(rb.get('crossing_required_events', 2))),
        branch_on_frames=max(1, int(rb.get('branch_on_frames', 3))),
        branch_off_frames=max(1, int(rb.get('branch_off_frames', 8))),
        crossing_on_frames=max(1, int(rb.get('crossing_on_frames', 2))),
        crossing_off_frames=max(1, int(rb.get('crossing_off_frames', 5))),
        roundabout_lookahead_m=max(
            0.05, float(rb.get('lookahead_m', 0.32))
        ),
        roundabout_throttle=float(
            np.clip(rb.get('throttle', 0.06), -1.0, 1.0)
        ),
        require_green_to_start=bool(signals.get('require_green_to_start', False)),
        stop_on_red=bool(signals.get('stop_on_red', False)),
        command_watchdog_sec=max(0.1, float(safety.get('command_watchdog_sec', 0.5))),
        log_state_changes=bool(debug.get('log_state_changes', True)),
        log_decision_changes=bool(debug.get('log_decision_changes', True)),
        debug_publish_hz=max(0.1, float(debug.get('publish_hz', 2.0))),
    )


class RisingEventCounter:
    """Debounced False→True event counter with independent re-arm timing."""

    def __init__(self, on_frames: int, off_frames: int):
        self.on_frames = max(1, int(on_frames))
        self.off_frames = max(1, int(off_frames))
        self.events = 0
        self._on_count = 0
        self._off_count = 0
        self.latched = False

    def reset(self) -> None:
        self.events = 0
        self._on_count = 0
        self._off_count = 0
        self.latched = False

    def update(self, visible: bool) -> bool:
        if visible:
            self._on_count += 1
            self._off_count = 0
            if not self.latched and self._on_count >= self.on_frames:
                self.latched = True
                self.events += 1
                return True
        else:
            self._off_count += 1
            self._on_count = 0
            if self.latched and self._off_count >= self.off_frames:
                self.latched = False
        return False


@dataclass(frozen=True)
class PursuitResult:
    valid: bool
    steering: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    path_points: int = 0
    target_distance: float = 0.0
    lookahead_m: float = 0.0
    desired_lookahead_m: float = 0.0
    path_curvature: float = 0.0
    curve_ratio: float = 0.0
    raw_steering: float = 0.0
    pp_steering: float = 0.0
    cross_track_error_m: float = 0.0
    cte_steering: float = 0.0
    heading_error_rad: float = 0.0
    heading_steering: float = 0.0
    path_extrapolation_m: float = 0.0


class MainPlanner:
    """Own all mission state and the single final steering/throttle decision."""

    def __init__(self, config: PlannerConfig | None = None, *, steer_trim: float = 0.0):
        self.config = config or load_planner_config()
        self.steer_trim = float(np.clip(steer_trim, -1.0, 1.0))
        self.state = (
            DrivingState.WAIT_GREEN
            if self.config.require_green_to_start
            else DrivingState.NORMAL
        )
        self.desired_turn = TurnSign.UNKNOWN
        self._sign_candidate = TurnSign.UNKNOWN
        self._sign_candidate_frames = 0
        self._fork_locked_turn = TurnSign.UNKNOWN
        self._fork_selected_rank: int | None = None
        self._fork_selection_reason = 'none'
        self._fork_cached_path = np.empty((0, 2), dtype=np.float32)
        self._fork_cached_source = PathSource.NONE
        self._fork_cached_confidence = 0.0
        self._fork_cooldown_until_sec = 0.0
        self.branch_counter = RisingEventCounter(
            self.config.branch_on_frames, self.config.branch_off_frames
        )
        self.crossing_counter = RisingEventCounter(
            self.config.crossing_on_frames, self.config.crossing_off_frames
        )
        self._roundabout_started_at: float | None = None
        self._yellow_valid_frames = 0
        self._path_lost_frames = 0
        self._fork_absent_frames = 0
        self._steering = 0.0
        self._lookahead_m = self.config.lookahead_m
        self._last_path_source = PathSource.NONE
        self._last_step_time_sec: float | None = None

    def neutralize_steering(self) -> None:
        """Synchronize planner state with an externally forced neutral command."""
        self._steering = 0.0

    def _step_dt(self, now_sec: float) -> float:
        if self._last_step_time_sec is None or now_sec <= self._last_step_time_sec:
            dt = self.config.nominal_control_dt_sec
        else:
            dt = now_sec - self._last_step_time_sec
        self._last_step_time_sec = now_sec
        # Avoid a long callback pause allowing an unconstrained steering jump.
        return float(np.clip(dt, 0.001, 0.25))

    def _return_steering_to_neutral(self, dt_sec: float) -> float:
        maximum_delta = self.config.path_lost_steering_return_rate_per_sec * dt_sec
        if abs(self._steering) <= maximum_delta:
            self._steering = 0.0
        else:
            self._steering -= math.copysign(maximum_delta, self._steering)
        return self._steering

    def _set_state(self, state: DrivingState, now_sec: float) -> None:
        if state == self.state:
            return
        previous_state = self.state
        self.state = state
        if state == DrivingState.ROUNDABOUT_CIRCLE:
            self._roundabout_started_at = now_sec
            self.branch_counter.reset()
            self.crossing_counter.reset()
            self._lookahead_m = self.config.roundabout_lookahead_m
        if state in (DrivingState.NORMAL, DrivingState.FINISHED):
            self._fork_absent_frames = 0
        if previous_state is DrivingState.FORK_TURN and state is DrivingState.NORMAL:
            self.desired_turn = TurnSign.UNKNOWN
            self._sign_candidate = TurnSign.UNKNOWN
            self._sign_candidate_frames = 0
            self._fork_locked_turn = TurnSign.UNKNOWN
            self._fork_selected_rank = None
            self._fork_selection_reason = 'none'
            self._fork_cached_path = np.empty((0, 2), dtype=np.float32)
            self._fork_cached_source = PathSource.NONE
            self._fork_cached_confidence = 0.0
            self._fork_cooldown_until_sec = (
                now_sec + self.config.fork_reentry_cooldown_sec
            )
            self.branch_counter.reset()

    def _update_desired_turn(self, observed: TurnSign) -> None:
        """Confirm a direction sign before latching; freeze it inside a fork."""
        if self.state is DrivingState.FORK_TURN:
            return
        if observed is TurnSign.UNKNOWN:
            self._sign_candidate = TurnSign.UNKNOWN
            self._sign_candidate_frames = 0
            return
        if observed is self._sign_candidate:
            self._sign_candidate_frames += 1
        else:
            self._sign_candidate = observed
            self._sign_candidate_frames = 1
        if self._sign_candidate_frames >= self.config.sign_confirm_frames:
            self.desired_turn = observed

    def _lock_fork_selection(self) -> None:
        """Freeze turn and branch rank for the complete fork manoeuvre.

        Contract: LEFT → ``lateral_rank`` / index **0**, RIGHT → **1**
        (exactly two layers). ``default_out_branch_rank`` only if sign unknown.
        """
        self._fork_locked_turn = self.desired_turn
        if self._fork_locked_turn is TurnSign.LEFT:
            self._fork_selected_rank = 0
            self._fork_selection_reason = 'sign_left'
        elif self._fork_locked_turn is TurnSign.RIGHT:
            self._fork_selected_rank = 1
            self._fork_selection_reason = 'sign_right'
        else:
            self._fork_selected_rank = int(self.config.default_out_branch_rank)
            self._fork_selection_reason = 'default_unknown'

    def force_fork_choice(
        self,
        turn: TurnSign,
        *,
        state: DrivingState | None = None,
    ) -> None:
        """Test/sim helper: latch turn + rank and optionally jump FSM state."""
        self.desired_turn = turn
        self._sign_candidate = turn
        self._sign_candidate_frames = self.config.sign_confirm_frames
        self._lock_fork_selection()
        if state is not None:
            self.state = state
            if state is DrivingState.FORK_TURN:
                self._fork_absent_frames = 0
            if state is DrivingState.ROUNDABOUT_EXIT:
                self._fork_absent_frames = 0

    def _selected_layer_path(
        self, lane: Any, rank: int
    ) -> tuple[np.ndarray, PathSource, float]:
        """PP path for one fork layer only (other branch ignored)."""
        return self._branch_path(lane, rank)

    def _color_or_selected_resume(
        self, lane: Any, color_path: np.ndarray, color_source: PathSource, color_conf: float
    ) -> tuple[np.ndarray, PathSource, float, str]:
        """After fork flicker: keep selected layer if still published, else color."""
        rank = self._fork_selected_rank
        if rank is not None and lane.fork_active and len(getattr(lane, 'branches', ())) >= 2:
            path, source, conf = self._selected_layer_path(lane, int(rank))
            if path.shape[0] >= self.config.min_points:
                return path, source, conf, 'selected_layer_resume'
        if color_path.shape[0] >= self.config.min_points:
            return color_path, color_source, color_conf, 'color_resume'
        return (
            np.empty((0, 2), dtype=np.float32),
            PathSource.NONE,
            0.0,
            'resume_lost',
        )

    def _valid_centerline(self, points: Any, confidence: float, minimum: float) -> bool:
        array = np.asarray(points, dtype=np.float32)
        return (
            confidence >= minimum
            and array.ndim == 2
            and array.shape[0] >= self.config.min_points
            and array.shape[1] >= 2
            and np.isfinite(array[:, :2]).all()
        )

    def _color_path(self, lane: Any) -> tuple[np.ndarray, PathSource, float]:
        yellow_valid = self._valid_centerline(
            lane.yellow_centerline,
            float(lane.yellow_confidence),
            self.config.yellow_min_confidence,
        )
        self._yellow_valid_frames = (
            self._yellow_valid_frames + 1 if yellow_valid else 0
        )
        if (
            self.config.route_mode is RouteMode.IN
            and self.config.prefer_yellow
            and self._yellow_valid_frames >= self.config.yellow_valid_on_frames
        ):
            return (
                np.asarray(lane.yellow_centerline, dtype=np.float32),
                PathSource.YELLOW_CENTERLINE,
                float(lane.yellow_confidence),
            )

        if self._valid_centerline(
            lane.white_centerline,
            float(lane.white_confidence),
            self.config.white_min_confidence,
        ):
            return (
                np.asarray(lane.white_centerline, dtype=np.float32),
                PathSource.WHITE_CENTERLINE,
                float(lane.white_confidence),
            )
        return np.empty((0, 2), dtype=np.float32), PathSource.NONE, 0.0

    def _branch_path(self, lane: Any, rank: int) -> tuple[np.ndarray, PathSource, float]:
        branch = self._ranked_branch(lane, rank)
        if branch is None:
            return np.empty((0, 2), dtype=np.float32), PathSource.NONE, 0.0
        # Prefer explicit lateral_rank when present (WonTae / fork pairs).
        source = PathSource.LEFT_BRANCH if int(rank) == 0 else PathSource.RIGHT_BRANCH
        return np.asarray(branch.points, dtype=np.float32), source, float(branch.confidence)

    @staticmethod
    def _ranked_branch(lane: Any, rank: int) -> Any | None:
        """Pick branch by ``lateral_rank`` when set, else list index (0=left, 1=right)."""
        branches = list(getattr(lane, 'branches', ()))
        if not branches:
            return None
        # Match lateral_rank first (stable if list order ever changes).
        for branch in branches:
            if hasattr(branch, 'lateral_rank') and int(branch.lateral_rank) == int(rank):
                return branch
        index = rank if rank >= 0 else len(branches) + rank
        if index < 0 or index >= len(branches):
            return None
        return branches[index]

    @staticmethod
    def _ordered_path(path: np.ndarray) -> np.ndarray:
        """Return finite rear-axle-frame points ordered from near to far."""
        points = np.asarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float32)
        xy = points[:, :2]
        xy = xy[np.isfinite(xy).all(axis=1) & (xy[:, 0] > 0.0)]
        if xy.shape[0] >= 2 and np.linalg.norm(xy[0]) > np.linalg.norm(xy[-1]):
            xy = xy[::-1]
        if xy.shape[0] >= 2:
            keep = np.concatenate(
                ([True], np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-5)
            )
            xy = xy[keep]
        return xy

    def _path_in_rear_axle_frame(self, path: np.ndarray) -> np.ndarray:
        """Translate a perception path into the PP rear-axle frame.

        Metric IPM currently measures ground distance from the camera. The
        configured positive x offset is therefore added to every path type
        before PP, curvature, heading and CTE consume it. Keeping this at the
        planner boundary prevents control terms from using different origins.
        """
        points = np.asarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float32)
        xy = points[:, :2].copy()
        xy[:, 0] += self.config.perception_to_rear_axle_x_m
        return xy

    def _extend_path_to_front_axle(
        self, xy: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Curve-aware extrapolation across the camera's near blind region.

        PP is referenced at the rear axle and CTE at the front axle, while the
        first IPM observation is ahead of the camera. Fit only the nearest
        observed section and extend it back to the front axle. This preserves
        the local turn direction instead of drawing a straight ray from the
        rear axle to the first visible point.
        """
        if xy.shape[0] < 2:
            return xy, 0.0

        first_x = float(xy[0, 0])
        front_x = self.config.wheelbase_m
        gap = first_x - front_x
        if gap <= 1e-4 or gap > self.config.near_path_extrapolation_max_m:
            return xy, 0.0

        local = xy[
            (xy[:, 0] >= first_x - 1e-5)
            & (xy[:, 0] <= first_x + self.config.near_path_fit_span_m)
        ].astype(np.float64)
        if local.shape[0] < 2:
            return xy, 0.0

        # Duplicate x samples make a polynomial ill-conditioned. Keep their
        # mean lateral position while retaining the near-to-far ordering.
        unique_x, inverse = np.unique(local[:, 0], return_inverse=True)
        if unique_x.size < 2:
            return xy, 0.0
        sums = np.bincount(inverse, weights=local[:, 1])
        counts = np.bincount(inverse)
        unique_y = sums / np.maximum(counts, 1)

        origin_x = first_x
        fit_x = unique_x - origin_x
        degree = 2 if unique_x.size >= 3 else 1
        coefficients = np.polynomial.polynomial.polyfit(
            fit_x,
            unique_y,
            degree,
        )

        observed_steps = np.diff(unique_x)
        positive_steps = observed_steps[observed_steps > 1e-5]
        step = float(np.median(positive_steps)) if positive_steps.size else 0.02
        sample_count = max(2, int(math.ceil(gap / max(step, 0.01))))
        extension_x = np.linspace(
            front_x,
            first_x,
            sample_count,
            endpoint=False,
            dtype=np.float64,
        )
        extension_y = np.polynomial.polynomial.polyval(
            extension_x - origin_x,
            coefficients,
        )
        extension = np.column_stack((extension_x, extension_y)).astype(np.float32)
        if not np.isfinite(extension).all():
            return xy, 0.0
        return np.vstack((extension, xy)), gap

    @staticmethod
    def _target_at_radius(xy: np.ndarray, radius: float) -> np.ndarray | None:
        """Intersect an ordered path polyline with a vehicle-centred LD circle."""
        if xy.shape[0] == 0:
            return None
        radius = max(1e-4, float(radius))
        norms = np.linalg.norm(xy, axis=1)

        # If the visible path starts beyond LD, extend the first ray toward the
        # vehicle. This keeps target distance deterministic instead of jumping.
        if norms[0] >= radius:
            return (xy[0] * (radius / max(norms[0], 1e-6))).astype(np.float32)

        for index in range(xy.shape[0] - 1):
            p0 = xy[index].astype(np.float64)
            p1 = xy[index + 1].astype(np.float64)
            if norms[index] > radius or norms[index + 1] < radius:
                continue
            direction = p1 - p0
            a = float(np.dot(direction, direction))
            b = 2.0 * float(np.dot(p0, direction))
            c = float(np.dot(p0, p0) - radius * radius)
            if a <= 1e-12:
                continue
            discriminant = max(0.0, b * b - 4.0 * a * c)
            roots = (
                (-b - math.sqrt(discriminant)) / (2.0 * a),
                (-b + math.sqrt(discriminant)) / (2.0 * a),
            )
            valid = [root for root in roots if 0.0 <= root <= 1.0]
            if valid:
                return (p0 + min(valid) * direction).astype(np.float32)

        return xy[-1].astype(np.float32)

    def _estimate_path_curvature(self, xy: np.ndarray) -> float:
        """Signed three-point curvature ahead of the vehicle (1/metre)."""
        samples = [
            self._target_at_radius(xy, distance)
            for distance in (
                self.config.curvature_near_m,
                self.config.curvature_mid_m,
                self.config.curvature_far_m,
            )
        ]
        if any(point is None for point in samples):
            return 0.0
        p1, p2, p3 = (np.asarray(point, dtype=np.float64) for point in samples)
        a = float(np.linalg.norm(p2 - p1))
        b = float(np.linalg.norm(p3 - p2))
        c = float(np.linalg.norm(p3 - p1))
        denominator = a * b * c
        if denominator <= 1e-8:
            return 0.0
        cross = float(np.cross(p2 - p1, p3 - p1))
        return 2.0 * cross / denominator

    def _adaptive_lookahead(self, curvature: float) -> tuple[float, float]:
        ratio = float(
            np.clip(abs(curvature) / self.config.curvature_full_scale, 0.0, 1.0)
        )
        desired = (
            self.config.lookahead_m * (1.0 - ratio)
            + self.config.curve_lookahead_m * ratio
        )
        delta = desired - self._lookahead_m
        if delta < 0.0:
            delta = max(delta, -self.config.lookahead_shrink_rate_m)
        else:
            delta = min(delta, self.config.lookahead_grow_rate_m)
        self._lookahead_m = float(
            np.clip(
                self._lookahead_m + delta,
                min(self.config.lookahead_m, self.config.curve_lookahead_m),
                max(self.config.lookahead_m, self.config.curve_lookahead_m),
            )
        )
        return desired, ratio

    def _cross_track_error(self, xy: np.ndarray) -> float:
        """Signed distance from the front axle to a rear-frame path segment.

        Positive means the reference path is to the vehicle's left.  Using a
        segment projection instead of a path point keeps the error insensitive
        to centerline sampling density.
        """
        if xy.shape[0] < 2:
            return 0.0
        # PP and this Stanley-shaped term share a rear-axle coordinate frame.
        # In that frame the front axle is exactly one wheelbase forward.
        front_axle = np.array([self.config.wheelbase_m, 0.0], dtype=np.float64)
        starts = xy[:-1].astype(np.float64)
        segments = xy[1:].astype(np.float64) - starts
        lengths_sq = np.einsum('ij,ij->i', segments, segments)
        valid = lengths_sq > 1e-8
        if not np.any(valid):
            return 0.0
        starts = starts[valid]
        segments = segments[valid]
        lengths_sq = lengths_sq[valid]
        ratios = np.einsum('ij,ij->i', front_axle - starts, segments) / lengths_sq
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = starts + ratios[:, None] * segments
        errors = projections - front_axle
        closest = int(np.argmin(np.einsum('ij,ij->i', errors, errors)))
        tangent = segments[closest] / math.sqrt(lengths_sq[closest])
        error = errors[closest]
        return float(tangent[0] * error[1] - tangent[1] * error[0])

    def _cte_correction(self, cte: float) -> float:
        effective_cte = math.copysign(
            max(0.0, abs(cte) - self.config.cte_deadband_m), cte
        )
        angle = math.atan2(
            self.config.cte_gain * effective_cte,
            self.config.cte_softening_m,
        )
        # Positive CTE means path-left; D-Racer uses negative steering for left.
        normalized = -angle / self.config.max_steer_angle_rad
        return float(
            np.clip(
                normalized,
                -self.config.max_cte_steering,
                self.config.max_cte_steering,
            )
        )

    def _heading_correction(self, xy: np.ndarray) -> tuple[float, float]:
        """Return path heading error and bounded D-Racer steering correction."""
        near = self._target_at_radius(xy, self.config.heading_preview_m)
        far = self._target_at_radius(
            xy,
            self.config.heading_preview_m + self.config.heading_sample_span_m,
        )
        if near is None or far is None:
            return 0.0, 0.0
        tangent = np.asarray(far, dtype=np.float64) - np.asarray(
            near, dtype=np.float64
        )
        if float(np.linalg.norm(tangent)) <= 1e-6:
            return 0.0, 0.0
        # Vehicle heading is zero in base_link; +angle means path turns left.
        error = math.atan2(float(tangent[1]), float(tangent[0]))
        normalized = (
            -self.config.heading_gain * error / self.config.max_steer_angle_rad
        )
        correction = float(
            np.clip(
                normalized,
                -self.config.max_heading_steering,
                self.config.max_heading_steering,
            )
        )
        return error, correction

    def _pure_pursuit(
        self, path: np.ndarray, dt_sec: float | None = None
    ) -> PursuitResult:
        rear_axle_path = self._path_in_rear_axle_frame(path)
        xy = self._ordered_path(rear_axle_path)
        if xy.shape[0] < self.config.min_points:
            return PursuitResult(False, path_points=int(xy.shape[0]))
        observed_path_points = int(xy.shape[0])
        xy, path_extrapolation_m = self._extend_path_to_front_axle(xy)
        path_curvature = self._estimate_path_curvature(xy)
        in_roundabout = self.state in (
            DrivingState.ROUNDABOUT_CIRCLE,
            DrivingState.ROUNDABOUT_EXIT_READY,
            DrivingState.ROUNDABOUT_EXIT,
        )
        if in_roundabout:
            desired_lookahead = self.config.roundabout_lookahead_m
            curve_ratio = float(
                np.clip(
                    abs(path_curvature) / self.config.curvature_full_scale,
                    0.0,
                    1.0,
                )
            )
            self._lookahead_m = desired_lookahead
        else:
            desired_lookahead, curve_ratio = self._adaptive_lookahead(path_curvature)
        target = self._target_at_radius(xy, self._lookahead_m)
        if target is None:
            return PursuitResult(False, path_points=int(xy.shape[0]))
        target_x, target_y = (float(value) for value in target)
        distance_sq = target_x * target_x + target_y * target_y
        if distance_sq <= 1e-6:
            return PursuitResult(False, path_points=int(xy.shape[0]))
        curvature = 2.0 * target_y / distance_sq
        steer_angle = math.atan(self.config.wheelbase_m * curvature)
        # base_link +y is left; D-Racer normalized steering is negative for left.
        pp_steering = -steer_angle / self.config.max_steer_angle_rad
        cte = self._cross_track_error(xy)
        cte_steering = self._cte_correction(cte)
        heading_error, heading_steering = self._heading_correction(xy)
        raw = pp_steering + heading_steering + cte_steering
        raw = float(
            np.clip(
                raw,
                -self.config.max_steering_command,
                self.config.max_steering_command,
            )
        )
        dt = self.config.nominal_control_dt_sec if dt_sec is None else max(0.0, dt_sec)
        maximum_delta = self.config.steering_rate_limit_per_sec * dt
        delta = float(
            np.clip(
                raw - self._steering,
                -maximum_delta,
                maximum_delta,
            )
        )
        self._steering = float(np.clip(self._steering + delta, -1.0, 1.0))
        return PursuitResult(
            valid=True,
            steering=self._steering,
            target_x=target_x,
            target_y=target_y,
            path_points=observed_path_points,
            target_distance=math.sqrt(distance_sq),
            lookahead_m=self._lookahead_m,
            desired_lookahead_m=desired_lookahead,
            path_curvature=path_curvature,
            curve_ratio=curve_ratio,
            raw_steering=raw,
            pp_steering=pp_steering,
            cross_track_error_m=cte,
            cte_steering=cte_steering,
            heading_error_rad=heading_error,
            heading_steering=heading_steering,
            path_extrapolation_m=path_extrapolation_m,
        )

    def _stop(self) -> ControlCommand:
        steering = float(np.clip(self._steering + self.steer_trim, -1.0, 1.0))
        return ControlCommand(steering=steering, throttle=0.0)

    def _drive(self, pursuit: PursuitResult) -> ControlCommand:
        if self.state in (
            DrivingState.ROUNDABOUT_CIRCLE,
            DrivingState.ROUNDABOUT_EXIT_READY,
            DrivingState.ROUNDABOUT_EXIT,
        ):
            throttle = self.config.roundabout_throttle
        else:
            throttle = (
                self.config.cruise_throttle * (1.0 - pursuit.curve_ratio)
                + self.config.curve_throttle * pursuit.curve_ratio
            )
        steering = float(np.clip(pursuit.steering + self.steer_trim, -1.0, 1.0))
        return ControlCommand(steering=steering, throttle=throttle)

    def step(self, frame: np.ndarray, *, now_sec: float) -> PlannerOutput:
        dt_sec = self._step_dt(now_sec)
        lane = lane_detection.detect(frame)
        traffic = traffic_sign.detect(frame)
        aruco = aruco_detection.detect(frame)

        self._update_desired_turn(traffic.turn)

        branch_event = self.branch_counter.update(bool(lane.fork_active))
        crossing_event = self.crossing_counter.update(bool(lane.yellow_crossing_line))
        elapsed = (
            0.0
            if self._roundabout_started_at is None
            else max(0.0, now_sec - self._roundabout_started_at)
        )

        if self.state is DrivingState.WAIT_GREEN and traffic.signal is TrafficSignal.GREEN:
            self._set_state(DrivingState.NORMAL, now_sec)

        color_path, path_source, path_confidence = self._color_path(lane)

        if (
            self.config.route_mode is RouteMode.IN
            and self.state is DrivingState.NORMAL
            and self.config.roundabout_entry_on_yellow
            and (
                path_source is PathSource.YELLOW_CENTERLINE
                or bool(lane.yellow_crossing_line)
            )
        ):
            self._set_state(DrivingState.ROUNDABOUT_CIRCLE, now_sec)
            elapsed = 0.0

        if (
            self.state is DrivingState.NORMAL
            and self.config.route_mode is RouteMode.OUT
            and branch_event
            and now_sec >= self._fork_cooldown_until_sec
        ):
            self._lock_fork_selection()
            self._set_state(DrivingState.FORK_TURN, now_sec)

        if self.state is DrivingState.ROUNDABOUT_CIRCLE:
            enough_time = elapsed >= self.config.min_lap_time_sec
            branch_ready = self.branch_counter.events >= self.config.branch_required_events
            crossing_ready = self.crossing_counter.events >= self.config.crossing_required_events
            if enough_time and (branch_ready or crossing_ready):
                self._set_state(DrivingState.ROUNDABOUT_EXIT_READY, now_sec)

        if self.state is DrivingState.ROUNDABOUT_EXIT_READY:
            if self._fork_selected_rank is None:
                self._lock_fork_selection()
            exit_rank = int(
                self._fork_selected_rank
                if self._fork_selected_rank is not None
                else self.config.exit_branch_rank
            )
            branch = self._ranked_branch(lane, exit_rank)
            if branch is not None and len(lane.branches) >= 2:
                self._set_state(DrivingState.ROUNDABOUT_EXIT, now_sec)

        pursuit = PursuitResult(False)
        selected_branch_rank: int | None = None
        branch_selection_reason = 'none'
        if aruco.should_stop:
            command = self._stop()
            path_source = PathSource.STOP
            decision = 'aruco_stop'
            pursuit = PursuitResult(False)
        elif self.state is DrivingState.WAIT_GREEN:
            command = self._stop()
            path_source = PathSource.STOP
            decision = 'wait_green'
            pursuit = PursuitResult(False)
        elif self.config.stop_on_red and traffic.signal is TrafficSignal.RED:
            command = self._stop()
            path_source = PathSource.STOP
            decision = 'red_signal_stop'
            pursuit = PursuitResult(False)
        elif self.state is DrivingState.FORK_TURN:
            if self._fork_selected_rank is None:
                self._lock_fork_selection()
            rank = int(self._fork_selected_rank)
            selected_branch_rank = rank
            branch_selection_reason = self._fork_selection_reason
            if lane.fork_active and len(lane.branches) >= 2:
                # Selected layer only — ignore the other fork path.
                path, path_source, path_confidence = self._selected_layer_path(
                    lane, rank
                )
                self._fork_cached_path = path.copy()
                self._fork_cached_source = path_source
                self._fork_cached_confidence = path_confidence
                pursuit = self._pure_pursuit(path, dt_sec)
                decision = f'out_fork_{self._fork_locked_turn.value}'
                self._fork_absent_frames = 0
            else:
                self._fork_absent_frames += 1
                if (
                    self._fork_cached_path.shape[0] >= self.config.min_points
                    and self._fork_absent_frames <= self.config.fork_path_hold_frames
                ):
                    path_source = self._fork_cached_source
                    path_confidence = self._fork_cached_confidence
                    pursuit = self._pure_pursuit(self._fork_cached_path, dt_sec)
                    decision = 'out_fork_cached_branch'
                else:
                    path, path_source, path_confidence, resume = (
                        self._color_or_selected_resume(
                            lane, color_path, path_source, path_confidence
                        )
                    )
                    pursuit = self._pure_pursuit(path, dt_sec)
                    decision = f'out_fork_{resume}'
            if self._fork_absent_frames >= self.config.fork_exit_off_frames:
                self._set_state(DrivingState.NORMAL, now_sec)
        elif self.state is DrivingState.ROUNDABOUT_EXIT:
            if self._fork_selected_rank is None:
                self._lock_fork_selection()
            rank = int(
                self._fork_selected_rank
                if self._fork_selected_rank is not None
                else self.config.exit_branch_rank
            )
            selected_branch_rank = rank
            branch_selection_reason = (
                self._fork_selection_reason
                if self._fork_selection_reason != 'none'
                else 'roundabout_exit'
            )
            if lane.fork_active and len(lane.branches) >= 2:
                path, path_source, path_confidence = self._selected_layer_path(
                    lane, rank
                )
                self._fork_cached_path = path.copy()
                self._fork_cached_source = path_source
                self._fork_cached_confidence = path_confidence
                pursuit = self._pure_pursuit(path, dt_sec)
                decision = f'roundabout_exit_rank{rank}'
                self._fork_absent_frames = 0
            else:
                self._fork_absent_frames += 1
                if (
                    self._fork_cached_path.shape[0] >= self.config.min_points
                    and self._fork_absent_frames <= self.config.fork_path_hold_frames
                ):
                    path_source = self._fork_cached_source
                    path_confidence = self._fork_cached_confidence
                    pursuit = self._pure_pursuit(self._fork_cached_path, dt_sec)
                    decision = 'roundabout_exit_cached'
                else:
                    path, path_source, path_confidence, resume = (
                        self._color_or_selected_resume(
                            lane, color_path, path_source, path_confidence
                        )
                    )
                    pursuit = self._pure_pursuit(path, dt_sec)
                    decision = f'roundabout_exit_{resume}'
            if self._fork_absent_frames >= self.config.fork_exit_off_frames:
                self._set_state(DrivingState.NORMAL, now_sec)
        else:
            pursuit = self._pure_pursuit(color_path, dt_sec)
            decision = (
                'roundabout_circle'
                if self.state is DrivingState.ROUNDABOUT_CIRCLE
                else 'roundabout_exit_wait_branch'
                if self.state is DrivingState.ROUNDABOUT_EXIT_READY
                else 'normal_lane_follow'
            )

        if pursuit.valid and path_source is not PathSource.STOP:
            self._path_lost_frames = 0
            command = self._drive(pursuit)
        elif path_source is not PathSource.STOP:
            self._path_lost_frames += 1
            if self._path_lost_frames <= self.config.path_lost_hold_frames:
                # A one-frame centerline dropout must not erase a valid corner
                # command before the actuator receives it. Hold briefly, then
                # return toward neutral if perception does not recover.
                held_steering = self._steering
                loss_action = 'hold'
            else:
                held_steering = self._return_steering_to_neutral(dt_sec)
                loss_action = 'return'
            command = (
                ControlCommand(
                    float(np.clip(held_steering + self.steer_trim, -1.0, 1.0)),
                    self.config.default_throttle,
                )
                if self._path_lost_frames >= self.config.path_lost_stop_frames
                else ControlCommand(
                    float(np.clip(held_steering + self.steer_trim, -1.0, 1.0)),
                    self.config.curve_throttle,
                )
            )
            path_source = PathSource.HOLD_PREVIOUS
            decision = f'{decision}_path_lost_{loss_action}'

        self._last_path_source = path_source
        debug = {
            'route': self.config.route_mode.value,
            'prefer_yellow': self.config.prefer_yellow,
            'state': self.state.value,
            'decision': decision,
            'path_source': path_source.value,
            'path_confidence': round(float(path_confidence), 3),
            'path_points': pursuit.path_points,
            'target_x': round(pursuit.target_x, 3),
            'target_y': round(pursuit.target_y, 3),
            'target_distance': round(pursuit.target_distance, 3),
            'lookahead_m': round(pursuit.lookahead_m, 3),
            'desired_lookahead_m': round(pursuit.desired_lookahead_m, 3),
            'path_curvature': round(pursuit.path_curvature, 3),
            'curve_ratio': round(pursuit.curve_ratio, 3),
            'raw_steering': round(pursuit.raw_steering, 3),
            'pp_steering': round(pursuit.pp_steering, 3),
            'cross_track_error_m': round(pursuit.cross_track_error_m, 3),
            'cte_steering': round(pursuit.cte_steering, 3),
            'heading_error_rad': round(pursuit.heading_error_rad, 3),
            'heading_steering': round(pursuit.heading_steering, 3),
            'path_extrapolation_m': round(pursuit.path_extrapolation_m, 3),
            'white_visible': bool(lane.white_visible),
            'white_confidence': round(float(lane.white_confidence), 3),
            'yellow_visible': bool(lane.yellow_visible),
            'yellow_confidence': round(float(lane.yellow_confidence), 3),
            'yellow_selected': path_source is PathSource.YELLOW_CENTERLINE,
            'fork_active': bool(lane.fork_active),
            'branch_count': len(lane.branches),
            'branch_event': branch_event,
            'branch_events': self.branch_counter.events,
            'crossing_active': bool(lane.yellow_crossing_line),
            'crossing_event': crossing_event,
            'crossing_events': self.crossing_counter.events,
            'roundabout_elapsed_sec': round(elapsed, 2),
            'traffic_signal': traffic.signal.value,
            'turn_sign': traffic.turn.value,
            'desired_turn': self.desired_turn.value,
            'sign_candidate': self._sign_candidate.value,
            'sign_candidate_frames': self._sign_candidate_frames,
            'fork_locked_turn': self._fork_locked_turn.value,
            'selected_branch_rank': selected_branch_rank,
            'branch_selection_reason': branch_selection_reason,
            'aruco_detected': aruco.detected,
            'aruco_stop': aruco.should_stop,
            'steering': round(command.steering, 3),
            'control_dt_sec': round(dt_sec, 4),
            'throttle': round(command.throttle, 3),
        }
        return PlannerOutput(
            command=command,
            lane=lane,
            traffic=traffic,
            aruco=aruco,
            state=self.state,
            path_source=path_source,
            decision=decision,
            debug=debug,
        )


def fuse_control(
    ctx: PipelineContext,
    *,
    steer_trim: float = 0.0,
    default_throttle: float = 0.0,
    cruise_throttle: float = 0.35,
) -> ControlCommand:
    """Legacy stateless fusion kept for focused unit-test compatibility."""
    if ctx.aruco.should_stop or ctx.traffic.signal is TrafficSignal.RED:
        return ControlCommand(steering=steer_trim, throttle=0.0)
    steering = float(np.clip(steer_trim + ctx.lane.steering_offset, -1.0, 1.0))
    throttle = (
        cruise_throttle * float(np.clip(ctx.lane.throttle_scale, 0.0, 1.0))
        if ctx.lane.confidence > 0.1
        else default_throttle
    )
    return ControlCommand(steering=steering, throttle=throttle)
