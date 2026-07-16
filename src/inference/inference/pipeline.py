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
    out_fork_require_sign: bool = True
    out_fork_sign_hold_sec: float = 3.0
    # OUT: also require ``out_fork_capture`` (tip+stretch) with the sign window.
    out_fork_require_capture: bool = True
    # If True, forced_turn keeps fork perception armed (old sim behavior).
    # Default False: forced_turn only picks L/R rank when a real sign arms the window.
    out_fork_forced_turn_arms: bool = False
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
    max_steer_angle_rad: float = 0.5574
    max_steering_command: float = 1.0
    steering_rate_limit_per_sec: float = 16.0
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
    # Zero out tiny PP command to stop straight micro-wobble (normalized [-1,1]).
    steer_command_deadband: float = 0.03
    cruise_throttle: float = 0.42
    curve_throttle: float = 0.28
    curve_steering_threshold: float = 0.35
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
    # IN: moment-driven keep then exit (rank 1 = right keep, 0 = left exit).
    in_exit_use_moment: bool = True
    in_keep_passes: int = 1
    in_keep_branch_rank: int = 1
    branch_required_events: int = 2
    crossing_required_events: int = 2
    branch_on_frames: int = 3
    branch_off_frames: int = 8
    crossing_on_frames: int = 2
    crossing_off_frames: int = 5
    roundabout_lookahead_m: float = 0.32
    roundabout_throttle: float = 0.06
    circle_ignore_fork_for_control: bool = True
    # ROUNDABOUT_CIRCLE control: 'pp' | 'stanley' | 'mask_p' | 'hybrid' | ''(=normal).
    # IN needs paint (yellow) centerline here — road COM alone drifts off the ring.
    circle_tracker: str = 'pp'
    require_green_to_start: bool = False
    stop_on_red: bool = False
    # When false: still detect ArUco for debug, but keep lane follow (no hard stop).
    stop_on_aruco: bool = True
    command_watchdog_sec: float = 0.5
    log_state_changes: bool = True
    log_decision_changes: bool = True
    debug_publish_hz: float = 2.0
    # NORMAL / roundabout-circle tracker: 'pp' | 'mask_p' | 'hybrid' | 'stanley'
    # hybrid = gated PP (straight) + hard-corridor mask COM (curve / large error).
    # stanley = path CTE/heading Stanley-lite + optional curvature FF (A/B vs mask_p).
    normal_tracker: str = 'pp'
    mask_steer_k: float = 2.0
    # sim_v2: limo_sim_code_v2 style (cx-half)/W·π·k clipped — strongest simple P.
    # image_p: error_norm * k. lateral_atan: Stanley-like δ=atan(k e_m/(v+ε))/δ_max.
    mask_steer_law: str = 'sim_v2'
    mask_steer_alpha: float = 0.4
    mask_near_band_ratio: float = 0.55
    # Farther BEV band (top of IPM). Blended into COM for earlier curve see-ahead.
    # Under hybrid, effective far_blend = mask_far_blend * blend_weight (0 on straight).
    mask_far_band_ratio: float = 0.90
    mask_far_blend: float = 0.0
    # Add white/yellow path CTE+heading on top of COM (mask_p only; hybrid uses PP CTE).
    mask_use_path_correction: bool = True
    # How to turn road_clean (black/red free-space) into a lateral target:
    # area = pixel COM (cuts inside on bends); row_mid = per-row L/R edge midpoint;
    # dist_ridge = distance-transform ridge (farthest from red/black boundary).
    mask_center_mode: str = 'row_mid'
    # Optional erode before center extract — keeps target off paint edges (px).
    mask_erode_px: int = 2
    # Nominal free-space width (black/red walls). Used when one side is FOV/occlusion
    # clipped so row_mid can rebuild center as visible_edge ± half_width.
    mask_lane_width_m: float = 0.35
    # Observed width below this fraction of lane_width → treat as single-sided.
    mask_single_side_width_ratio: float = 0.72
    mask_min_area_px: float = 100.0
    mask_curve_steer_threshold: float = 0.35
    mask_curve_speed_scale: float = 0.80
    # Course-color / fork guard for mask_p (see mask_pursuit YAML).
    # off | hard (AND dilate around color path) | soft (distance-weighted COM).
    mask_corridor_mode: str = 'off'
    mask_corridor_half_width_m: float = 0.28
    mask_path_weight_sigma_m: float = 0.20
    # When fork/branch≥2 visible, skip mask COM and use color/branch PP.
    mask_fork_force_pp: bool = True
    # If color_path missing under hard/soft corridor, fail mask → PP fallback.
    mask_require_color_path: bool = True
    # When OUT fork perception is armed (sign hold), auto-enable hard corridor.
    mask_corridor_near_fork: bool = True
    # Confidence-gated paint pull on top of drivable COM (mask_p).
    # w = paint_blend_max * smoothstep(path_conf, lo, hi); raw = COM + w*(CTE+heading).
    mask_paint_blend_max: float = 0.0
    mask_paint_blend_lo: float = 0.20
    mask_paint_blend_hi: float = 0.55
    # When mask drops in a corner, hold last COM steer this many frames before
    # white PP (PP often yanks off when rails vanish).
    mask_occlusion_hold_frames: int = 24
    # Hybrid gate: w = max(smoothstep(|e|-deadband), smoothstep(|κ|)).
    mask_error_deadband: float = 0.04
    mask_blend_error_lo: float = 0.08
    mask_blend_error_hi: float = 0.35
    mask_blend_curvature_lo: float = 0.40
    mask_blend_curvature_hi: float = 1.20
    # EMA on hybrid blended/PP-only command before rate limit (straight stickiness).
    hybrid_steer_alpha: float = 0.32
    # Shrink cruise by |CTE| / |steer| / hybrid_w (helps late corrections settle).
    error_speed_cte_full_m: float = 0.16
    error_speed_steer_full: float = 0.50
    error_speed_min_scale: float = 0.48
    # Phase A: temporal lateral track state (COM norm + path near-y).
    track_err_alpha: float = 0.40
    track_err_max_jump: float = 0.35
    track_path_y_alpha: float = 0.35
    track_path_y_max_jump_m: float = 0.12
    track_half_width_m: float = 0.175
    track_enable_path_hold: bool = True
    # Phase B: Stanley-lite (δ = ψ + atan(k e / (v+eps)) + κ FF).
    stanley_k_cte: float = 1.20
    stanley_k_yaw: float = 1.0
    stanley_v_soft: float = 0.18
    stanley_curvature_ff_gain: float = 0.35
    stanley_steer_alpha: float = 0.35
    # Phase C: predict CTE across vision/servo delay (seconds). 0 = off.
    delay_pred_sec: float = 0.0
    # Approximate |v| from |throttle| for Stanley / delay (m/s at throttle=1).
    throttle_speed_scale_mps: float = 1.0


def load_planner_config(
    path: str | Path | None = None,
    *,
    route_mode: str | None = None,
    traffic_pass: bool | None = None,
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
    tracker = _section(data, 'tracker')
    mask = _section(data, 'mask_pursuit')
    track = _section(data, 'track_state')
    stanley = _section(data, 'stanley')
    selected_route = route_mode or route.get('mode', RouteMode.OUT.value)
    mode = RouteMode(str(selected_route).lower())
    # IN → yellow priority by default; OUT → white (prefer_yellow false).
    if mode is RouteMode.IN:
        prefer_yellow = bool(route.get('prefer_yellow', True))
    else:
        prefer_yellow = bool(route.get('prefer_yellow', False))
    # Mid-track tests: skip green-wait and red-stop (ArUco still active).
    require_green = bool(signals.get('require_green_to_start', False))
    stop_red = bool(signals.get('stop_on_red', False))
    if traffic_pass is True or (
        traffic_pass is None and bool(signals.get('traffic_pass', False))
    ):
        require_green = False
        stop_red = False
    return PlannerConfig(
        route_mode=mode,
        prefer_yellow=prefer_yellow,
        default_out_branch_rank=int(route.get('default_out_branch_rank', 0)),
        sign_confirm_frames=max(1, int(route.get('sign_confirm_frames', 3))),
        out_fork_require_sign=bool(route.get('out_fork_require_sign', True)),
        out_fork_sign_hold_sec=max(
            0.0, float(route.get('out_fork_sign_hold_sec', 3.0))
        ),
        out_fork_require_capture=bool(
            route.get('out_fork_require_capture', True)
        ),
        out_fork_forced_turn_arms=bool(
            route.get('out_fork_forced_turn_arms', False)
        ),
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
        max_steer_angle_rad=max(0.01, float(pp.get('max_steer_angle_rad', 0.5574))),
        max_steering_command=float(
            np.clip(pp.get('max_steering_command', 1.0), 0.1, 1.0)
        ),
        steering_rate_limit_per_sec=max(
            0.01, float(pp.get('steering_rate_limit_per_sec', 16.0))
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
        steer_command_deadband=float(
            np.clip(pp.get('steer_command_deadband', 0.03), 0.0, 0.2)
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
        in_exit_use_moment=bool(rb.get('in_exit_use_moment', True)),
        in_keep_passes=max(0, int(rb.get('in_keep_passes', 1))),
        in_keep_branch_rank=int(rb.get('in_keep_branch_rank', 1)),
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
        circle_ignore_fork_for_control=bool(
            rb.get('circle_ignore_fork_for_control', True)
        ),
        circle_tracker=str(rb.get('circle_tracker', 'pp') or 'pp').strip().lower(),
        require_green_to_start=require_green,
        stop_on_red=stop_red,
        stop_on_aruco=bool(signals.get('stop_on_aruco', True)),
        command_watchdog_sec=max(0.1, float(safety.get('command_watchdog_sec', 0.5))),
        log_state_changes=bool(debug.get('log_state_changes', True)),
        log_decision_changes=bool(debug.get('log_decision_changes', True)),
        debug_publish_hz=max(0.1, float(debug.get('publish_hz', 2.0))),
        normal_tracker=str(tracker.get('normal', 'pp')).strip().lower(),
        mask_steer_k=max(0.01, float(mask.get('steer_k', 2.0))),
        mask_steer_law=str(mask.get('steer_law', 'sim_v2') or 'sim_v2').lower(),
        mask_steer_alpha=float(np.clip(mask.get('steer_alpha', 0.4), 0.01, 1.0)),
        mask_near_band_ratio=float(
            np.clip(mask.get('near_band_ratio', 0.55), 0.1, 1.0)
        ),
        mask_far_band_ratio=float(
            np.clip(mask.get('far_band_ratio', 0.90), 0.15, 1.0)
        ),
        mask_far_blend=float(np.clip(mask.get('far_blend', 0.0), 0.0, 0.8)),
        mask_use_path_correction=bool(mask.get('use_path_correction', True)),
        mask_center_mode=str(mask.get('center_mode', 'row_mid')).strip().lower(),
        mask_erode_px=max(0, int(mask.get('erode_px', 2))),
        mask_lane_width_m=max(0.15, float(mask.get('lane_width_m', 0.35))),
        mask_single_side_width_ratio=float(
            np.clip(mask.get('single_side_width_ratio', 0.72), 0.4, 0.95)
        ),
        mask_min_area_px=max(1.0, float(mask.get('min_area_px', 100.0))),
        mask_curve_steer_threshold=float(
            np.clip(mask.get('curve_steer_threshold', 0.50), 0.0, 1.0)
        ),
        mask_curve_speed_scale=float(
            np.clip(mask.get('curve_speed_scale', 0.80), 0.05, 1.0)
        ),
        mask_corridor_mode=str(mask.get('corridor_mode', 'off')).strip().lower(),
        mask_corridor_half_width_m=max(
            0.05, float(mask.get('corridor_half_width_m', 0.28))
        ),
        mask_path_weight_sigma_m=max(
            0.02, float(mask.get('path_weight_sigma_m', 0.20))
        ),
        mask_fork_force_pp=bool(mask.get('fork_force_pp', True)),
        mask_require_color_path=bool(mask.get('require_color_path', True)),
        mask_corridor_near_fork=bool(mask.get('corridor_near_fork', True)),
        mask_paint_blend_max=float(
            np.clip(mask.get('paint_blend_max', 0.0), 0.0, 1.0)
        ),
        mask_paint_blend_lo=max(0.0, float(mask.get('paint_blend_lo', 0.20))),
        mask_paint_blend_hi=max(
            0.01, float(mask.get('paint_blend_hi', 0.55))
        ),
        mask_occlusion_hold_frames=max(
            0, int(mask.get('occlusion_hold_frames', 12))
        ),
        mask_error_deadband=max(0.0, float(mask.get('error_deadband', 0.04))),
        mask_blend_error_lo=max(0.0, float(mask.get('blend_error_lo', 0.08))),
        mask_blend_error_hi=max(
            0.01, float(mask.get('blend_error_hi', 0.35))
        ),
        mask_blend_curvature_lo=max(
            0.0, float(mask.get('blend_curvature_lo', 0.40))
        ),
        mask_blend_curvature_hi=max(
            0.01, float(mask.get('blend_curvature_hi', 1.20))
        ),
        hybrid_steer_alpha=float(
            np.clip(mask.get('hybrid_steer_alpha', 0.32), 0.05, 1.0)
        ),
        error_speed_cte_full_m=max(
            0.02, float(speed.get('error_speed_cte_full_m', 0.16))
        ),
        error_speed_steer_full=float(
            np.clip(speed.get('error_speed_steer_full', 0.50), 0.05, 1.0)
        ),
        error_speed_min_scale=float(
            np.clip(speed.get('error_speed_min_scale', 0.48), 0.15, 1.0)
        ),
        track_err_alpha=float(np.clip(track.get('err_alpha', 0.40), 0.01, 1.0)),
        track_err_max_jump=max(0.02, float(track.get('err_max_jump', 0.35))),
        track_path_y_alpha=float(
            np.clip(track.get('path_y_alpha', 0.35), 0.01, 1.0)
        ),
        track_path_y_max_jump_m=max(
            0.02, float(track.get('path_y_max_jump_m', 0.12))
        ),
        track_half_width_m=max(
            0.05, float(track.get('half_width_m', 0.175))
        ),
        track_enable_path_hold=bool(track.get('enable_path_hold', True)),
        stanley_k_cte=max(0.0, float(stanley.get('k_cte', 1.20))),
        stanley_k_yaw=max(0.0, float(stanley.get('k_yaw', 1.0))),
        stanley_v_soft=max(0.01, float(stanley.get('v_soft', 0.18))),
        stanley_curvature_ff_gain=float(
            np.clip(stanley.get('curvature_ff_gain', 0.35), 0.0, 2.0)
        ),
        stanley_steer_alpha=float(
            np.clip(stanley.get('steer_alpha', 0.35), 0.01, 1.0)
        ),
        delay_pred_sec=max(0.0, float(track.get('delay_pred_sec', 0.0))),
        throttle_speed_scale_mps=max(
            0.1, float(track.get('throttle_speed_scale_mps', 1.0))
        ),
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
        self._forced_turn = TurnSign.UNKNOWN
        self._last_out_sign_sec: float | None = None
        self._fork_perception_enabled = True
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
        # IN keep/exit moment risings (same debounce as branch_on/off).
        self.moment_counter = RisingEventCounter(
            self.config.branch_on_frames, self.config.branch_off_frames
        )
        self._in_fork_pass_count = 0
        self._out_capture_latched = False
        self._last_fork_arm_reason = 'none'
        self._roundabout_started_at: float | None = None
        self._yellow_valid_frames = 0
        self._path_lost_frames = 0
        self._fork_absent_frames = 0
        self._steering = 0.0
        self._steer_f = 0.0
        self._hybrid_steer_f = 0.0
        self._stanley_steer_f = 0.0
        self._track_e_norm: float | None = None
        self._track_cte_m: float | None = None
        self._track_path_y_m: float | None = None
        self._track_psi: float | None = None
        self._track_com_cx: float | None = None
        self._mask_occlusion_hold_frames = 0
        self._lookahead_m = self.config.lookahead_m
        self._last_path_source = PathSource.NONE
        self._last_mask_debug: dict[str, Any] = {}
        self._last_step_time_sec: float | None = None

    def neutralize_steering(self) -> None:
        """Synchronize planner state with an externally forced neutral command."""
        self._steering = 0.0
        self._steer_f = 0.0
        self._hybrid_steer_f = 0.0
        self._stanley_steer_f = 0.0
        self._track_e_norm = None
        self._track_cte_m = None
        self._track_path_y_m = None
        self._track_psi = None
        self._track_com_cx = None
        self._mask_occlusion_hold_frames = 0

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
            self.moment_counter.reset()
            self._in_fork_pass_count = 0
            self._lookahead_m = self.config.roundabout_lookahead_m
        if state in (DrivingState.NORMAL, DrivingState.FINISHED):
            self._fork_absent_frames = 0
        if previous_state is DrivingState.FORK_TURN and state is DrivingState.NORMAL:
            # Sign is past — resume white-only; do not re-arm on stale hold.
            self._last_out_sign_sec = None
            if self._forced_turn is not TurnSign.UNKNOWN:
                # Keep the sim override latched across fork completion.
                self.apply_forced_turn(self._forced_turn)
            else:
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
        """Confirm a direction sign before latching; freeze it inside a fork.

        ``_forced_turn`` (sim/test override) blocks camera sign updates.
        """
        if self._forced_turn is not TurnSign.UNKNOWN:
            return
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

    def apply_forced_turn(self, turn: TurnSign) -> None:
        """Latch LEFT/RIGHT for the whole run (sim experiment override).

        In-route contract for tests:
          LEFT  → exit the roundabout (rank 0)
          RIGHT → stay circulating (do not arm exit)
        Out-route: LEFT→rank0 / RIGHT→rank1 as usual.
        """
        if turn is TurnSign.UNKNOWN:
            self._forced_turn = TurnSign.UNKNOWN
            return
        self._forced_turn = turn
        self.desired_turn = turn
        self._sign_candidate = turn
        self._sign_candidate_frames = self.config.sign_confirm_frames
        self._lock_fork_selection()
        self._fork_selection_reason = f'forced_{turn.value}'

    def _wants_roundabout_exit(self) -> bool:
        """IN-mode policy: LEFT exits; RIGHT keeps circling; else default exit.

        When ``in_exit_use_moment`` is on, keep-passes must already be finished
        (``_in_fork_pass_count > in_keep_passes``) before exit is allowed.
        """
        if self.config.route_mode is not RouteMode.IN:
            return True
        if self.config.in_exit_use_moment:
            from inference.modules.perception.fork.judgment import (
                in_wants_exit_from_passes,
            )

            if not in_wants_exit_from_passes(
                self._in_fork_pass_count,
                keep_passes=self.config.in_keep_passes,
            ):
                return False
        latched = (
            self._forced_turn
            if self._forced_turn is not TurnSign.UNKNOWN
            else self.desired_turn
        )
        if latched is TurnSign.RIGHT:
            return False
        if latched is TurnSign.LEFT:
            return True
        # No pre-choice: after moment keep-passes, attempt exit when armed.
        return True

    def _apply_in_moment_pass(self, moment_rising: bool) -> None:
        """On IN moment rising: pass1→right keep, pass2→left exit (+ fork follow)."""

        if not moment_rising:
            return
        if self.config.route_mode is not RouteMode.IN:
            return
        if not self.config.in_exit_use_moment:
            return
        if self.state is not DrivingState.ROUNDABOUT_CIRCLE:
            return
        # Sim/test override: forced_turn owns keep/exit.
        if self._forced_turn is not TurnSign.UNKNOWN:
            return
        from inference.modules.perception.fork.judgment import decide_in_exit_pass

        decision = decide_in_exit_pass(
            self._in_fork_pass_count,
            keep_passes=self.config.in_keep_passes,
            keep_rank=self.config.in_keep_branch_rank,
            exit_rank=self.config.exit_branch_rank,
        )
        self._in_fork_pass_count = decision.pass_index
        self._fork_selected_rank = int(decision.select_rank)
        self._fork_selection_reason = decision.reason
        if decision.wants_exit:
            self.desired_turn = TurnSign.LEFT
            self._fork_locked_turn = TurnSign.LEFT
        else:
            self.desired_turn = TurnSign.RIGHT
            self._fork_locked_turn = TurnSign.RIGHT

    def force_fork_choice(
        self,
        turn: TurnSign,
        *,
        state: DrivingState | None = None,
    ) -> None:
        """Test/sim helper: latch turn + rank and optionally jump FSM state.

        Also sets ``_forced_turn`` so camera signs cannot overwrite the choice
        (same as launch ``forced_turn:=``).
        """
        self.apply_forced_turn(turn)
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

    def _strict_ranked_branch(self, lane: Any, rank: int) -> Any | None:
        """Match ``lateral_rank`` only — never fall back to list index.

        Used after fork lock / ego-only publish so a lone wrong-side remnant
        cannot steal the path (index 0 != selected LEFT).
        """
        for branch in getattr(lane, 'branches', ()) or ():
            if (
                hasattr(branch, 'lateral_rank')
                and int(branch.lateral_rank) == int(rank)
            ):
                return branch
        return None

    def _locked_ego_path(
        self, lane: Any, rank: int
    ) -> tuple[np.ndarray, PathSource, float]:
        """Single selected corridor after lock (no opposite fork required)."""
        branch = self._strict_ranked_branch(lane, rank)
        if branch is not None:
            points = np.asarray(branch.points, dtype=np.float32)
            if points.ndim == 2 and points.shape[0] >= self.config.min_points:
                source = (
                    PathSource.LEFT_BRANCH
                    if int(rank) == 0
                    else PathSource.RIGHT_BRANCH
                )
                return points, source, float(getattr(branch, 'confidence', 0.0))
        # Collapse wrote the selected centerline onto the follow color.
        return self._color_path(lane)

    def _color_or_selected_resume(
        self, lane: Any, color_path: np.ndarray, color_source: PathSource, color_conf: float
    ) -> tuple[np.ndarray, PathSource, float, str]:
        """After fork flicker: keep selected layer if still published, else color."""
        rank = self._fork_selected_rank
        if rank is not None:
            path, source, conf = self._locked_ego_path(lane, int(rank))
            # Prefer strict branch over color when present.
            branch = self._strict_ranked_branch(lane, int(rank))
            if branch is not None and path.shape[0] >= self.config.min_points:
                return path, source, conf, 'selected_layer_resume'
            if (
                branch is None
                and color_path.shape[0] >= self.config.min_points
            ):
                return color_path, color_source, color_conf, 'color_resume'
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
        """Course color contract for normal follow.

        * Out: white only (never yellow).
        * In (route_mode=in): **yellow when available** (priority), else white.
          Mission flow is white approach → yellow roundabout → white merge;
          that emerges naturally as yellow appears/disappears. Not a
          yellow→white→yellow ladder.
        """

        yellow_valid = self._valid_centerline(
            lane.yellow_centerline,
            float(lane.yellow_confidence),
            self.config.yellow_min_confidence,
        )
        white_valid = self._valid_centerline(
            lane.white_centerline,
            float(lane.white_confidence),
            self.config.white_min_confidence,
        )
        self._yellow_valid_frames = (
            self._yellow_valid_frames + 1 if yellow_valid else 0
        )

        if self.config.route_mode is RouteMode.OUT or not self.config.prefer_yellow:
            # Out SSOT: white only — yellow paint must not pull the robot.
            if white_valid:
                return (
                    np.asarray(lane.white_centerline, dtype=np.float32),
                    PathSource.WHITE_CENTERLINE,
                    float(lane.white_confidence),
                )
            return np.empty((0, 2), dtype=np.float32), PathSource.NONE, 0.0

        # In: prefer yellow once stable; otherwise keep white (approach / merge).
        if self._yellow_valid_frames >= self.config.yellow_valid_on_frames:
            return (
                np.asarray(lane.yellow_centerline, dtype=np.float32),
                PathSource.YELLOW_CENTERLINE,
                float(lane.yellow_confidence),
            )
        if white_valid:
            return (
                np.asarray(lane.white_centerline, dtype=np.float32),
                PathSource.WHITE_CENTERLINE,
                float(lane.white_confidence),
            )
        # Yellow visible but not yet debounced, and no white: take yellow early.
        if yellow_valid:
            return (
                np.asarray(lane.yellow_centerline, dtype=np.float32),
                PathSource.YELLOW_CENTERLINE,
                float(lane.yellow_confidence),
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

        Metric IPM measures ground distance from the camera. Add
        ``perception_to_rear_axle_x_m`` (= rear_axle→camera = L + ahead_front)
        once here so PP / heading / CTE share one origin. See
        docs/vehicle-geometry.md §4.1.1 (sim 0.265 m, real 0.200 m).
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

    def _apply_rate_limited_steering(self, raw: float, dt_sec: float) -> float:
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
        return self._steering

    @staticmethod
    def _ema_with_jump(
        previous: float | None,
        observation: float,
        *,
        alpha: float,
        max_jump: float,
    ) -> tuple[float, bool]:
        """Low-pass filter with jump reject (keeps previous on flicker / L-R flip)."""
        if previous is None or not math.isfinite(previous):
            return float(observation), False
        jumped = abs(float(observation) - float(previous)) > float(max_jump)
        sample = float(previous) if jumped else float(observation)
        alpha = float(np.clip(alpha, 0.01, 1.0))
        return (1.0 - alpha) * float(previous) + alpha * sample, jumped

    def _estimate_speed_mps(self) -> float:
        """Throttle→speed scale for Stanley / delay prediction (no odom required)."""
        cruise = abs(float(self.config.cruise_throttle))
        scale = float(self.config.throttle_speed_scale_mps)
        return max(0.05, cruise * scale)

    def _harden_color_path(self, path_xy: np.ndarray) -> np.ndarray:
        """Temporal near-y hold + half-width soft clamp on color centerline.

        Rejects sudden lateral jumps (occlusion / L-R flip) and gently pulls
        paths that sit beyond ~1.35× half-width back toward the prior center.
        Perception already synthesizes missing rails; this is a control-layer
        safety net that does not touch fork branch geometry when unused.
        """
        arr = np.asarray(path_xy, dtype=np.float32)
        if (
            not self.config.track_enable_path_hold
            or arr.ndim != 2
            or arr.shape[0] < 2
            or arr.shape[1] < 2
        ):
            return arr
        rear = self._path_in_rear_axle_frame(arr)
        near_mask = rear[:, 0] <= max(0.9, float(self.config.heading_preview_m) + 0.4)
        near = rear[near_mask] if np.any(near_mask) else rear[: min(4, rear.shape[0])]
        raw_y = float(np.mean(near[:, 1]))
        filtered_y, jumped = self._ema_with_jump(
            self._track_path_y_m,
            raw_y,
            alpha=self.config.track_path_y_alpha,
            max_jump=self.config.track_path_y_max_jump_m,
        )
        half = float(self.config.track_half_width_m)
        # Soft half-width prior: if filtered magnitude exceeds rail half*1.35,
        # blend toward sign(filtered)*half (virtual single-rail center).
        soft_limit = 1.35 * half
        if abs(filtered_y) > soft_limit:
            target = math.copysign(half, filtered_y)
            filtered_y = 0.7 * filtered_y + 0.3 * target
        self._track_path_y_m = filtered_y
        dy = filtered_y - raw_y
        if abs(dy) < 1e-5:
            return arr
        out = arr.copy()
        out[:, 1] = out[:, 1] + float(dy)
        self._last_mask_debug = {
            **getattr(self, '_last_mask_debug', {}),
            'track_path_y_raw': round(raw_y, 4),
            'track_path_y_f': round(filtered_y, 4),
            'track_path_jump': bool(jumped),
        }
        return out

    def _stanley_pursuit(
        self,
        path: np.ndarray,
        dt_sec: float | None = None,
        *,
        apply_rate_limit: bool = True,
    ) -> PursuitResult:
        """Stanley-lite on hardened color path + optional curvature feedforward."""
        hardened = self._harden_color_path(path)
        rear_axle_path = self._path_in_rear_axle_frame(hardened)
        xy = self._ordered_path(rear_axle_path)
        if xy.shape[0] < self.config.min_points:
            return PursuitResult(False, path_points=int(xy.shape[0]))
        xy, path_extrapolation_m = self._extend_path_to_front_axle(xy)
        path_curvature = self._estimate_path_curvature(xy)
        cte = self._cross_track_error(xy)
        heading_error, _heading_cmd = self._heading_correction(xy)

        cte_f, cte_jump = self._ema_with_jump(
            self._track_cte_m,
            cte,
            alpha=self.config.track_err_alpha,
            max_jump=self.config.track_path_y_max_jump_m,
        )
        self._track_cte_m = cte_f
        psi_f, _ = self._ema_with_jump(
            self._track_psi,
            float(heading_error),
            alpha=self.config.track_err_alpha,
            max_jump=0.55,
        )
        self._track_psi = psi_f

        v = self._estimate_speed_mps()
        delay = float(self.config.delay_pred_sec)
        e_pred = float(cte_f)
        if delay > 1e-6:
            e_pred = e_pred + delay * v * math.sin(float(psi_f))

        effective_e = math.copysign(
            max(0.0, abs(e_pred) - self.config.cte_deadband_m), e_pred
        )
        stanley_term = math.atan2(
            self.config.stanley_k_cte * effective_e,
            v + float(self.config.stanley_v_soft),
        )
        yaw_term = float(self.config.stanley_k_yaw) * float(psi_f)
        ff = float(self.config.stanley_curvature_ff_gain) * math.atan(
            self.config.wheelbase_m * float(path_curvature)
        )
        # Positive path-left angles → negative D-Racer steering.
        steer_angle = yaw_term + stanley_term + ff
        raw = float(
            np.clip(
                -steer_angle / self.config.max_steer_angle_rad,
                -self.config.max_steering_command,
                self.config.max_steering_command,
            )
        )
        if abs(raw) < self.config.steer_command_deadband:
            raw = 0.0

        alpha = float(self.config.stanley_steer_alpha)
        self._stanley_steer_f = (1.0 - alpha) * self._stanley_steer_f + alpha * raw
        filtered = float(self._stanley_steer_f)
        dt = self.config.nominal_control_dt_sec if dt_sec is None else max(0.0, dt_sec)
        if apply_rate_limit:
            steered = self._apply_rate_limited_steering(filtered, dt)
        else:
            steered = float(np.clip(filtered, -1.0, 1.0))

        curve_ratio = float(
            np.clip(
                abs(path_curvature) / max(self.config.curvature_full_scale, 1e-3),
                0.0,
                1.0,
            )
        )
        target = self._target_at_radius(xy, max(0.35, self.config.heading_preview_m))
        if target is None:
            target_x, target_y = 0.4, float(-cte_f)
        else:
            target_x, target_y = float(target[0]), float(target[1])

        self._last_mask_debug = {
            **getattr(self, '_last_mask_debug', {}),
            'stanley_cte_m': round(float(cte), 4),
            'stanley_cte_f': round(float(cte_f), 4),
            'stanley_cte_jump': bool(cte_jump),
            'stanley_psi': round(float(psi_f), 4),
            'stanley_e_pred': round(float(e_pred), 4),
            'stanley_ff': round(float(ff), 4),
            'stanley_v': round(float(v), 3),
        }
        return PursuitResult(
            valid=True,
            steering=steered,
            target_x=target_x,
            target_y=target_y,
            path_points=int(xy.shape[0]),
            target_distance=math.hypot(target_x, target_y),
            lookahead_m=float(self.config.heading_preview_m),
            desired_lookahead_m=float(self.config.heading_preview_m),
            path_curvature=float(path_curvature),
            curve_ratio=curve_ratio,
            raw_steering=filtered,
            pp_steering=raw,
            cross_track_error_m=float(cte_f),
            cte_steering=float(
                -stanley_term / self.config.max_steer_angle_rad
            ),
            heading_error_rad=float(psi_f),
            heading_steering=float(-yaw_term / self.config.max_steer_angle_rad),
            path_extrapolation_m=float(path_extrapolation_m),
        )

    def _mask_com_pursuit(
        self,
        lane: Any,
        dt_sec: float | None = None,
        *,
        color_path: np.ndarray | None = None,
        path_confidence: float | None = None,
        far_blend_override: float | None = None,
        apply_rate_limit: bool = True,
        update_ema: bool = True,
        use_path_correction_override: bool | None = None,
    ) -> PursuitResult:
        """Free-space mask COM proportional steer (limo_sim BEV follower style).

        Uses Metric-IPM ``drivable_area``. Optional ``color_path`` (course
        white/yellow centerline in base_link) constrains COM via
        ``mask_corridor_mode`` so OUT/IN forks do not average into the wrong
        course. CTE metrics always prefer the color path when present.
        """
        import cv2
        from inference.modules import lane_detection as ld

        mask = np.asarray(getattr(lane, 'drivable_area', None), dtype=np.uint8)
        if mask.ndim != 2 or mask.size == 0:
            return self._mask_occlusion_hold_or_fail(dt_sec)
        binary = (mask > 0).astype(np.uint8)
        height, width = binary.shape
        if height < 4 or width < 4:
            return self._mask_occlusion_hold_or_fail(dt_sec)

        path_xy = np.empty((0, 2), dtype=np.float32)
        if color_path is not None:
            arr = np.asarray(color_path, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                path_xy = self._harden_color_path(arr)

        corridor_mode, require_color_path = self._effective_mask_corridor_mode(lane)
        if corridor_mode in ('hard', 'soft') and path_xy.shape[0] < 2:
            if require_color_path:
                return self._mask_occlusion_hold_or_fail(dt_sec)

        mpp = float(getattr(lane, 'meters_per_pixel', 0.0) or 0.0)
        if mpp <= 1e-9:
            mpp = float(getattr(ld, 'METERS_PER_PIXEL', 0.01) or 0.01)

        work = binary
        weight = None
        center_mode = str(self.config.mask_center_mode or 'row_mid').lower()
        erode_px = int(self.config.mask_erode_px)
        if erode_px > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
            )
            work = cv2.erode(work, kernel, iterations=1)
        if corridor_mode in ('hard', 'soft') and path_xy.shape[0] >= 2:
            corridor = self._bev_corridor_mask(
                path_xy, height, width, mpp, self.config.mask_corridor_half_width_m
            )
            if corridor_mode == 'hard':
                work = cv2.bitwise_and(work, corridor)
            else:
                # Soft: distance to path skeleton → Gaussian weight on road.
                skeleton = self._bev_path_skeleton(path_xy, height, width)
                inv = np.where(skeleton > 0, 0, 255).astype(np.uint8)
                dist_px = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
                dist_m = dist_px.astype(np.float32) * float(mpp)
                sigma = float(self.config.mask_path_weight_sigma_m)
                weight = np.exp(-(dist_m * dist_m) / (2.0 * sigma * sigma))
                weight = weight * (work > 0).astype(np.float32)

        band = float(np.clip(self.config.mask_near_band_ratio, 0.1, 1.0))
        far_band = float(np.clip(self.config.mask_far_band_ratio, band, 1.0))
        if far_blend_override is None:
            far_blend = float(np.clip(self.config.mask_far_blend, 0.0, 0.8))
        else:
            far_blend = float(np.clip(far_blend_override, 0.0, 0.8))

        # limo_sim_code_v2: drive on the largest road blob only (drop side flakes).
        if str(self.config.mask_steer_law or '').lower() == 'sim_v2':
            nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
                (work > 0).astype(np.uint8), connectivity=8
            )
            if nlab > 1:
                best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                work = np.where(labels == best, np.uint8(255), np.uint8(0))
                if weight is not None:
                    weight = weight * (work > 0).astype(np.float32)

        def _roi_area_com(y_lo: int, y_hi: int) -> tuple[float, float, float] | None:
            if y_hi - y_lo < 2:
                return None
            if weight is not None:
                w_roi = weight[y_lo:y_hi, :]
                area_v = float(np.sum(w_roi))
                if area_v < self.config.mask_min_area_px:
                    return None
                ys, xs = np.indices(w_roi.shape, dtype=np.float64)
                cx_v = float(np.sum(xs * w_roi) / area_v)
                cy_v = float(np.sum(ys * w_roi) / area_v) + float(y_lo)
                return cx_v, cy_v, area_v
            moments = cv2.moments(work[y_lo:y_hi, :], binaryImage=True)
            area_v = float(moments.get('m00', 0.0))
            if area_v < self.config.mask_min_area_px:
                return None
            cx_v = float(moments['m10'] / area_v)
            cy_v = float(moments['m01'] / area_v) + float(y_lo)
            return cx_v, cy_v, area_v

        def _roi_row_mid(y_lo: int, y_hi: int) -> tuple[float, float, float] | None:
            """Per-row L/R midpoint with single-side width prior (S-curve occlusion).

            Only treat FOV-touch as a clipped rail when the blob is also
            *narrower* than the nominal lane — a full-width road that reaches
            the BEV edge must keep geometric mid (otherwise cornering yanks off).
            When single-sided, prefer the edge that stays near the temporal
            ``_track_com_cx`` prior.
            """
            if y_hi - y_lo < 2:
                return None
            roi = work[y_lo:y_hi, :]
            expected_w = max(
                4.0, float(self.config.mask_lane_width_m) / max(mpp, 1e-6)
            )
            half_lane = 0.5 * expected_w
            edge_tol = 2
            min_ratio = float(self.config.mask_single_side_width_ratio)
            prev_cx = self._track_com_cx
            mids: list[float] = []
            rows_used: list[float] = []
            areas = 0.0
            single_hits = 0
            for i in range(roi.shape[0]):
                cols = np.flatnonzero(roi[i] > 0)
                if cols.size < 2:
                    continue
                left = float(cols[0])
                right = float(cols[-1])
                width_px = right - left + 1.0
                clipped_l = left <= edge_tol
                clipped_r = right >= float(width - 1 - edge_tol)
                narrow = width_px < min_ratio * expected_w
                geom = 0.5 * (left + right)
                mid = geom
                if narrow and clipped_l and not clipped_r:
                    mid = right - half_lane
                    single_hits += 1
                elif narrow and clipped_r and not clipped_l:
                    mid = left + half_lane
                    single_hits += 1
                elif narrow:
                    # Mid-frame occlusion: pick the reconstruction closest to prior.
                    c_left = left + half_lane
                    c_right = right - half_lane
                    if prev_cx is not None:
                        mid = (
                            c_left
                            if abs(c_left - prev_cx) <= abs(c_right - prev_cx)
                            else c_right
                        )
                    else:
                        img_c = 0.5 * float(width - 1)
                        mid = c_left if geom >= img_c else c_right
                    single_hits += 1
                # else: full-width (even if touching FOV) → geometric mid
                # Mild prior only when reconstruct jumps hard (avoid corner freeze).
                if prev_cx is not None and narrow and abs(mid - float(prev_cx)) > 0.35 * expected_w:
                    mid = 0.35 * float(prev_cx) + 0.65 * mid
                mid = float(np.clip(mid, 0.0, float(width - 1)))
                mids.append(mid)
                rows_used.append(float(y_lo + i))
                areas += float(cols.size)
            if len(mids) < 2 or areas < self.config.mask_min_area_px:
                return None
            mids_a = np.asarray(mids, dtype=np.float64)
            rows_a = np.asarray(rows_used, dtype=np.float64)
            wrow = (rows_a - float(y_lo) + 1.0)
            wrow = wrow / max(float(np.sum(wrow)), 1e-6)
            cx_v = float(np.sum(mids_a * wrow))
            cy_v = float(np.sum(rows_a * wrow))
            single_ratio = float(single_hits) / max(len(mids), 1)
            self._last_mask_debug = {
                **getattr(self, '_last_mask_debug', {}),
                'mask_single_side_rows': int(single_hits),
                'mask_single_side_ratio': round(single_ratio, 3),
            }
            return cx_v, cy_v, areas

        def _roi_dist_ridge(y_lo: int, y_hi: int) -> tuple[float, float, float] | None:
            """Distance-transform ridge: farthest column from road boundary per band."""
            if y_hi - y_lo < 2:
                return None
            roi = work[y_lo:y_hi, :]
            if int(np.count_nonzero(roi)) < int(self.config.mask_min_area_px):
                return None
            dist = cv2.distanceTransform(roi, cv2.DIST_L2, 3)
            mids: list[float] = []
            rows_used: list[float] = []
            areas = 0.0
            for i in range(dist.shape[0]):
                row = dist[i]
                if float(np.max(row)) <= 1e-6:
                    continue
                u = float(np.argmax(row))
                mids.append(u)
                rows_used.append(float(y_lo + i))
                areas += float(np.count_nonzero(roi[i]))
            if len(mids) < 2 or areas < self.config.mask_min_area_px:
                return None
            mids_a = np.asarray(mids, dtype=np.float64)
            rows_a = np.asarray(rows_used, dtype=np.float64)
            wrow = (rows_a - float(y_lo) + 1.0)
            wrow = wrow / max(float(np.sum(wrow)), 1e-6)
            return float(np.sum(mids_a * wrow)), float(np.sum(rows_a * wrow)), areas

        def _roi_com(y_lo: int, y_hi: int) -> tuple[float, float, float] | None:
            if center_mode == 'dist_ridge':
                hit = _roi_dist_ridge(y_lo, y_hi)
                if hit is not None:
                    return hit
                return _roi_row_mid(y_lo, y_hi) or _roi_area_com(y_lo, y_hi)
            if center_mode == 'area':
                return _roi_area_com(y_lo, y_hi)
            # Default: row_mid (boundary-aware). Fall back to area if sparse.
            return _roi_row_mid(y_lo, y_hi) or _roi_area_com(y_lo, y_hi)

        # BEV: bottom = near, top = far. Near ROI drives lateral; far COM is a
        # light anticipatory bias so road bends show up before they enter
        # the near window (without the old full-height cut-in).
        y_near0 = int(max(0, math.floor(height * (1.0 - band))))
        near = _roi_com(y_near0, height)
        corridor_recover = 'none'
        # Hard corridor emptied by oversteer / paint flicker → widen, then soft
        # free-space rather than drop to PP path-lost.
        if near is None and corridor_mode == 'hard' and path_xy.shape[0] >= 2:
            wide_hw = float(self.config.mask_corridor_half_width_m) * 1.35
            wide = self._bev_corridor_mask(path_xy, height, width, mpp, wide_hw)
            work = cv2.bitwise_and(binary if erode_px <= 0 else work, wide)
            if erode_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
                )
                work = cv2.bitwise_and(
                    cv2.erode(binary, kernel, iterations=1), wide
                )
            near = _roi_com(y_near0, height)
            corridor_recover = 'widen' if near is not None else 'widen_fail'
        if near is None and corridor_mode == 'hard':
            work = binary
            if erode_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1)
                )
                work = cv2.erode(work, kernel, iterations=1)
            weight = None
            near = _roi_com(y_near0, height)
            if near is not None:
                corridor_recover = 'free_fallback'
        if near is None:
            return self._mask_occlusion_hold_or_fail(dt_sec)
        self._mask_occlusion_hold_frames = 0
        cx_near, cy_near, area = near
        half_w = 0.5 * float(max(width - 1, 1))
        error_near = (cx_near - half_w) / half_w
        cx, cy = cx_near, cy_near
        single_ratio = float(
            (getattr(self, '_last_mask_debug', {}) or {}).get(
                'mask_single_side_ratio', 0.0
            )
        )
        # Far blend is dangerous when one rail/FOV is clipped — kills S-curve.
        far_blend_eff = far_blend * max(0.0, 1.0 - 1.5 * single_ratio)
        if far_blend_eff > 1e-6 and far_band > band + 1e-3:
            y_far0 = int(max(0, math.floor(height * (1.0 - far_band))))
            far = _roi_com(y_far0, y_near0)
            if far is not None:
                cx = (1.0 - far_blend_eff) * cx + far_blend_eff * far[0]
                cy = (1.0 - far_blend_eff) * cy + far_blend_eff * far[1]
                area = float(area + far[2])

        law = str(self.config.mask_steer_law or 'sim_v2').lower()
        # sim_v2 matches limo_sim_code_v2: no temporal COM freeze (it lags corners).
        if law != 'sim_v2' and update_ema and self._track_com_cx is not None:
            jump_u = abs(float(cx) - float(self._track_com_cx))
            max_jump_u = max(
                6.0,
                float(self.config.mask_lane_width_m) / max(mpp, 1e-6) * 0.45,
            )
            if jump_u > max_jump_u:
                cx = 0.70 * float(self._track_com_cx) + 0.30 * float(cx)
            elif single_ratio >= 0.55:
                cx = 0.40 * float(self._track_com_cx) + 0.60 * float(cx)
            else:
                cx = 0.55 * float(self._track_com_cx) + 0.45 * float(cx)
        if update_ema:
            self._track_com_cx = float(cx)

        # Image-center error: road left of center → need left (negative) steer.
        # normalized_error in [-1, 1]: +1 means road is fully to the right.
        error_norm_raw = (cx - half_w) / half_w
        if law == 'sim_v2':
            # External BEV P: trust one frame COM (EMA only on command below).
            error_norm = float(error_norm_raw)
            err_jumped = False
            if update_ema:
                self._track_e_norm = float(error_norm)
        elif update_ema:
            jump_lim = float(self.config.track_err_max_jump)
            if single_ratio >= 0.55:
                jump_lim = min(jump_lim, 0.28)
            error_norm, err_jumped = self._ema_with_jump(
                self._track_e_norm,
                float(error_norm_raw),
                alpha=self.config.track_err_alpha,
                max_jump=jump_lim,
            )
            self._track_e_norm = float(error_norm)
        else:
            error_norm = float(error_norm_raw)
            err_jumped = False
        e_y_m = -(float(cx) - half_w) * mpp
        if law == 'sim_v2':
            # D-Racer: (cx-half)/W · π · k  ≡  -angular_z of limo_sim_code_v2.
            w_px = float(max(width - 1, 1))
            raw_p = float(
                np.clip(
                    ((float(cx) - half_w) / w_px)
                    * math.pi
                    * self.config.mask_steer_k,
                    -self.config.max_steering_command,
                    self.config.max_steering_command,
                )
            )
        elif law == 'image_p':
            raw_p = float(error_norm) * self.config.mask_steer_k
        else:
            # Lateral target in meters (BEV +u=right). Road left of center → +e_y
            # (path-left convention) → negative D-Racer steer after /δ_max.
            e_eff = math.copysign(
                max(0.0, abs(e_y_m) - float(self.config.cte_deadband_m)),
                e_y_m,
            )
            v = self._estimate_speed_mps() if update_ema else max(
                0.05, abs(float(self.config.cruise_throttle))
            )
            delta = math.atan2(
                self.config.mask_steer_k * e_eff,
                v + float(self.config.stanley_v_soft),
            )
            raw_p = float(
                np.clip(
                    -delta / max(self.config.max_steer_angle_rad, 1e-6),
                    -self.config.max_steering_command,
                    self.config.max_steering_command,
                )
            )
        if update_ema:
            alpha = self.config.mask_steer_alpha
            self._steer_f = (1.0 - alpha) * self._steer_f + alpha * raw_p
            filtered = float(self._steer_f)
        else:
            filtered = float(raw_p)

        use_path_corr = (
            self.config.mask_use_path_correction
            if use_path_correction_override is None
            else bool(use_path_correction_override)
        )
        cte = 0.0
        cte_steering = 0.0
        heading_error = 0.0
        heading_steering = 0.0
        paint_w = 0.0
        if path_xy.shape[0] >= 2:
            rear = self._path_in_rear_axle_frame(path_xy)
            xy = self._ordered_path(rear)
            if xy.shape[0] >= 2:
                cte = self._cross_track_error(xy)
                cte_steering = self._cte_correction(cte)
                heading_error, heading_steering = self._heading_correction(xy)

        # Always-on additive path correction (legacy). Prefer paint_blend_* instead.
        paint_add = 0.0
        blend_max = float(np.clip(self.config.mask_paint_blend_max, 0.0, 1.0))
        if use_path_corr:
            paint_add = float(cte_steering) + float(heading_steering)
        elif blend_max > 1e-6 and path_xy.shape[0] >= 2:
            # Confidence-gated paint pull: drivable COM first; clean paint → more weight.
            if path_confidence is None:
                conf = max(
                    float(getattr(lane, 'white_confidence', 0.0) or 0.0),
                    float(getattr(lane, 'yellow_confidence', 0.0) or 0.0),
                    float(getattr(lane, 'confidence', 0.0) or 0.0),
                )
            else:
                conf = float(path_confidence)
            paint_w = blend_max * self._smoothstep(
                conf,
                float(self.config.mask_paint_blend_lo),
                float(self.config.mask_paint_blend_hi),
            )
            prefer_y = bool(getattr(lane, 'yellow_visible', False)) and bool(
                self.config.prefer_yellow
            )
            paint_vis = (
                bool(getattr(lane, 'yellow_visible', False))
                if prefer_y
                else bool(getattr(lane, 'white_visible', False))
            )
            if not paint_vis:
                paint_w = 0.0
            paint_add = paint_w * (float(cte_steering) + float(heading_steering))

        raw = float(filtered) + float(paint_add)
        dt = self.config.nominal_control_dt_sec if dt_sec is None else max(0.0, dt_sec)
        if apply_rate_limit:
            steered = self._apply_rate_limited_steering(raw, dt)
        else:
            steered = float(np.clip(raw, -1.0, 1.0))

        curve_ratio = float(
            np.clip(
                abs(steered) / max(self.config.mask_curve_steer_threshold, 1e-3),
                0.0,
                1.0,
            )
        )

        # Approximate mask COM in vehicle frame for debug (y left).
        target_y = -(cx - half_w) * mpp if mpp > 1e-6 else error_norm
        target_x = max(0.05, float(getattr(lane, 'x_forward_max', 0.0) or 0.0) * 0.35)
        if target_x < 0.06:
            target_x = 0.35

        result = PursuitResult(
            valid=True,
            steering=steered,
            target_x=target_x,
            target_y=float(target_y),
            path_points=int(area),
            target_distance=math.hypot(target_x, float(target_y)),
            lookahead_m=0.0,
            desired_lookahead_m=0.0,
            path_curvature=0.0,
            curve_ratio=curve_ratio,
            raw_steering=raw,
            pp_steering=raw_p,
            cross_track_error_m=float(cte),
            cte_steering=float(cte_steering),
            heading_error_rad=float(heading_error),
            heading_steering=float(heading_steering),
            path_extrapolation_m=0.0,
        )
        # Attach transient debug for the caller (step()).
        self._last_mask_debug = {
            **getattr(self, '_last_mask_debug', {}),
            'mask_corridor_mode': corridor_mode,
            'mask_area_px': round(float(area), 1),
            'mask_com_cx': round(float(cx), 2),
            'mask_com_cy': round(float(cy), 2),
            'mask_error_norm': round(float(error_norm), 4),
            'mask_error_norm_raw': round(float(error_norm_raw), 4),
            'mask_error_jump': bool(err_jumped),
            'mask_error_near': round(float(error_near), 4),
            'mask_error_y_m': round(float(e_y_m), 4),
            'mask_steer_law': law,
            'mask_far_blend': round(float(far_blend_eff), 3),
            'mask_far_blend_cfg': round(float(far_blend), 3),
            'mask_path_correction': bool(use_path_corr),
            'mask_paint_blend_w': round(float(paint_w), 3),
            'mask_paint_blend_max': round(float(blend_max), 3),
            'mask_paint_add': round(float(paint_add), 4),
            'mask_color_path_points': int(path_xy.shape[0]),
            'mask_corridor_recover': corridor_recover,
            'mask_center_mode': center_mode,
            'mask_erode_px': int(erode_px),
            'mask_single_side_ratio': round(float(single_ratio), 3),
        }
        return result

    def _mask_occlusion_hold_or_fail(
        self, dt_sec: float | None
    ) -> PursuitResult:
        """Hold last mask steer briefly when free-space vanishes (corner FOV)."""
        hold_n = int(self.config.mask_occlusion_hold_frames)
        can_hold = (
            hold_n > 0
            and self._mask_occlusion_hold_frames < hold_n
            and (
                self._track_com_cx is not None
                or abs(float(self._steer_f)) > 0.02
            )
        )
        if not can_hold:
            return PursuitResult(False)
        self._mask_occlusion_hold_frames += 1
        held = float(self._steer_f)
        dt = (
            self.config.nominal_control_dt_sec
            if dt_sec is None
            else max(0.0, float(dt_sec))
        )
        steered = self._apply_rate_limited_steering(held, dt)
        # Force curve throttle — do not blast cruise while free-space is gone.
        self._last_mask_debug = {
            **getattr(self, '_last_mask_debug', {}),
            'mask_occlusion_hold': True,
            'mask_occlusion_hold_frames': int(self._mask_occlusion_hold_frames),
        }
        return PursuitResult(
            valid=True,
            steering=steered,
            target_x=0.35,
            target_y=0.0,
            path_points=0,
            target_distance=0.35,
            curve_ratio=1.0,
            raw_steering=held,
            pp_steering=held,
        )

    @staticmethod
    def _bev_path_skeleton(
        path_xy: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        import cv2
        from inference.modules import lane_detection as ld

        canvas = np.zeros((height, width), dtype=np.uint8)
        pts: list[list[int]] = []
        for x, y in path_xy:
            u, v = ld.vehicle_xy_to_bev_uv(float(x), float(y))
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < width and 0 <= vi < height:
                pts.append([ui, vi])
        if len(pts) < 2:
            return canvas
        cv2.polylines(
            canvas,
            [np.asarray(pts, dtype=np.int32)],
            isClosed=False,
            color=255,
            thickness=1,
            lineType=cv2.LINE_8,
        )
        return canvas

    @staticmethod
    def _bev_corridor_mask(
        path_xy: np.ndarray,
        height: int,
        width: int,
        mpp: float,
        half_width_m: float,
    ) -> np.ndarray:
        import cv2

        skeleton = MainPlanner._bev_path_skeleton(path_xy, height, width)
        if not np.any(skeleton):
            return skeleton
        thickness = max(1, int(round((2.0 * float(half_width_m)) / max(mpp, 1e-6))))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(1, thickness), max(1, thickness)),
        )
        return cv2.dilate(skeleton, kernel)

    def _note_out_sign_sighting(self, observed: TurnSign, now_sec: float) -> None:
        """Refresh the OUT fork-arm timer whenever a turn sign is visible."""
        if observed is not TurnSign.UNKNOWN:
            self._last_out_sign_sec = now_sec

    def _fork_perception_allowed(
        self,
        *,
        now_sec: float,
        observed_sign: TurnSign,
        out_capture: bool = False,
    ) -> bool:
        """Whether detect() should publish fork/branches this frame.

        OUT: turn-sign window AND (optional) ``out_fork_capture`` tip+stretch.
        IN: always allow perception so moment+yellow_alt can run; control is
        gated by circle_ignore / EXIT states (see §5.1.4).
        """
        from inference.modules.perception.fork.judgment import decide_out_fork_arm

        mid = self.state in (
            DrivingState.FORK_TURN,
            DrivingState.ROUNDABOUT_EXIT,
            DrivingState.ROUNDABOUT_EXIT_READY,
        )
        if self.config.route_mode is not RouteMode.OUT:
            self._last_fork_arm_reason = 'in_open'
            return True
        if (
            self.config.out_fork_forced_turn_arms
            and self._forced_turn is not TurnSign.UNKNOWN
        ):
            self._last_fork_arm_reason = 'forced_turn_arms'
            return True
        self._note_out_sign_sighting(observed_sign, now_sec)
        sign_window = False
        if self._last_out_sign_sec is not None:
            sign_window = (now_sec - self._last_out_sign_sec) <= float(
                self.config.out_fork_sign_hold_sec
            )
        decision = decide_out_fork_arm(
            sign_window=sign_window,
            capture=bool(out_capture),
            require_sign=self.config.out_fork_require_sign,
            require_capture=self.config.out_fork_require_capture,
            force_mid_manoeuvre=mid,
        )
        self._last_fork_arm_reason = decision.reason
        return bool(decision.arm)
    def _forkish_for_mask(self, lane: Any) -> bool:
        if not self.config.mask_fork_force_pp:
            return False
        # Circulating: stay on mask/yellow unless IN moment already selected a
        # keep/exit rank (pass policy wants fork follow ON — §5.1.4).
        if (
            self.config.circle_ignore_fork_for_control
            and self.state is DrivingState.ROUNDABOUT_CIRCLE
            and not (
                self.config.in_exit_use_moment
                and self._fork_selected_rank is not None
            )
        ):
            return False
        if not self._fork_perception_enabled:
            return False
        if bool(getattr(lane, 'fork_active', False)):
            return True
        return len(getattr(lane, 'branches', ()) or ()) >= 2
    def _effective_mask_corridor_mode(self, lane: Any) -> tuple[str, bool]:
        """Return (corridor_mode, require_color_path) for mask_p this frame."""

        base = str(self.config.mask_corridor_mode or 'off').lower()
        require_path = bool(self.config.mask_require_color_path)
        if base in ('hard', 'soft'):
            return base, require_path
        if not self.config.mask_corridor_near_fork:
            return 'off', require_path
        if self.config.route_mode is not RouteMode.OUT:
            return 'off', require_path
        if self.state is not DrivingState.NORMAL:
            return 'off', require_path
        if not self._fork_perception_enabled:
            return 'off', require_path
        # Only clip when fork geometry is visible — sign-hold alone must not
        # force hard corridor (white path flicker → path_lost).
        if bool(getattr(lane, 'fork_active', False)):
            return 'hard', True
        if len(getattr(lane, 'branches', ()) or ()) >= 2:
            return 'hard', True
        return 'off', require_path

    def _track_normal_path(
        self,
        lane: Any,
        color_path: np.ndarray,
        dt_sec: float,
        path_confidence: float | None = None,
    ) -> PursuitResult:
        tracker = str(self.config.normal_tracker or 'pp').lower()
        if tracker == 'hybrid':
            if self._forkish_for_mask(lane):
                return self._pure_pursuit(color_path, dt_sec)
            return self._hybrid_pursuit(
                lane, color_path, dt_sec, path_confidence=path_confidence
            )
        if tracker == 'mask_p':
            if self._forkish_for_mask(lane):
                return self._pure_pursuit(color_path, dt_sec)
            return self._mask_com_pursuit(
                lane,
                dt_sec,
                color_path=color_path,
                path_confidence=path_confidence,
            )
        if tracker == 'stanley':
            if self._forkish_for_mask(lane):
                return self._pure_pursuit(color_path, dt_sec)
            return self._stanley_pursuit(color_path, dt_sec)
        return self._pure_pursuit(color_path, dt_sec)

    @staticmethod
    def _smoothstep(x: float, lo: float, hi: float) -> float:
        if hi <= lo + 1e-9:
            return 1.0 if x >= lo else 0.0
        t = float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    def _apply_hybrid_smoothed_steering(self, raw: float, dt: float) -> float:
        """EMA then rate-limit — damps straight wobble without killing corner response."""
        alpha = float(np.clip(self.config.hybrid_steer_alpha, 0.05, 1.0))
        self._hybrid_steer_f = (1.0 - alpha) * self._hybrid_steer_f + alpha * float(raw)
        return self._apply_rate_limited_steering(float(self._hybrid_steer_f), dt)

    def _hybrid_pursuit(
        self,
        lane: Any,
        color_path: np.ndarray,
        dt_sec: float,
        path_confidence: float | None = None,
    ) -> PursuitResult:
        """Gated blend: PP (+CTE/heading) on straights, mask COM on curve/error."""
        dt = self.config.nominal_control_dt_sec if dt_sec is None else max(0.0, dt_sec)
        pp = self._pure_pursuit(color_path, dt, apply_rate_limit=False)
        if not pp.valid:
            return self._mask_com_pursuit(
                lane,
                dt,
                color_path=color_path,
                path_confidence=path_confidence,
                far_blend_override=0.0,
                use_path_correction_override=False,
            )

        # Probe near-only COM for lateral error (no EMA / rate / paint blend).
        probe = self._mask_com_pursuit(
            lane,
            dt,
            color_path=color_path,
            path_confidence=0.0,
            far_blend_override=0.0,
            apply_rate_limit=False,
            update_ema=False,
            use_path_correction_override=False,
        )
        if not probe.valid:
            steered = self._apply_hybrid_smoothed_steering(float(pp.raw_steering), dt)
            return PursuitResult(
                valid=True,
                steering=steered,
                target_x=pp.target_x,
                target_y=pp.target_y,
                path_points=pp.path_points,
                target_distance=pp.target_distance,
                lookahead_m=pp.lookahead_m,
                desired_lookahead_m=pp.desired_lookahead_m,
                path_curvature=pp.path_curvature,
                curve_ratio=pp.curve_ratio,
                raw_steering=float(pp.raw_steering),
                pp_steering=pp.pp_steering,
                cross_track_error_m=pp.cross_track_error_m,
                cte_steering=pp.cte_steering,
                heading_error_rad=pp.heading_error_rad,
                heading_steering=pp.heading_steering,
                path_extrapolation_m=pp.path_extrapolation_m,
            )

        err = abs(float(self._last_mask_debug.get('mask_error_near', 0.0)))
        e_eff = max(0.0, err - float(self.config.mask_error_deadband))
        kappa = abs(float(pp.path_curvature))
        w_e = self._smoothstep(
            e_eff,
            float(self.config.mask_blend_error_lo),
            float(self.config.mask_blend_error_hi),
        )
        w_k = self._smoothstep(
            kappa,
            float(self.config.mask_blend_curvature_lo),
            float(self.config.mask_blend_curvature_hi),
        )
        w = float(np.clip(max(w_e, w_k), 0.0, 1.0))
        far_eff = float(self.config.mask_far_blend) * w

        if w < 1e-3:
            steered = self._apply_hybrid_smoothed_steering(float(pp.raw_steering), dt)
            self._last_mask_debug = {
                **getattr(self, '_last_mask_debug', {}),
                'hybrid_w': 0.0,
                'hybrid_far_blend': 0.0,
                'hybrid_mode': 'pp',
            }
            return PursuitResult(
                valid=True,
                steering=steered,
                target_x=pp.target_x,
                target_y=pp.target_y,
                path_points=pp.path_points,
                target_distance=pp.target_distance,
                lookahead_m=pp.lookahead_m,
                desired_lookahead_m=pp.desired_lookahead_m,
                path_curvature=pp.path_curvature,
                curve_ratio=pp.curve_ratio,
                raw_steering=float(pp.raw_steering),
                pp_steering=pp.pp_steering,
                cross_track_error_m=pp.cross_track_error_m,
                cte_steering=pp.cte_steering,
                heading_error_rad=pp.heading_error_rad,
                heading_steering=pp.heading_steering,
                path_extrapolation_m=pp.path_extrapolation_m,
            )

        mask = self._mask_com_pursuit(
            lane,
            dt,
            color_path=color_path,
            path_confidence=path_confidence,
            far_blend_override=far_eff,
            apply_rate_limit=False,
            update_ema=True,
            use_path_correction_override=False,
        )
        if not mask.valid:
            steered = self._apply_hybrid_smoothed_steering(float(pp.raw_steering), dt)
            return PursuitResult(
                valid=True,
                steering=steered,
                target_x=pp.target_x,
                target_y=pp.target_y,
                path_points=pp.path_points,
                target_distance=pp.target_distance,
                lookahead_m=pp.lookahead_m,
                desired_lookahead_m=pp.desired_lookahead_m,
                path_curvature=pp.path_curvature,
                curve_ratio=pp.curve_ratio,
                raw_steering=float(pp.raw_steering),
                pp_steering=pp.pp_steering,
                cross_track_error_m=pp.cross_track_error_m,
                cte_steering=pp.cte_steering,
                heading_error_rad=pp.heading_error_rad,
                heading_steering=pp.heading_steering,
                path_extrapolation_m=pp.path_extrapolation_m,
            )

        raw = (1.0 - w) * float(pp.raw_steering) + w * float(mask.raw_steering)
        steered = self._apply_hybrid_smoothed_steering(raw, dt)
        curve_ratio = float(
            np.clip(max(pp.curve_ratio, mask.curve_ratio, w), 0.0, 1.0)
        )
        self._last_mask_debug = {
            **getattr(self, '_last_mask_debug', {}),
            'hybrid_w': round(w, 4),
            'hybrid_w_error': round(w_e, 4),
            'hybrid_w_curv': round(w_k, 4),
            'hybrid_far_blend': round(far_eff, 4),
            'hybrid_mode': 'blend',
        }
        return PursuitResult(
            valid=True,
            steering=steered,
            target_x=pp.target_x if w < 0.5 else mask.target_x,
            target_y=pp.target_y if w < 0.5 else mask.target_y,
            path_points=pp.path_points,
            target_distance=pp.target_distance,
            lookahead_m=pp.lookahead_m,
            desired_lookahead_m=pp.desired_lookahead_m,
            path_curvature=pp.path_curvature,
            curve_ratio=curve_ratio,
            raw_steering=float(raw),
            pp_steering=pp.pp_steering,
            cross_track_error_m=pp.cross_track_error_m,
            cte_steering=pp.cte_steering,
            heading_error_rad=pp.heading_error_rad,
            heading_steering=pp.heading_steering,
            path_extrapolation_m=pp.path_extrapolation_m,
        )

    def _pure_pursuit(
        self, path: np.ndarray, dt_sec: float | None = None, *, apply_rate_limit: bool = True
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
        dead = float(self.config.steer_command_deadband)
        if dead > 0.0 and abs(raw) < dead:
            raw = 0.0
        dt = self.config.nominal_control_dt_sec if dt_sec is None else max(0.0, dt_sec)
        if apply_rate_limit:
            self._apply_rate_limited_steering(raw, dt)
            steered = self._steering
        else:
            steered = float(np.clip(raw, -1.0, 1.0))
        return PursuitResult(
            valid=True,
            steering=steered,
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
        elif str(self.config.normal_tracker or 'pp').lower() in ('mask_p',):
            # limo_sim style: shrink cruise when |steer| is large.
            scale = 1.0 - (
                1.0 - self.config.mask_curve_speed_scale
            ) * float(np.clip(pursuit.curve_ratio, 0.0, 1.0))
            throttle = self.config.cruise_throttle * scale
            # Extra slowdown on big COM error / hard turn — reduces merge path-lost.
            steer_abs = abs(float(pursuit.steering))
            err_abs = abs(
                float(
                    (getattr(self, '_last_mask_debug', {}) or {}).get(
                        'mask_error_norm', 0.0
                    )
                )
            )
            err = max(
                float(
                    np.clip(
                        steer_abs / max(self.config.error_speed_steer_full, 1e-3),
                        0.0,
                        1.0,
                    )
                ),
                float(np.clip(err_abs / 0.55, 0.0, 1.0)),
            )
            if err > 0.2:
                throttle = float(throttle) * float(
                    np.clip(
                        1.0
                        - (1.0 - float(self.config.error_speed_min_scale)) * err,
                        0.0,
                        1.0,
                    )
                )
        else:
            throttle = (
                self.config.cruise_throttle * (1.0 - pursuit.curve_ratio)
                + self.config.curve_throttle * pursuit.curve_ratio
            )
            # Mild slowdown only when clearly off-path (keep usable cruise).
            cte_abs = abs(float(pursuit.cross_track_error_m))
            steer_abs = abs(float(pursuit.steering))
            cte_term = float(
                np.clip(
                    cte_abs / max(self.config.error_speed_cte_full_m, 1e-3),
                    0.0,
                    1.0,
                )
            )
            steer_term = float(
                np.clip(
                    steer_abs / max(self.config.error_speed_steer_full, 1e-3),
                    0.0,
                    1.0,
                )
            )
            err = max(cte_term, steer_term)
            if err > 0.15:
                scale = 1.0 - (
                    1.0 - float(self.config.error_speed_min_scale)
                ) * err
                throttle = float(throttle) * float(np.clip(scale, 0.0, 1.0))
        steering = float(np.clip(pursuit.steering + self.steer_trim, -1.0, 1.0))
        return ControlCommand(steering=steering, throttle=throttle)

    def step(self, frame: np.ndarray, *, now_sec: float) -> PlannerOutput:
        dt_sec = self._step_dt(now_sec)
        # Feed last lock into perception so overlays / publish drop the
        # opposite fork and ignore merge-lane starts (active_lane policy).
        # EXIT_READY stays in explore so both candidates remain until exit starts.
        active_rank = None
        if (
            self.state
            in (
                DrivingState.FORK_TURN,
                DrivingState.ROUNDABOUT_EXIT,
                DrivingState.ROUNDABOUT_CIRCLE,
            )
            and self._fork_selected_rank is not None
        ):
            # CIRCLE keep/exit: drop opposite fork in perception once moment latched.
            active_rank = int(self._fork_selected_rank)
        # Course contract: Out → white-only forks; In → yellow-first.
        prefer_yellow = (
            bool(self.config.prefer_yellow)
            if self.config.route_mode is RouteMode.IN
            else False
        )
        # Sign first so OUT can gate fork perception before detect().
        # Capture uses previous-frame latch (stretch lasts ≫1 frame).
        traffic = traffic_sign.detect(frame)
        aruco = aruco_detection.detect(frame)
        # Freeze mission counters/FSM while stopped so ArUco/red cannot
        # advance moment passes or arm forks under zero throttle.
        mission_freeze = bool(
            (self.config.stop_on_aruco and aruco.should_stop)
            or (
                self.config.stop_on_red
                and traffic.signal is TrafficSignal.RED
                and self.state is not DrivingState.WAIT_GREEN
            )
        )
        if not mission_freeze:
            self._note_out_sign_sighting(traffic.turn, now_sec)
            self._update_desired_turn(traffic.turn)
        enable_fork = self._fork_perception_allowed(
            now_sec=now_sec,
            observed_sign=traffic.turn,
            out_capture=bool(self._out_capture_latched),
        )
        self._fork_perception_enabled = enable_fork
        lane = lane_detection.detect(
            frame,
            active_branch_rank=active_rank,
            prefer_yellow=prefer_yellow,
            enable_fork=enable_fork and not mission_freeze,
        )
        # Refresh capture latch from this frame for next arm decision.
        # Hold while capture stays true; drop when stretch ends (unless mid FORK).
        if not mission_freeze:
            if bool(getattr(lane, 'out_fork_capture', False)):
                self._out_capture_latched = True
            elif self.state is not DrivingState.FORK_TURN:
                self._out_capture_latched = False

        branch_event = False
        crossing_event = False
        moment_rising = False
        if not mission_freeze:
            branch_event = self.branch_counter.update(bool(lane.fork_active))
            crossing_event = self.crossing_counter.update(
                bool(lane.yellow_crossing_line)
            )
            moment_rising = self.moment_counter.update(
                bool(getattr(lane, 'in_circle_fork_moment', False))
            )
            self._apply_in_moment_pass(moment_rising)
        elapsed = (
            0.0
            if self._roundabout_started_at is None
            else max(0.0, now_sec - self._roundabout_started_at)
        )

        if self.state is DrivingState.WAIT_GREEN and traffic.signal is TrafficSignal.GREEN:
            self._set_state(DrivingState.NORMAL, now_sec)

        color_path, path_source, path_confidence = self._color_path(lane)

        if not mission_freeze:
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
                and enable_fork
                and now_sec >= self._fork_cooldown_until_sec
                and (
                    branch_event
                    or bool(getattr(lane, 'fork_active', False))
                )
            ):
                # Require a confirmed L/R before locking — never default_unknown
                # when a sign is still being confirmed. Allow level fork_active
                # so confirm can finish after the rising edge.
                if self._forced_turn is not TurnSign.UNKNOWN:
                    self.apply_forced_turn(self._forced_turn)
                    self._set_state(DrivingState.FORK_TURN, now_sec)
                elif self.desired_turn is not TurnSign.UNKNOWN:
                    self._lock_fork_selection()
                    self._set_state(DrivingState.FORK_TURN, now_sec)

            if self.state is DrivingState.ROUNDABOUT_CIRCLE:
                enough_time = elapsed >= self.config.min_lap_time_sec
                if self.config.in_exit_use_moment:
                    # Moment pass only — do not let yellow crossing alone exit.
                    branch_ready = self._in_fork_pass_count > int(
                        self.config.in_keep_passes
                    )
                    exit_gate = branch_ready
                else:
                    branch_ready = (
                        self.branch_counter.events
                        >= self.config.branch_required_events
                    )
                    crossing_ready = (
                        self.crossing_counter.events
                        >= self.config.crossing_required_events
                    )
                    exit_gate = branch_ready or crossing_ready
                if (
                    enough_time
                    and exit_gate
                    and self._wants_roundabout_exit()
                ):
                    self._set_state(DrivingState.ROUNDABOUT_EXIT_READY, now_sec)

            if self.state is DrivingState.ROUNDABOUT_EXIT_READY:
                if self._fork_selected_rank is None:
                    if self._forced_turn is not TurnSign.UNKNOWN:
                        self.apply_forced_turn(self._forced_turn)
                    else:
                        self._lock_fork_selection()
                exit_rank = int(
                    self._fork_selected_rank
                    if self._fork_selected_rank is not None
                    else self.config.exit_branch_rank
                )
                branch = self._ranked_branch(lane, exit_rank)
                # Dual-branch preferred; single ranked branch OK after moment latch.
                if branch is not None and (
                    len(lane.branches) >= 2 or self.config.in_exit_use_moment
                ):
                    self._set_state(DrivingState.ROUNDABOUT_EXIT, now_sec)

        pursuit = PursuitResult(False)
        selected_branch_rank: int | None = None
        branch_selection_reason = 'none'
        if self.config.stop_on_aruco and aruco.should_stop:
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
                # Locked ego corridor (opposite fork already dropped by perception).
                path, path_source, path_confidence = self._locked_ego_path(
                    lane, rank
                )
                strict = self._strict_ranked_branch(lane, rank)
                if (
                    strict is not None
                    and path.shape[0] >= self.config.min_points
                ):
                    self._fork_cached_path = path.copy()
                    self._fork_cached_source = path_source
                    self._fork_cached_confidence = path_confidence
                    pursuit = self._pure_pursuit(path, dt_sec)
                    decision = f'out_fork_ego_follow_rank{rank}'
                    self._fork_absent_frames = 0
                else:
                    self._fork_absent_frames += 1
                    if (
                        self._fork_cached_path.shape[0] >= self.config.min_points
                        and self._fork_absent_frames
                        <= self.config.fork_path_hold_frames
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
                path, path_source, path_confidence = self._locked_ego_path(
                    lane, rank
                )
                strict = self._strict_ranked_branch(lane, rank)
                if (
                    strict is not None
                    and path.shape[0] >= self.config.min_points
                ):
                    self._fork_cached_path = path.copy()
                    self._fork_cached_source = path_source
                    self._fork_cached_confidence = path_confidence
                    pursuit = self._pure_pursuit(path, dt_sec)
                    decision = f'roundabout_exit_ego_follow_rank{rank}'
                    self._fork_absent_frames = 0
                else:
                    self._fork_absent_frames += 1
                    if (
                        self._fork_cached_path.shape[0] >= self.config.min_points
                        and self._fork_absent_frames
                        <= self.config.fork_path_hold_frames
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
            # IN CIRCLE: after moment latched keep/exit, follow that branch while
            # fork geometry is visible (not just yellow centerline).
            circle_rank_follow = False
            if (
                self.state is DrivingState.ROUNDABOUT_CIRCLE
                and self._fork_selected_rank is not None
                and (
                    bool(getattr(lane, 'fork_active', False))
                    or len(getattr(lane, 'branches', ()) or ()) >= 2
                )
            ):
                rank = int(self._fork_selected_rank)
                selected_branch_rank = rank
                branch_selection_reason = self._fork_selection_reason
                path = np.empty((0, 2), dtype=np.float32)
                if lane.fork_active and len(lane.branches) >= 2:
                    path, path_source, path_confidence = self._selected_layer_path(
                        lane, rank
                    )
                if path.shape[0] < self.config.min_points:
                    path, path_source, path_confidence = self._locked_ego_path(
                        lane, rank
                    )
                if path.shape[0] >= self.config.min_points:
                    self._fork_cached_path = path.copy()
                    self._fork_cached_source = path_source
                    self._fork_cached_confidence = path_confidence
                    pursuit = self._pure_pursuit(path, dt_sec)
                    tag = (
                        'keep'
                        if rank == int(self.config.in_keep_branch_rank)
                        else 'exit'
                    )
                    decision = f'roundabout_circle_{tag}_rank{rank}'
                    circle_rank_follow = True

            if circle_rank_follow:
                pass
            else:
                forkish = self._forkish_for_mask(lane)
                tracker = str(self.config.normal_tracker or 'pp').lower()
                # Circle: follow yellow/white paint path (PP/Stanley), not free-space COM.
                if self.state is DrivingState.ROUNDABOUT_CIRCLE:
                    ct = str(self.config.circle_tracker or 'pp').strip().lower()
                    if ct in ('pp', 'stanley', 'mask_p', 'hybrid'):
                        tracker = ct
                use_mask = (
                    tracker == 'mask_p'
                    and self.state
                    in (DrivingState.NORMAL, DrivingState.ROUNDABOUT_CIRCLE)
                    and not forkish
                )
                use_stanley = (
                    tracker == 'stanley'
                    and self.state
                    in (DrivingState.NORMAL, DrivingState.ROUNDABOUT_CIRCLE)
                    and not forkish
                )
                use_hybrid = (
                    tracker == 'hybrid'
                    and self.state
                    in (DrivingState.NORMAL, DrivingState.ROUNDABOUT_CIRCLE)
                    and not forkish
                )
                if use_hybrid:
                    self._last_mask_debug = {}
                    pursuit = self._hybrid_pursuit(
                        lane, color_path, dt_sec, path_confidence=path_confidence
                    )
                    if pursuit.valid:
                        w = float(
                            (getattr(self, '_last_mask_debug', {}) or {}).get(
                                'hybrid_w', 0.0
                            )
                        )
                        path_source = (
                            PathSource.MASK_DRIVABLE
                            if w >= 0.5
                            else path_source
                        )
                        decision = (
                            'roundabout_circle_hybrid'
                            if self.state is DrivingState.ROUNDABOUT_CIRCLE
                            else 'normal_hybrid'
                        )
                    else:
                        pursuit = self._pure_pursuit(color_path, dt_sec)
                        decision = (
                            'roundabout_circle_hybrid_fallback_pp'
                            if self.state is DrivingState.ROUNDABOUT_CIRCLE
                            else 'normal_hybrid_fallback_pp'
                        )
                elif use_stanley:
                    self._last_mask_debug = {}
                    pursuit = self._stanley_pursuit(color_path, dt_sec)
                    if pursuit.valid:
                        decision = (
                            'roundabout_circle_stanley'
                            if self.state is DrivingState.ROUNDABOUT_CIRCLE
                            else 'normal_stanley'
                        )
                    else:
                        pursuit = self._pure_pursuit(color_path, dt_sec)
                        decision = (
                            'roundabout_circle_stanley_fallback_pp'
                            if self.state is DrivingState.ROUNDABOUT_CIRCLE
                            else 'normal_stanley_fallback_pp'
                        )
                elif use_mask:
                    self._last_mask_debug = {}
                    pursuit = self._mask_com_pursuit(
                        lane,
                        dt_sec,
                        color_path=color_path,
                        path_confidence=path_confidence,
                    )
                    if pursuit.valid:
                        holding = bool(
                            (self._last_mask_debug or {}).get('mask_occlusion_hold')
                        )
                        in_circle = self.state is DrivingState.ROUNDABOUT_CIRCLE
                        if holding:
                            path_source = PathSource.HOLD_PREVIOUS
                            decision = (
                                'roundabout_circle_mask_occlusion_hold'
                                if in_circle
                                else 'normal_mask_occlusion_hold'
                            )
                        else:
                            path_source = PathSource.MASK_DRIVABLE
                            decision = (
                                'roundabout_circle_mask'
                                if in_circle
                                else 'normal_mask_follow'
                            )
                    else:
                        pursuit = self._pure_pursuit(color_path, dt_sec)
                        decision = (
                            'roundabout_circle_mask_fallback_pp'
                            if self.state is DrivingState.ROUNDABOUT_CIRCLE
                            else 'normal_mask_fallback_pp'
                        )
                elif (
                    tracker in ('mask_p', 'hybrid', 'stanley')
                    and forkish
                    and self.state
                    in (DrivingState.NORMAL, DrivingState.ROUNDABOUT_CIRCLE)
                ):
                    # Keep color path_source so logs show fork guard, not mask COM.
                    pursuit = self._pure_pursuit(color_path, dt_sec)
                    decision = (
                        'roundabout_circle_mask_fork_pp'
                        if self.state is DrivingState.ROUNDABOUT_CIRCLE
                        else 'normal_mask_fork_pp'
                    )
                else:
                    pursuit = self._pure_pursuit(color_path, dt_sec)
                    if self.state is DrivingState.ROUNDABOUT_CIRCLE:
                        # Paint centerline PP (yellow IN / white fallback).
                        decision = (
                            'roundabout_circle_lane_pp'
                            if path_source
                            in (
                                PathSource.YELLOW_CENTERLINE,
                                PathSource.WHITE_CENTERLINE,
                            )
                            else 'roundabout_circle'
                        )
                    elif self.state is DrivingState.ROUNDABOUT_EXIT_READY:
                        decision = 'roundabout_exit_wait_branch'
                    else:
                        decision = 'normal_lane_follow'

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
            'traffic_pass': (
                not self.config.require_green_to_start
                and not self.config.stop_on_red
            ),
            'state': self.state.value,
            'decision': decision,
            'mission_freeze': bool(mission_freeze),
            'path_source': path_source.value,
            'path_confidence': round(float(path_confidence), 3),
            'normal_tracker': str(self.config.normal_tracker),
            'circle_tracker': str(self.config.circle_tracker),
            'mask_corridor_mode': str(self.config.mask_corridor_mode),
            'mask_fork_force_pp': bool(self.config.mask_fork_force_pp),
            'mask_forkish': bool(self._forkish_for_mask(lane)),
            **{
                k: v
                for k, v in (self._last_mask_debug or {}).items()
                if k.startswith('mask_')
                or k.startswith('track_')
                or k.startswith('stanley_')
                or k.startswith('hybrid_')
            },
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
            'fork_perception': bool(enable_fork),
            'fork_arm_reason': str(self._last_fork_arm_reason),
            'out_fork_capture': bool(getattr(lane, 'out_fork_capture', False)),
            'in_circle_fork_moment': bool(
                getattr(lane, 'in_circle_fork_moment', False)
            ),
            'in_fork_pass_count': int(self._in_fork_pass_count),
            'moment_rising': bool(moment_rising),
            'branch_count': len(lane.branches),
            'branch_event': branch_event,
            'branch_events': self.branch_counter.events,
            'moment_events': self.moment_counter.events,
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
            'forced_turn': self._forced_turn.value,
            'selected_branch_rank': selected_branch_rank,
            'branch_selection_reason': branch_selection_reason,
            'lane_policy': str(getattr(lane, 'lane_policy', 'explore') or 'explore'),
            'aruco_detected': aruco.detected,
            'aruco_stop': bool(self.config.stop_on_aruco and aruco.should_stop),
            'aruco_should_stop': aruco.should_stop,
            'stop_on_aruco': bool(self.config.stop_on_aruco),
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
