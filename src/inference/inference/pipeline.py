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
    lookahead_m: float = 1.00
    wheelbase_m: float = 0.24
    max_steer_angle_rad: float = 0.5236
    max_steering_command: float = 1.0
    steering_rate_limit: float = 0.15
    cruise_throttle: float = 0.25
    curve_throttle: float = 0.18
    curve_steering_threshold: float = 0.50
    default_throttle: float = 0.0
    min_points: int = 5
    white_min_confidence: float = 0.15
    yellow_min_confidence: float = 0.15
    yellow_valid_on_frames: int = 3
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
    require_green_to_start: bool = False
    stop_on_red: bool = False
    command_watchdog_sec: float = 0.5
    log_state_changes: bool = True
    log_decision_changes: bool = True


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
    return PlannerConfig(
        route_mode=RouteMode(str(selected_route).lower()),
        prefer_yellow=bool(route.get('prefer_yellow', False)),
        default_out_branch_rank=int(route.get('default_out_branch_rank', 0)),
        lookahead_m=max(0.05, float(pp.get('lookahead_m', 0.60))),
        wheelbase_m=max(0.01, float(pp.get('wheelbase_m', 0.24))),
        max_steer_angle_rad=max(0.01, float(pp.get('max_steer_angle_rad', 0.5236))),
        max_steering_command=float(np.clip(pp.get('max_steering_command', 1.0), 0.1, 1.0)),
        steering_rate_limit=float(np.clip(pp.get('steering_rate_limit', 0.15), 0.01, 1.0)),
        cruise_throttle=float(np.clip(speed.get('cruise_throttle', 0.25), -1.0, 1.0)),
        curve_throttle=float(np.clip(speed.get('curve_throttle', 0.18), -1.0, 1.0)),
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
        require_green_to_start=bool(signals.get('require_green_to_start', False)),
        stop_on_red=bool(signals.get('stop_on_red', False)),
        command_watchdog_sec=max(0.1, float(safety.get('command_watchdog_sec', 0.5))),
        log_state_changes=bool(debug.get('log_state_changes', True)),
        log_decision_changes=bool(debug.get('log_decision_changes', True)),
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
        self._last_path_source = PathSource.NONE

    def _set_state(self, state: DrivingState, now_sec: float) -> None:
        if state == self.state:
            return
        self.state = state
        if state == DrivingState.ROUNDABOUT_CIRCLE:
            self._roundabout_started_at = now_sec
            self.branch_counter.reset()
            self.crossing_counter.reset()
        if state in (DrivingState.NORMAL, DrivingState.FINISHED):
            self._fork_absent_frames = 0

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

    @staticmethod
    def _ranked_branch(lane: Any, rank: int) -> Any | None:
        branches = list(getattr(lane, 'branches', ()))
        if not branches:
            return None
        index = rank if rank >= 0 else len(branches) + rank
        if index < 0 or index >= len(branches):
            return None
        return branches[index]

    def _branch_path(self, lane: Any, rank: int) -> tuple[np.ndarray, PathSource, float]:
        branch = self._ranked_branch(lane, rank)
        if branch is None:
            return np.empty((0, 2), dtype=np.float32), PathSource.NONE, 0.0
        source = PathSource.LEFT_BRANCH if rank == 0 else PathSource.RIGHT_BRANCH
        return np.asarray(branch.points, dtype=np.float32), source, float(branch.confidence)

    def _pure_pursuit(self, path: np.ndarray) -> PursuitResult:
        points = np.asarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return PursuitResult(False)
        xy = points[:, :2]
        xy = xy[np.isfinite(xy).all(axis=1) & (xy[:, 0] > 0.0)]
        if xy.shape[0] < self.config.min_points:
            return PursuitResult(False, path_points=int(xy.shape[0]))
        distances = np.linalg.norm(xy, axis=1)
        ahead = np.flatnonzero(distances >= self.config.lookahead_m)
        target_index = (
            int(ahead[np.argmin(distances[ahead])])
            if ahead.size
            else int(np.argmax(distances))
        )
        target_x, target_y = (float(value) for value in xy[target_index])
        distance_sq = target_x * target_x + target_y * target_y
        if distance_sq <= 1e-6:
            return PursuitResult(False, path_points=int(xy.shape[0]))
        curvature = 2.0 * target_y / distance_sq
        steer_angle = math.atan(self.config.wheelbase_m * curvature)
        # base_link +y is left; D-Racer normalized steering is negative for left.
        raw = -steer_angle / self.config.max_steer_angle_rad
        raw = float(
            np.clip(
                raw,
                -self.config.max_steering_command,
                self.config.max_steering_command,
            )
        )
        delta = float(
            np.clip(
                raw - self._steering,
                -self.config.steering_rate_limit,
                self.config.steering_rate_limit,
            )
        )
        self._steering = float(np.clip(self._steering + delta, -1.0, 1.0))
        return PursuitResult(True, self._steering, target_x, target_y, int(xy.shape[0]))

    def _stop(self) -> ControlCommand:
        steering = float(np.clip(self._steering + self.steer_trim, -1.0, 1.0))
        return ControlCommand(steering=steering, throttle=0.0)

    def _drive(self, pursuit: PursuitResult) -> ControlCommand:
        throttle = (
            self.config.curve_throttle
            if abs(pursuit.steering) >= self.config.curve_steering_threshold
            else self.config.cruise_throttle
        )
        steering = float(np.clip(pursuit.steering + self.steer_trim, -1.0, 1.0))
        return ControlCommand(steering=steering, throttle=throttle)

    def step(self, frame: np.ndarray, *, now_sec: float) -> PlannerOutput:
        lane = lane_detection.detect(frame)
        traffic = traffic_sign.detect(frame)
        aruco = aruco_detection.detect(frame)

        if traffic.turn is not TurnSign.UNKNOWN:
            self.desired_turn = traffic.turn

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
            and lane.fork_active
        ):
            self._set_state(DrivingState.FORK_TURN, now_sec)

        if self.state is DrivingState.ROUNDABOUT_CIRCLE:
            enough_time = elapsed >= self.config.min_lap_time_sec
            branch_ready = self.branch_counter.events >= self.config.branch_required_events
            crossing_ready = self.crossing_counter.events >= self.config.crossing_required_events
            if enough_time and (branch_ready or crossing_ready):
                self._set_state(DrivingState.ROUNDABOUT_EXIT_READY, now_sec)

        if self.state is DrivingState.ROUNDABOUT_EXIT_READY:
            branch = self._ranked_branch(lane, self.config.exit_branch_rank)
            if branch is not None and len(lane.branches) >= 2:
                self._set_state(DrivingState.ROUNDABOUT_EXIT, now_sec)

        pursuit = PursuitResult(False)
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
            if self.desired_turn is TurnSign.LEFT:
                rank = 0
            elif self.desired_turn is TurnSign.RIGHT:
                rank = -1
            else:
                rank = self.config.default_out_branch_rank
            path, path_source, path_confidence = self._branch_path(lane, rank)
            pursuit = self._pure_pursuit(path)
            decision = f'out_fork_{self.desired_turn.value}'
            self._fork_absent_frames = 0 if lane.fork_active else self._fork_absent_frames + 1
            if self._fork_absent_frames >= self.config.fork_exit_off_frames:
                self._set_state(DrivingState.NORMAL, now_sec)
        elif self.state is DrivingState.ROUNDABOUT_EXIT:
            path, path_source, path_confidence = self._branch_path(
                lane, self.config.exit_branch_rank
            )
            pursuit = self._pure_pursuit(path)
            decision = 'roundabout_exit_branch'
            self._fork_absent_frames = 0 if lane.fork_active else self._fork_absent_frames + 1
            if self._fork_absent_frames >= self.config.fork_exit_off_frames:
                self._set_state(DrivingState.NORMAL, now_sec)
        else:
            pursuit = self._pure_pursuit(color_path)
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
            command = (
                ControlCommand(
                    float(np.clip(self._steering + self.steer_trim, -1.0, 1.0)),
                    self.config.default_throttle,
                )
                if self._path_lost_frames >= self.config.path_lost_stop_frames
                else ControlCommand(
                    float(np.clip(self._steering + self.steer_trim, -1.0, 1.0)),
                    self.config.curve_throttle,
                )
            )
            path_source = PathSource.HOLD_PREVIOUS
            decision = f'{decision}_path_lost'

        self._last_path_source = path_source
        debug = {
            'route': self.config.route_mode.value,
            'state': self.state.value,
            'decision': decision,
            'path_source': path_source.value,
            'path_confidence': round(float(path_confidence), 3),
            'path_points': pursuit.path_points,
            'target_x': round(pursuit.target_x, 3),
            'target_y': round(pursuit.target_y, 3),
            'white_visible': bool(lane.white_visible),
            'white_confidence': round(float(lane.white_confidence), 3),
            'yellow_visible': bool(lane.yellow_visible),
            'yellow_confidence': round(float(lane.yellow_confidence), 3),
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
            'aruco_detected': aruco.detected,
            'aruco_stop': aruco.should_stop,
            'steering': round(command.steering, 3),
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
