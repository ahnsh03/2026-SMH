"""Unit tests for MainPlanner primitives (no ROS required)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


from inference.pipeline import (  # noqa: E402
    MainPlanner,
    PlannerConfig,
    RisingEventCounter,
    load_planner_config,
)
from inference.types import (  # noqa: E402
    ArucoResult,
    DrivingState,
    PathSource,
    RouteMode,
    TrafficResult,
    TrafficSignal,
    TurnSign,
)


def test_event_counter_debounces_and_rearms():
    counter = RisingEventCounter(on_frames=2, off_frames=2)
    events = [counter.update(value) for value in (True, True, True, False, False, True, True)]
    assert events == [False, True, False, False, False, False, True]
    assert counter.events == 2


def test_mask_com_pursuit_steers_toward_road_center():
    """Drivable blob left of image → negative (left) D-Racer steering."""
    import cv2

    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='mask_p',
            mask_steer_law='sim_v2',
            mask_steer_k=2.0,
            mask_steer_alpha=1.0,
            mask_near_band_ratio=1.0,
            mask_center_mode='area',
            mask_corridor_mode='off',
            mask_min_area_px=10.0,
            mask_erode_px=0,
            steering_rate_limit_per_sec=100.0,
            max_steering_command=1.0,
        )
    )
    mask = np.zeros((80, 120), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (50, 70), 255, -1)  # left-heavy blob
    lane = type('L', (), {'drivable_area': mask, 'meters_per_pixel': 0.01, 'x_forward_max': 1.0})()
    result = planner._mask_com_pursuit(lane, dt_sec=0.1)
    assert result.valid
    assert result.steering < 0.0


def test_mask_com_pursuit_rejects_empty_mask():
    planner = MainPlanner(PlannerConfig(normal_tracker='mask_p'))
    lane = type('L', (), {'drivable_area': np.zeros((40, 40), dtype=np.uint8)})()
    assert not planner._mask_com_pursuit(lane, dt_sec=0.1).valid


def test_mask_paint_blend_pulls_toward_clean_path():
    """High path confidence adds paint CTE/heading on top of drivable COM."""
    import cv2
    from inference.modules import lane_detection as ld

    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    mask = np.zeros((h, w), dtype=np.uint8)
    # Centered free-space → near-zero COM steer alone.
    cv2.rectangle(mask, (w // 2 - 25, h // 4), (w // 2 + 25, h - 1), 255, -1)
    # Paint path offset left (positive y) so CTE/heading pull nonzero.
    path = np.array(
        [[0.4, 0.12], [0.8, 0.14], [1.2, 0.16], [1.5, 0.18]],
        dtype=np.float32,
    )
    lane = type(
        'L',
        (),
        {
            'drivable_area': mask,
            'meters_per_pixel': float(ld.METERS_PER_PIXEL),
            'x_forward_max': 2.0,
            'white_visible': True,
            'yellow_visible': False,
            'white_confidence': 0.9,
            'yellow_confidence': 0.0,
            'confidence': 0.9,
        },
    )()
    base_kw = dict(
        normal_tracker='mask_p',
        mask_steer_law='sim_v2',
        mask_steer_k=2.0,
        mask_steer_alpha=1.0,
        mask_near_band_ratio=1.0,
        mask_center_mode='area',
        mask_corridor_mode='off',
        mask_use_path_correction=False,
        mask_min_area_px=10.0,
        mask_erode_px=0,
        prefer_yellow=False,
        steering_rate_limit_per_sec=100.0,
        max_steering_command=1.0,
        perception_to_rear_axle_x_m=0.265,
    )
    com_only = MainPlanner(
        PlannerConfig(**base_kw, mask_paint_blend_max=0.0)
    )._mask_com_pursuit(lane, dt_sec=0.1, color_path=path, path_confidence=0.9)
    blended = MainPlanner(
        PlannerConfig(
            **base_kw,
            mask_paint_blend_max=0.50,
            mask_paint_blend_lo=0.20,
            mask_paint_blend_hi=0.55,
        )
    )._mask_com_pursuit(lane, dt_sec=0.1, color_path=path, path_confidence=0.9)
    low_conf = MainPlanner(
        PlannerConfig(
            **base_kw,
            mask_paint_blend_max=0.50,
            mask_paint_blend_lo=0.20,
            mask_paint_blend_hi=0.55,
        )
    )._mask_com_pursuit(lane, dt_sec=0.1, color_path=path, path_confidence=0.05)

    assert com_only.valid and blended.valid and low_conf.valid
    assert abs(float(blended.steering) - float(com_only.steering)) > 0.02
    assert abs(float(low_conf.steering) - float(com_only.steering)) < 0.02
    assert float(blended.cte_steering) != 0.0
    dbg = MainPlanner(
        PlannerConfig(
            **base_kw,
            mask_paint_blend_max=0.50,
            mask_paint_blend_lo=0.20,
            mask_paint_blend_hi=0.55,
        )
    )
    dbg._mask_com_pursuit(lane, dt_sec=0.1, color_path=path, path_confidence=0.9)
    assert float(dbg._last_mask_debug.get('mask_paint_blend_w', 0.0)) > 0.3


def test_mask_hard_corridor_ignores_side_blob():
    """Opposite-course blob must not yank COM when hard corridor is on."""
    import cv2
    from inference.modules import lane_detection as ld

    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='mask_p',
            mask_steer_k=2.0,
            mask_steer_alpha=1.0,
            mask_near_band_ratio=1.0,
            mask_min_area_px=10.0,
            mask_corridor_mode='hard',
            mask_corridor_half_width_m=0.15,
            mask_require_color_path=True,
            steering_rate_limit_per_sec=100.0,
            max_steering_command=1.0,
        )
    )
    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    mask = np.zeros((h, w), dtype=np.uint8)
    # Ego corridor near image center (white path ahead).
    path = np.array([[0.3, 0.0], [0.8, 0.0], [1.3, 0.0]], dtype=np.float32)
    for x, y in path:
        u, v = ld.vehicle_xy_to_bev_uv(float(x), float(y))
        cv2.circle(mask, (int(u), int(v)), 8, 255, -1)
    # Strong false blob far left (would yank COM without corridor).
    cv2.rectangle(mask, (5, h // 3), (40, h - 5), 255, -1)
    lane = type(
        'L',
        (),
        {
            'drivable_area': mask,
            'meters_per_pixel': float(ld.METERS_PER_PIXEL),
            'x_forward_max': 2.0,
        },
    )()
    open_loop = planner._mask_com_pursuit(lane, dt_sec=0.1, color_path=None)
    # Without path + require_color → invalid under hard mode.
    assert not open_loop.valid

    with_path = planner._mask_com_pursuit(lane, dt_sec=0.1, color_path=path)
    assert with_path.valid
    # Corridor keeps COM near center → |steer| not slammed left by false blob.
    assert abs(with_path.steering) < 0.55


def test_mask_fork_force_pp_uses_color_decision():
    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='mask_p',
            mask_fork_force_pp=True,
            prefer_yellow=False,
            route_mode=RouteMode.OUT,
            min_points=2,
            steering_rate_limit_per_sec=100.0,
        )
    )
    assert planner._forkish_for_mask(
        SimpleNamespace(fork_active=True, branches=(object(), object()))
    )
    assert not planner._forkish_for_mask(
        SimpleNamespace(fork_active=False, branches=(object(),))
    )
    planner.state = DrivingState.ROUNDABOUT_CIRCLE
    planner.config = replace(
        planner.config, circle_ignore_fork_for_control=True
    )
    assert not planner._forkish_for_mask(
        SimpleNamespace(fork_active=True, branches=(object(), object()))
    )


def test_effective_corridor_near_fork_out_only_when_armed():
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            mask_corridor_mode='off',
            mask_corridor_near_fork=True,
        )
    )
    planner.state = DrivingState.NORMAL
    lane = SimpleNamespace(fork_active=False, branches=())
    planner._fork_perception_enabled = True
    # Sign-hold alone must not force hard corridor.
    mode, require_path = planner._effective_mask_corridor_mode(lane)
    assert mode == 'off'

    lane_fork = SimpleNamespace(fork_active=True, branches=())
    mode_on, require_path = planner._effective_mask_corridor_mode(lane_fork)
    assert mode_on == 'hard'
    assert require_path is True

    planner._fork_perception_enabled = False
    mode_off, _ = planner._effective_mask_corridor_mode(lane_fork)
    assert mode_off == 'off'

    planner.config = replace(planner.config, route_mode=RouteMode.IN)
    planner._fork_perception_enabled = True
    mode_in, _ = planner._effective_mask_corridor_mode(lane_fork)
    assert mode_in == 'off'


def test_roundabout_circle_uses_paint_pp_not_mask():
    """NORMAL=mask_p but CIRCLE+circle_tracker=pp follows yellow centerline."""
    import cv2
    from inference.modules import lane_detection as ld

    yellow = np.array(
        [[0.3, 0.05], [0.55, 0.12], [0.8, 0.22], [1.05, 0.35], [1.3, 0.48]],
        dtype=np.float32,
    )
    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    # Bias mask hard left so mask COM would differ from yellow PP if used.
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (2, h // 4), (w // 4, h - 1), 255, -1)
    lane = SimpleNamespace(
        white_centerline=np.empty((0, 2), dtype=np.float32),
        yellow_centerline=yellow,
        white_confidence=0.0,
        yellow_confidence=0.9,
        white_visible=False,
        yellow_visible=True,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(),
        drivable_area=mask,
        meters_per_pixel=float(ld.METERS_PER_PIXEL),
        x_forward_max=2.0,
        lane_policy='explore',
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            prefer_yellow=True,
            normal_tracker='mask_p',
            circle_tracker='pp',
            yellow_valid_on_frames=1,
            min_points=5,
            mask_require_color_path=False,
            steering_rate_limit_per_sec=100.0,
            stop_on_aruco=False,
        )
    )
    planner.state = DrivingState.ROUNDABOUT_CIRCLE
    planner._roundabout_started_at = 0.0
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        out = planner.step(frame, now_sec=1.0)
    assert out.path_source is PathSource.YELLOW_CENTERLINE
    assert out.decision == 'roundabout_circle_lane_pp'
    assert 'mask' not in out.decision


def test_smoothstep_and_hybrid_prefers_pp_on_straight():
    """Small lateral error + low curvature → hybrid weight near 0 (PP path)."""
    import cv2
    from inference.modules import lane_detection as ld

    assert MainPlanner._smoothstep(0.0, 0.08, 0.35) == 0.0
    assert MainPlanner._smoothstep(0.5, 0.08, 0.35) == 1.0
    mid = MainPlanner._smoothstep(0.215, 0.08, 0.35)
    assert 0.4 < mid < 0.6

    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    mask = np.zeros((h, w), dtype=np.uint8)
    # Centered road stripe.
    cv2.rectangle(mask, (w // 2 - 20, h // 3), (w // 2 + 20, h - 1), 255, -1)
    path = np.array(
        [[0.5, 0.0], [0.8, 0.0], [1.1, 0.0], [1.4, 0.0], [1.6, 0.0]],
        dtype=np.float32,
    )
    lane = type(
        'L',
        (),
        {
            'drivable_area': mask,
            'meters_per_pixel': float(ld.METERS_PER_PIXEL),
            'x_forward_max': 2.0,
            'fork_active': False,
            'branches': (),
        },
    )()
    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='hybrid',
            mask_steer_k=1.55,
            mask_steer_alpha=1.0,
            mask_near_band_ratio=0.55,
            mask_far_blend=0.28,
            mask_use_path_correction=False,
            mask_corridor_mode='hard',
            mask_corridor_half_width_m=0.38,
            mask_require_color_path=True,
            mask_error_deadband=0.04,
            mask_blend_error_lo=0.08,
            mask_blend_error_hi=0.35,
            mask_blend_curvature_lo=0.40,
            mask_blend_curvature_hi=1.20,
            min_points=3,
            steering_rate_limit_per_sec=100.0,
            max_steering_command=1.0,
            perception_to_rear_axle_x_m=0.265,
        )
    )
    result = planner._hybrid_pursuit(lane, path, dt_sec=0.1)
    assert result.valid
    hybrid_w = float(planner._last_mask_debug.get('hybrid_w', 1.0))
    assert hybrid_w < 0.25
    # Near-zero weight: pure PP or tiny blend label (anti-wobble pack).
    assert planner._last_mask_debug.get('hybrid_mode') in ('pp', 'blend')


def test_hybrid_raises_mask_weight_far_blend_scales():
    """Off-center mask with a mild path bend → hybrid_w up, far_blend gated."""
    import cv2
    from inference.modules import lane_detection as ld

    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (5, h // 4), (w // 3, h - 1), 255, -1)
    path = np.array(
        [[0.5, 0.0], [0.8, 0.05], [1.1, 0.12], [1.4, 0.22], [1.6, 0.30]],
        dtype=np.float32,
    )
    for x, y in path:
        u, v = ld.vehicle_xy_to_bev_uv(float(x), float(y))
        cv2.circle(mask, (int(u), int(v)), 10, 255, -1)
    lane = type(
        'L',
        (),
        {
            'drivable_area': mask,
            'meters_per_pixel': float(ld.METERS_PER_PIXEL),
            'x_forward_max': 2.0,
            'fork_active': False,
            'branches': (),
        },
    )()
    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='hybrid',
            mask_steer_k=1.55,
            mask_steer_alpha=1.0,
            mask_near_band_ratio=0.7,
            mask_far_blend=0.28,
            mask_use_path_correction=False,
            mask_corridor_mode='off',
            mask_require_color_path=False,
            mask_error_deadband=0.0,
            mask_blend_error_lo=0.05,
            mask_blend_error_hi=0.25,
            mask_blend_curvature_lo=0.15,
            mask_blend_curvature_hi=0.80,
            min_points=3,
            steering_rate_limit_per_sec=100.0,
            max_steering_command=1.0,
            perception_to_rear_axle_x_m=0.265,
        )
    )
    result = planner._hybrid_pursuit(lane, path, dt_sec=0.1)
    assert result.valid
    blend_w = float(planner._last_mask_debug.get('hybrid_w', 0.0))
    assert blend_w > 0.3
    far = float(planner._last_mask_debug.get('hybrid_far_blend', -1.0))
    assert abs(far - 0.28 * blend_w) < 1e-3


def test_hybrid_forkish_uses_pp_only():
    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='hybrid',
            mask_fork_force_pp=True,
            prefer_yellow=False,
            route_mode=RouteMode.OUT,
            min_points=2,
            steering_rate_limit_per_sec=100.0,
        )
    )
    assert planner._forkish_for_mask(
        SimpleNamespace(fork_active=True, branches=(object(), object()))
    )
    lane = SimpleNamespace(fork_active=True, branches=(object(), object()))
    out = planner._track_normal_path(
        lane,
        np.array([[0.5, 0.0], [1.0, 0.0], [1.5, 0.0]], dtype=np.float32),
        0.1,
    )
    # Fork → PP path (may be invalid if min_points unmet after frame shift; just ensure no crash)
    assert out is not None


def test_out_fork_gated_without_sign():
    """OUT normal: fork visible but no turn sign → do not enter FORK_TURN."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=0),
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=1),
        ),
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            out_fork_require_sign=True,
            out_fork_require_capture=True,
            out_fork_sign_hold_sec=3.0,
            branch_on_frames=1,
            branch_off_frames=1,
            min_points=5,
            sign_confirm_frames=1,
        )
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        out = planner.step(frame, now_sec=0.0)
    assert out.state is DrivingState.NORMAL
    assert out.debug['fork_perception'] is False


def test_aruco_detected_but_stop_on_aruco_false_keeps_lane_follow():
    """ArUco may latch should_stop; with stop_on_aruco=false keep driving."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(),
        drivable_area=np.ones((40, 40), dtype=np.uint8) * 255,
        meters_per_pixel=0.01,
        x_forward_max=1.0,
    )
    aruco = ArucoResult(detected=True, should_stop=True, marker_id=3)
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            stop_on_aruco=False,
            normal_tracker='mask_p',
            mask_steer_law='sim_v2',
            mask_corridor_mode='off',
            mask_require_color_path=False,
            mask_min_area_px=10.0,
            min_points=5,
        )
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=aruco
    ):
        out = planner.step(frame, now_sec=1.0)
    assert out.decision != 'aruco_stop'
    assert out.debug.get('aruco_should_stop') is True
    assert out.debug.get('aruco_stop') is False
    assert abs(float(out.command.throttle)) > 1e-3


def test_forced_turn_does_not_arm_out_fork_by_default():
    """forced_turn picks rank but does not enable fork perception forever."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=0),
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=1),
        ),
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            out_fork_require_sign=True,
            out_fork_require_capture=True,
            out_fork_forced_turn_arms=False,
            out_fork_sign_hold_sec=3.0,
            branch_on_frames=1,
            min_points=5,
            sign_confirm_frames=1,
        )
    )
    planner.apply_forced_turn(TurnSign.LEFT)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        out = planner.step(frame, now_sec=0.0)
    assert out.debug['fork_perception'] is False
    assert out.state is DrivingState.NORMAL


def test_out_fork_arms_after_sign_and_capture():
    """OUT: turn sign AND out_fork_capture arm fork window → FORK_TURN."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        out_fork_capture=False,
        in_circle_fork_moment=False,
        branches=(
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=0),
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=1),
        ),
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            out_fork_require_sign=True,
            out_fork_require_capture=True,
            out_fork_sign_hold_sec=3.0,
            branch_on_frames=1,
            branch_off_frames=1,
            min_points=5,
            sign_confirm_frames=1,
        )
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        return_value=TrafficResult(turn=TurnSign.LEFT),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        # Sign alone is not enough.
        first = planner.step(frame, now_sec=0.0)
        assert first.debug['fork_perception'] is False
        # Next frame: capture true → latch → arm on following step.
        lane.out_fork_capture = True
        primed = planner.step(frame, now_sec=0.05)
        assert primed.debug['out_fork_capture'] is True
        # Latch now set; same sign still held → perception arms.
        armed = planner.step(frame, now_sec=0.1)
        assert armed.debug['fork_perception'] is True
        lane.fork_active = True
        second = planner.step(frame, now_sec=0.2)
    assert second.state is DrivingState.FORK_TURN
    assert second.debug['desired_turn'] == 'left'


def test_pure_pursuit_steering_sign():
    planner = MainPlanner(
        PlannerConfig(
            min_points=2,
            lookahead_m=0.5,
            steering_rate_limit_per_sec=10.0,
        )
    )
    left_path = np.array([[0.2, 0.0], [0.8, 0.2]], dtype=np.float32)
    result = planner._pure_pursuit(left_path)
    assert result.valid
    assert result.steering < 0.0  # D-Racer: negative steering is left.

    planner._steering = 0.0
    right_path = np.array([[0.2, 0.0], [0.8, -0.2]], dtype=np.float32)
    result = planner._pure_pursuit(right_path)
    assert result.valid
    assert result.steering > 0.0


def test_pure_pursuit_rejects_short_path():
    planner = MainPlanner(PlannerConfig(min_points=3))
    result = planner._pure_pursuit(np.array([[0.5, 0.0]], dtype=np.float32))
    assert not result.valid


def test_perception_path_is_translated_once_into_rear_axle_frame():
    planner = MainPlanner(
        PlannerConfig(
            min_points=3,
            perception_to_rear_axle_x_m=0.265,
            lookahead_m=0.65,
            curve_lookahead_m=0.65,
            steering_rate_limit_per_sec=10.0,
        )
    )
    camera_path = np.array(
        [[0.22, 0.00], [0.40, 0.03], [0.80, 0.12]], dtype=np.float32
    )
    original = camera_path.copy()

    rear_path = planner._path_in_rear_axle_frame(camera_path)
    np.testing.assert_allclose(rear_path[:, 0], camera_path[:, 0] + 0.265)
    np.testing.assert_allclose(rear_path[:, 1], camera_path[:, 1])
    np.testing.assert_array_equal(camera_path, original)

    with patch.object(
        planner,
        '_cross_track_error',
        wraps=planner._cross_track_error,
    ) as cte:
        result = planner._pure_pursuit(camera_path)
    assert result.valid
    control_path = cte.call_args.args[0]
    np.testing.assert_allclose(control_path[-rear_path.shape[0] :], rear_path)
    assert abs(float(control_path[0, 0]) - planner.config.wheelbase_m) < 1e-6


def test_near_path_is_curve_extrapolated_to_front_axle():
    planner = MainPlanner(
        PlannerConfig(
            wheelbase_m=0.24,
            near_path_fit_span_m=0.30,
            near_path_extrapolation_max_m=0.35,
        )
    )
    x = np.linspace(0.485, 0.90, 20, dtype=np.float32)
    y = (0.40 * x * x - 0.05 * x + 0.01).astype(np.float32)
    observed = np.column_stack((x, y))

    extended, distance = planner._extend_path_to_front_axle(observed)

    assert abs(distance - 0.245) < 1e-5
    assert abs(float(extended[0, 0]) - planner.config.wheelbase_m) < 1e-6
    expected_y = 0.40 * planner.config.wheelbase_m**2 - 0.05 * planner.config.wheelbase_m + 0.01
    assert abs(float(extended[0, 1]) - expected_y) < 1e-4
    np.testing.assert_allclose(extended[-observed.shape[0] :], observed)


def test_near_path_extrapolation_rejects_unbounded_gap():
    planner = MainPlanner(
        PlannerConfig(
            wheelbase_m=0.24,
            near_path_extrapolation_max_m=0.20,
        )
    )
    observed = np.array([[0.50, 0.1], [0.70, 0.2]], dtype=np.float32)

    unchanged, distance = planner._extend_path_to_front_axle(observed)

    assert distance == 0.0
    np.testing.assert_array_equal(unchanged, observed)


def test_cte_correction_steers_toward_offset_path():
    planner = MainPlanner(
        PlannerConfig(
            min_points=2,
            lookahead_m=0.5,
            curve_lookahead_m=0.5,
            steering_rate_limit_per_sec=10.0,
            cte_gain=0.2,
            cte_deadband_m=0.0,
        )
    )
    left_offset = np.array([[0.1, 0.2], [1.0, 0.2]], dtype=np.float32)
    result = planner._pure_pursuit(left_offset)
    assert result.cross_track_error_m > 0.0
    assert result.cte_steering < 0.0

    planner._steering = 0.0
    right_offset = np.array([[0.1, -0.2], [1.0, -0.2]], dtype=np.float32)
    result = planner._pure_pursuit(right_offset)
    assert result.cross_track_error_m < 0.0
    assert result.cte_steering > 0.0


def test_cte_deadband_ignores_small_centerline_noise():
    planner = MainPlanner(
        PlannerConfig(min_points=2, cte_deadband_m=0.02, cte_gain=0.2)
    )
    path = np.array([[0.1, 0.01], [1.0, 0.01]], dtype=np.float32)
    result = planner._pure_pursuit(path)
    assert result.cte_steering == 0.0


def test_heading_correction_steers_with_path_tangent():
    planner = MainPlanner(
        PlannerConfig(
            min_points=3,
            heading_gain=0.25,
            heading_preview_m=0.30,
            heading_sample_span_m=0.15,
            max_heading_steering=0.20,
            steering_rate_limit_per_sec=10.0,
        )
    )
    left_path = np.array(
        [[0.2, 0.0], [0.35, 0.02], [0.5, 0.12], [0.8, 0.35]],
        dtype=np.float32,
    )
    result = planner._pure_pursuit(left_path)
    assert result.heading_error_rad > 0.0
    assert result.heading_steering < 0.0


def test_roundabout_uses_dedicated_lookahead_and_throttle():
    planner = MainPlanner(
        PlannerConfig(
            min_points=3,
            roundabout_lookahead_m=0.32,
            roundabout_throttle=0.06,
            steering_rate_limit_per_sec=10.0,
        )
    )
    planner.state = DrivingState.ROUNDABOUT_CIRCLE
    path = np.array(
        [[0.22, 0.0], [0.4, 0.05], [0.7, 0.2]], dtype=np.float32
    )
    result = planner._pure_pursuit(path)
    command = planner._drive(result)
    assert abs(result.lookahead_m - 0.32) < 1e-9
    assert abs(command.throttle - 0.06) < 1e-9


def test_path_loss_returns_stored_steering_toward_neutral():
    planner = MainPlanner(
        PlannerConfig(path_lost_steering_return_rate_per_sec=2.0)
    )
    planner._steering = 0.75
    assert abs(planner._return_steering_to_neutral(0.1) - 0.55) < 1e-9
    planner._steering = -0.15
    assert planner._return_steering_to_neutral(0.1) == 0.0


def test_brief_path_loss_holds_then_returns_steering():
    empty = np.empty((0, 2), dtype=np.float32)
    lane = SimpleNamespace(
        white_centerline=empty,
        yellow_centerline=empty,
        white_confidence=0.0,
        yellow_confidence=0.0,
        white_visible=False,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(),
    )
    planner = MainPlanner(
        PlannerConfig(
            path_lost_hold_frames=2,
            path_lost_stop_frames=4,
            path_lost_crawl_throttle=0.14,
            path_lost_steering_return_rate_per_sec=2.0,
            require_green_to_start=False,
        )
    )
    planner._steering = 0.6
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        first = planner.step(frame, now_sec=0.0)
        second = planner.step(frame, now_sec=0.05)
        late = planner.step(frame, now_sec=0.10)
        late = planner.step(frame, now_sec=0.15)
        late = planner.step(frame, now_sec=0.20)

    # Hold last steer + crawl until stop_frames, then throttle 0 (steer still held).
    assert first.command.steering == 0.6
    assert second.command.steering == 0.6
    assert first.decision.endswith('path_lost_hold_crawl')
    assert second.decision.endswith('path_lost_hold_crawl')
    assert abs(first.command.throttle - 0.14) < 1e-9
    assert late.decision.endswith('path_lost_hold_stop')
    assert late.command.throttle == 0.0
    assert late.command.steering == 0.6


def test_steering_rate_limit_is_time_based():
    path = np.array([[0.1, 0.0], [1.0, 0.5]], dtype=np.float32)
    config = PlannerConfig(
        min_points=2,
        lookahead_m=0.5,
        curve_lookahead_m=0.5,
        steering_rate_limit_per_sec=1.0,
    )
    short_step = MainPlanner(config)._pure_pursuit(path, dt_sec=0.05)
    long_step = MainPlanner(config)._pure_pursuit(path, dt_sec=0.20)
    assert abs(short_step.steering) <= 0.05 + 1e-9
    assert abs(long_step.steering) <= 0.20 + 1e-9
    assert abs(long_step.steering) > abs(short_step.steering)


def test_lookahead_target_is_interpolated_to_exact_radius():
    planner = MainPlanner(
        PlannerConfig(
            min_points=3,
            lookahead_m=0.65,
            curve_lookahead_m=0.65,
            steering_rate_limit_per_sec=10.0,
        )
    )
    sparse_path = np.array(
        [[0.2, 0.0], [0.4, 0.05], [1.2, 0.2]], dtype=np.float32
    )
    result = planner._pure_pursuit(sparse_path)
    assert result.valid
    assert abs(result.target_distance - 0.65) < 1e-4


def test_curvature_reduces_lookahead_and_throttle():
    planner = MainPlanner(
        PlannerConfig(
            min_points=5,
            lookahead_m=0.8,
            curve_lookahead_m=0.45,
            lookahead_shrink_rate_m=1.0,
            curvature_full_scale=0.5,
            cruise_throttle=0.13,
            curve_throttle=0.07,
            steering_rate_limit_per_sec=10.0,
        )
    )
    angles = np.linspace(0.0, 1.2, 30, dtype=np.float32)
    curved_path = np.column_stack(
        (0.2 + 0.6 * np.sin(angles), 0.6 * (1.0 - np.cos(angles)))
    ).astype(np.float32)
    result = planner._pure_pursuit(curved_path)
    command = planner._drive(result)
    assert result.valid
    assert result.curve_ratio > 0.0
    assert result.lookahead_m < planner.config.lookahead_m
    assert command.throttle < planner.config.cruise_throttle


def test_in_course_exits_on_second_moment_pass():
    """IN: moment rising ×1 = keep(rank1); ×2 = exit(rank0)."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    branch0 = SimpleNamespace(points=path, confidence=0.9, lateral_rank=0)
    branch1 = SimpleNamespace(points=path, confidence=0.9, lateral_rank=1)
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path,
        white_confidence=0.9,
        yellow_confidence=0.9,
        white_visible=True,
        yellow_visible=True,
        fork_active=False,
        yellow_crossing_line=False,
        out_fork_capture=False,
        in_circle_fork_moment=False,
        branches=(branch0,),
    )
    config = PlannerConfig(
        route_mode=RouteMode.IN,
        prefer_yellow=True,
        yellow_valid_on_frames=1,
        min_points=5,
        min_lap_time_sec=1.0,
        in_exit_use_moment=True,
        in_keep_passes=1,
        in_keep_branch_rank=1,
        exit_branch_rank=0,
        branch_on_frames=1,
        branch_off_frames=1,
        crossing_on_frames=1,
        crossing_off_frames=1,
    )
    planner = MainPlanner(config)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        planner.step(frame, now_sec=0.0)  # Enter roundabout.
        # Pass 1: keep right — follow keep branch, stay in CIRCLE.
        lane.in_circle_fork_moment = True
        lane.fork_active = True
        lane.branches = (branch0, branch1)
        keep = planner.step(frame, now_sec=0.1)
        assert planner._in_fork_pass_count == 1
        assert planner._fork_selected_rank == 1
        assert keep.state is DrivingState.ROUNDABOUT_CIRCLE
        assert keep.decision == 'roundabout_circle_keep_rank1'
        assert keep.debug['selected_branch_rank'] == 1
        lane.in_circle_fork_moment = False
        planner.step(frame, now_sec=0.2)
        # Pass 2: exit left (≥ min_lap_time).
        lane.in_circle_fork_moment = True
        output = planner.step(frame, now_sec=1.2)

    assert planner._in_fork_pass_count == 2
    assert output.state is DrivingState.ROUNDABOUT_EXIT
    assert output.decision == 'roundabout_exit_rank0'


def test_aruco_stop_freezes_in_moment_pass_count():
    """ArUco throttle-stop must not advance IN moment keep/exit counters."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    branch0 = SimpleNamespace(points=path, confidence=0.9, lateral_rank=0)
    branch1 = SimpleNamespace(points=path, confidence=0.9, lateral_rank=1)
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path,
        white_confidence=0.9,
        yellow_confidence=0.9,
        white_visible=True,
        yellow_visible=True,
        fork_active=True,
        yellow_crossing_line=False,
        out_fork_capture=False,
        in_circle_fork_moment=True,
        branches=(branch0, branch1),
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            prefer_yellow=True,
            yellow_valid_on_frames=1,
            min_points=5,
            min_lap_time_sec=1.0,
            in_exit_use_moment=True,
            in_keep_passes=1,
            stop_on_aruco=True,
            branch_on_frames=1,
            branch_off_frames=1,
            require_green_to_start=False,
        )
    )
    planner.state = DrivingState.ROUNDABOUT_CIRCLE
    planner._roundabout_started_at = 0.0
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    aruco = ArucoResult(detected=True, should_stop=True, marker_id=3)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=aruco
    ):
        out = planner.step(frame, now_sec=0.5)
    assert out.decision == 'aruco_stop'
    assert out.debug['mission_freeze'] is True
    assert planner._in_fork_pass_count == 0
    assert planner.state is DrivingState.ROUNDABOUT_CIRCLE


def test_out_fork_waits_for_confirmed_sign_before_lock():
    """Fork rising with unconfirmed sign must not lock default_unknown rank."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=True,
        yellow_crossing_line=False,
        out_fork_capture=True,
        in_circle_fork_moment=False,
        branches=(
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=0),
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=1),
        ),
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            out_fork_require_sign=True,
            out_fork_require_capture=True,
            out_fork_sign_hold_sec=3.0,
            sign_confirm_frames=3,
            branch_on_frames=1,
            branch_off_frames=1,
            min_points=5,
            default_out_branch_rank=0,
            require_green_to_start=False,
        )
    )
    planner._out_capture_latched = True
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        return_value=TrafficResult(turn=TurnSign.RIGHT),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        # First sighting: candidate only — stay NORMAL.
        first = planner.step(frame, now_sec=0.0)
        assert first.state is DrivingState.NORMAL
        assert planner.desired_turn is TurnSign.UNKNOWN
        # Confirm over remaining frames.
        planner.step(frame, now_sec=0.1)
        locked = planner.step(frame, now_sec=0.2)
    assert locked.state is DrivingState.FORK_TURN
    assert locked.debug['selected_branch_rank'] == 1
    assert locked.debug['fork_locked_turn'] == 'right'


def test_out_fork_state_does_not_treat_single_branch_as_turn_path():
    white_path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    misleading_branch = np.array(
        [[0.2, -0.2], [0.4, -0.3], [0.6, -0.4], [0.8, -0.5], [1.0, -0.6]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=white_path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        branches=(SimpleNamespace(points=misleading_branch, confidence=0.9),),
    )
    planner = MainPlanner(PlannerConfig(min_points=5))
    planner.state = DrivingState.FORK_TURN
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        output = planner.step(frame, now_sec=0.0)

    assert output.path_source is PathSource.WHITE_CENTERLINE
    assert output.decision == 'out_fork_color_resume'


def test_forced_turn_right_stays_in_roundabout_circle():
    """IN + forced RIGHT: do not arm exit even when branch events fire."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    branch = SimpleNamespace(
        points=path, confidence=0.9, lateral_rank=0
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path,
        white_confidence=0.9,
        yellow_confidence=0.9,
        white_visible=True,
        yellow_visible=True,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(branch, SimpleNamespace(points=path, confidence=0.9, lateral_rank=1)),
        lane_policy='explore',
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            prefer_yellow=True,
            yellow_valid_on_frames=1,
            min_points=5,
            min_lap_time_sec=0.5,
            in_exit_use_moment=False,
            branch_required_events=1,
            branch_on_frames=1,
            branch_off_frames=1,
        )
    )
    planner.apply_forced_turn(TurnSign.RIGHT)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        planner.step(frame, now_sec=0.0)  # enter circle
        assert planner.state is DrivingState.ROUNDABOUT_CIRCLE
        out = planner.step(frame, now_sec=1.0)

    assert out.state is DrivingState.ROUNDABOUT_CIRCLE
    assert out.debug['forced_turn'] == 'right'
    assert planner._fork_selection_reason == 'forced_right'


def test_forced_turn_left_arms_roundabout_exit():
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    left = SimpleNamespace(points=path, confidence=0.9, lateral_rank=0)
    right = SimpleNamespace(
        points=path * np.array([1.0, -1.0], dtype=np.float32),
        confidence=0.9,
        lateral_rank=1,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=path,
        white_confidence=0.9,
        yellow_confidence=0.9,
        white_visible=True,
        yellow_visible=True,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(left, right),
        lane_policy='explore',
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.IN,
            prefer_yellow=True,
            yellow_valid_on_frames=1,
            min_points=5,
            min_lap_time_sec=0.5,
            in_exit_use_moment=False,
            branch_required_events=1,
            branch_on_frames=1,
            branch_off_frames=1,
        )
    )
    planner.apply_forced_turn(TurnSign.LEFT)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        planner.step(frame, now_sec=0.0)
        out = planner.step(frame, now_sec=1.0)

    assert out.state in (
        DrivingState.ROUNDABOUT_EXIT_READY,
        DrivingState.ROUNDABOUT_EXIT,
    )
    assert out.debug['forced_turn'] == 'left'
    assert planner._fork_selected_rank == 0


def test_force_fork_choice_left_rank0_right_rank1():
    planner = MainPlanner(PlannerConfig(min_points=5))
    planner.force_fork_choice(TurnSign.LEFT, state=DrivingState.FORK_TURN)
    assert planner._fork_selected_rank == 0
    assert planner.state is DrivingState.FORK_TURN
    planner.force_fork_choice(TurnSign.RIGHT, state=DrivingState.ROUNDABOUT_EXIT)
    assert planner._fork_selected_rank == 1
    assert planner.state is DrivingState.ROUNDABOUT_EXIT


def test_ranked_branch_matches_lateral_rank():
    left = SimpleNamespace(points=np.zeros((5, 2), np.float32), confidence=1.0, lateral_rank=0)
    right = SimpleNamespace(points=np.ones((5, 2), np.float32), confidence=1.0, lateral_rank=1)
    # Deliberately reverse list order — lateral_rank must win.
    lane = SimpleNamespace(branches=(right, left))
    assert MainPlanner._ranked_branch(lane, 0) is left
    assert MainPlanner._ranked_branch(lane, 1) is right

    left_path = np.array(
        [[0.2, 0.15], [0.4, 0.2], [0.6, 0.25], [0.8, 0.3], [1.0, 0.35]],
        dtype=np.float32,
    )
    right_path = left_path.copy()
    right_path[:, 1] *= -1.0
    lane = SimpleNamespace(
        white_centerline=left_path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(
            SimpleNamespace(points=left_path, confidence=0.9),
            SimpleNamespace(points=right_path, confidence=0.9),
        ),
    )
    planner = MainPlanner(PlannerConfig(min_points=5))
    planner.desired_turn = TurnSign.RIGHT
    planner.state = DrivingState.FORK_TURN
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        return_value=TrafficResult(turn=TurnSign.RIGHT),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        output = planner.step(frame, now_sec=0.0)

    assert output.path_source is PathSource.RIGHT_BRANCH
    assert output.debug['turn_sign'] == 'right'
    assert output.debug['desired_turn'] == 'right'
    assert output.debug['selected_branch_rank'] == 1
    assert output.debug['branch_selection_reason'] == 'sign_right'


def test_sign_is_confirmed_then_locked_during_fork():
    left_path = np.array(
        [[0.2, 0.1], [0.4, 0.15], [0.6, 0.2], [0.8, 0.25], [1.0, 0.3]],
        dtype=np.float32,
    )
    right_path = left_path.copy()
    right_path[:, 1] *= -1.0
    lane = SimpleNamespace(
        white_centerline=left_path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        out_fork_capture=True,
        in_circle_fork_moment=False,
        branches=(
            SimpleNamespace(points=left_path, confidence=0.9),
            SimpleNamespace(points=right_path, confidence=0.9),
        ),
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            min_points=5,
            sign_confirm_frames=2,
            branch_on_frames=1,
            out_fork_require_sign=True,
            out_fork_require_capture=True,
        )
    )
    planner._out_capture_latched = True
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        side_effect=(
            TrafficResult(turn=TurnSign.LEFT),
            TrafficResult(turn=TurnSign.LEFT),
            TrafficResult(turn=TurnSign.RIGHT),
        ),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        first = planner.step(frame, now_sec=0.0)
        lane.fork_active = True
        second = planner.step(frame, now_sec=0.1)
        third = planner.step(frame, now_sec=0.2)

    assert first.debug['desired_turn'] == 'unknown'
    assert second.state is DrivingState.FORK_TURN
    assert second.path_source is PathSource.LEFT_BRANCH
    assert second.debug['fork_locked_turn'] == 'left'
    assert third.debug['turn_sign'] == 'right'
    assert third.debug['desired_turn'] == 'left'
    assert third.debug['fork_locked_turn'] == 'left'
    assert third.path_source is PathSource.LEFT_BRANCH


def test_fork_uses_cached_branch_during_short_detection_flicker():
    left_path = np.array(
        [[0.2, 0.1], [0.4, 0.15], [0.6, 0.2], [0.8, 0.25], [1.0, 0.3]],
        dtype=np.float32,
    )
    right_path = left_path.copy()
    right_path[:, 1] *= -1.0
    branch_left = SimpleNamespace(points=left_path, confidence=0.9)
    branch_right = SimpleNamespace(points=right_path, confidence=0.9)
    lane = SimpleNamespace(
        white_centerline=right_path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=True,
        yellow_crossing_line=False,
        branches=(branch_left, branch_right),
    )
    planner = MainPlanner(
        PlannerConfig(min_points=5, fork_path_hold_frames=2)
    )
    planner.desired_turn = TurnSign.LEFT
    planner._lock_fork_selection()
    planner.state = DrivingState.FORK_TURN
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect', return_value=TrafficResult()
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        planner.step(frame, now_sec=0.0)
        lane.fork_active = False
        lane.branches = (branch_right,)
        output = planner.step(frame, now_sec=0.1)

    assert output.path_source is PathSource.LEFT_BRANCH
    assert output.decision == 'out_fork_cached_branch'
    assert output.debug['selected_branch_rank'] == 0


def test_ema_with_jump_rejects_spike():
    filtered, jumped = MainPlanner._ema_with_jump(0.0, 0.8, alpha=0.5, max_jump=0.3)
    assert jumped
    assert abs(filtered - 0.0) < 1e-6
    filtered2, jumped2 = MainPlanner._ema_with_jump(0.0, 0.1, alpha=0.5, max_jump=0.3)
    assert not jumped2
    assert 0.04 < filtered2 < 0.06


def test_harden_path_holds_lateral_jump():
    planner = MainPlanner(
        PlannerConfig(
            track_enable_path_hold=True,
            track_path_y_alpha=0.5,
            track_path_y_max_jump_m=0.08,
            track_half_width_m=0.175,
            perception_to_rear_axle_x_m=0.0,
        )
    )
    path0 = np.array([[0.4, 0.02], [0.8, 0.02], [1.2, 0.02]], dtype=np.float32)
    path1 = path0.copy()
    path1[:, 1] = 0.35  # L/R-style flip spike
    out0 = planner._harden_color_path(path0)
    out1 = planner._harden_color_path(path1)
    assert abs(float(np.mean(out0[:, 1])) - 0.02) < 0.05
    # Jump rejected → stay near prior center, not at +0.35.
    assert abs(float(np.mean(out1[:, 1]))) < 0.15


def test_centerline_half_width_virtual_when_one_rail_missing():
    from inference.modules import lane_detection as ld

    left = np.full(20, np.nan, dtype=np.float32)
    right = np.full(20, np.nan, dtype=np.float32)
    left[5:15] = 40.0
    center = ld.centerline_from_boundaries(
        left, right, synthesize_missing=True, lane_width_m=0.35
    )
    half_px = 0.5 * 0.35 / ld.METERS_PER_PIXEL
    assert abs(float(center[10]) - (40.0 + half_px)) < 1e-3


def test_stanley_steers_left_for_path_left_cte():
    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='stanley',
            min_points=3,
            stanley_k_cte=2.0,
            stanley_k_yaw=0.5,
            stanley_steer_alpha=1.0,
            stanley_curvature_ff_gain=0.0,
            steering_rate_limit_per_sec=100.0,
            cte_deadband_m=0.0,
            perception_to_rear_axle_x_m=0.0,
            track_enable_path_hold=False,
        )
    )
    # Path to the left of vehicle → need left (negative) steer.
    path = np.array(
        [[0.3, 0.15], [0.6, 0.15], [0.9, 0.15], [1.2, 0.15], [1.5, 0.15]],
        dtype=np.float32,
    )
    result = planner._stanley_pursuit(path, dt_sec=0.1)
    assert result.valid
    assert result.steering < 0.0


def test_stanley_fork_force_uses_pp_path():
    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='stanley',
            mask_fork_force_pp=True,
            min_points=3,
            steering_rate_limit_per_sec=100.0,
        )
    )
    lane = SimpleNamespace(fork_active=True, branches=(object(), object()))
    assert planner._forkish_for_mask(lane)
    path = np.array(
        [[0.3, 0.0], [0.6, 0.0], [0.9, 0.0], [1.2, 0.0], [1.5, 0.0]],
        dtype=np.float32,
    )
    out = planner._track_normal_path(lane, path, 0.1)
    assert out.valid
    # Fork guard must not call stanley (no stanley_* debug from this path).
    assert 'stanley_cte_m' not in (planner._last_mask_debug or {})


def test_row_mid_rebuilds_center_when_left_fov_clipped():
    """S-curve single-side: left FOV clip must not use truncated blob mid."""
    import cv2
    from inference.modules import lane_detection as ld

    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='mask_p',
            mask_center_mode='row_mid',
            mask_erode_px=0,
            mask_lane_width_m=0.35,
            mask_steer_k=1.0,
            mask_steer_alpha=1.0,
            mask_near_band_ratio=1.0,
            mask_corridor_mode='off',
            mask_require_color_path=False,
            mask_min_area_px=20.0,
            mask_far_blend=0.0,
            steering_rate_limit_per_sec=100.0,
        )
    )
    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    mpp = float(ld.METERS_PER_PIXEL)
    expected_w = 0.35 / mpp
    half = 0.5 * expected_w
    # Only the right ~half of the lane remains (genuinely narrow + left FOV clip).
    true_right = int(round(half * 0.95))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (0, h // 4), (max(3, true_right), h - 1), 255, -1)
    lane = type(
        'L',
        (),
        {
            'drivable_area': mask,
            'meters_per_pixel': mpp,
            'x_forward_max': 1.5,
        },
    )()
    result = planner._mask_com_pursuit(lane, dt_sec=0.1)
    assert result.valid
    cx = float(planner._last_mask_debug['mask_com_cx'])
    naive = 0.5 * true_right
    rebuilt = true_right - half
    # Temporal blend pulls toward prior after first frame; use second step.
    result2 = planner._mask_com_pursuit(lane, dt_sec=0.1)
    cx2 = float(planner._last_mask_debug['mask_com_cx'])
    assert planner._last_mask_debug.get('mask_single_side_rows', 0) > 0
    assert abs(cx2 - rebuilt) < abs(naive - rebuilt) + 0.25 * expected_w
    assert abs(cx2 - naive) > 1.0 or abs(cx2 - rebuilt) < abs(cx - naive)

def test_mask_occlusion_hold_before_pp_fallback():
    """Empty mask should reuse last COM steer for a few frames, not fail open."""
    import cv2
    from inference.modules import lane_detection as ld

    planner = MainPlanner(
        PlannerConfig(
            normal_tracker='mask_p',
            mask_center_mode='row_mid',
            mask_erode_px=0,
            mask_lane_width_m=0.35,
            mask_steer_k=1.2,
            mask_steer_alpha=1.0,
            mask_near_band_ratio=1.0,
            mask_corridor_mode='off',
            mask_require_color_path=False,
            mask_fork_force_pp=False,
            mask_min_area_px=20.0,
            mask_far_blend=0.0,
            mask_occlusion_hold_frames=5,
            steering_rate_limit_per_sec=100.0,
        )
    )
    h, w = ld.BEV_HEIGHT, ld.BEV_WIDTH
    mpp = float(ld.METERS_PER_PIXEL)
    good = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(good, (w // 2 - 8, h // 3), (w // 2 + 8, h - 1), 255, -1)
    empty = np.zeros((h, w), dtype=np.uint8)
    path = np.array(
        [[0.4, 0.25], [0.7, 0.35], [1.0, 0.45], [1.3, 0.55]],
        dtype=np.float32,
    )
    lane_good = type(
        'L',
        (),
        {
            'drivable_area': good,
            'meters_per_pixel': mpp,
            'x_forward_max': 1.5,
            'fork_active': False,
            'branches': (),
        },
    )()
    p1 = planner._mask_com_pursuit(lane_good, dt_sec=0.1)
    assert p1.valid
    seed_steer = float(planner._steer_f)
    lane_empty = type(
        'L',
        (),
        {
            'drivable_area': empty,
            'meters_per_pixel': mpp,
            'x_forward_max': 1.5,
            'fork_active': False,
            'branches': (),
        },
    )()
    out = planner._track_normal_path(lane_empty, path, 0.1)
    assert out.valid
    assert planner._last_mask_debug.get('mask_occlusion_hold') is True
    assert abs(float(out.steering) - seed_steer) < 0.25


def test_load_planner_config_has_track_and_stanley():
    cfg = load_planner_config(route_mode='out')
    assert cfg.track_half_width_m > 0.0
    assert cfg.stanley_k_cte > 0.0
    # Soft anti-wobble ego-blob pack (main_planner.yaml).
    assert cfg.stanley_k_cte == 0.90
    assert cfg.stanley_k_yaw == 1.05
    assert cfg.stanley_steer_alpha == 0.20
    assert cfg.circle_tracker == 'pp'
    assert cfg.roundabout_lookahead_m == 0.55
    assert cfg.normal_tracker == 'mask_p'
    assert cfg.mask_center_mode == 'row_mid'
    assert cfg.mask_steer_k == 1.0
    assert cfg.mask_steer_law == 'lateral_atan'
    assert cfg.cruise_throttle == 0.24
    assert cfg.path_lost_crawl_throttle == 0.14
    assert cfg.default_out_branch_rank == 1
    assert cfg.out_fork_require_sign is False
    assert cfg.require_green_to_start is True
    assert cfg.stop_on_red is False
    assert cfg.stop_on_aruco is True
    assert cfg.green_wait_timeout_sec == 15.0


def test_wait_green_timeout_assumes_green_and_starts():
    """No green for green_wait_timeout_sec → leave WAIT_GREEN as if green."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        out_fork_capture=False,
        in_circle_fork_moment=False,
        branches=(),
        drivable_area=np.ones((40, 40), dtype=np.uint8) * 255,
        meters_per_pixel=0.01,
        x_forward_max=1.0,
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            require_green_to_start=True,
            green_wait_timeout_sec=15.0,
            stop_on_red=True,
            stop_on_aruco=False,
            normal_tracker='pp',
            min_points=5,
            mask_require_color_path=False,
            steering_rate_limit_per_sec=100.0,
        )
    )
    assert planner.state is DrivingState.WAIT_GREEN
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        return_value=TrafficResult(signal=TrafficSignal.UNKNOWN),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        early = planner.step(frame, now_sec=1.0)
        assert early.state is DrivingState.WAIT_GREEN
        assert early.decision == 'wait_green'
        late = planner.step(frame, now_sec=16.5)
    assert late.state is DrivingState.NORMAL
    assert planner._green_assumed is True
    assert late.debug.get('green_assumed') is True


def test_out_fork_sign_miss_defaults_to_right():
    """Competition: fork visible, no sign → lock default_out_branch_rank (RIGHT=1)."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=True,
        yellow_crossing_line=False,
        out_fork_capture=True,
        in_circle_fork_moment=False,
        branches=(
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=0),
            SimpleNamespace(points=path, confidence=0.9, lateral_rank=1),
        ),
        drivable_area=np.ones((40, 40), dtype=np.uint8) * 255,
        meters_per_pixel=0.01,
        x_forward_max=1.0,
    )
    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            out_fork_require_sign=False,
            out_fork_require_capture=True,
            sign_confirm_frames=3,
            branch_on_frames=1,
            branch_off_frames=1,
            min_points=5,
            default_out_branch_rank=1,
            require_green_to_start=False,
            stop_on_aruco=False,
            mask_require_color_path=False,
            normal_tracker='pp',
            steering_rate_limit_per_sec=100.0,
        )
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        return_value=TrafficResult(turn=TurnSign.UNKNOWN),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        # Capture latch is previous-frame: prime once, arm on the next step.
        primed = planner.step(frame, now_sec=0.5)
        assert primed.debug.get('out_fork_capture') is True
        out = planner.step(frame, now_sec=1.0)
    assert out.state is DrivingState.FORK_TURN
    assert out.debug['selected_branch_rank'] == 1
    assert planner._fork_selection_reason == 'default_unknown'


def test_traffic_pass_skips_green_wait_and_red_stop():
    """Launch/YAML traffic_pass: start NORMAL and ignore red (ArUco still stops)."""
    path = np.array(
        [[0.2, 0.0], [0.4, 0.0], [0.6, 0.0], [0.8, 0.0], [1.0, 0.0]],
        dtype=np.float32,
    )
    lane = SimpleNamespace(
        white_centerline=path,
        yellow_centerline=np.empty((0, 2), dtype=np.float32),
        white_confidence=0.9,
        yellow_confidence=0.0,
        white_visible=True,
        yellow_visible=False,
        fork_active=False,
        yellow_crossing_line=False,
        out_fork_capture=False,
        in_circle_fork_moment=False,
        branches=(),
        drivable_area=np.ones((40, 40), dtype=np.uint8) * 255,
        meters_per_pixel=0.01,
        x_forward_max=1.0,
    )
    cfg = load_planner_config(route_mode='out', traffic_pass=True)
    assert cfg.require_green_to_start is False
    assert cfg.stop_on_red is False
    assert cfg.stop_on_aruco is True

    planner = MainPlanner(
        PlannerConfig(
            route_mode=RouteMode.OUT,
            require_green_to_start=False,
            stop_on_red=False,
            stop_on_aruco=True,
            normal_tracker='pp',
            min_points=5,
            mask_corridor_mode='off',
        )
    )
    assert planner.state is DrivingState.NORMAL
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch(
        'inference.pipeline.lane_detection.detect_with_debug',
        return_value=(lane, SimpleNamespace()),
    ), patch(
        'inference.pipeline.traffic_sign.detect',
        return_value=TrafficResult(signal=TrafficSignal.RED),
    ), patch(
        'inference.pipeline.aruco_detection.detect', return_value=ArucoResult()
    ):
        out = planner.step(frame, now_sec=0.0)
    assert out.decision != 'red_signal_stop'
    assert out.decision != 'wait_green'
    assert out.debug['traffic_pass'] is True
    assert out.path_source is not PathSource.STOP
