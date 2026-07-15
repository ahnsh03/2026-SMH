"""Unit tests for D-Racer ↔ Gazebo control mapping."""

import math

from dracer_sim.control_mapping import CMD_MODE_DIFF, control_to_cmd_vel


MAX_V = 1.2
MAX_STEER = 0.5574
WHEELBASE = 0.24


def test_positive_steering_is_right_turn_negative_angular_z():
  """D-Racer +steering = right; Gazebo +angular.z = left → must negate."""
  linear, angular = control_to_cmd_vel(
    0.5, 0.3, max_linear_speed=MAX_V, max_steer_angle_rad=MAX_STEER
  )
  assert linear == 0.3 * MAX_V
  assert angular == -0.5 * MAX_STEER


def test_negative_steering_is_left_turn_positive_angular_z():
  linear, angular = control_to_cmd_vel(
    -1.0, 0.2, max_linear_speed=MAX_V, max_steer_angle_rad=MAX_STEER
  )
  assert linear == 0.2 * MAX_V
  assert angular == MAX_STEER


def test_angular_z_is_steer_angle_not_yaw_rate():
  """gazebo_ros_ackermann_drive treats angular.z as steer angle (rad)."""
  _, angular_slow = control_to_cmd_vel(
    0.4, 0.1, max_linear_speed=MAX_V, max_steer_angle_rad=MAX_STEER
  )
  _, angular_fast = control_to_cmd_vel(
    0.4, 1.0, max_linear_speed=MAX_V, max_steer_angle_rad=MAX_STEER
  )
  assert angular_slow == angular_fast == -0.4 * MAX_STEER


def test_reverse_undoes_plugin_copysign_so_steer_direction_matches_real():
  """Plugin multiplies steer by sign(linear); undo so reverse matches servo."""
  _, angular_fwd = control_to_cmd_vel(
    0.5, 0.3, max_linear_speed=MAX_V, max_steer_angle_rad=MAX_STEER
  )
  linear_rev, angular_rev = control_to_cmd_vel(
    0.5, -0.3, max_linear_speed=MAX_V, max_steer_angle_rad=MAX_STEER
  )
  assert linear_rev == -0.3 * MAX_V
  assert angular_rev == -angular_fwd


def test_clip_steering_and_throttle():
  linear, angular = control_to_cmd_vel(
    2.0, -2.0, max_linear_speed=MAX_V, max_steer_angle_rad=MAX_STEER
  )
  assert linear == -MAX_V
  assert angular == MAX_STEER


def test_diff_mode_yaw_rate_keeps_right_negative():
  linear, angular = control_to_cmd_vel(
    0.5,
    0.5,
    max_linear_speed=MAX_V,
    max_steer_angle_rad=MAX_STEER,
    cmd_mode=CMD_MODE_DIFF,
    wheelbase_m=WHEELBASE,
  )
  expected = (0.5 * MAX_V / WHEELBASE) * math.tan(-0.5 * MAX_STEER)
  assert linear == 0.5 * MAX_V
  assert abs(angular - expected) < 1e-9
