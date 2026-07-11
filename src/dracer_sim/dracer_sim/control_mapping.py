"""Pure D-Racer Control → Gazebo Twist field mapping (no ROS deps)."""

from __future__ import annotations

import math

CMD_MODE_ACKERMANN = 'ackermann_steer'
CMD_MODE_DIFF = 'diff_yaw_rate'


def clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
  return lo if value < lo else hi if value > hi else value


def control_to_cmd_vel(
  steering: float,
  throttle: float,
  *,
  max_linear_speed: float,
  max_steer_angle_rad: float,
  cmd_mode: str = CMD_MODE_ACKERMANN,
  wheelbase_m: float = 0.24,
) -> tuple[float, float]:
  """Map D-Racer Control norms to Twist fields.

  D-Racer: +steering = right, +throttle = forward.
  Gazebo +angular.z = left (REP-103).

  Modes:
    ackermann_steer — gazebo_ros_ackermann_drive (angular.z = steer angle rad)
    diff_yaw_rate   — gazebo_ros_diff_drive (angular.z = yaw rate rad/s)

  Returns (linear.x, angular.z).
  """
  linear = clip(throttle) * max_linear_speed
  # Left-positive steer angle matching Gazebo / ROS (negate D-Racer right-positive).
  steer_left = -clip(steering) * max_steer_angle_rad

  if cmd_mode == CMD_MODE_DIFF:
    if abs(linear) < 1e-4:
      return linear, 0.0
    return linear, (linear / wheelbase_m) * math.tan(steer_left)

  # Ackermann: plugin multiplies steer by copysign(1, linear) — undo when reversing
  # so physical steer matches the real servo (independent of throttle sign).
  if linear < -1e-4:
    steer_left = -steer_left
  return linear, steer_left
