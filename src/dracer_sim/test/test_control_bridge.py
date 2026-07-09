"""Unit tests for dracer_sim helpers."""

import math


def bicycle_angular_rate(linear: float, steer_angle: float, wheelbase: float) -> float:
  if abs(linear) < 1e-4:
    return steer_angle
  return (linear / wheelbase) * math.tan(steer_angle)


def test_bicycle_model_turn_rate():
  linear = 0.5
  steer_angle = 0.5
  wheelbase = 0.2
  expected = (linear / wheelbase) * math.tan(steer_angle)
  assert abs(bicycle_angular_rate(linear, steer_angle, wheelbase) - expected) < 1e-6
