"""Auto-driving stack for an already-running Gazebo bringup.

Does **not** include ``sim_bringup`` — keep Gazebo in another terminal and
Ctrl+C this launch to restart joystick + inference + lane preview without
killing the world.

Visualization profiles (``viz``)::

  lane   — Lane/Fork Perception 창만 (기본)
  off    — OpenCV 창 없음
  debug  — lane + inference ``Lane drive`` 1창
  all    — lane + ``Lane drive`` + ``HSV masks``

Typical::

  # T1 — 카메라/BEV 끄고 갈림만 (권장)
  ros2 launch dracer_sim sim_bringup.launch.py spawn_pose:=out_fork view:=none

  # T2
  ros2 launch dracer_sim sim_auto_stack.launch.py \\
    route_mode:=out forced_turn:=left viz:=lane
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_vehicle_config_path() -> str:
  for base_path in Path(__file__).resolve().parents:
    candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
    if candidate.exists():
      return str(candidate)
  return '/home/topst/2026-SMH/src/config/vehicle_config.yaml'


def get_planner_config_path() -> str:
  for base_path in Path(__file__).resolve().parents:
    candidate = base_path / 'config' / 'main_planner.yaml'
    if candidate.exists():
      return str(candidate)
  return '/home/topst/2026-SMH/config/main_planner.yaml'


def _viz_settings(viz: str) -> tuple[bool, str]:
  """Return (use_lane_view, LANE_VISUALIZE env)."""
  mode = (viz or 'lane').strip().lower()
  if mode in ('off', 'none', '0', 'false'):
    return False, 'off'
  if mode in ('debug', 'control'):
    return True, 'control'
  if mode in ('all', 'on', 'full'):
    return True, 'on'
  # lane (preview only) — no inference OpenCV window
  return True, 'off'


def _launch_setup(context, *args, **kwargs):
  vehicle_config_path = get_vehicle_config_path()
  planner_config_path = get_planner_config_path()
  route_mode = LaunchConfiguration('route_mode')
  use_sim_time = LaunchConfiguration('use_sim_time')

  viz_raw = LaunchConfiguration('viz').perform(context)
  use_lane_view, lane_visualize = _viz_settings(viz_raw)
  # Legacy override: use_lane_view:=false still wins.
  legacy = LaunchConfiguration('use_lane_view').perform(context).strip().lower()
  if legacy in ('false', '0', 'no', 'off'):
    use_lane_view = False
  elif legacy in ('true', '1', 'yes', 'on') and viz_raw.strip().lower() in (
    '',
    'lane',
  ):
    use_lane_view = True

  nodes = [
    Node(
      package='joystick',
      executable='joystick_node',
      name='gamepad_publisher',
      output='screen',
      parameters=[
        {
          'calibration_mode': False,
          'vehicle_config_file': vehicle_config_path,
          'use_sim_time': use_sim_time,
        },
      ],
    ),
    Node(
      package='inference',
      executable='inference_node',
      name='inference_node',
      output='screen',
      additional_env={'LANE_VISUALIZE': lane_visualize},
      parameters=[
        {
          'vehicle_config_file': vehicle_config_path,
          'planner_config_file': planner_config_path,
          'route_mode': route_mode,
          'forced_turn': LaunchConfiguration('forced_turn'),
          'aruco_debug_topic': '/debug/aruco',
          'planner_debug_topic': '/debug/planner',
          'aruco_debug_log': True,
          # Gazebo steering=0 is mechanically centred; ignore real-car trim.
          'use_vehicle_steer_trim': False,
          'steer_trim': 0.0,
          'use_sim_time': use_sim_time,
        },
      ],
    ),
  ]

  if use_lane_view:
    nodes.append(
      Node(
        package='inference',
        executable='lane_preview_node',
        name='lane_preview_node',
        output='screen',
        additional_env={'LANE_VISUALIZE': 'off'},
        parameters=[
          {
            'use_sim_time': use_sim_time,
            'image_topic': '/camera/image_raw',
            'planner_debug_topic': '/debug/planner',
            'control_topic': '/control',
            'window_name': 'Lane / Fork Perception',
            'window_scale': LaunchConfiguration('lane_view_scale'),
            'window_x': LaunchConfiguration('lane_view_x'),
            'window_y': LaunchConfiguration('lane_view_y'),
            'focus': LaunchConfiguration('lane_view_focus'),
            'max_hz': LaunchConfiguration('lane_view_max_hz'),
            'route_mode': route_mode,
          },
        ],
      )
    )

  return nodes


def generate_launch_description():
  return LaunchDescription([
    DeclareLaunchArgument('use_sim_time', default_value='true'),
    DeclareLaunchArgument(
      'route_mode', default_value='', choices=['', 'in', 'out'],
      description='Optional route override; empty uses main_planner.yaml',
    ),
    DeclareLaunchArgument(
      'forced_turn',
      default_value='',
      choices=['', 'left', 'right'],
      description=(
        'Pre-latch turn for experiments (camera signs ignored). '
        'IN: left=exit, right=stay. OUT: left=rank0, right=rank1.'
      ),
    ),
    DeclareLaunchArgument(
      'viz',
      default_value='debug',
      choices=['lane', 'off', 'debug', 'all'],
      description=(
        'Windows: lane=Fork preview only; off=none; '
        'debug=preview + Lane drive (control SSOT); all=+HSV'
      ),
    ),
    DeclareLaunchArgument(
      'use_lane_view',
      default_value='true',
      description='Legacy; prefer viz:=. false forces lane preview off.',
    ),
    DeclareLaunchArgument(
      'lane_view_focus',
      default_value='all',
      description='Initial fork focus: all|left|right (keys 0/1/2 live)',
    ),
    DeclareLaunchArgument('lane_view_scale', default_value='1.5'),
    DeclareLaunchArgument('lane_view_x', default_value='40'),
    DeclareLaunchArgument('lane_view_y', default_value='40'),
    DeclareLaunchArgument('lane_view_max_hz', default_value='12.0'),
    OpaqueFunction(function=_launch_setup),
  ])
