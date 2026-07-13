import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
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


def generate_launch_description():
  pkg_share = get_package_share_directory('dracer_sim')
  vehicle_config_path = get_vehicle_config_path()
  planner_config_path = get_planner_config_path()
  route_mode = LaunchConfiguration('route_mode')
  use_sim_time = LaunchConfiguration('use_sim_time')

  return LaunchDescription([
    DeclareLaunchArgument('use_sim_time', default_value='true'),
    DeclareLaunchArgument(
      'route_mode', default_value='', choices=['', 'in', 'out'],
      description='Optional route override; empty uses main_planner.yaml',
    ),

    IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(pkg_share, 'launch', 'sim_bringup.launch.py')
      ),
      launch_arguments={
        'use_sim_time': use_sim_time,
        'use_monitor': 'false',
      }.items(),
    ),

    # Same stack as auto_driving.launch.py but without hardware camera/control/battery.
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
      parameters=[
        {
          'vehicle_config_file': vehicle_config_path,
          'planner_config_file': planner_config_path,
          'route_mode': route_mode,
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
  ])
