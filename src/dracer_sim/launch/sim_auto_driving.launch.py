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


def generate_launch_description():
  pkg_share = get_package_share_directory('dracer_sim')
  vehicle_config_path = get_vehicle_config_path()
  cruise_throttle = LaunchConfiguration('cruise_throttle')
  use_sim_time = LaunchConfiguration('use_sim_time')

  return LaunchDescription([
    DeclareLaunchArgument('use_sim_time', default_value='true'),
    DeclareLaunchArgument('cruise_throttle', default_value='0.35'),

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
          'cruise_throttle': cruise_throttle,
          'aruco_debug_topic': '/debug/aruco',
          'aruco_debug_log': True,
          'use_sim_time': use_sim_time,
        },
      ],
    ),
  ])
