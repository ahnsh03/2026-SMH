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
  use_sim_time = LaunchConfiguration('use_sim_time')
  spawn_pose = LaunchConfiguration('spawn_pose')
  spawn_x = LaunchConfiguration('spawn_x')
  spawn_y = LaunchConfiguration('spawn_y')
  spawn_z = LaunchConfiguration('spawn_z')
  spawn_yaw = LaunchConfiguration('spawn_yaw')

  return LaunchDescription([
    DeclareLaunchArgument('use_sim_time', default_value='true'),
    DeclareLaunchArgument(
      'spawn_pose', default_value='start',
      description='Mission spawn preset (see config/spawn_poses.yaml)',
    ),
    DeclareLaunchArgument('spawn_x', default_value='2.6'),
    DeclareLaunchArgument('spawn_y', default_value='-3.92'),
    DeclareLaunchArgument('spawn_z', default_value='0.15'),
    DeclareLaunchArgument('spawn_yaw', default_value='-3.14'),

    IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(pkg_share, 'launch', 'sim_bringup.launch.py')
      ),
      launch_arguments={
        'use_sim_time': use_sim_time,
        'use_monitor': 'false',
        'spawn_pose': spawn_pose,
        'spawn_x': spawn_x,
        'spawn_y': spawn_y,
        'spawn_z': spawn_z,
        'spawn_yaw': spawn_yaw,
      }.items(),
    ),

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
      package='dracer_sim',
      executable='sim_joystick_bridge',
      name='sim_joystick_bridge',
      output='screen',
      parameters=[{'use_sim_time': use_sim_time}],
    ),
  ])
