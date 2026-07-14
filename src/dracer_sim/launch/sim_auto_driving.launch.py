"""Full sim auto-driving: Gazebo bringup + auto stack (one command).

For experiments that restart driving often without killing Gazebo, prefer::

  # T1 — leave up
  ros2 launch dracer_sim sim_bringup.launch.py spawn_pose:=out_fork
  # T2 — toggle
  ros2 launch dracer_sim sim_auto_stack.launch.py route_mode:=out
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
  pkg_share = get_package_share_directory('dracer_sim')
  use_sim_time = LaunchConfiguration('use_sim_time')
  route_mode = LaunchConfiguration('route_mode')
  spawn_pose = LaunchConfiguration('spawn_pose')
  spawn_x = LaunchConfiguration('spawn_x')
  spawn_y = LaunchConfiguration('spawn_y')
  spawn_z = LaunchConfiguration('spawn_z')
  spawn_yaw = LaunchConfiguration('spawn_yaw')
  use_lane_view = LaunchConfiguration('use_lane_view')

  return LaunchDescription([
    DeclareLaunchArgument('use_sim_time', default_value='true'),
    DeclareLaunchArgument(
      'route_mode', default_value='', choices=['', 'in', 'out'],
      description='Optional route override; empty uses main_planner.yaml',
    ),
    DeclareLaunchArgument(
      'spawn_pose', default_value='start',
      description='Mission spawn preset (see config/spawn_poses.yaml)',
    ),
    DeclareLaunchArgument('spawn_x', default_value='2.6'),
    DeclareLaunchArgument('spawn_y', default_value='-3.92'),
    DeclareLaunchArgument('spawn_z', default_value='0.15'),
    DeclareLaunchArgument('spawn_yaw', default_value='-3.14'),
    DeclareLaunchArgument(
      'use_lane_view',
      default_value='true',
      description='Open live lane/fork perception overlay',
    ),
    DeclareLaunchArgument(
      'forced_turn',
      default_value='',
      choices=['', 'left', 'right'],
      description='IN: left=exit, right=stay; OUT: left/right fork rank',
    ),

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
    IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(pkg_share, 'launch', 'sim_auto_stack.launch.py')
      ),
      launch_arguments={
        'use_sim_time': use_sim_time,
        'route_mode': route_mode,
        'use_lane_view': use_lane_view,
        'forced_turn': LaunchConfiguration('forced_turn'),
      }.items(),
    ),
  ])
