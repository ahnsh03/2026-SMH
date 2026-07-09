import os
from pathlib import Path

import xacro
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_vehicle_config_path() -> str:
  for base_path in Path(__file__).resolve().parents:
    candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
    if candidate.exists():
      return str(candidate)
  return '/home/topst/2026-SMH/src/config/vehicle_config.yaml'


def resolve_gazebo_meshes(urdf: str) -> str:
  """Gazebo Classic does not resolve package:// mesh URIs; RViz does."""
  try:
    limo_share = get_package_share_directory('limo_car')
  except PackageNotFoundError:
    return urdf
  return urdf.replace('package://limo_car/', f'file://{limo_share}/')


def write_gazebo_urdf(urdf: str, robot: str) -> str:
  path = f'/tmp/dracer_sim_{robot}_gazebo.urdf'
  with open(path, 'w', encoding='utf-8') as urdf_file:
    urdf_file.write(resolve_gazebo_meshes(urdf))
  return path


def load_robot_description(robot: str) -> str:
  pkg_share = get_package_share_directory('dracer_sim')
  if robot == 'dracer':
    urdf_path = os.path.join(pkg_share, 'urdf', 'dracer_sim.urdf')
    with open(urdf_path, encoding='utf-8') as urdf_file:
      return urdf_file.read()

  xacro_path = os.path.join(pkg_share, 'urdf', 'limo_dracer_sim.xacro')
  return xacro.process_file(xacro_path).toxml()


def launch_setup(context, *args, **kwargs):
  pkg_share = get_package_share_directory('dracer_sim')
  control_config = os.path.join(pkg_share, 'config', 'control_bridge.yaml')
  camera_config = os.path.join(pkg_share, 'config', 'camera_republish.yaml')
  battery_config = os.path.join(pkg_share, 'config', 'battery_stub.yaml')
  rviz_config = os.path.join(pkg_share, 'rviz', 'sim_camera.rviz')
  vehicle_config_path = get_vehicle_config_path()

  robot = LaunchConfiguration('robot').perform(context)
  use_sim_time = LaunchConfiguration('use_sim_time')
  spawn_x = LaunchConfiguration('spawn_x')
  spawn_y = LaunchConfiguration('spawn_y')
  spawn_yaw = LaunchConfiguration('spawn_yaw')
  use_monitor = LaunchConfiguration('use_monitor').perform(context)
  use_rviz = LaunchConfiguration('use_rviz').perform(context)

  robot_description = load_robot_description(robot)
  gazebo_urdf_path = write_gazebo_urdf(robot_description, robot)
  entity_name = 'limo' if robot == 'limo' else 'dracer_sim'
  spawn_z_val = LaunchConfiguration('spawn_z').perform(context)
  if robot == 'dracer':
    spawn_z_val = '0.05'

  nodes = [
    Node(
      package='robot_state_publisher',
      executable='robot_state_publisher',
      name='robot_state_publisher',
      output='screen',
      parameters=[
        {'robot_description': robot_description, 'use_sim_time': use_sim_time},
      ],
    ),
    Node(
      package='dracer_sim',
      executable='sim_robot_description_publisher',
      name='robot_description_publisher',
      output='screen',
      parameters=[
        {'robot_description': robot_description, 'use_sim_time': use_sim_time},
      ],
    ),
    Node(
      package='gazebo_ros',
      executable='spawn_entity.py',
      name='spawn_robot',
      output='screen',
      arguments=[
        '-entity', entity_name,
        '-file', gazebo_urdf_path,
        '-x', spawn_x,
        '-y', spawn_y,
        '-z', spawn_z_val,
        '-Y', spawn_yaw,
      ],
      parameters=[{'use_sim_time': use_sim_time}],
    ),
    Node(
      package='dracer_sim',
      executable='sim_control_bridge',
      name='sim_control_bridge',
      output='screen',
      parameters=[control_config, {'use_sim_time': use_sim_time}],
    ),
    Node(
      package='dracer_sim',
      executable='sim_camera_republish',
      name='sim_camera_republish',
      output='screen',
      parameters=[camera_config, {'use_sim_time': use_sim_time}],
    ),
    Node(
      package='dracer_sim',
      executable='sim_battery_stub',
      name='sim_battery_stub',
      output='screen',
      parameters=[battery_config, {'use_sim_time': use_sim_time}],
    ),
  ]

  if use_monitor == 'true':
    nodes.append(
      Node(
        package='monitor',
        executable='monitor_node',
        name='monitor_node',
        output='screen',
        parameters=[
          {
            'vehicle_config_file': vehicle_config_path,
            'debug_image': False,
            'web_host': '0.0.0.0',
            'image_source_width': 320,
            'image_source_height': 180,
            'use_sim_time': use_sim_time,
          },
        ],
      )
    )

  if use_rviz == 'true':
    nodes.append(
      Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        additional_env={
          'QT_X11_NO_MITSHM': '1',
          'DISPLAY': os.environ.get('DISPLAY', ':0'),
        },
      )
    )

  return nodes


def generate_launch_description():
  pkg_share = get_package_share_directory('dracer_sim')
  world_path = os.path.join(pkg_share, 'worlds', 'track_cw.world')
  models_path = os.path.join(pkg_share, 'models')
  headless = LaunchConfiguration('headless')

  return LaunchDescription([
    DeclareLaunchArgument('use_sim_time', default_value='true'),
    DeclareLaunchArgument('robot', default_value='limo',
                          choices=['limo', 'dracer'],
                          description='Spawn LIMO Ackermann or lightweight D-Racer box'),
    DeclareLaunchArgument('spawn_x', default_value='2.6'),
    DeclareLaunchArgument('spawn_y', default_value='-3.92'),
    DeclareLaunchArgument('spawn_z', default_value='0.15'),
    DeclareLaunchArgument('spawn_yaw', default_value='-3.14'),
    DeclareLaunchArgument(
      'headless', default_value='false',
      description='true면 gzclient(GUI) 없이 gzserver만 실행 — CPU 부하 감소',
    ),
    DeclareLaunchArgument(
      'use_monitor', default_value='false',
      description='D-Racer 웹 모니터 (시뮬 기본 OFF — RViz 사용, 실기는 manual/auto launch)',
    ),
    DeclareLaunchArgument(
      'use_rviz', default_value='true',
      description='RViz2로 /camera/image_raw(320x180) + 로봇 모델 표시',
    ),

    SetEnvironmentVariable(
      name='GAZEBO_MODEL_PATH',
      value=models_path + ':' + os.environ.get('GAZEBO_MODEL_PATH', ''),
    ),

    IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
      ),
      launch_arguments={
        'world': world_path,
        'verbose': 'false',
        'gui': 'false',
        'server_required': 'true',
        'gui_required': 'false',
      }.items(),
      condition=IfCondition(headless),
    ),

    IncludeLaunchDescription(
      PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
      ),
      launch_arguments={
        'world': world_path,
        'verbose': 'false',
        'gui': 'true',
      }.items(),
      condition=UnlessCondition(headless),
    ),

    OpaqueFunction(function=launch_setup),
  ])
