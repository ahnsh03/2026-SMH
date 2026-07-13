import os
from pathlib import Path

import xacro
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, SetEnvironmentVariable
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


def spawn_poses_path(pkg_share: str) -> str:
  return os.path.join(pkg_share, 'config', 'spawn_poses.yaml')


def load_spawn_pose_catalog(pkg_share: str) -> dict:
  path = spawn_poses_path(pkg_share)
  try:
    with open(path, encoding='utf-8') as f:
      data = yaml.safe_load(f) or {}
  except OSError:
    data = {}
  poses = data.get('poses') or {}
  default = str(data.get('default') or 'start')
  return {'default': default, 'poses': poses, 'path': path}


def resolve_spawn(context, pkg_share: str) -> tuple[str, str, str, str, str, str]:
  """Return (x, y, z, yaw, pose_id, label) for spawn_entity."""

  catalog = load_spawn_pose_catalog(pkg_share)
  poses = catalog['poses']
  pose_id = LaunchConfiguration('spawn_pose').perform(context).strip()
  if not pose_id:
    pose_id = catalog['default']

  if pose_id != 'custom' and pose_id in poses:
    pose = poses[pose_id]
    return (
      f"{float(pose['x']):.6g}",
      f"{float(pose['y']):.6g}",
      f"{float(pose.get('z', 0.15)):.6g}",
      f"{float(pose['yaw']):.6g}",
      pose_id,
      str(pose.get('label') or pose_id),
    )

  if pose_id != 'custom' and pose_id not in poses:
    # Unknown name → fall back to start if present, else CLI coords.
    if 'start' in poses:
      pose = poses['start']
      return (
        f"{float(pose['x']):.6g}",
        f"{float(pose['y']):.6g}",
        f"{float(pose.get('z', 0.15)):.6g}",
        f"{float(pose['yaw']):.6g}",
        'start',
        f"unknown '{pose_id}' → start",
      )

  return (
    LaunchConfiguration('spawn_x').perform(context),
    LaunchConfiguration('spawn_y').perform(context),
    LaunchConfiguration('spawn_z').perform(context),
    LaunchConfiguration('spawn_yaw').perform(context),
    'custom',
    'custom (spawn_x/y/z/yaw)',
  )


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
  camera_preview_config = os.path.join(pkg_share, 'config', 'camera_preview.yaml')
  bev_preview_config = os.path.join(pkg_share, 'config', 'bev_preview.yaml')
  battery_config = os.path.join(pkg_share, 'config', 'battery_stub.yaml')
  vehicle_config_path = get_vehicle_config_path()

  robot = LaunchConfiguration('robot').perform(context)
  use_sim_time = LaunchConfiguration('use_sim_time')
  use_monitor = LaunchConfiguration('use_monitor').perform(context)
  use_camera_view = LaunchConfiguration('use_camera_view').perform(context)
  use_bev_view = LaunchConfiguration('use_bev_view').perform(context)
  camera_view_topic = LaunchConfiguration('camera_view_topic').perform(context)
  camera_view_width = int(LaunchConfiguration('camera_view_width').perform(context))
  camera_view_height = int(LaunchConfiguration('camera_view_height').perform(context))
  bev_view_scale = float(LaunchConfiguration('bev_view_scale').perform(context))

  spawn_x, spawn_y, spawn_z_val, spawn_yaw, pose_id, pose_label = resolve_spawn(
    context, pkg_share
  )
  if robot == 'dracer':
    spawn_z_val = '0.05'

  # Team Metric IPM SSOT (locked in config/lane_vision.yaml).
  lane_vision_config = ''
  for base_path in Path(__file__).resolve().parents:
    candidate = base_path / 'config' / 'lane_vision.yaml'
    if candidate.is_file():
      lane_vision_config = str(candidate)
      break

  robot_description = load_robot_description(robot)
  gazebo_urdf_path = write_gazebo_urdf(robot_description, robot)
  entity_name = 'limo' if robot == 'limo' else 'dracer_sim'

  nodes = [
    LogInfo(
      msg=(
        f'[dracer_sim] spawn_pose={pose_id} ({pose_label}) '
        f'x={spawn_x} y={spawn_y} z={spawn_z_val} yaw={spawn_yaw}'
      )
    ),
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
      parameters=[
        control_config,
        {
          'use_sim_time': use_sim_time,
          # limo = Ackermann (steer angle); lightweight box = diff_drive (yaw rate)
          'cmd_mode': 'diff_yaw_rate' if robot == 'dracer' else 'ackermann_steer',
        },
      ],
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

  if use_camera_view == 'true':
    nodes.append(
      Node(
        package='dracer_sim',
        executable='sim_camera_preview',
        name='sim_camera_preview',
        output='screen',
        parameters=[
          camera_preview_config,
          {
            'use_sim_time': use_sim_time,
            'image_topic': camera_view_topic,
            'window_width': camera_view_width,
            'window_height': camera_view_height,
          },
        ],
        additional_env={
          'QT_X11_NO_MITSHM': '1',
          'DISPLAY': os.environ.get('DISPLAY', ':0'),
        },
      )
    )

  if use_bev_view == 'true':
    bev_params = {
      'use_sim_time': use_sim_time,
      'image_topic': camera_view_topic,
      'window_scale': bev_view_scale,
    }
    if lane_vision_config:
      bev_params['config_path'] = lane_vision_config
    nodes.append(
      Node(
        package='dracer_sim',
        executable='sim_bev_preview',
        name='sim_bev_preview',
        output='screen',
        parameters=[
          bev_preview_config,
          bev_params,
        ],
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
  catalog = load_spawn_pose_catalog(pkg_share)
  pose_choices = sorted(catalog['poses'].keys()) + ['custom']
  default_pose = catalog['default'] if catalog['default'] in catalog['poses'] else 'start'

  return LaunchDescription([
    DeclareLaunchArgument('use_sim_time', default_value='true'),
    DeclareLaunchArgument('robot', default_value='limo',
                          choices=['limo', 'dracer'],
                          description='Spawn LIMO Ackermann or lightweight D-Racer box'),
    DeclareLaunchArgument(
      'spawn_pose',
      default_value=default_pose,
      choices=pose_choices,
      description=(
        'Mission spawn preset from config/spawn_poses.yaml '
        f'({", ".join(pose_choices)}). Use custom + spawn_x/y/z/yaw for manual.'
      ),
    ),
    DeclareLaunchArgument(
      'spawn_x', default_value='2.6',
      description='Used when spawn_pose:=custom',
    ),
    DeclareLaunchArgument(
      'spawn_y', default_value='-3.92',
      description='Used when spawn_pose:=custom',
    ),
    DeclareLaunchArgument(
      'spawn_z', default_value='0.15',
      description='Used when spawn_pose:=custom (limo); dracer forces 0.05',
    ),
    DeclareLaunchArgument(
      'spawn_yaw', default_value='-3.14',
      description='Used when spawn_pose:=custom (rad)',
    ),
    DeclareLaunchArgument(
      'headless', default_value='false',
      description='true면 gzclient(GUI) 없이 gzserver만 실행 — CPU 부하 감소',
    ),
    DeclareLaunchArgument(
      'use_monitor', default_value='false',
      description='D-Racer 웹 모니터 (시뮬 기본 OFF — OpenCV 카메라 프리뷰 사용, 실기는 manual/auto launch)',
    ),
    DeclareLaunchArgument(
      'use_camera_view', default_value='true',
      description='OpenCV 카메라 프리뷰 창 (/camera/image_raw, 16:9)',
    ),
    DeclareLaunchArgument(
      'use_bev_view', default_value='true',
      description='Metric IPM BEV 프리뷰 창 (config/lane_vision.yaml SSOT)',
    ),
    DeclareLaunchArgument(
      'camera_view_topic', default_value='/camera/image_raw',
      description='카메라·BEV 프리뷰 구독 토픽',
    ),
    DeclareLaunchArgument(
      'camera_view_width', default_value='640',
      description='프리뷰 창 가로 (기본 640 = 320x180의 2배)',
    ),
    DeclareLaunchArgument(
      'camera_view_height', default_value='360',
      description='프리뷰 창 세로 (기본 360, 16:9)',
    ),
    DeclareLaunchArgument(
      'bev_view_scale', default_value='2.0',
      description='BEV 창 배율 (Metric IPM 그리드 ≈386×321 × scale)',
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
