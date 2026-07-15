"""자율주행 통합 launch (board: inference/launch/auto_driving.launch.py 대응).

주행 체인:
  camera_node(D-Racer) → inference_node(인지) → lane_control_node(제어) → control_node(D-Racer 서보)

  /camera/image/compressed → /perception/lane → /control

D-Racer-Kit(camera/control/battery/monitor)는 언더레이로 먼저 빌드/소싱되어 있어야 한다.
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def find_vehicle_config():
    """D-Racer-Kit 의 vehicle_config.yaml 경로를 탐색."""
    candidates = [
        Path.home() / 'D-Racer-Kit' / 'src' / 'config' / 'vehicle_config.yaml',
        Path.home() / 'D-Racer' / 'src' / 'config' / 'vehicle_config.yaml',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def generate_launch_description():
    cfg = get_package_share_directory('inference')
    lane_vision = os.path.join(cfg, 'config', 'lane_vision.yaml')
    lane_control = os.path.join(cfg, 'config', 'lane_control.yaml')
    main_planner = os.path.join(cfg, 'config', 'main_planner.yaml')
    vehicle_config = find_vehicle_config()

    use_monitor = LaunchConfiguration('use_monitor')
    use_battery = LaunchConfiguration('use_battery')

    return LaunchDescription([
        DeclareLaunchArgument('use_monitor', default_value='true'),
        DeclareLaunchArgument('use_battery', default_value='true'),

        # 1) 카메라 (D-Racer) — /camera/image/compressed 발행
        Node(
            package='camera', executable='camera_node', name='camera_node',
            output='screen',
            parameters=[{'vehicle_config_file': vehicle_config}],
        ),

        # 2) 인지 (perception) — 영상 → /perception/lane
        Node(
            package='inference', executable='inference_node', name='inference_node',
            output='screen',
            parameters=[lane_vision, {'vehicle_config_file': vehicle_config}],
        ),

        # 3) 제어 (driving) — /perception/lane → /control
        Node(
            package='driving', executable='control_node', name='lane_control_node',
            output='screen',
            parameters=[lane_control, main_planner],
        ),

        # 4) 액추에이터 (D-Racer) — /control → 서보/스로틀
        Node(
            package='control', executable='control_node', name='control_node',
            output='screen',
            parameters=[{
                'use_joystick_control': False,
                'vehicle_config_file': vehicle_config,
            }],
        ),

        # 5) 배터리 모니터 (D-Racer, 선택)
        _optional('battery', 'battery_node', use_battery,
                  [{'vehicle_config_file': vehicle_config}]),

        # 6) 웹 모니터 (D-Racer, 선택)
        _optional('monitor', 'monitor_node', use_monitor,
                  [{'vehicle_config_file': vehicle_config}]),
    ])


def _optional(package, executable, cond, params):
    from launch.conditions import IfCondition
    return Node(
        package=package, executable=executable, name=executable,
        output='screen', parameters=params, condition=IfCondition(cond),
    )
