"""수동주행 통합 launch (board: inference/launch/manual_driving.launch.py 대응).

조이스틱 teleop 만 구동 (인지/제어 계층 없음):
  joystick_node → /joystick → control_node(D-Racer, use_joystick_control=True) → 서보
카메라/모니터는 데이터 수집·디버그용으로 함께 띄운다.
"""
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def find_vehicle_config():
    candidates = [
        Path.home() / 'D-Racer-Kit' / 'src' / 'config' / 'vehicle_config.yaml',
        Path.home() / 'D-Racer' / 'src' / 'config' / 'vehicle_config.yaml',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(candidates[0])


def generate_launch_description():
    vehicle_config = find_vehicle_config()
    use_monitor = LaunchConfiguration('use_monitor')

    return LaunchDescription([
        DeclareLaunchArgument('use_monitor', default_value='true'),

        Node(
            package='camera', executable='camera_node', name='camera_node',
            output='screen',
            parameters=[{'vehicle_config_file': vehicle_config}],
        ),
        Node(
            package='joystick', executable='joystick_node', name='joystick_node',
            output='screen',
            parameters=[{'vehicle_config_file': vehicle_config}],
        ),
        Node(
            package='control', executable='control_node', name='control_node',
            output='screen',
            parameters=[{
                'use_joystick_control': True,
                'vehicle_config_file': vehicle_config,
            }],
        ),
        Node(
            package='monitor', executable='monitor_node', name='monitor_node',
            output='screen',
            parameters=[{'vehicle_config_file': vehicle_config}],
            condition=IfCondition(use_monitor),
        ),
    ])
