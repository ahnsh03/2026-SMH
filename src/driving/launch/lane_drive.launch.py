"""경량 차선추종 주행 launch — 통합 노드(인지+제어 in-process).

  camera_node(D-Racer) → lane_drive_node(인지+제어) → control_node(D-Racer 서보)

인지→제어를 토픽 없이 한 프로세스에서 처리한다. course_mode 로 In/Out 선택.

  ros2 launch driving lane_drive.launch.py course_mode:=out
  ros2 launch driving lane_drive.launch.py course_mode:=in
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def find_vehicle_config():
    for c in (Path.home() / 'D-Racer-Kit' / 'src' / 'config' / 'vehicle_config.yaml',
              Path.home() / 'D-Racer' / 'src' / 'config' / 'vehicle_config.yaml'):
        if c.exists():
            return str(c)
    return ''


def generate_launch_description():
    vision_share = get_package_share_directory('inference')
    lane_vision = os.path.join(vision_share, 'config', 'lane_vision.yaml')
    drive_share = get_package_share_directory('driving')
    lane_control = os.path.join(drive_share, 'config', 'lane_control.yaml')
    vehicle_config = find_vehicle_config()

    course_mode = LaunchConfiguration('course_mode')

    return LaunchDescription([
        DeclareLaunchArgument('course_mode', default_value='out',
                              description="'out'(흰선) 또는 'in'(노란선 진입+회전교차로)"),
        DeclareLaunchArgument('use_camera', default_value='true'),

        # 카메라 (D-Racer)
        Node(
            package='camera', executable='camera_node', name='camera_node',
            output='screen',
            parameters=[{'vehicle_config_file': vehicle_config}],
            condition=IfCondition(LaunchConfiguration('use_camera')),
        ),

        # 통합 주행노드 (인지+제어 in-process) — /camera → /control
        Node(
            package='driving', executable='lane_drive_node', name='lane_drive_node',
            output='screen',
            parameters=[lane_control, {
                'course_mode': course_mode,
                'vision_config_file': lane_vision,
            }],
        ),

        # 액추에이터 (D-Racer) — /control → 서보/스로틀
        Node(
            package='control', executable='control_node', name='control_node',
            output='screen',
            parameters=[{
                'use_joystick_control': False,
                'vehicle_config_file': vehicle_config,
            }],
        ),
    ])
