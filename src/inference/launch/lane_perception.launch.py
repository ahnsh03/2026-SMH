"""차선 인지 단독 launch — 카메라 + inference_node (제어 없음).

  camera_node(D-Racer) → inference_node(인지) → /perception/lane

metric BEV 차선검출을 실차에서 단독 검증할 때 사용. 제어까지 함께 띄우려면
auto_driving.launch.py 를 쓴다.

주의: lane_vision.yaml 의 metric_ipm/hsv/lane_detect 블록은 (노드 ros__parameters
아래가 아니라) 최상위에 있어 ROS 파라미터로 자동 주입되지 않는다. 그래서
vision_config_file 로 '파일 경로'를 넘겨 LaneDetector 가 직접 읽게 한다.
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
    share = get_package_share_directory('inference')
    lane_vision = os.path.join(share, 'config', 'lane_vision.yaml')
    vehicle_config = find_vehicle_config()

    return LaunchDescription([
        DeclareLaunchArgument('use_camera', default_value='true',
                              description='D-Racer camera_node 동시 실행'),
        DeclareLaunchArgument('vision_config_file', default_value=lane_vision),

        # 카메라 (D-Racer) — /camera/image/compressed
        Node(
            package='camera', executable='camera_node', name='camera_node',
            output='screen',
            parameters=[{'vehicle_config_file': vehicle_config}],
            condition=IfCondition(LaunchConfiguration('use_camera')),
        ),

        # 인지 — 영상 → /perception/lane
        Node(
            package='inference', executable='inference_node', name='inference_node',
            output='screen',
            parameters=[lane_vision, {
                'vision_config_file': LaunchConfiguration('vision_config_file'),
                'vehicle_config_file': vehicle_config,
            }],
        ),
    ])
