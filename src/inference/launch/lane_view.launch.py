"""lane_view — auto_driving(또는 manual_driving) 과 **동시에** 띄우는 차선 뷰.

주행 런치가 이미 띄운 카메라(/camera/image/compressed)를 공유하므로 여기서는
카메라를 다시 띄우지 않는다(장치 /dev/video1 충돌 방지). 자체 모니터를 주행
대시보드와 **다른 포트(5001)**에 debug 패널 on 으로 띄운다(포트 충돌 방지).

  # 터미널 1: 주행
  ros2 launch inference auto_driving.launch.py route_mode:=out
  # 터미널 2: 차선 뷰
  ros2 launch inference lane_view.launch.py

  브라우저:
    http://<board-ip>:5000  = 주행 대시보드 (auto_driving 모니터)
    http://<board-ip>:5001  = 차선 뷰 패널 (WHITE=HSV|경계, YELLOW=HSV|경계, EDGE=주행가능)

주의: auto_driving 의 inference 검출 + lane_view 의 detect_with_debug = 검출 2회 +
모니터 2개라 CPU-bound 보드엔 무겁다. 버벅이면 max_hz 를 낮춘다:
  ros2 launch inference lane_view.launch.py max_hz:=1.0
튜닝용이며, 대회 주행 시엔 lane_view 를 끈다.
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / 'config' / 'lane_vision.yaml').is_file():
            return parent
    return Path('/home/topst/2026-SMH')


def generate_launch_description():
    root = _repo_root()
    monitor_config = str(root / 'config' / 'vehicle_config.lane_view.yaml')
    max_hz = LaunchConfiguration('max_hz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'max_hz',
            default_value='1.5',
            description='lane_view preview rate (keep ≤2; detect_with_debug is heavy)',
        ),
        LogInfo(
            msg=[
                'lane_view: monitor(port 5001, debug) + lane_view_web_node @ ',
                max_hz,
                ' Hz. Run alongside auto_driving. Panels → http://<board-ip>:5001',
            ]
        ),
        # 주행 대시보드(5000)와 별개 포트(5001)의 디버그 모니터. 카메라는 공유.
        Node(
            package='monitor',
            executable='monitor_node',
            name='lane_view_monitor',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': monitor_config,
                    'debug_image': True,
                    'image_refresh_interval_ms': 400,
                    'refresh_interval_ms': 1000,
                },
            ],
        ),
        Node(
            package='inference',
            executable='lane_view_web_node',
            name='lane_view_web_node',
            output='screen',
            parameters=[
                {
                    'compressed_topic': '/camera/image/compressed',
                    'max_hz': max_hz,
                    'jpeg_quality': 55,
                },
            ],
        ),
    ])
