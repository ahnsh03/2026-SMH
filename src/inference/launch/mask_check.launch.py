"""Headless HSV mask check on the D3-G monitor (no inference / no control).

Lightweight IPM + inRange only (~1.5 Hz) so the dashboard stays responsive.

  ./scripts/board_sync.sh --no-pull
  source install/setup.bash
  ros2 launch inference mask_check.launch.py

Browser: http://<board-ip>:5000
  camera   = live frame
  grayscale = white HSV mask
  blur      = yellow HSV mask
  edge      = 2x2 (white | yellow / black_road | red_road)
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
    vehicle_config = str(root / 'config' / 'vehicle_config.mask_check.yaml')
    vision_config = str(root / 'config' / 'lane_vision.yaml')
    max_hz = LaunchConfiguration('max_hz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'max_hz',
            default_value='1.5',
            description='HSV preview rate (keep ≤2 to avoid starving the board CPU)',
        ),
        LogInfo(
            msg=[
                'HSV mask check: camera + monitor(debug) + hsv_mask_web @ ',
                max_hz,
                ' Hz. Open http://<board-ip>:5000 — grayscale=WHITE blur=YELLOW edge=ROAD(black|red) masked BEV',
            ]
        ),
        Node(
            package='camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config,
                    'debug_log': False,
                },
            ],
        ),
        Node(
            package='monitor',
            executable='monitor_node',
            name='monitor_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config,
                    'debug_image': True,
                    # Poll debug panels slower than driving camera (~less Wi-Fi).
                    'image_refresh_interval_ms': 400,
                    'refresh_interval_ms': 1000,
                },
            ],
        ),
        Node(
            package='inference',
            executable='hsv_mask_web_node',
            name='hsv_mask_web_node',
            output='screen',
            parameters=[
                {
                    'compressed_topic': '/camera/image/compressed',
                    'max_hz': max_hz,
                    'jpeg_quality': 55,
                    'vision_config': vision_config,
                },
            ],
        ),
    ])
