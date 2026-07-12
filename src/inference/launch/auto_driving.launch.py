from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def get_vehicle_config_path() -> str:
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'src' / 'config' / 'vehicle_config.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/2026-SMH/src/config/vehicle_config.yaml'


def get_lane_control_config_path() -> str:
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'config' / 'lane_control.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/2026-SMH/config/lane_control.yaml'


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    control_config_path = get_lane_control_config_path()
    cruise_throttle = LaunchConfiguration('cruise_throttle')

    return LaunchDescription([
        DeclareLaunchArgument(
            'cruise_throttle',
            default_value='0.35',
            description='Default forward throttle when lane following is active',
        ),
        Node(
            package='camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[{'vehicle_config_file': vehicle_config_path}],
        ),
        Node(
            package='control',
            executable='control_node',
            name='control_node',
            output='screen',
            parameters=[
                {
                    'use_joystick_control': False,
                    'vehicle_config_file': vehicle_config_path,
                },
            ],
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
                },
            ],
        ),
        Node(
            package='battery',
            executable='battery_node',
            name='battery_node',
            output='screen',
        ),
        Node(
            package='monitor',
            executable='monitor_node',
            name='monitor_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    'debug_image': False,
                },
            ],
        ),
        Node(
            package='inference',
            executable='inference_node',
            name='inference_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    # ArUco 보드 테스트: ros2 topic echo /debug/aruco
                    'aruco_debug_topic': '/debug/aruco',
                    'aruco_debug_log': True,
                },
            ],
        ),
        # Real car: STEER_TRIM from vehicle_config (steer_trim_override=false).
        Node(
            package='inference',
            executable='lane_control_node',
            name='lane_control_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    'control_config_file': control_config_path,
                    'cruise_throttle': cruise_throttle,
                    'steer_trim_override': False,
                    'lane_timeout_sec': 0.5,
                },
            ],
        ),
    ])
