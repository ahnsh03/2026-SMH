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


def get_planner_config_path() -> str:
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'config' / 'main_planner.yaml'
        if candidate.exists():
            return str(candidate)
    return '/home/topst/2026-SMH/config/main_planner.yaml'


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    planner_config_path = get_planner_config_path()
    route_mode = LaunchConfiguration('route_mode')

    return LaunchDescription([
        DeclareLaunchArgument(
            'route_mode',
            default_value='',
            choices=['', 'in', 'out'],
            description='Optional route override; empty uses main_planner.yaml',
        ),
        Node(
            package='camera',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[
                {
                    'vehicle_config_file': vehicle_config_path,
                    # Board is CPU-bound: a log line per frame is 30 writes/sec.
                    'debug_log': False,
                },
            ],
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
                    'planner_config_file': planner_config_path,
                    'route_mode': route_mode,
                    # ArUco 보드 테스트: ros2 topic echo /debug/aruco
                    'aruco_debug_topic': '/debug/aruco',
                    'planner_debug_topic': '/debug/planner',
                    'aruco_debug_log': True,
                    # Real servo neutral is calibrated in vehicle_config.yaml.
                    'use_vehicle_steer_trim': True,
                },
            ],
        ),
    ])
