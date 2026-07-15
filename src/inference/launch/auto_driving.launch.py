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


def get_rt_wrap_prefix():
    """i2c 노드(control/battery)를 SCHED_RR로 실행하는 래퍼 경로.
    권한 없으면 래퍼가 알아서 일반 실행으로 폴백한다. 못 찾으면 prefix 미적용."""
    for base_path in Path(__file__).resolve().parents:
        candidate = base_path / 'scripts' / 'rt_wrap.sh'
        if candidate.exists():
            return [str(candidate)]
    fallback = Path('/home/topst/2026-SMH/scripts/rt_wrap.sh')
    return [str(fallback)] if fallback.exists() else None


def generate_launch_description():
    vehicle_config_path = get_vehicle_config_path()
    planner_config_path = get_planner_config_path()
    route_mode = LaunchConfiguration('route_mode')
    rt_prefix = get_rt_wrap_prefix()

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
            # i2c 타이머를 CPU 스타베이션에서 보호(SCHED_RR). 권한 없으면 자동 폴백.
            prefix=rt_prefix,
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
            # i2c 타이머를 CPU 스타베이션에서 보호(SCHED_RR). 권한 없으면 자동 폴백.
            prefix=rt_prefix,
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
                    # Real car: measured geometry + timing for the board's
                    # ~2.2 Hz perception. See profiles.real in main_planner.yaml.
                    'planner_profile': 'real',
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
