"""Debug / monitor launch for board-race trackside checks.

Same stack as ``auto_driving.launch.py``, with defaults tuned for debugging:

* ``traffic_pass:=true`` — skip green wait / red stop (ArUco still stops)
* ``publish_bev_debug:=true`` — White / IN ego / OUT ego on web monitor
* ``drive_debug_log:=true``

Web UI: http://<board-ip>:5000  
Terminal: ``python3 scripts/board_monitor_term.py --hz``

    ros2 launch inference debug_monitor.launch.py route_mode:=out
    ros2 launch inference debug_monitor.launch.py route_mode:=in traffic_pass:=false
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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
    traffic_pass = LaunchConfiguration('traffic_pass')
    publish_bev = LaunchConfiguration('publish_bev_debug')

    return LaunchDescription([
        DeclareLaunchArgument(
            'route_mode',
            default_value='out',
            choices=['', 'in', 'out'],
            description='Course override for debug runs',
        ),
        DeclareLaunchArgument(
            'traffic_pass',
            default_value='true',
            choices=['true', 'false'],
            description='Skip WAIT_GREEN / red stop (ArUco still stops)',
        ),
        DeclareLaunchArgument(
            'publish_bev_debug',
            default_value='true',
            choices=['true', 'false'],
            description='Publish White/IN/OUT BEV masks to monitor panels',
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
                    'debug_image': True,
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
                    'traffic_pass': ParameterValue(traffic_pass, value_type=bool),
                    'aruco_debug_topic': '/debug/aruco',
                    'planner_debug_topic': '/debug/planner',
                    'aruco_debug_log': True,
                    'publish_bev_debug': ParameterValue(
                        publish_bev, value_type=bool
                    ),
                    'bev_lane_topic': '/debug/bev/white/compressed',
                    'bev_road_topic': '/debug/bev/in/compressed',
                    'bev_out_topic': '/debug/bev/out/compressed',
                    'bev_debug_hz': 5.0,
                    'bringup_crawl_throttle': 0.20,
                    'drive_debug_log': True,
                    'drive_debug_hz': 2.0,
                    'use_vehicle_steer_trim': True,
                },
            ],
        ),
    ])
