#!/usr/bin/env python3

"""
real_arm_scan.launch.py

Brings up the real xArm7 (MoveIt + hardware driver) headlessly -
i.e. without the default xarm_moveit_config RViz window - and
instead opens our own scan_visualization.rviz config, which shows
the robot model/TF plus the live scan_record/points cloud.

Usage:

    ros2 launch laundry_control real_arm_scan.launch.py

    ros2 launch laundry_control real_arm_scan.launch.py \\
        robot_ip:=192.168.1.207
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    robot_ip = LaunchConfiguration("robot_ip")

    rviz_delay = LaunchConfiguration("rviz_delay")

    csv_path = LaunchConfiguration("csv_path")

    declare_robot_ip = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.1.207",
        description="IP address of the real xArm7 controller.",
    )

    declare_rviz_delay = DeclareLaunchArgument(
        "rviz_delay",
        default_value="3.0",
        description=(
            "Seconds to wait before starting RViz, so the robot "
            "description / TF are already available when it opens."
        ),
    )

    declare_csv_path = DeclareLaunchArgument(
        "csv_path",
        default_value="",
        description=(
            "Path scan_recorder_node saves the ToF scan CSV to, on "
            "save_scan service requests from scan_move.py. Left "
            "empty (default), it auto-saves under scan_records/ at "
            "the package root, named with this run's start time."
        ),
    )

    # =========================================================
    # Real-arm MoveIt stack, headless (no bundled RViz window).
    #
    # xarm_moveit_config/launch/xarm7_moveit_realmove.launch.py
    # =========================================================

    real_arm_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("xarm_moveit_config"),
                    "launch",
                    "xarm7_moveit_realmove.launch.py",
                ]
            )
        ),
        launch_arguments={
            "robot_ip": robot_ip,
            "show_rviz": "false",
        }.items(),
    )

    # =========================================================
    # Scan recorder - a separate node/process from the arm's own
    # xarm7_controller node, so ToF capture, TF lookups and CSV/
    # point-cloud work can never stall arm-control's spin loop.
    # =========================================================

    scan_recorder_node = Node(
        package="laundry_control",
        executable="scan_recorder_node",
        name="scan_recorder_node",
        output="screen",
        parameters=[{"csv_path": csv_path}],
    )

    # =========================================================
    # Our own RViz config: robot model + TF + scan_record/points.
    # =========================================================

    rviz_config_file = PathJoinSubstitution(
        [
            FindPackageShare("laundry_control"),
            "rviz",
            "scan_visualization.rviz",
        ]
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
    )

    delayed_rviz_node = TimerAction(
        period=rviz_delay,
        actions=[rviz_node],
    )

    return LaunchDescription(
        [
            declare_robot_ip,
            declare_rviz_delay,
            declare_csv_path,
            real_arm_moveit_launch,
            scan_recorder_node,
            delayed_rviz_node,
        ]
    )
