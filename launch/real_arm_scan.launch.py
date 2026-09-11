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
            real_arm_moveit_launch,
            delayed_rviz_node,
        ]
    )
