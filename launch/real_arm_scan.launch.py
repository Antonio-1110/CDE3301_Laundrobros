#!/usr/bin/env python3

"""
real_arm_scan.launch.py

Brings up the entire real-xArm7 scanning stack on a single host
(the Pi, which both controls the arm and handles the ToF sensor):

    - xArm7 hardware driver + MoveIt (headless - no bundled RViz
      window; xarm_moveit_config/launch/xarm7_moveit_realmove.launch.py)
    - tof_sensor         - reads the VL53L0X, publishes
                            sensor_msgs/Range on tof_sensor/range,
                            plus the static flange_link -> sensor
                            mounting transform.
    - scan_recorder_node - listens to that Range topic + /tf,
                            accumulates points, publishes
                            scan_record/points, and serves the
                            save_scan / clear_scan services used by
                            scan_move.py.
    - our own RViz config (scan_visualization.rviz), showing the
      robot model/TF plus the live scan_record/points cloud.

Usage:

    ros2 launch laundry_control real_arm_scan.launch.py

    ros2 launch laundry_control real_arm_scan.launch.py \\
        robot_ip:=192.168.1.207 offset_cm:=-8.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# scan_recorder_node's default auto-save directory is derived from
# wherever its own installed .py file lives - which is inside
# install/laundry_control/..., NOT the source tree. This launch
# file itself gets copied into install/.../share/... too (colcon
# build doesn't use --symlink-install here), so even __file__ of
# *this* file can't be used to find the source tree at runtime.
# Both locations get wiped by `rm -rf install/...` before every
# rebuild, which would silently delete saved scans. So this is
# hardcoded to the actual source-tree location instead of derived
# from anything under install/ - update it if the workspace moves.
_SCAN_RECORDS_DIR = "/home/cde3301a/ros2_ws/src/CDE3301_Laundrobros/scan_records"


def generate_launch_description():

    robot_ip = LaunchConfiguration("robot_ip")

    rviz_delay = LaunchConfiguration("rviz_delay")

    csv_path = LaunchConfiguration("csv_path")

    records_dir = LaunchConfiguration("records_dir")

    offset_cm = LaunchConfiguration("offset_cm")

    publish_rate_hz = LaunchConfiguration("publish_rate_hz")

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
            "Exact path scan_recorder_node saves the ToF scan CSV "
            "to, on save_scan service requests from scan_move.py. "
            "Left empty (default), a file is auto-named under "
            "records_dir with each scan's start time."
        ),
    )

    declare_records_dir = DeclareLaunchArgument(
        "records_dir",
        default_value=_SCAN_RECORDS_DIR,
        description=(
            "Directory auto-named scan CSVs are saved under, when "
            "csv_path is left empty. Defaults to the source-tree "
            "scan_records/ dir, since scan_recorder_node's own "
            "install-space location isn't a safe place to store "
            "data - it gets deleted by `rm -rf install/...` before "
            "a rebuild."
        ),
    )

    declare_offset_cm = DeclareLaunchArgument(
        "offset_cm",
        default_value="-8.5",
        description="Calibration offset (cm) added to raw VL53L0X readings.",
    )

    declare_publish_rate_hz = DeclareLaunchArgument(
        "publish_rate_hz",
        default_value="5.0",
        description="Rate (Hz) at which scan_recorder_node republishes the accumulated PointCloud2.",
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
    # ToF sensor - reads the VL53L0X and broadcasts the static
    # flange_link -> sensor mounting transform.
    # =========================================================

    tof_sensor_node = Node(
        package="laundry_control",
        executable="tof_sensor",
        name="tof_sensor",
        output="screen",
        parameters=[{"offset_cm": offset_cm}],
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
        parameters=[
            {
                "csv_path": csv_path,
                "records_dir": records_dir,
                "publish_rate_hz": publish_rate_hz,
            }
        ],
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
            declare_records_dir,
            declare_offset_cm,
            declare_publish_rate_hz,
            real_arm_moveit_launch,
            tof_sensor_node,
            scan_recorder_node,
            delayed_rviz_node,
        ]
    )
