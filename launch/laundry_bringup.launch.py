#!/usr/bin/env python3

r"""
Bring up everything `laundry run` needs, on one host.

Real rig (default, fake:=false) - the Pi controls the arm and owns
the ToF sensor and the gripper servo:

    - xArm7 hardware driver + MoveIt, headless
      (xarm_moveit_config/launch/_robot_moveit_realmove.launch.py, unmodified;
       our gripper mesh and joint limits go in as its launch arguments)
    - tof_sensor         - reads the VL53L0X, publishes
                           sensor_msgs/Range on tof_sensor/range,
                           plus the static flange -> sensor transform.
    - scan_recorder_node - turns Range + /tf into scan points,
                           publishes scan_record/points, serves
                           save_scan / clear_scan.
    - gripper_node       - owns the servo GPIO, serves
                           open_gripper / close_gripper.
    - `laundry scene apply` - adds the padded bucket and table
      (config.OBSTACLES) to MoveIt as world objects, then exits.
    - RViz with scan_visualization.rviz (rviz:=false to skip).

No hardware (fake:=true) - only the MoveIt fake controller, the
obstacles (and RViz if asked for), for use with `laundry ... --fake-hardware`, which
replaces the ToF recorder and the gripper with in-process stand-ins:

    ros2 launch laundry_control laundry_bringup.launch.py fake:=true rviz:=false
    laundry run --fake-hardware --scan-from baseline_scans/<one>.csv

Usage:

    ros2 launch laundry_control laundry_bringup.launch.py
    ros2 launch laundry_control laundry_bringup.launch.py \\
        robot_ip:=192.168.1.207 offset_cm:=-8.5

Scan CSVs: scan_recorder_node's records_dir defaults (when left "")
to config.scan_records_dir() - the SOURCE tree's scan_records/,
found from the installed module's real path under --symlink-install.
It must not be anywhere under install/ or build/, which get wiped by
`rm -rf build install log` before a clean rebuild.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _moveit_include(launch_file, condition, extra_arguments):
    # The manufacturer's underlying launch, not its xarm7_* wrapper: the
    # wrapper forwards only a few arguments, and ours (the gripper mesh,
    # the joint limits - config.xarm_description_arguments) must reach
    # the robot description. xarm_ros2 itself stays unmodified.
    from laundry_control import config

    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('xarm_moveit_config'), 'launch', launch_file]
            )
        ),
        launch_arguments=dict(
            config.xarm_description_arguments(),
            show_rviz='false',
            **extra_arguments,
        ).items(),
        condition=condition,
    )


def generate_launch_description():
    fake = LaunchConfiguration('fake')
    rviz = LaunchConfiguration('rviz')
    robot_ip = LaunchConfiguration('robot_ip')
    rviz_delay = LaunchConfiguration('rviz_delay')
    csv_path = LaunchConfiguration('csv_path')
    records_dir = LaunchConfiguration('records_dir')
    offset_cm = LaunchConfiguration('offset_cm')
    publish_rate_hz = LaunchConfiguration('publish_rate_hz')

    arguments = [
        DeclareLaunchArgument(
            'fake',
            default_value='false',
            description=(
                'true: MoveIt fake controller only, no hardware nodes '
                '(pair with `laundry ... --fake-hardware`).'
            ),
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Start RViz with scan_visualization.rviz.',
        ),
        DeclareLaunchArgument(
            'robot_ip',
            default_value='192.168.1.207',
            description='IP address of the real xArm7 controller.',
        ),
        DeclareLaunchArgument(
            'rviz_delay',
            default_value='3.0',
            description=(
                'Seconds to wait before starting RViz, so the robot '
                'description / TF are already available when it opens.'
            ),
        ),
        DeclareLaunchArgument(
            'csv_path',
            default_value='',
            description=(
                'Exact path scan_recorder_node saves to. Normally left '
                'empty: `laundry scan --save` / `laundry run` pin it per '
                'scan, and otherwise files are auto-named by start time '
                'under records_dir.'
            ),
        ),
        DeclareLaunchArgument(
            'records_dir',
            default_value='',
            description=(
                'Directory auto-named scan CSVs go under '
                '(default "": <repo>/scan_records).'
            ),
        ),
        DeclareLaunchArgument(
            'offset_cm',
            default_value='-8.5',
            description='Calibration offset (cm) added to raw VL53L0X readings.',
        ),
        DeclareLaunchArgument(
            'publish_rate_hz',
            default_value='5.0',
            description=(
                'Rate (Hz) at which scan_recorder_node republishes the '
                'accumulated PointCloud2.'
            ),
        ),
    ]

    real_moveit = _moveit_include(
        '_robot_moveit_realmove.launch.py',
        UnlessCondition(fake),
        {'robot_ip': robot_ip},
    )

    fake_moveit = _moveit_include(
        '_robot_moveit_fake.launch.py',
        IfCondition(fake),
        {},
    )

    # =========================================================
    # Hardware nodes - real rig only. Each is its own process, so
    # ToF capture, TF lookups, CSV I/O and GPIO can never stall the
    # arm-control node's spin loop.
    # =========================================================

    tof_sensor_node = Node(
        package='laundry_control',
        executable='tof_sensor',
        name='tof_sensor',
        output='screen',
        parameters=[{'offset_cm': offset_cm}],
        condition=UnlessCondition(fake),
    )

    scan_recorder_node = Node(
        package='laundry_control',
        executable='scan_recorder_node',
        name='scan_recorder_node',
        output='screen',
        parameters=[
            {
                'csv_path': csv_path,
                'records_dir': records_dir,
                'publish_rate_hz': publish_rate_hz,
            }
        ],
        condition=UnlessCondition(fake),
    )

    gripper_node = Node(
        package='laundry_control',
        executable='gripper_node',
        name='gripper_node',
        output='screen',
        condition=UnlessCondition(fake),
    )

    # One-shot: waits for move_group, adds the obstacles, exits. Every
    # `laundry` command does the same on connect (skipped when already
    # there); doing it here too means RViz and hand planning see them
    # from the start.
    scene_node = Node(
        package='laundry_control',
        executable='laundry',
        arguments=['scene', 'apply'],
        output='screen',
        condition=UnlessCondition(fake),
    )

    # The mock hardware starts at all-zeros, where the modelled gripper
    # is in the table: move the fake arm to HOME (see cli.py
    # _fake_start_at_home).
    fake_scene_node = Node(
        package='laundry_control',
        executable='laundry',
        arguments=['scene', 'apply', '--fake-start-home'],
        output='screen',
        condition=IfCondition(fake),
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d',
            PathJoinSubstitution(
                [
                    FindPackageShare('laundry_control'),
                    'rviz',
                    'scan_visualization.rviz',
                ]
            ),
        ],
    )

    delayed_rviz = TimerAction(
        period=rviz_delay,
        actions=[rviz_node],
        condition=IfCondition(rviz),
    )

    return LaunchDescription(
        arguments
        + [
            real_moveit,
            fake_moveit,
            tof_sensor_node,
            scan_recorder_node,
            gripper_node,
            scene_node,
            fake_scene_node,
            delayed_rviz,
        ]
    )
