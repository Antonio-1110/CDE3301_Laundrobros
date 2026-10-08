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

The arm's ros2_control loop runs at config.ARM_CONTROL_RATE_HZ (100),
not the stock 150 Hz; to compare against stock:

    ros2 launch laundry_control laundry_bringup.launch.py control_rate_hz:=150

ToF sensor and gripper on the ESP32 (over MQTT) instead of this Pi's
I2C/GPIO - defaults from config.TOF_SOURCE / GRIPPER_BACKEND:

    ros2 launch laundry_control laundry_bringup.launch.py \\
        tof_source:=mqtt gripper_backend:=mqtt

Scan CSVs: scan_recorder_node's records_dir defaults (when left "")
to config.scan_records_dir() - the SOURCE tree's scan_records/,
found from the installed module's real path under --symlink-install.
It must not be anywhere under install/ or build/, which get wiped by
`rm -rf build install log` before a clean rebuild.
"""

from launch import LaunchDescription, LaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.actions import OpaqueFunction, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _moveit_launch_file(launch_file):
    return PythonLaunchDescriptionSource(
        PathJoinSubstitution(
            [FindPackageShare('xarm_moveit_config'), 'launch', launch_file]
        )
    )


def _realmove_with_control_rate():
    """
    Return the manufacturer's real-arm launch, at our controller rate.

    Their _robot_moveit_realmove.launch.py, unmodified, loaded from its
    installed file - except that its call building the ros2_control
    parameters (uf_robot_utils.generate_ros2_control_params_temp_file)
    also gets update_rate, from the control_rate_hz launch argument.
    That helper already supports it; the launch just never passes it,
    so the rate would otherwise be the stock xarm7_controllers.yaml's
    150 Hz (config.ARM_CONTROL_RATE_HZ says why we run slower).
    """
    import importlib.util
    import os

    from ament_index_python.packages import get_package_share_directory

    path = os.path.join(
        get_package_share_directory('xarm_moveit_config'),
        'launch',
        '_robot_moveit_realmove.launch.py',
    )
    spec = importlib.util.spec_from_file_location('xarm_realmove', path)
    realmove = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(realmove)

    # Loud, not silently 150 Hz, if an xarm_ros2 update renames these.
    stock_params = realmove.generate_ros2_control_params_temp_file
    stock_setup = realmove.launch_setup

    def launch_setup(context, *args, **kwargs):
        rate = int(LaunchConfiguration('control_rate_hz').perform(context))

        def params_at_our_rate(params_path, **params_kwargs):
            params_kwargs['update_rate'] = rate
            return stock_params(params_path, **params_kwargs)

        # Only this private copy of their module is changed.
        realmove.generate_ros2_control_params_temp_file = params_at_our_rate
        return stock_setup(context, *args, **kwargs)

    return LaunchDescriptionSource(
        LaunchDescription([OpaqueFunction(function=launch_setup)])
    )


def _moveit_include(source, condition, extra_arguments):
    # The manufacturer's underlying launch, not its xarm7_* wrapper: the
    # wrapper forwards only a few arguments, and ours (the gripper mesh,
    # the joint limits - config.xarm_description_arguments) must reach
    # the robot description. xarm_ros2 itself stays unmodified.
    from laundry_control import config

    return IncludeLaunchDescription(
        source,
        launch_arguments=dict(
            config.xarm_description_arguments(),
            show_rviz='false',
            **extra_arguments,
        ).items(),
        condition=condition,
    )


# The MoveIt parameters the manufacturer's own RViz gets
# (xarm_moveit_config/launch/_robot_moveit_common2.launch.py).
RVIZ_MOVEIT_PARAMETERS = (
    'robot_description',
    'robot_description_semantic',
    'robot_description_kinematics',
    'robot_description_planning',
    'planning_pipelines',
)


def _rviz_node(context):
    # RViz's PlanningScene display needs the SRDF as a parameter:
    # move_group does not publish /robot_description_semantic, so
    # without it the display waits forever and never shows the
    # obstacles. Built by the manufacturer's builder from the same
    # arguments as move_group's. Not the manufacturer's RViz node
    # itself: closing that one shuts down the whole launch, arm
    # driver included.
    from laundry_control import config
    from uf_ros_lib.moveit_configs_builder import MoveItConfigsBuilder

    moveit_config = MoveItConfigsBuilder(
        context=context, **config.xarm_description_arguments()
    ).to_moveit_configs().to_dict()

    return [
        Node(
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
            parameters=[
                {name: moveit_config[name] for name in RVIZ_MOVEIT_PARAMETERS}
            ],
        )
    ]


def generate_launch_description():
    from laundry_control import config

    fake = LaunchConfiguration('fake')
    rviz = LaunchConfiguration('rviz')
    robot_ip = LaunchConfiguration('robot_ip')
    rviz_delay = LaunchConfiguration('rviz_delay')
    csv_path = LaunchConfiguration('csv_path')
    records_dir = LaunchConfiguration('records_dir')
    offset_cm = LaunchConfiguration('offset_cm')
    publish_rate_hz = LaunchConfiguration('publish_rate_hz')
    tof_source = LaunchConfiguration('tof_source')
    gripper_backend = LaunchConfiguration('gripper_backend')
    mqtt_host = LaunchConfiguration('mqtt_host')
    mqtt_port = LaunchConfiguration('mqtt_port')
    control_rate_hz = LaunchConfiguration('control_rate_hz')

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
            'control_rate_hz',
            default_value=str(config.ARM_CONTROL_RATE_HZ),
            description=(
                'Real arm only: ros2_control update rate (Hz), i.e. how '
                'often a new target goes to the arm. Stock xarm_ros2: 150.'
            ),
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
        DeclareLaunchArgument(
            'tof_source',
            default_value=config.TOF_SOURCE,
            choices=list(config.TOF_SOURCES),
            description=(
                "Where ToF readings come from: 'i2c' (this Pi's VL53L0X) "
                "or 'mqtt' (the ESP32)."
            ),
        ),
        DeclareLaunchArgument(
            'gripper_backend',
            default_value=config.GRIPPER_BACKEND,
            choices=list(config.GRIPPER_BACKENDS),
            description=(
                "Who drives the gripper servo: 'gpio' (this Pi) or "
                "'mqtt' (the ESP32)."
            ),
        ),
        DeclareLaunchArgument(
            'mqtt_host',
            default_value=config.MQTT_HOST,
            description='MQTT broker the ESP32 talks through (mqtt only).',
        ),
        DeclareLaunchArgument(
            'mqtt_port',
            default_value=str(config.MQTT_PORT),
            description='MQTT broker port (mqtt only).',
        ),
    ]

    mqtt_parameters = {'mqtt_host': mqtt_host, 'mqtt_port': mqtt_port}

    real_moveit = _moveit_include(
        _realmove_with_control_rate(),
        UnlessCondition(fake),
        {'robot_ip': robot_ip, 'control_rate_hz': control_rate_hz},
    )

    # The fake controller keeps the stock rate: there is no arm to wait
    # for, so it never overruns the way the real one did.
    fake_moveit = _moveit_include(
        _moveit_launch_file('_robot_moveit_fake.launch.py'),
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
        parameters=[
            {'offset_cm': offset_cm, 'source': tof_source, **mqtt_parameters}
        ],
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
        parameters=[{'backend': gripper_backend, **mqtt_parameters}],
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
    # is in the table: move the fake arm to HOME (see cli/scene.py
    # _fake_start_at_home).
    fake_scene_node = Node(
        package='laundry_control',
        executable='laundry',
        arguments=['scene', 'apply', '--fake-start-home'],
        output='screen',
        condition=IfCondition(fake),
    )

    delayed_rviz = TimerAction(
        period=rviz_delay,
        actions=[OpaqueFunction(function=_rviz_node)],
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
