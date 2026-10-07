#!/usr/bin/env python3

"""Moving and checking the arm: `laundry move`, `gripper`, `check-flange`."""

import math
import sys

from .common import (
    add_fake_arguments,
    add_observed_state_argument,
    make_gripper,
    RosSession,
)


def cmd_move(args):
    """Run `laundry move ...`."""
    from .. import config

    # Resolve everything that can fail before touching ROS/MoveIt,
    # so a typo is reported immediately instead of after waiting on
    # MoveIt interfaces to come up.
    kind = args.move_kind

    joint_speed = (
        args.velocity if args.velocity is not None else 0.3,
        args.acceleration if args.acceleration is not None else 0.3,
    )

    cartesian_speed = (
        args.velocity if args.velocity is not None else 0.1,
        args.acceleration if args.acceleration is not None else 0.1,
    )

    if kind == 'joints':
        target = list(args.angles)

        if args.degrees:
            target = [math.radians(x) for x in target]

    elif kind in config.named_poses():
        target = config.get_named_pose(kind)

    with RosSession(args=args) as arm:
        if kind == 'joint6':
            velocity, acceleration = joint_speed
            ok = arm.rotate_joint6(
                args.angle, velocity=velocity, acceleration=acceleration
            )

        elif kind == 'joint7':
            velocity, acceleration = joint_speed
            ok = arm.rotate_joint7(
                args.angle, velocity=velocity, acceleration=acceleration
            )

        elif kind == 'linear':
            velocity, acceleration = cartesian_speed
            ok = arm.move_tool_z(
                args.distance,
                max_step=args.step,
                velocity=velocity,
                acceleration=acceleration,
            )

        elif kind == 'twist':
            velocity, acceleration = cartesian_speed
            ok = arm.move_tool_z_with_twist(
                args.distance,
                args.angle,
                max_step=args.step,
                velocity=velocity,
                acceleration=acceleration,
            )

        elif kind == 'joints':
            velocity, acceleration = joint_speed
            ok = arm.move_joints(
                target, velocity=velocity, acceleration=acceleration
            )

        else:
            # Named pose: baked transfer, else straight checked joint
            # move, else the planner (arm/transfers.go_to).
            from ..arm.transfers import go_to

            ok = go_to(arm, kind, time_scale=args.speed)

    return 0 if ok else 1


def _gripper_angle_over_mqtt(angle):
    """Move the ESP32's servo to `angle` (config.GRIPPER_BACKEND 'mqtt')."""
    from .. import config
    from ..hardware.gripper_node import make_mqtt_servo
    from ..hardware.mqtt_servo import GripperLinkError

    driver = make_mqtt_servo(
        config.MQTT_HOST,
        config.MQTT_PORT,
        config.MQTT_TOPIC_PREFIX,
        client_id='laundry_gripper_cli',
    )

    try:
        if not driver.wait_connected(5.0):
            print('Could not reach the MQTT broker.', file=sys.stderr)
            return 1

        print(f'Moving the ESP32 servo to {angle:.1f} degrees')
        driver.move(angle, hold=False)

    except GripperLinkError as exc:
        print(f'Gripper move failed: {exc}', file=sys.stderr)
        return 1

    finally:
        driver.close()

    return 0


def cmd_gripper(args):
    """Run `laundry gripper open|close|ANGLE`."""
    action = args.action.strip().lower()

    if action not in ('open', 'close'):
        try:
            angle = float(action)
        except ValueError:
            print(
                f"Expected 'open', 'close' or an angle; got {args.action!r}",
                file=sys.stderr,
            )
            return 2

        from ..hardware import servo

        # Validate before touching GPIO, so a typo never moves it.
        servo.validate_angle(angle)

        if args.fake_hardware:
            print(f'[fake gripper] servo -> {angle:.1f} deg')
            return 0

        # Drives the servo directly, not through gripper_node: there
        # is no set-angle service, and this is for calibrating the
        # open/close angles in the first place. Don't run it while
        # the pipeline is actively using the gripper.
        from .. import config

        if config.GRIPPER_BACKEND == 'mqtt':
            return _gripper_angle_over_mqtt(angle)

        servo.set_servo_angle(angle)
        return 0

    with RosSession(need_arm=False, node_name='laundry_gripper') as node:
        gripper = make_gripper(node, args.fake_hardware)

        if action == 'open':
            ok = gripper.open_blocking()
        else:
            ok = gripper.close_blocking()

    return 0 if ok else 1


def cmd_check_flange(args):
    """Run `laundry check-flange`."""
    from ..perception import flange_check

    flange_check.main()

    return 0


def add_move_parser(subparsers):
    """Add `laundry move` and its kinds (named poses, joints, ...)."""
    from .. import config

    move = subparsers.add_parser(
        'move',
        help='Move the arm (named pose, joints, J6/J7, linear, twist).',
    )

    move.add_argument(
        '--velocity',
        type=float,
        default=None,
        help='MoveIt velocity scaling (default 0.3 joint, 0.1 Cartesian).',
    )

    move.add_argument(
        '--acceleration',
        type=float,
        default=None,
        help='MoveIt acceleration scaling (default 0.3 joint, 0.1 Cartesian).',
    )

    add_observed_state_argument(move)

    kinds = move.add_subparsers(dest='move_kind', required=True)

    for name in config.named_poses():
        pose = kinds.add_parser(
            name, help=f'Move to the recorded {name.upper()} pose.'
        )
        pose.add_argument(
            '--speed',
            type=float,
            default=1.0,
            help=(
                'Fraction of the baked transfer / straight-move speed, '
                '(0, 1] (default: 1). Use e.g. 0.3 for first runs.'
            ),
        )

    joints = kinds.add_parser('joints', help='Move all seven joints.')
    joints.add_argument(
        'angles', type=float, nargs=7, metavar='J',
        help='Seven absolute target joint angles (radians).',
    )
    joints.add_argument(
        '--degrees', action='store_true',
        help='Interpret angles as degrees.',
    )

    for joint in ('joint6', 'joint7'):
        parser = kinds.add_parser(
            joint, help=f'Rotate {joint.upper()} relative to where it is.'
        )
        parser.add_argument(
            'angle', type=float, help='Relative rotation in degrees.'
        )

    linear = kinds.add_parser('linear', help='Move along current tool Z.')
    linear.add_argument('distance', type=float, help='Distance in metres.')
    linear.add_argument(
        '--step', type=float, default=0.005,
        help='Cartesian interpolation step (default: 0.005).',
    )

    twist = kinds.add_parser(
        'twist', help='Move along tool Z while rotating J7.'
    )
    twist.add_argument('distance', type=float, help='Distance in metres.')
    twist.add_argument(
        'angle', type=float, help='Additional J7 rotation in degrees.'
    )
    twist.add_argument(
        '--step', type=float, default=0.005,
        help='Cartesian interpolation step (default: 0.005).',
    )

    move.set_defaults(func=cmd_move)


def add_gripper_parser(subparsers):
    """Add `laundry gripper`."""
    gripper = subparsers.add_parser(
        'gripper', help='Open/close the gripper, or set a raw servo angle.'
    )
    gripper.add_argument(
        'action', help="'open', 'close' (via gripper_node) or ANGLE in degrees."
    )
    add_fake_arguments(gripper)
    gripper.set_defaults(func=cmd_gripper)


def add_check_flange_parser(subparsers):
    """Add `laundry check-flange`."""
    check = subparsers.add_parser(
        'check-flange',
        help="Report the insertion axis's alignment with the bucket.",
    )
    check.set_defaults(func=cmd_check_flange)
