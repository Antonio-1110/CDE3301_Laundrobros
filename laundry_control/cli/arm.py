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


# `laundry move jointN`: N = 1..7.
SINGLE_JOINTS = tuple(f'joint{n}' for n in range(1, 8))

# Default --speed for a single-joint move: the full straight-move
# speed (45 deg/s peak), like the other straight moves.
SINGLE_JOINT_DEFAULT_SPEED = 1.0


def _limits_deg(index):
    """Return a joint's usable range in degrees: its limits minus the margin."""
    from .. import config

    margin = config.JOINT_LIMIT_MARGIN_RAD

    return (
        math.degrees(config.JOINT_LOWER_LIMITS_RAD[index] + margin),
        math.degrees(config.JOINT_UPPER_LIMITS_RAD[index] - margin),
    )


def single_joint_target(current, index, angle_deg, absolute):
    """
    Return the seven-joint target that moves only joint `index`, or raise.

    angle_deg is relative to current[index], or the target itself when
    absolute. Raises ValueError, saying why, if it is outside the
    joint's range: config.JOINT_*_LIMITS_RAD less the margin, or for a
    joint already inside the margin, its current angle on that side.
    """
    from .. import config
    from ..arm.joint_path import limits_no_closer_than

    target = list(current)
    target[index] = (
        math.radians(angle_deg) if absolute
        else current[index] + math.radians(angle_deg)
    )

    # A joint already inside its margin may move out, not further in.
    lower, upper = limits_no_closer_than(
        current,
        config.JOINT_LOWER_LIMITS_RAD,
        config.JOINT_UPPER_LIMITS_RAD,
        config.JOINT_LIMIT_MARGIN_RAD,
    )
    low, high = math.degrees(lower[index]), math.degrees(upper[index])
    wanted = math.degrees(target[index])

    if not low <= wanted <= high:
        raise ValueError(
            f'J{index + 1} to {wanted:+.1f} deg is outside its range '
            f'{low:+.1f} .. {high:+.1f} deg; not moving.'
        )

    return target


def _move_single_joint(arm, index, angle_deg, absolute, time_scale):
    """Move only joint `index`, in a straight collision-checked move."""
    current = arm.get_current_joints()

    if current is None:
        print('Could not read the current joint angles.', file=sys.stderr)
        return False

    try:
        target = single_joint_target(current, index, angle_deg, absolute)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return False

    print(
        f'J{index + 1}: {math.degrees(current[index]):+.1f} -> '
        f'{math.degrees(target[index]):+.1f} deg '
        f'({math.degrees(target[index] - current[index]):+.1f}); '
        'the other joints stay put.'
    )

    # Straight joint-space line: only this joint moves, every 1 deg
    # of it is collision- and limit-checked first, and nothing moves
    # if any state fails - unlike the planner, which may move the
    # other joints on the way. A joint already inside its limit
    # margin may stay there or move out, never further in, so the
    # arm can always be moved out of such a pose by hand.
    return arm.move_joints_linear(
        target, time_scale=time_scale, no_closer_to_limits=True
    )


def cmd_move(args):
    """Run `laundry move ...`."""
    from .. import config

    # Resolve everything that can fail before touching ROS/MoveIt,
    # so a typo is reported immediately instead of after waiting on
    # MoveIt interfaces to come up.
    kind = args.move_kind

    if kind in SINGLE_JOINTS:
        index = SINGLE_JOINTS.index(kind)

        if not 0.0 < args.speed <= 1.0:
            print('--speed must be in (0, 1].', file=sys.stderr)
            return 2

        if args.to:
            low, high = _limits_deg(index)

            if not low <= args.angle <= high:
                print(
                    f'J{index + 1} to {args.angle:+.1f} deg is outside its '
                    f'range {low:+.1f} .. {high:+.1f} deg; not moving.',
                    file=sys.stderr,
                )
                return 2

        with RosSession(args=args) as arm:
            ok = _move_single_joint(
                arm, index, args.angle, args.to, args.speed
            )

        return 0 if ok else 1

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
        if kind == 'linear':
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
        help='Move the arm (named pose, joints, one joint, linear, twist).',
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

    for index, joint in enumerate(SINGLE_JOINTS):
        low, high = _limits_deg(index)
        parser = kinds.add_parser(
            joint,
            help=(
                f'Turn J{index + 1} alone by DEG (or to DEG with --to); '
                'straight, collision-checked.'
            ),
        )
        parser.add_argument(
            'angle', type=float, metavar='DEG',
            help=(
                f'Degrees to turn J{index + 1} by (relative), or with --to '
                f'the angle to turn it to; range {low:+.0f} .. {high:+.0f}.'
            ),
        )
        parser.add_argument(
            '--to', action='store_true',
            help='DEG is the target angle, not a relative turn.',
        )
        parser.add_argument(
            '--speed', type=float, default=SINGLE_JOINT_DEFAULT_SPEED,
            help=(
                'Fraction of the straight-move speed (45 deg/s), (0, 1] '
                f'(default: {SINGLE_JOINT_DEFAULT_SPEED:g}).'
            ),
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
