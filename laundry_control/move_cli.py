#!/usr/bin/env python3

"""
move_cli.py

Terminal interface for XArm7Controller (move.py): ad-hoc joint,
J6/J7 rotation, linear, twist, and named-position moves from the
command line.

Usage:
    python3 move_cli.py joints <7 angles> [--degrees]
    python3 move_cli.py joint6 <angle>
    python3 move_cli.py joint7 <angle>
    python3 move_cli.py linear <distance> [--step ...]
    python3 move_cli.py twist <distance> <angle> [--step ...]
    python3 move_cli.py position <name>
"""

import argparse
import math
import sys

import rclpy

import arm_position
from move import XArm7Controller


def get_named_position(name):
    """
    Look up a named joint configuration in arm_position.py
    (case-insensitive), e.g. "home" -> arm_position.HOME.

    Returns a list of 7 joint angles in radians.
    Raises KeyError, listing what IS available, if not found.
    """

    positions = {
        attr: value
        for attr, value in vars(arm_position).items()
        if attr.isupper() and isinstance(value, (list, tuple))
    }

    key = name.strip().upper()

    if key not in positions:

        available = ", ".join(sorted(positions)) or "(none defined)"

        raise KeyError(
            f"Unknown arm position '{name}'. "
            f"Available positions in arm_position.py: {available}"
        )

    return list(positions[key])


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Unified xArm7 MoveIt controller."
        )
    )

    parser.add_argument(
        "--velocity",
        type=float,
        default=None,
        help="MoveIt velocity scaling.",
    )

    parser.add_argument(
        "--acceleration",
        type=float,
        default=None,
        help="MoveIt acceleration scaling.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ---------------------------------------------------------
    # joints
    # ---------------------------------------------------------

    joints_parser = subparsers.add_parser(
        "joints",
        help="Move all seven joints.",
    )

    joints_parser.add_argument(
        "angles",
        type=float,
        nargs=7,
        metavar="J",
        help="Seven target joint angles.",
    )

    joints_parser.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret angles as degrees.",
    )

    # ---------------------------------------------------------
    # joint6
    # ---------------------------------------------------------

    j6_parser = subparsers.add_parser(
        "joint6",
        help="Rotate J6 relative to current position.",
    )

    j6_parser.add_argument(
        "angle",
        type=float,
        help="Relative J6 rotation in degrees.",
    )

    # ---------------------------------------------------------
    # joint7
    # ---------------------------------------------------------

    j7_parser = subparsers.add_parser(
        "joint7",
        help="Rotate J7 relative to current position.",
    )

    j7_parser.add_argument(
        "angle",
        type=float,
        help="Relative J7 rotation in degrees.",
    )

    # ---------------------------------------------------------
    # linear
    # ---------------------------------------------------------

    linear_parser = subparsers.add_parser(
        "linear",
        help="Move along current tool Z.",
    )

    linear_parser.add_argument(
        "distance",
        type=float,
        help="Distance in metres.",
    )

    linear_parser.add_argument(
        "--step",
        type=float,
        default=0.005,
        help="Cartesian interpolation step.",
    )

    # ---------------------------------------------------------
    # twist
    # ---------------------------------------------------------

    twist_parser = subparsers.add_parser(
        "twist",
        help="Move along tool Z while rotating J7.",
    )

    twist_parser.add_argument(
        "distance",
        type=float,
        help="Distance in metres.",
    )

    twist_parser.add_argument(
        "angle",
        type=float,
        help="Additional J7 rotation in degrees.",
    )

    twist_parser.add_argument(
        "--step",
        type=float,
        default=0.005,
        help="Cartesian interpolation step.",
    )

    # ---------------------------------------------------------
    # position
    # ---------------------------------------------------------

    position_parser = subparsers.add_parser(
        "position",
        help="Move to a named joint configuration from arm_position.py.",
    )

    position_parser.add_argument(
        "name",
        type=str,
        help="Position name, e.g. home, inter (case-insensitive).",
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    # Resolve the named position (if any) before touching ROS/MoveIt
    # at all, so a typo is reported immediately instead of after
    # waiting on MoveIt interfaces to come up.
    target_position = None

    if args.command == "position":

        try:
            target_position = get_named_position(args.name)
        except KeyError as exc:
            print(exc, file=sys.stderr)
            raise SystemExit(1)

    rclpy.init()

    arm = XArm7Controller()

    success = False

    try:

        # =====================================================
        # JOINTS
        # =====================================================

        if args.command == "joints":

            joint_angles = list(
                args.angles
            )

            if args.degrees:

                joint_angles = [
                    math.radians(x)
                    for x in joint_angles
                ]

            velocity = (
                args.velocity
                if args.velocity is not None
                else 0.3
            )

            acceleration = (
                args.acceleration
                if args.acceleration is not None
                else 0.3
            )

            success = arm.move_joints(
                joint_angles,
                velocity=velocity,
                acceleration=acceleration,
            )

        # =====================================================
        # J6
        # =====================================================

        elif args.command == "joint6":

            velocity = (
                args.velocity
                if args.velocity is not None
                else 0.3
            )

            acceleration = (
                args.acceleration
                if args.acceleration is not None
                else 0.3
            )

            success = arm.rotate_joint6(
                args.angle,
                velocity=velocity,
                acceleration=acceleration,
            )

        # =====================================================
        # J7
        # =====================================================

        elif args.command == "joint7":

            velocity = (
                args.velocity
                if args.velocity is not None
                else 0.3
            )

            acceleration = (
                args.acceleration
                if args.acceleration is not None
                else 0.3
            )

            success = arm.rotate_joint7(
                args.angle,
                velocity=velocity,
                acceleration=acceleration,
            )

        # =====================================================
        # LINEAR
        # =====================================================

        elif args.command == "linear":

            velocity = (
                args.velocity
                if args.velocity is not None
                else 0.1
            )

            acceleration = (
                args.acceleration
                if args.acceleration is not None
                else 0.1
            )

            success = arm.move_tool_z(
                args.distance,
                max_step=args.step,
                velocity=velocity,
                acceleration=acceleration,
            )

        # =====================================================
        # TWIST
        # =====================================================

        elif args.command == "twist":

            velocity = (
                args.velocity
                if args.velocity is not None
                else 0.1
            )

            acceleration = (
                args.acceleration
                if args.acceleration is not None
                else 0.1
            )

            success = arm.move_tool_z_with_twist(
                args.distance,
                args.angle,
                max_step=args.step,
                velocity=velocity,
                acceleration=acceleration,
            )

        # =====================================================
        # POSITION
        # =====================================================

        elif args.command == "position":

            velocity = (
                args.velocity
                if args.velocity is not None
                else 0.3
            )

            acceleration = (
                args.acceleration
                if args.acceleration is not None
                else 0.3
            )

            success = arm.move_joints(
                target_position,
                velocity=velocity,
                acceleration=acceleration,
            )

    except KeyboardInterrupt:

        arm.get_logger().warning(
            "Movement interrupted."
        )

        success = False

    finally:

        arm.destroy_node()
        rclpy.shutdown()

    raise SystemExit(
        0 if success else 1
    )


if __name__ == "__main__":
    main()