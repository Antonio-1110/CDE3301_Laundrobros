#!/usr/bin/env python3

"""
gripper_cli.py

Direct terminal control of the gripper, via gripper_node.py's
open_gripper/close_gripper services (see gripper_client.py) - no
need to hand-type `ros2 service call ... std_srvs/srv/Trigger`.

gripper_node.py must already be running separately:

    ros2 run laundry_control gripper_node

Usage:
    ros2 run laundry_control gripper_cli open
    ros2 run laundry_control gripper_cli close
    python3 -m laundry_control.gripper_cli open
"""

import argparse

import rclpy
from rclpy.node import Node

from .gripper_client import GripperClient


def build_parser():

    parser = argparse.ArgumentParser(
        description="Open or close the gripper via gripper_node.py."
    )

    parser.add_argument(
        "command",
        choices=["open", "close"],
        help="Which action to perform.",
    )

    return parser


def main():

    args = build_parser().parse_args()

    rclpy.init()

    node = Node("gripper_cli")
    gripper = GripperClient(node)

    try:

        if args.command == "open":
            success = gripper.open_blocking()
        else:
            success = gripper.close_blocking()

    finally:

        node.destroy_node()
        rclpy.shutdown()

    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
