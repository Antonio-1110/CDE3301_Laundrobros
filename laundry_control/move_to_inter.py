#!/usr/bin/env python3

import rclpy

from laundry_control.arm_position import INTER
from laundry_control.move import XArm7Controller


def move_to_inter(arm=None):

    # If an existing arm controller was provided,
    # use it directly.
    if arm is not None:
        return arm.move_joints(INTER)

    # Otherwise this script is responsible for ROS.
    rclpy.init()

    arm = XArm7Controller()

    try:
        return arm.move_joints(INTER)

    finally:
        arm.destroy_node()
        rclpy.shutdown()


def main():

    success = move_to_inter()

    if success:
        print("Robot reached INTER.")
        return 0

    print("Failed to reach INTER.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())