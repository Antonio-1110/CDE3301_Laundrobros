#!/usr/bin/env python3

"""
preplanned_motion.py

Pre-planned "clear the bucket" sweep, with NO sensor data and no
detection at all - just visits a fixed sequence of previously
recorded joint configurations (arm_position.py's RETRIEVE_0..3),
closing the gripper at each one (as if grabbing something there)
and opening it at DROP in between.

Sequence:

    INTER
    -> RETRIEVE_3 -> close gripper -> DROP -> open gripper
    -> RETRIEVE_2 -> close gripper -> DROP -> open gripper
    -> RETRIEVE_1 -> close gripper -> DROP -> open gripper
    -> RETRIEVE_0 -> close gripper -> DROP -> open gripper
    -> INTER

(highest RETRIEVE_n first, descending to RETRIEVE_0; the final
return to INTER is just a clean resting state, not otherwise
requested - remove it if you don't want it.)

gripper_node.py must be running separately:

    ros2 run laundry_control gripper_node

Usage:
    python3 preplanned_motion.py
"""

import rclpy

import arm_position
from gripper_client import GripperClient
from move import XArm7Controller

# Highest RETRIEVE_n first, descending to lowest.
RETRIEVE_POSITIONS = [
    ("RETRIEVE_3", arm_position.RETRIEVE_3),
    ("RETRIEVE_2", arm_position.RETRIEVE_2),
    ("RETRIEVE_1", arm_position.RETRIEVE_1),
    ("RETRIEVE_0", arm_position.RETRIEVE_0),
]


def main():

    rclpy.init()

    arm = XArm7Controller()
    gripper = GripperClient(arm)

    try:

        print("Moving to INTER...")

        if not arm.move_joints(arm_position.INTER):

            print("Failed to reach INTER; aborting.")

            return

        for name, position in RETRIEVE_POSITIONS:

            print(f"Moving to {name}...")

            if not arm.move_joints(position):

                print(f"Failed to reach {name}; aborting.")

                return

            print("Closing gripper...")

            gripper.close_blocking()

            print("Moving to DROP...")

            if not arm.move_joints(arm_position.DROP):

                print("Failed to reach DROP; aborting.")

                return

            print("Opening gripper...")

            gripper.open_blocking()

        print("Returning to INTER...")

        arm.move_joints(arm_position.INTER)

    finally:

        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
