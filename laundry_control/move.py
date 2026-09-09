#!/usr/bin/env python3

import argparse
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint


class XArm7MoveGroup(Node):
    """
    Send collision-aware joint-space goals to MoveIt's /move_action.

    Equivalent to sending a MoveGroup action goal from the ROS 2 terminal.
    """

    JOINT_NAMES = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    def __init__(self):
        super().__init__("xarm7_move_group_client")

        self._action_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )

        self.get_logger().info(
            "Waiting for /move_action..."
        )

        self._action_client.wait_for_server()

        self.get_logger().info(
            "Connected to MoveIt /move_action."
        )

    def move_joints(
        self,
        joint_angles,
        velocity=0.3,
        acceleration=0.3,
        planning_attempts=10,
        planning_time=5.0,
        tolerance=0.01,
        replan=True,
        replan_attempts=5,
    ):
        """
        Plan and execute a collision-aware motion to the supplied
        xArm7 joint configuration.

        Parameters
        ----------
        joint_angles : list[float]
            Seven target joint angles in RADIANS.

        velocity : float
            MoveIt velocity scaling factor, 0.0-1.0.

        acceleration : float
            MoveIt acceleration scaling factor, 0.0-1.0.

        Returns
        -------
        bool
            True if MoveIt reports SUCCESS, otherwise False.
        """

        if len(joint_angles) != 7:
            raise ValueError(
                "xArm7 requires exactly 7 joint angles."
            )

        # --------------------------------------------------
        # Construct goal constraints
        # --------------------------------------------------

        constraints = Constraints()

        for joint_name, angle in zip(
            self.JOINT_NAMES,
            joint_angles,
        ):
            joint_constraint = JointConstraint()

            joint_constraint.joint_name = joint_name
            joint_constraint.position = float(angle)

            joint_constraint.tolerance_above = tolerance
            joint_constraint.tolerance_below = tolerance
            joint_constraint.weight = 1.0

            constraints.joint_constraints.append(
                joint_constraint
            )

        # --------------------------------------------------
        # Construct MoveGroup action goal
        # --------------------------------------------------

        goal = MoveGroup.Goal()

        # Same as:
        #
        # request:
        #   group_name: 'xarm7'
        #   num_planning_attempts: 10
        #   allowed_planning_time: 5.0
        #   ...
        #

        goal.request.group_name = "xarm7"

        goal.request.num_planning_attempts = (
            planning_attempts
        )

        goal.request.allowed_planning_time = (
            planning_time
        )

        goal.request.max_velocity_scaling_factor = (
            velocity
        )

        goal.request.max_acceleration_scaling_factor = (
            acceleration
        )

        goal.request.goal_constraints = [
            constraints
        ]

        # --------------------------------------------------
        # Planning options
        # --------------------------------------------------

        # Same settings as your working CLI command.
        goal.planning_options.plan_only = False
        goal.planning_options.replan = replan
        goal.planning_options.replan_attempts = (
            replan_attempts
        )

        # --------------------------------------------------
        # Send goal
        # --------------------------------------------------

        self.get_logger().info(
            "Sending target joint configuration:"
        )

        for name, angle in zip(
            self.JOINT_NAMES,
            joint_angles,
        ):
            self.get_logger().info(
                f"  {name}: "
                f"{angle:+.4f} rad "
                f"({math.degrees(angle):+.2f} deg)"
            )

        send_future = self._action_client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self,
            send_future,
        )

        goal_handle = send_future.result()

        if goal_handle is None:
            self.get_logger().error(
                "Failed to communicate with MoveIt."
            )
            return False

        if not goal_handle.accepted:
            self.get_logger().error(
                "MoveIt rejected the goal."
            )
            return False

        self.get_logger().info(
            "Goal accepted. Planning/executing..."
        )

        # --------------------------------------------------
        # Wait for execution result
        # --------------------------------------------------

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        wrapped_result = result_future.result()

        if wrapped_result is None:
            self.get_logger().error(
                "MoveIt returned no result."
            )
            return False

        result = wrapped_result.result

        error_code = result.error_code.val

        # MoveItErrorCodes.SUCCESS == 1
        if error_code == 1:
            self.get_logger().info(
                "Motion completed successfully."
            )
            return True

        self.get_logger().error(
            f"MoveIt failed. Error code: {error_code}"
        )

        return False


# ==========================================================
# Terminal interface
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Send a collision-aware joint goal "
            "to xArm7 through MoveIt."
        )
    )

    parser.add_argument(
        "joints",
        type=float,
        nargs=7,
        metavar="J",
        help="Seven target joint angles",
    )

    parser.add_argument(
        "--degrees",
        action="store_true",
        help=(
            "Interpret supplied joint angles as degrees "
            "instead of radians."
        ),
    )

    parser.add_argument(
        "--velocity",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--acceleration",
        type=float,
        default=0.3,
    )

    args = parser.parse_args()

    joint_angles = args.joints

    if args.degrees:
        joint_angles = [
            math.radians(x)
            for x in joint_angles
        ]

    rclpy.init()

    arm = XArm7MoveGroup()

    try:
        success = arm.move_joints(
            joint_angles,
            velocity=args.velocity,
            acceleration=args.acceleration,
        )

    finally:
        arm.destroy_node()
        rclpy.shutdown()

    if success:
        raise SystemExit(0)

    raise SystemExit(1)


if __name__ == "__main__":
    main()