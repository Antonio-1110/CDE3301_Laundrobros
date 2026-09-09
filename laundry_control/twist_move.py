#!/usr/bin/env python3

import math
import argparse

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory

from move_linear import XArm7LinearMove


class XArm7TwistMove(XArm7LinearMove):
    """
    Adds a screw/auger-style twist to the base linear move:
    the flange still translates along its local Z axis
    (inherited TF lookup / tool-Z math from
    move_linear.XArm7LinearMove), but can now also rotate
    about that SAME local Z axis at the same time. Since
    the rotation axis passes through the flange origin,
    this changes only orientation, not position.

    move_linear.py itself is left completely untouched -
    this subclass overrides move_along_tool_z() rather than
    modifying the parent.
    """

    # ======================================================
    # Quaternion helpers (not needed by the plain linear
    # move, so they live here rather than in the parent)
    # ======================================================

    @staticmethod
    def _quat_from_axis_angle(axis, angle_rad):
        """
        Quaternion representing a rotation of angle_rad
        about a LOCAL (tool-frame) axis: "x", "y", or "z".

        Returned as (x, y, z, w).
        """

        half = angle_rad / 2.0
        s = math.sin(half)
        c = math.cos(half)

        if axis == "x":
            return (s, 0.0, 0.0, c)
        elif axis == "y":
            return (0.0, s, 0.0, c)
        elif axis == "z":
            return (0.0, 0.0, s, c)
        else:
            raise ValueError(
                f"axis must be 'x', 'y', or 'z', got "
                f"{axis!r}"
            )

    @staticmethod
    def _quat_multiply(q_world, q_local):
        """
        Hamilton product: q_world * q_local.

        Post-multiplying applies q_local in the flange's
        OWN (body) frame - i.e. "rotate the last link about
        its own axis" rather than about a world axis.

        Each quaternion is (x, y, z, w).
        """

        x1, y1, z1, w1 = q_world
        x2, y2, z2, w2 = q_local

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return (x, y, z, w)

    @staticmethod
    def _normalize_quat(q):

        x, y, z, w = q

        n = math.sqrt(x * x + y * y + z * z + w * w)

        return (x / n, y / n, z / n, w / n)

    def _send_cartesian_waypoints(
        self,
        waypoints,
        max_step,
        velocity,
        acceleration,
        log_label="Cartesian path",
    ):
        """
        The "ask MoveIt, check fraction, execute" sequence
        from the parent's move_along_tool_z, factored out
        so both this class's twist move and scan_move.py's
        oscillating move (which subclasses this class) can
        reuse it instead of duplicating it.
        """

        request = GetCartesianPath.Request()

        request.header.frame_id = self.base_frame
        request.group_name = self.group_name
        request.link_name = self.flange_link

        request.waypoints = waypoints

        request.max_step = max_step

        request.jump_threshold = 0.0

        request.avoid_collisions = True
        request.max_velocity_scaling_factor = velocity
        request.max_acceleration_scaling_factor = acceleration

        self.get_logger().info(
            f"Computing collision-aware {log_label}..."
        )

        future = self.cartesian_client.call_async(
            request
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        response = future.result()

        if response is None:
            self.get_logger().error(
                "No response from MoveIt."
            )
            return False

        fraction = response.fraction

        self.get_logger().info(
            f"Cartesian path fraction: "
            f"{fraction * 100:.1f}%"
        )

        if fraction < 0.999:
            self.get_logger().error(
                "MoveIt could not generate the complete "
                "collision-free path."
            )

            self.get_logger().error(
                "Trajectory will NOT be executed."
            )

            return False

        goal = ExecuteTrajectory.Goal()

        goal.trajectory = response.solution

        self.get_logger().info(
            f"Full {log_label} found. Executing..."
        )

        send_future = self.execute_client.send_goal_async(
            goal
        )

        rclpy.spin_until_future_complete(
            self,
            send_future,
        )

        goal_handle = send_future.result()

        if goal_handle is None:
            self.get_logger().error(
                "Failed to communicate with "
                "trajectory execution server."
            )
            return False

        if not goal_handle.accepted:
            self.get_logger().error(
                "Trajectory execution was rejected."
            )
            return False

        result_future = goal_handle.get_result_async()

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        wrapped_result = result_future.result()

        if wrapped_result is None:
            self.get_logger().error(
                "No execution result received."
            )
            return False

        result = wrapped_result.result

        if result.error_code.val == 1:
            self.get_logger().info(
                f"{log_label} completed successfully."
            )
            return True

        self.get_logger().error(
            "Trajectory execution failed. "
            f"MoveIt error code: "
            f"{result.error_code.val}"
        )

        return False

    # ======================================================
    # Override: translate along tool Z, optionally twisting
    # about that same local Z axis at the same time.
    # ======================================================

    def move_along_tool_z(
        self,
        distance,
        twist_deg=None,
        twist_per_meter=0.0,
        max_step=0.005,
        velocity=0.1,
        acceleration=0.1
    ):
        """
        Same translation behaviour as
        XArm7LinearMove.move_along_tool_z, plus an optional
        simultaneous twist about the local tool Z axis.

        distance:
            meters. +0.25 -> 25 cm along tool +Z.
                    -0.25 -> 25 cm along tool -Z.

        twist_deg:
            If given, this exact twist (degrees) is applied
            about the local tool Z axis for this call,
            regardless of distance.

        twist_per_meter:
            If twist_deg is NOT given, the twist is derived
            from distance travelled:
                twist_deg = distance * twist_per_meter
            Example: "150 deg every 5 cm" -> 150/0.05=3000.
            Default 0.0 -> no twist.

        max_step:
            Cartesian interpolation resolution (meters).
        """

        tf = self._get_flange_transform()

        translation = tf.transform.translation
        rotation = tf.transform.rotation

        start_x = translation.x
        start_y = translation.y
        start_z = translation.z

        zx, zy, zz = self._tool_z_from_quaternion(
            rotation
        )

        target_x = start_x + distance * zx
        target_y = start_y + distance * zy
        target_z = start_z + distance * zz

        if twist_deg is None:
            twist_deg = distance * twist_per_meter

        twist_rad = math.radians(twist_deg)

        self.get_logger().info(
            "Linear + twist Cartesian movement:"
        )

        self.get_logger().info(
            f"  distance: {distance:+.3f} m"
        )

        self.get_logger().info(
            f"  twist:    {twist_deg:+.2f} deg about "
            f"local tool Z"
        )

        current_q = (
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )

        twist_q = self._quat_from_axis_angle(
            "z",
            twist_rad,
        )

        target_q = self._quat_multiply(
            current_q,
            twist_q,
        )

        target_q = self._normalize_quat(target_q)

        target_pose = Pose()

        target_pose.position.x = target_x
        target_pose.position.y = target_y
        target_pose.position.z = target_z

        target_pose.orientation.x = target_q[0]
        target_pose.orientation.y = target_q[1]
        target_pose.orientation.z = target_q[2]
        target_pose.orientation.w = target_q[3]

        return self._send_cartesian_waypoints(
            waypoints=[target_pose],
            max_step=max_step,
            velocity=velocity,
            acceleration=acceleration,
            log_label="linear+twist path",
        )


# ==========================================================
# Terminal interface
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Move xArm7 linearly along the flange's local "
            "Z axis, twisting the last link about that "
            "same axis (a screw motion)."
        )
    )

    parser.add_argument(
        "distance",
        type=float,
        help=(
            "Distance in meters. "
            "Positive = tool +Z, negative = tool -Z."
        ),
    )

    parser.add_argument(
        "--twist",
        type=float,
        default=None,
        help=(
            "Exact twist in degrees about the local tool "
            "Z axis for this move. Overrides "
            "--twist-per-meter if given."
        ),
    )

    parser.add_argument(
        "--twist-per-meter",
        type=float,
        default=0.0,
        help=(
            "Twist in degrees, scaled by distance moved. "
            "Example: for '150 deg every 5 cm', pass "
            "3000 (i.e. 150 / 0.05). Default: 0.0."
        ),
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--velocity",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--acceleration",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    rclpy.init()

    twister = XArm7TwistMove()

    try:
        success = twister.move_along_tool_z(
            args.distance,
            twist_deg=args.twist,
            twist_per_meter=args.twist_per_meter,
            max_step=args.step,
            velocity=args.velocity,
            acceleration=args.acceleration,
        )

    finally:
        twister.destroy_node()
        rclpy.shutdown()

    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()