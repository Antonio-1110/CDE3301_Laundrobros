# #!/usr/bin/env python3

# import argparse
# import math

# import rclpy

# from geometry_msgs.msg import Pose
# from moveit_msgs.srv import GetCartesianPath
# from moveit_msgs.action import ExecuteTrajectory

# from move_linear import XArm7LinearMove


# class XArm7TwistMove(XArm7LinearMove):
#     """
#     Linear motion along the current tool +Z axis, with an additional
#     deliberate rotation applied specifically to joint7.

#     Method
#     ------
#     1. Ask MoveIt for a normal straight Cartesian trajectory while
#        keeping the flange orientation fixed.

#     2. Keep the COMPLETE MoveIt solution as the baseline:
#            q1_moveit(t) ... q7_moveit(t)

#        In particular, q7_moveit(t) may contain compensation required
#        by MoveIt to maintain the fixed flange orientation.

#     3. Add the desired process rotation ON TOP of MoveIt's joint7:

#            q7_final(t)
#                = q7_moveit(t) + q7_twist(t)

#     4. Execute the resulting trajectory as ONE 7-joint trajectory.

#     This avoids running two controllers simultaneously.
#     """

#     # ==========================================================
#     # Helpers
#     # ==========================================================

#     @staticmethod
#     def _duration_to_seconds(duration):
#         return (
#             float(duration.sec)
#             + float(duration.nanosec) * 1e-9
#         )

#     # ==========================================================
#     # Plan pure Cartesian translation
#     # ==========================================================

#     def _plan_linear_waypoint(
#         self,
#         target_pose,
#         max_step,
#         velocity,
#         acceleration,
#     ):
#         """
#         Generate the orientation-preserving Cartesian baseline.

#         This intentionally DOES NOT contain the desired twist.
#         """

#         request = GetCartesianPath.Request()

#         request.header.frame_id = self.base_frame
#         request.group_name = self.group_name
#         request.link_name = self.flange_link

#         request.waypoints = [target_pose]

#         request.max_step = max_step
#         request.jump_threshold = 0.0

#         request.avoid_collisions = True

#         request.max_velocity_scaling_factor = velocity
#         request.max_acceleration_scaling_factor = acceleration

#         self.get_logger().info(
#             "Computing fixed-orientation Cartesian baseline..."
#         )

#         future = self.cartesian_client.call_async(request)

#         rclpy.spin_until_future_complete(
#             self,
#             future,
#         )

#         response = future.result()

#         if response is None:
#             self.get_logger().error(
#                 "No response from MoveIt."
#             )
#             return None

#         self.get_logger().info(
#             f"Cartesian path fraction: "
#             f"{response.fraction * 100.0:.1f}%"
#         )

#         if response.fraction < 0.999:
#             self.get_logger().error(
#                 "MoveIt could not generate the complete "
#                 "Cartesian path."
#             )

#             self.get_logger().error(
#                 "Trajectory will NOT be executed."
#             )

#             return None

#         trajectory = response.solution

#         if not trajectory.joint_trajectory.points:
#             self.get_logger().error(
#                 "MoveIt returned an empty joint trajectory."
#             )
#             return None

#         return trajectory

#     # ==========================================================
#     # Add J7 process rotation
#     # ==========================================================

#     def _add_joint7_twist(
#         self,
#         robot_trajectory,
#         twist_deg,
#         joint7_name="joint7",
#     ):
#         """
#         Add the desired rotation to MoveIt's existing J7 solution.

#         IMPORTANT:

#             q7_final(t)
#                 = q7_moveit(t)
#                 + desired_twist(t)

#         MoveIt's J7 is therefore NOT discarded.

#         The twist progress is tied to trajectory progress from
#         start to finish.

#         Because this Cartesian path is a straight line generated
#         by GetCartesianPath, normalized path progress corresponds
#         to normalized translation progress for the requested move.
#         """

#         traj = robot_trajectory.joint_trajectory

#         if not traj.points:
#             self.get_logger().error(
#                 "Cannot modify an empty trajectory."
#             )
#             return False

#         joint_names = list(traj.joint_names)

#         if joint7_name not in joint_names:
#             self.get_logger().error(
#                 f"{joint7_name!r} not found."
#             )

#             self.get_logger().error(
#                 f"Trajectory joints: {joint_names}"
#             )

#             return False

#         j7_index = joint_names.index(joint7_name)

#         twist_rad = math.radians(twist_deg)

#         # ------------------------------------------------------
#         # Save the ORIGINAL MoveIt J7 baseline for diagnostics
#         # ------------------------------------------------------

#         q7_moveit_start = (
#             traj.points[0].positions[j7_index]
#         )

#         q7_moveit_end = (
#             traj.points[-1].positions[j7_index]
#         )

#         q7_moveit_delta = (
#             q7_moveit_end - q7_moveit_start
#         )

#         total_time = self._duration_to_seconds(
#             traj.points[-1].time_from_start
#         )

#         if total_time <= 0.0:
#             self.get_logger().error(
#                 "Trajectory duration is zero or invalid."
#             )
#             return False

#         self.get_logger().info(
#             "MoveIt J7 baseline:"
#         )

#         self.get_logger().info(
#             f"  start       = "
#             f"{math.degrees(q7_moveit_start):+.2f} deg"
#         )

#         self.get_logger().info(
#             f"  end         = "
#             f"{math.degrees(q7_moveit_end):+.2f} deg"
#         )

#         self.get_logger().info(
#             f"  compensation= "
#             f"{math.degrees(q7_moveit_delta):+.2f} deg"
#         )

#         self.get_logger().info(
#             "Desired additional J7 rotation:"
#         )

#         self.get_logger().info(
#             f"  process twist = {twist_deg:+.2f} deg"
#         )

#         self.get_logger().info(
#             "Expected final J7:"
#         )

#         self.get_logger().info(
#             f"  {math.degrees(q7_moveit_end):+.2f} "
#             f"+ {twist_deg:+.2f} "
#             f"= "
#             f"{math.degrees(q7_moveit_end + twist_rad):+.2f} deg"
#         )

#         # ------------------------------------------------------
#         # Add desired twist to each trajectory point
#         # ------------------------------------------------------
#         #
#         # We use normalized trajectory time here:
#         #
#         #       progress = t / T
#         #
#         # Because GetCartesianPath generates the straight-line
#         # Cartesian path and MoveIt time-parameterizes that path,
#         # this preserves the same start/end synchronization.
#         #
#         # q7_final =
#         #       q7_moveit
#         #       + twist_total * progress
#         #
#         # ------------------------------------------------------

#         for point in traj.points:

#             t = self._duration_to_seconds(
#                 point.time_from_start
#             )

#             progress = t / total_time

#             progress = max(
#                 0.0,
#                 min(1.0, progress),
#             )

#             # ----------------------------------------------
#             # Position
#             # ----------------------------------------------

#             positions = list(point.positions)

#             q7_moveit = positions[j7_index]

#             q7_twist = (
#                 twist_rad * progress
#             )

#             q7_final = (
#                 q7_moveit + q7_twist
#             )

#             positions[j7_index] = q7_final

#             point.positions = positions

#             # ----------------------------------------------
#             # Velocity
#             #
#             # q_twist = twist_rad * t/T
#             #
#             # therefore:
#             #
#             # dq_twist/dt = twist_rad/T
#             # ----------------------------------------------

#             # if len(point.velocities) == len(joint_names):

#             #     velocities = list(
#             #         point.velocities
#             #     )

#             #     moveit_velocity = (
#             #         velocities[j7_index]
#             #     )

#             #     twist_velocity = (
#             #         twist_rad / total_time
#             #     )

#             #     velocities[j7_index] = (
#             #         moveit_velocity
#             #         + twist_velocity
#             #     )

#             #     point.velocities = velocities

#             # ----------------------------------------------
#             # Acceleration
#             #
#             # Linear twist vs time has zero additional
#             # acceleration between trajectory points.
#             #
#             # Therefore MoveIt's original J7 acceleration
#             # remains untouched.
#             # ----------------------------------------------

#         return True

#     # ==========================================================
#     # Execute trajectory
#     # ==========================================================

#     def _execute_trajectory(
#         self,
#         trajectory,
#     ):

#         goal = ExecuteTrajectory.Goal()

#         goal.trajectory = trajectory

#         self.get_logger().info(
#             "Executing combined trajectory:"
#         )

#         self.get_logger().info(
#             "  J1-J6 = MoveIt Cartesian solution"
#         )

#         self.get_logger().info(
#             "  J7    = MoveIt baseline + process twist"
#         )

#         send_future = (
#             self.execute_client.send_goal_async(goal)
#         )

#         rclpy.spin_until_future_complete(
#             self,
#             send_future,
#         )

#         goal_handle = send_future.result()

#         if goal_handle is None:
#             self.get_logger().error(
#                 "Failed to communicate with "
#                 "trajectory execution server."
#             )
#             return False

#         if not goal_handle.accepted:
#             self.get_logger().error(
#                 "Trajectory execution rejected."
#             )
#             return False

#         result_future = (
#             goal_handle.get_result_async()
#         )

#         rclpy.spin_until_future_complete(
#             self,
#             result_future,
#         )

#         wrapped_result = result_future.result()

#         if wrapped_result is None:
#             self.get_logger().error(
#                 "No execution result received."
#             )
#             return False

#         result = wrapped_result.result

#         if result.error_code.val == 1:
#             self.get_logger().info(
#                 "Linear + J7 twist completed successfully."
#             )
#             return True

#         self.get_logger().error(
#             "Trajectory execution failed. "
#             f"MoveIt error code: "
#             f"{result.error_code.val}"
#         )

#         return False

#     # ==========================================================
#     # Main public function
#     # ==========================================================

#     def move_along_tool_z(
#         self,
#         distance,
#         twist_deg=None,
#         twist_per_meter=0.0,
#         max_step=0.005,
#         velocity=0.1,
#         acceleration=0.1,
#     ):
#         """
#         Translate along the current flange local +Z axis while
#         adding an explicit J7 rotation.

#         Example:

#             distance  = 0.05 m
#             twist_deg = 150 deg

#         produces approximately:

#             translation progress     added J7 twist

#                   0%                       0 deg
#                  20%                      30 deg
#                  40%                      60 deg
#                  60%                      90 deg
#                  80%                     120 deg
#                 100%                     150 deg

#         ON TOP OF MoveIt's own J7 compensation.
#         """

#         # ------------------------------------------------------
#         # 1. Current flange transform
#         # ------------------------------------------------------

#         tf = self._get_flange_transform()

#         translation = tf.transform.translation
#         rotation = tf.transform.rotation

#         start_x = translation.x
#         start_y = translation.y
#         start_z = translation.z

#         # ------------------------------------------------------
#         # 2. Tool +Z
#         #
#         # Reuse parent implementation from move_linear.py.
#         # ------------------------------------------------------

#         zx, zy, zz = (
#             self._tool_z_from_quaternion(rotation)
#         )

#         # ------------------------------------------------------
#         # 3. Translation target
#         # ------------------------------------------------------

#         target_x = (
#             start_x + distance * zx
#         )

#         target_y = (
#             start_y + distance * zy
#         )

#         target_z = (
#             start_z + distance * zz
#         )

#         # ------------------------------------------------------
#         # 4. Desired process rotation
#         # ------------------------------------------------------

#         if twist_deg is None:
#             twist_deg = (
#                 distance * twist_per_meter
#             )

#         self.get_logger().info(
#             "Requested synchronized movement:"
#         )

#         self.get_logger().info(
#             f"  translation = {distance:+.4f} m"
#         )

#         self.get_logger().info(
#             f"  added J7    = {twist_deg:+.2f} deg"
#         )

#         if abs(distance) > 1e-9:

#             ratio = twist_deg / distance

#             self.get_logger().info(
#                 f"  ratio       = "
#                 f"{ratio:+.2f} deg/m"
#             )

#             self.get_logger().info(
#                 f"              = "
#                 f"{ratio / 100.0:+.2f} deg/cm"
#             )

#         # ------------------------------------------------------
#         # 5. PURE Cartesian target
#         #
#         # CRITICAL:
#         #
#         # Orientation is copied exactly.
#         #
#         # MoveIt therefore calculates whatever J7 compensation
#         # it thinks is necessary to maintain orientation.
#         # ------------------------------------------------------

#         target_pose = Pose()

#         target_pose.position.x = target_x
#         target_pose.position.y = target_y
#         target_pose.position.z = target_z

#         target_pose.orientation.x = rotation.x
#         target_pose.orientation.y = rotation.y
#         target_pose.orientation.z = rotation.z
#         target_pose.orientation.w = rotation.w

#         # ------------------------------------------------------
#         # 6. Generate baseline
#         # ------------------------------------------------------


#         robot_trajectory = self._plan_linear_waypoint(
#             target_pose=target_pose,
#             max_step=max_step,
#             velocity=velocity,
#             acceleration=acceleration,
#         )

#         if robot_trajectory is None:
#             return False

#         # ------------------------------------------------------
#         # 7. Add J7 process rotation ON TOP of baseline
#         # ------------------------------------------------------

#         success = self._add_joint7_twist(
#             robot_trajectory=robot_trajectory,
#             twist_deg=twist_deg,
#         )

#         if not success:
#             return False

#         # ------------------------------------------------------
#         # 8. Execute ONE combined trajectory
#         # ------------------------------------------------------

#         return self._execute_trajectory(
#             robot_trajectory
#         )


# # ==============================================================
# # CLI
# # ==============================================================

# def main():

#     parser = argparse.ArgumentParser(
#         description=(
#             "Move xArm7 along the flange local Z axis "
#             "while adding an explicit synchronized J7 rotation."
#         )
#     )

#     parser.add_argument(
#         "distance",
#         type=float,
#         help=(
#             "Distance in meters. "
#             "Positive = tool +Z, negative = tool -Z."
#         ),
#     )

#     parser.add_argument(
#         "--twist",
#         type=float,
#         default=None,
#         help=(
#             "Additional J7 rotation in degrees. "
#             "Overrides --twist-per-meter."
#         ),
#     )

#     parser.add_argument(
#         "--twist-per-meter",
#         type=float,
#         default=0.0,
#         help=(
#             "Additional J7 rotation per meter. "
#             "For 150 deg per 5 cm: 3000."
#         ),
#     )

#     parser.add_argument(
#         "--step",
#         type=float,
#         default=0.005,
#         help=(
#             "Cartesian interpolation step in meters."
#         ),
#     )

#     parser.add_argument(
#         "--velocity",
#         type=float,
#         default=0.3,
#     )

#     parser.add_argument(
#         "--acceleration",
#         type=float,
#         default=0.2,
#     )

#     args = parser.parse_args()

#     rclpy.init()

#     twister = XArm7TwistMove()

#     success = False

#     try:

#         success = twister.move_along_tool_z(
#             distance=args.distance,
#             twist_deg=args.twist,
#             twist_per_meter=args.twist_per_meter,
#             max_step=args.step,
#             velocity=args.velocity,
#             acceleration=args.acceleration,
#         )

#     finally:

#         twister.destroy_node()
#         rclpy.shutdown()

#     raise SystemExit(
#         0 if success else 1
#     )


# if __name__ == "__main__":
#     main()
