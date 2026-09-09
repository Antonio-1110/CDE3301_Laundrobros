# #!/usr/bin/env python3

# import math
# import argparse

# import rclpy
# from rclpy.node import Node
# from rclpy.action import ActionClient

# from geometry_msgs.msg import Pose
# from moveit_msgs.srv import GetCartesianPath
# from moveit_msgs.action import ExecuteTrajectory

# import tf2_ros


# class XArm7LinearMove(Node):

#     def __init__(
#         self,
#         group_name="xarm7",
#         base_frame="link_base",
#         flange_link="link7",
#     ):
#         super().__init__("xarm7_linear_move")

#         self.group_name = group_name
#         self.base_frame = base_frame
#         self.flange_link = flange_link

#         # --------------------------------------------------
#         # TF
#         # --------------------------------------------------

#         self.tf_buffer = tf2_ros.Buffer()

#         self.tf_listener = tf2_ros.TransformListener(
#             self.tf_buffer,
#             self,
#         )

#         # --------------------------------------------------
#         # Cartesian path service
#         # --------------------------------------------------

#         self.cartesian_client = self.create_client(
#             GetCartesianPath,
#             "/compute_cartesian_path",
#         )

#         self.get_logger().info(
#             "Waiting for /compute_cartesian_path..."
#         )

#         while not self.cartesian_client.wait_for_service(
#             timeout_sec=1.0
#         ):
#             self.get_logger().info(
#                 "Still waiting for MoveIt..."
#             )

#         # --------------------------------------------------
#         # Execute trajectory action
#         # --------------------------------------------------

#         self.execute_client = ActionClient(
#             self,
#             ExecuteTrajectory,
#             "/execute_trajectory",
#         )

#         self.get_logger().info(
#             "Waiting for /execute_trajectory..."
#         )

#         self.execute_client.wait_for_server()

#         self.get_logger().info(
#             "Linear motion interface ready."
#         )

#     # ======================================================
#     # Current flange pose
#     # ======================================================

#     def _get_flange_transform(self):

#         future = self.tf_buffer.wait_for_transform_async(
#             self.base_frame,
#             self.flange_link,
#             rclpy.time.Time(),
#         )

#         rclpy.spin_until_future_complete(
#             self,
#             future,
#         )

#         transform = self.tf_buffer.lookup_transform(
#             self.base_frame,
#             self.flange_link,
#             rclpy.time.Time(),
#         )

#         return transform

#     # ======================================================
#     # Quaternion -> local Z axis
#     # ======================================================

#     @staticmethod
#     def _tool_z_from_quaternion(q):
#         """
#         Return the tool's local +Z unit vector expressed
#         in the base coordinate frame.

#         q is geometry_msgs/Quaternion.
#         """

#         x = q.x
#         y = q.y
#         z = q.z
#         w = q.w

#         # Third column of quaternion rotation matrix.
#         #
#         # This is the tool-frame +Z axis represented
#         # in the base frame.

#         zx = 2.0 * (x * z + w * y)
#         zy = 2.0 * (y * z - w * x)
#         zz = 1.0 - 2.0 * (x * x + y * y)

#         # Normalize just to protect against numerical error.

#         length = math.sqrt(
#             zx * zx +
#             zy * zy +
#             zz * zz
#         )

#         return (
#             zx / length,
#             zy / length,
#             zz / length,
#         )

#     # ======================================================
#     # Main function
#     # ======================================================

#     def move_along_tool_z(
#         self,
#         distance,
#         max_step=0.005,
#         velocity=0.1,
#         acceleration=0.1
#     ):
#         """
#         Move the flange in a straight Cartesian line along
#         its current local Z axis.

#         distance:
#             meters

#             +0.25 -> 25 cm along tool +Z
#             -0.25 -> 25 cm along tool -Z

#         max_step:
#             Cartesian interpolation resolution.
#             Default = 5 mm.
#         """

#         # --------------------------------------------------
#         # 1. Get current flange transform
#         # --------------------------------------------------

#         tf = self._get_flange_transform()

#         translation = tf.transform.translation
#         rotation = tf.transform.rotation

#         start_x = translation.x
#         start_y = translation.y
#         start_z = translation.z

#         # --------------------------------------------------
#         # 2. Determine tool +Z direction in base frame
#         # --------------------------------------------------

#         zx, zy, zz = self._tool_z_from_quaternion(
#             rotation
#         )

#         self.get_logger().info(
#             "Tool +Z direction in base frame:"
#         )

#         self.get_logger().info(
#             f"  X: {zx:+.4f}"
#         )

#         self.get_logger().info(
#             f"  Y: {zy:+.4f}"
#         )

#         self.get_logger().info(
#             f"  Z: {zz:+.4f}"
#         )

#         # --------------------------------------------------
#         # 3. Calculate target position
#         # --------------------------------------------------

#         target_x = start_x + distance * zx
#         target_y = start_y + distance * zy
#         target_z = start_z + distance * zz

#         self.get_logger().info(
#             "Linear Cartesian movement:"
#         )

#         self.get_logger().info(
#             f"  distance: {distance:+.3f} m"
#         )

#         self.get_logger().info(
#             "Start:"
#         )

#         self.get_logger().info(
#             f"  ({start_x:.4f}, "
#             f"{start_y:.4f}, "
#             f"{start_z:.4f})"
#         )

#         self.get_logger().info(
#             "Target:"
#         )

#         self.get_logger().info(
#             f"  ({target_x:.4f}, "
#             f"{target_y:.4f}, "
#             f"{target_z:.4f})"
#         )

#         # --------------------------------------------------
#         # 4. Construct target pose
#         # --------------------------------------------------

#         target_pose = Pose()

#         target_pose.position.x = target_x
#         target_pose.position.y = target_y
#         target_pose.position.z = target_z

#         # CRITICAL:
#         #
#         # Copy current orientation exactly.
#         #
#         # This keeps the flange orientation fixed while
#         # translating.

#         target_pose.orientation.x = rotation.x
#         target_pose.orientation.y = rotation.y
#         target_pose.orientation.z = rotation.z
#         target_pose.orientation.w = rotation.w

#         # --------------------------------------------------
#         # 5. Ask MoveIt for Cartesian path
#         # --------------------------------------------------

#         request = GetCartesianPath.Request()

#         request.header.frame_id = self.base_frame

#         request.group_name = self.group_name

#         request.link_name = self.flange_link

#         request.waypoints = [
#             target_pose
#         ]

#         request.max_step = max_step

#         # Disable jump detection.
#         #
#         # Can be made stricter later if required.

#         request.jump_threshold = 0.0

#         # IMPORTANT:
#         # False means collision checking IS enabled.

#         request.avoid_collisions = True
#         request.max_velocity_scaling_factor = velocity
#         request.max_acceleration_scaling_factor = acceleration

#         self.get_logger().info(
#             "Computing collision-aware Cartesian path..."
#         )

#         future = self.cartesian_client.call_async(
#             request
#         )

#         rclpy.spin_until_future_complete(
#             self,
#             future,
#         )

#         response = future.result()

#         if response is None:
#             self.get_logger().error(
#                 "No response from MoveIt."
#             )
#             return False

#         # --------------------------------------------------
#         # 6. Check how much of the path MoveIt could make
#         # --------------------------------------------------

#         fraction = response.fraction

#         self.get_logger().info(
#             f"Cartesian path fraction: "
#             f"{fraction * 100:.1f}%"
#         )

#         # DO NOT execute partial insertion paths.

#         if fraction < 0.999:
#             self.get_logger().error(
#                 "MoveIt could not generate the complete "
#                 "collision-free Cartesian path."
#             )

#             self.get_logger().error(
#                 "Trajectory will NOT be executed."
#             )

#             return False

#         # --------------------------------------------------
#         # 7. Execute trajectory
#         # --------------------------------------------------

#         goal = ExecuteTrajectory.Goal()

#         goal.trajectory = response.solution

#         self.get_logger().info(
#             "Full Cartesian path found."
#         )

#         self.get_logger().info(
#             "Executing trajectory..."
#         )

#         send_future = self.execute_client.send_goal_async(
#             goal
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
#                 "Trajectory execution was rejected."
#             )
#             return False

#         # --------------------------------------------------
#         # 8. Wait for completion
#         # --------------------------------------------------

#         result_future = goal_handle.get_result_async()

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
#                 "Linear movement completed successfully."
#             )
#             return True

#         self.get_logger().error(
#             "Trajectory execution failed. "
#             f"MoveIt error code: "
#             f"{result.error_code.val}"
#         )

#         return False


# # ==========================================================
# # Terminal interface
# # ==========================================================

# def main():

#     parser = argparse.ArgumentParser(
#         description=(
#             "Move xArm7 linearly along the flange's "
#             "local Z axis."
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
#         "--step",
#         type=float,
#         default=0.005,
#         help=(
#             "Cartesian interpolation step in meters "
#             "(default: 0.005)"
#         ),
#     )

#     parser.add_argument(
#         "--velocity",
#         type=float,
#         default=0.3,
#     )

#     parser.add_argument(
#             "--acceleration",
#             type=float,
#             default=0.2,
#     )

#     args = parser.parse_args()

#     rclpy.init()

#     linear = XArm7LinearMove()

#     try:
#         success = linear.move_along_tool_z(
#             args.distance,
#             max_step=args.step,
#             velocity=args.velocity,
#             acceleration=args.acceleration,
#         )

#     finally:
#         linear.destroy_node()
#         rclpy.shutdown()

#     raise SystemExit(0 if success else 1)


# if __name__ == "__main__":
#     main()