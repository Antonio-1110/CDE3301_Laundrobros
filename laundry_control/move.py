#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import tf2_ros

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint
from moveit_msgs.srv import GetCartesianPath


class XArm7Controller(Node):
    """
    Unified xArm7 motion controller.

    Provides:
        1. Current joint-state access
        2. Joint-space MoveIt motion
        3. Relative J7-only rotation
        4. Straight Cartesian tool-Z motion
        5. Straight tool-Z motion + explicit J7 twist

    The same controller can be:
        - imported by scan_move.py / main.py
        - run directly from the terminal
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

    def __init__(
        self,
        group_name="xarm7",
        base_frame="link_base",
        flange_link="link7",
        joint_state_topic="/joint_states",
    ):
        super().__init__("xarm7_controller")

        self.group_name = group_name
        self.base_frame = base_frame
        self.flange_link = flange_link

        # =====================================================
        # Current joint states
        # =====================================================

        self._latest_joint_state = None

        self._joint_state_sub = self.create_subscription(
            JointState,
            joint_state_topic,
            self._joint_state_callback,
            10,
        )

        # =====================================================
        # TF
        # =====================================================

        self.tf_buffer = tf2_ros.Buffer()

        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )

        # =====================================================
        # MoveGroup action
        #
        # Used for collision-aware joint-space movements.
        # =====================================================

        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )

        # =====================================================
        # Cartesian path service
        # =====================================================

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            "/compute_cartesian_path",
        )

        # =====================================================
        # Trajectory execution
        # =====================================================

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            "/execute_trajectory",
        )

        # =====================================================
        # Wait for MoveIt
        # =====================================================

        self.get_logger().info(
            "Waiting for MoveIt interfaces..."
        )

        self.move_group_client.wait_for_server()

        while not self.cartesian_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                "Waiting for /compute_cartesian_path..."
            )

        self.execute_client.wait_for_server()

        self.get_logger().info(
            "XArm7Controller ready."
        )

    # =========================================================
    # JOINT STATE
    # =========================================================

    def _joint_state_callback(self, msg):
        self._latest_joint_state = msg

    def get_current_joints(self, timeout=2.0):
        """
        Return current joints in JOINT_NAMES order:

            [J1, J2, J3, J4, J5, J6, J7]

        Units: radians.
        """

        self._latest_joint_state = None

        start_time = self.get_clock().now()

        while rclpy.ok():

            rclpy.spin_once(
                self,
                timeout_sec=0.05,
            )

            if self._latest_joint_state is not None:
                break

            elapsed = (
                self.get_clock().now() - start_time
            ).nanoseconds * 1e-9

            if elapsed >= timeout:

                self.get_logger().error(
                    "Timed out waiting for joint states."
                )

                return None

        state = self._latest_joint_state

        state_map = dict(
            zip(
                state.name,
                state.position,
            )
        )

        missing = [
            name
            for name in self.JOINT_NAMES
            if name not in state_map
        ]

        if missing:

            self.get_logger().error(
                f"Missing joints in joint state: {missing}"
            )

            return None

        return [
            state_map[name]
            for name in self.JOINT_NAMES
        ]

    # =========================================================
    # TF / CARTESIAN HELPERS
    # =========================================================

    def get_flange_transform(self):
        """
        Return transform:

            base_frame -> flange_link
        """

        future = self.tf_buffer.wait_for_transform_async(
            self.base_frame,
            self.flange_link,
            rclpy.time.Time(),
        )

        rclpy.spin_until_future_complete(
            self,
            future,
        )

        return self.tf_buffer.lookup_transform(
            self.base_frame,
            self.flange_link,
            rclpy.time.Time(),
        )

    @staticmethod
    def _tool_z_from_quaternion(q):
        """
        Return flange local +Z expressed in the base frame.
        """

        x = q.x
        y = q.y
        z = q.z
        w = q.w

        zx = 2.0 * (x * z + w * y)
        zy = 2.0 * (y * z - w * x)
        zz = 1.0 - 2.0 * (x * x + y * y)

        length = math.sqrt(
            zx * zx
            + zy * zy
            + zz * zz
        )

        if length <= 1e-12:
            raise RuntimeError(
                "Invalid flange quaternion."
            )

        return (
            zx / length,
            zy / length,
            zz / length,
        )

    @staticmethod
    def _duration_to_seconds(duration):

        return (
            float(duration.sec)
            + float(duration.nanosec) * 1e-9
        )

    # =========================================================
    # GENERIC JOINT MOVEMENT
    # =========================================================

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
        Collision-aware joint-space movement.

        joint_angles:
            Seven absolute target angles in RADIANS.
        """

        if len(joint_angles) != 7:

            raise ValueError(
                "xArm7 requires exactly 7 joint angles."
            )

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

        goal = MoveGroup.Goal()

        goal.request.group_name = self.group_name

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

        goal.planning_options.plan_only = False
        goal.planning_options.replan = replan

        goal.planning_options.replan_attempts = (
            replan_attempts
        )

        self.get_logger().info(
            "Joint-space target:"
        )

        for name, angle in zip(
            self.JOINT_NAMES,
            joint_angles,
        ):

            self.get_logger().info(
                f"  {name}: "
                f"{math.degrees(angle):+.2f} deg"
            )

        send_future = (
            self.move_group_client.send_goal_async(
                goal
            )
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
                "MoveIt rejected joint-space goal."
            )

            return False

        result_future = (
            goal_handle.get_result_async()
        )

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

        error_code = (
            wrapped_result.result.error_code.val
        )

        if error_code == 1:

            self.get_logger().info(
                "Joint movement completed successfully."
            )

            return True

        self.get_logger().error(
            f"Joint movement failed. "
            f"MoveIt error code: {error_code}"
        )

        return False

    # =========================================================
    # RELATIVE SINGLE-JOINT MOVEMENT
    # =========================================================

    def _rotate_joint_relative(
        self,
        joint_index,
        delta_deg,
        velocity=0.3,
        acceleration=0.3,
    ):
        """
        Rotate ONLY the joint at joint_index relative to its
        current position.

        All other joints are read from the current robot state
        and used unchanged as the MoveIt target.
        """

        current = self.get_current_joints()

        if current is None:
            return False

        target = list(current)

        joint_name = self.JOINT_NAMES[joint_index]

        old_angle = target[joint_index]

        target[joint_index] += math.radians(
            delta_deg
        )

        self.get_logger().info(
            f"{joint_name}-only relative movement:"
        )

        self.get_logger().info(
            f"  current: "
            f"{math.degrees(old_angle):+.2f} deg"
        )

        self.get_logger().info(
            f"  delta:   "
            f"{delta_deg:+.2f} deg"
        )

        self.get_logger().info(
            f"  target:  "
            f"{math.degrees(target[joint_index]):+.2f} deg"
        )

        return self.move_joints(
            target,
            velocity=velocity,
            acceleration=acceleration,
        )

    def rotate_joint6(
        self,
        delta_deg,
        velocity=0.3,
        acceleration=0.3,
    ):
        """
        Rotate ONLY J6 relative to its current position.

        Example:

            rotate_joint6(-75)

        means:

            J1 -> current J1
            ...
            J5 -> current J5
            J6 -> current J6 - 75 deg
            J7 -> current J7
        """

        return self._rotate_joint_relative(
            joint_index=5,
            delta_deg=delta_deg,
            velocity=velocity,
            acceleration=acceleration,
        )

    def rotate_joint7(
        self,
        delta_deg,
        velocity=0.3,
        acceleration=0.3,
    ):
        """
        Rotate ONLY J7 relative to its current position.

        Example:

            rotate_joint7(-75)

        means:

            J1 -> current J1
            ...
            J6 -> current J6
            J7 -> current J7 - 75 deg
        """

        return self._rotate_joint_relative(
            joint_index=6,
            delta_deg=delta_deg,
            velocity=velocity,
            acceleration=acceleration,
        )

    # =========================================================
    # CARTESIAN PLANNING
    # =========================================================

    def _plan_to_pose(
        self,
        target_pose,
        max_step=0.005,
        velocity=0.1,
        acceleration=0.1,
    ):
        """
        Generate but DO NOT execute a straight Cartesian
        trajectory from the flange's CURRENT actual position to
        target_pose.

        Returns:
            (trajectory, fraction)

            trajectory is a RobotTrajectory, or None if planning
            failed, the path was incomplete (fraction < 0.999), or
            MoveIt returned an empty trajectory.

            fraction is ALWAYS returned (0.0 if there was no
            response at all), even when trajectory is None, so
            callers doing reachability probing can inspect partial
            feasibility instead of only getting a binary None.
        """

        request = GetCartesianPath.Request()

        request.header.frame_id = self.base_frame
        request.group_name = self.group_name
        request.link_name = self.flange_link

        request.waypoints = [
            target_pose
        ]

        request.max_step = max_step
        request.jump_threshold = 0.0

        request.avoid_collisions = True

        request.max_velocity_scaling_factor = (
            velocity
        )

        request.max_acceleration_scaling_factor = (
            acceleration
        )

        self.get_logger().info(
            "Computing Cartesian path..."
        )

        self.get_logger().info(
            f"  target: "
            f"({target_pose.position.x:.4f}, "
            f"{target_pose.position.y:.4f}, "
            f"{target_pose.position.z:.4f})"
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
                "No Cartesian-path response from MoveIt."
            )

            return None, 0.0

        self.get_logger().info(
            f"Cartesian path fraction: "
            f"{response.fraction * 100.0:.1f}%"
        )

        if response.fraction < 0.999:

            self.get_logger().error(
                "Complete Cartesian path could not "
                "be generated."
            )

            return None, response.fraction

        if not response.solution.joint_trajectory.points:

            self.get_logger().error(
                "MoveIt returned an empty trajectory."
            )

            return None, response.fraction

        return response.solution, response.fraction

    def _build_pose(
        self,
        x,
        y,
        z,
        orientation=None,
    ):
        """
        Build a Pose at (x, y, z) in base_frame.

        orientation:
            An object with .x/.y/.z/.w (e.g. a
            geometry_msgs/Quaternion), or None to reuse the
            flange's CURRENT orientation (read live via TF).

            None is what callers computing a grasp/approach target
            should pass, since gripper.py's mounting offsets are
            only valid while the reference orientation (INTER) is
            held fixed.
        """

        if orientation is None:
            orientation = self.get_flange_transform().transform.rotation

        pose = Pose()

        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        pose.orientation.x = orientation.x
        pose.orientation.y = orientation.y
        pose.orientation.z = orientation.z
        pose.orientation.w = orientation.w

        return pose

    def _plan_tool_z(
        self,
        distance,
        max_step=0.005,
        velocity=0.1,
        acceleration=0.1,
    ):
        """
        Generate but DO NOT execute a straight Cartesian
        trajectory along the flange's current local Z axis.

        Orientation is kept fixed.

        Returns:
            RobotTrajectory or None
        """

        if abs(distance) <= 1e-12:

            self.get_logger().error(
                "_plan_tool_z() requires non-zero distance."
            )

            return None

        tf = self.get_flange_transform()

        translation = tf.transform.translation
        rotation = tf.transform.rotation

        zx, zy, zz = (
            self._tool_z_from_quaternion(
                rotation
            )
        )

        target_pose = self._build_pose(
            x=translation.x + distance * zx,
            y=translation.y + distance * zy,
            z=translation.z + distance * zz,
            orientation=rotation,
        )

        trajectory, _fraction = self._plan_to_pose(
            target_pose,
            max_step=max_step,
            velocity=velocity,
            acceleration=acceleration,
        )

        return trajectory

    # =========================================================
    # TRAJECTORY EXECUTION
    # =========================================================

    def _execute_trajectory(
        self,
        trajectory,
    ):

        goal = ExecuteTrajectory.Goal()

        goal.trajectory = trajectory

        send_future = (
            self.execute_client.send_goal_async(
                goal
            )
        )

        rclpy.spin_until_future_complete(
            self,
            send_future,
        )

        goal_handle = send_future.result()

        if goal_handle is None:

            self.get_logger().error(
                "Failed to communicate with "
                "/execute_trajectory."
            )

            return False

        if not goal_handle.accepted:

            self.get_logger().error(
                "Trajectory execution rejected."
            )

            return False

        result_future = (
            goal_handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future,
        )

        wrapped_result = result_future.result()

        if wrapped_result is None:

            self.get_logger().error(
                "No trajectory execution result."
            )

            return False

        error_code = (
            wrapped_result.result.error_code.val
        )

        if error_code == 1:

            self.get_logger().info(
                "Trajectory completed successfully."
            )

            return True

        self.get_logger().error(
            f"Trajectory execution failed. "
            f"MoveIt error code: {error_code}"
        )

        return False

    # =========================================================
    # PURE LINEAR TOOL-Z MOVEMENT
    # =========================================================

    def move_tool_z(
        self,
        distance,
        max_step=0.005,
        velocity=0.1,
        acceleration=0.1,
    ):
        """
        Straight Cartesian movement along current tool Z,
        maintaining flange orientation.
        """

        trajectory = self._plan_tool_z(
            distance=distance,
            max_step=max_step,
            velocity=velocity,
            acceleration=acceleration,
        )

        if trajectory is None:
            return False

        return self._execute_trajectory(
            trajectory
        )

    # =========================================================
    # ARBITRARY-POSE CARTESIAN MOVEMENT
    #
    # Unlike move_tool_z() (offset along whatever the CURRENT
    # local Z happens to be), these accept an absolute (x, y, z)
    # target in base_frame. Needed for grasp targets whose mount
    # offset (see gripper.py) has no X/Y component: since that
    # offset sits on link7's own rotation axis, reaching it
    # requires genuine XY translation, not just a Z offset plus
    # J7 rotation the way scanning works.
    # =========================================================

    def move_to_pose(
        self,
        x,
        y,
        z,
        orientation=None,
        max_step=0.005,
        velocity=0.1,
        acceleration=0.1,
    ):
        """
        Straight Cartesian movement of the flange to an absolute
        (x, y, z) target in base_frame.

        orientation:
            See _build_pose() - None reuses the flange's CURRENT
            orientation.
        """

        target_pose = self._build_pose(
            x, y, z, orientation
        )

        trajectory, fraction = self._plan_to_pose(
            target_pose,
            max_step=max_step,
            velocity=velocity,
            acceleration=acceleration,
        )

        if trajectory is None:

            self.get_logger().error(
                f"move_to_pose: infeasible "
                f"(fraction={fraction:.3f})."
            )

            return False

        return self._execute_trajectory(
            trajectory
        )

    def check_pose_reachable(
        self,
        x,
        y,
        z,
        orientation=None,
        max_step=0.005,
        velocity=0.1,
        acceleration=0.1,
    ):
        """
        PLAN-ONLY reachability probe: identical planning to
        move_to_pose(), but NEVER executes - safe to call
        repeatedly on candidate targets that may turn out
        infeasible.

        Returns MoveIt's raw Cartesian fraction in [0, 1].
        Callers should treat >= 0.999 as "fully reachable" (the
        same threshold _plan_to_pose applies internally).

        IMPORTANT: GetCartesianPath plans from the arm's actual
        current live state, not a hypothetical one - callers must
        have already moved the arm to whatever pose they intend as
        the real starting point (e.g. INTER) before probing, or
        the probed fraction will describe the wrong path.
        """

        target_pose = self._build_pose(
            x, y, z, orientation
        )

        _trajectory, fraction = self._plan_to_pose(
            target_pose,
            max_step=max_step,
            velocity=velocity,
            acceleration=acceleration,
        )

        return fraction

    # =========================================================
    # J7 TWIST MODIFICATION
    # =========================================================

    def _add_joint7_twist(
        self,
        trajectory,
        twist_deg,
    ):
        """
        Add deliberate J7 rotation ON TOP OF MoveIt's
        orientation-preserving J7 trajectory.

        IMPORTANT:

            q7_final(t)
                =
            q7_moveit(t)
                +
            desired_twist(t)

        We intentionally modify POSITIONS ONLY.

        During testing with the fake xArm controller, manually
        modifying J7 velocities caused CONTROL_FAILED (-4).

        Therefore MoveIt's velocity/acceleration fields are
        left untouched.
        """

        traj = trajectory.joint_trajectory

        if not traj.points:

            self.get_logger().error(
                "Cannot twist an empty trajectory."
            )

            return False

        joint_names = list(
            traj.joint_names
        )

        if "joint7" not in joint_names:

            self.get_logger().error(
                "joint7 not present in trajectory."
            )

            return False

        j7_index = joint_names.index(
            "joint7"
        )

        total_time = (
            self._duration_to_seconds(
                traj.points[-1].time_from_start
            )
        )

        if total_time <= 0.0:

            self.get_logger().error(
                "Trajectory duration is zero or invalid."
            )

            return False

        twist_rad = math.radians(
            twist_deg
        )

        # Diagnostic baseline
        q7_start = (
            traj.points[0].positions[j7_index]
        )

        q7_end = (
            traj.points[-1].positions[j7_index]
        )

        self.get_logger().info(
            "J7 synchronized twist:"
        )

        self.get_logger().info(
            f"  MoveIt baseline: "
            f"{math.degrees(q7_start):+.2f} -> "
            f"{math.degrees(q7_end):+.2f} deg"
        )

        self.get_logger().info(
            f"  Added twist: "
            f"{twist_deg:+.2f} deg"
        )

        self.get_logger().info(
            f"  Final target: "
            f"{math.degrees(q7_end + twist_rad):+.2f} deg"
        )

        # -----------------------------------------------------
        # Add twist according to normalized trajectory time.
        #
        # This guarantees:
        #
        #   translation starts when twist starts
        #   translation ends when twist ends
        #
        # It does NOT yet guarantee exact angular displacement
        # per physical centimetre throughout the stroke.
        # -----------------------------------------------------

        for point in traj.points:

            t = self._duration_to_seconds(
                point.time_from_start
            )

            progress = t / total_time

            progress = max(
                0.0,
                min(1.0, progress),
            )

            positions = list(
                point.positions
            )

            q7_moveit = (
                positions[j7_index]
            )

            q7_added = (
                twist_rad * progress
            )

            positions[j7_index] = (
                q7_moveit + q7_added
            )

            point.positions = positions

        return True

    # =========================================================
    # LINEAR + J7 TWIST
    # =========================================================

    def move_tool_z_with_twist(
        self,
        distance,
        twist_deg,
        max_step=0.005,
        velocity=0.1,
        acceleration=0.1,
    ):
        """
        Straight tool-Z translation + explicit J7 rotation.

        Example:

            move_tool_z_with_twist(
                0.05,
                150,
            )

        means:

            translate +5 cm

        while:

            adding +150 deg to J7

        MoveIt's original J7 trajectory is preserved as a
        baseline and the requested rotation is added to it.
        """

        if abs(distance) <= 1e-12:

            self.get_logger().error(
                "For J7-only movement use rotate_joint7()."
            )

            return False

        trajectory = self._plan_tool_z(
            distance=distance,
            max_step=max_step,
            velocity=velocity,
            acceleration=acceleration,
        )

        if trajectory is None:
            return False

        success = self._add_joint7_twist(
            trajectory=trajectory,
            twist_deg=twist_deg,
        )

        if not success:
            return False

        return self._execute_trajectory(
            trajectory
        )
