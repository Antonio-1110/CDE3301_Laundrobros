#!/usr/bin/env python3

"""
XArm7Controller: every arm motion the package makes goes through here.

Joint-space moves go through MoveGroup (Pilz PTP first, OMPL
fallback - see config.PILZ_PIPELINE_ID for why), straight tool-Z
strokes and absolute-pose reaches through GetCartesianPath +
ExecuteTrajectory, and the scan's helical strokes add a J7 twist on
top of a planned tool-Z stroke.
"""

import math
import time

from action_msgs.srv import CancelGoal
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath, GetPositionIK, GetStateValidity
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .geometry import tool_z_from_quaternion
from .joint_path import densify, time_path
from .. import config
from ..config import (
    OMPL_PIPELINE_ID,
    OMPL_PLANNER_ID,
    PILZ_PIPELINE_ID,
    PILZ_PTP_PLANNER_ID,
)

# moveit_msgs/MoveItErrorCodes, for logging a failure as something
# readable instead of a bare integer. "why did that move fail" is
# otherwise a trip to the message definition every time.
MOVEIT_ERROR_CODES = {
    1: 'SUCCESS',
    -1: 'FAILURE',
    -2: 'PLANNING_FAILED',
    -3: 'INVALID_MOTION_PLAN',
    -4: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
    -5: 'CONTROL_FAILED',
    -6: 'UNABLE_TO_AQUIRE_SENSOR_DATA',
    -7: 'TIMED_OUT',
    -8: 'PREEMPTED',
    -10: 'START_STATE_IN_COLLISION',
    -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
    -12: 'GOAL_IN_COLLISION',
    -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
    -14: 'GOAL_CONSTRAINTS_VIOLATED',
    -15: 'INVALID_GROUP_NAME',
    -16: 'INVALID_GOAL_CONSTRAINTS',
    -17: 'INVALID_ROBOT_STATE',
    -18: 'INVALID_LINK_NAME',
    -19: 'INVALID_OBJECT_NAME',
    -21: 'FRAME_TRANSFORM_FAILURE',
    -22: 'COLLISION_CHECKING_UNAVAILABLE',
    -23: 'ROBOT_STATE_STALE',
    -24: 'SENSOR_INFO_STALE',
    -25: 'COMMUNICATION_FAILURE',
    -31: 'NO_IK_SOLUTION',
}


def describe_moveit_error(code):

    return f"{code} ({MOVEIT_ERROR_CODES.get(code, 'UNKNOWN')})"


# Error codes MoveIt returns before anything has moved: the request
# could not be planned. Only these justify retrying on the fallback
# pipeline. Anything else (CONTROL_FAILED, PREEMPTED, TIMED_OUT, a
# plan invalidated mid-motion, ...) can mean the arm was stopped
# part-way - by its own collision detection, the e-stop or a cancel -
# and must never be answered by commanding the motion again.
PLANNING_STAGE_ERROR_CODES = frozenset({
    -2,   # PLANNING_FAILED
    -3,   # INVALID_MOTION_PLAN
    -11,  # START_STATE_VIOLATES_PATH_CONSTRAINTS
    -12,  # GOAL_IN_COLLISION
    -13,  # GOAL_VIOLATES_PATH_CONSTRAINTS
    -14,  # GOAL_CONSTRAINTS_VIOLATED
    -15,  # INVALID_GROUP_NAME
    -16,  # INVALID_GOAL_CONSTRAINTS
    -17,  # INVALID_ROBOT_STATE (Pilz: non-zero start velocity)
    -31,  # NO_IK_SOLUTION
})


def failure_is_retryable(result):
    """
    Return True if a failed MoveGroup result cannot have moved the arm.

    That is a planning-stage error code, or no planned trajectory at
    all: nothing was planned, so nothing was executed. The second case
    covers codes MoveIt uses outside the planning-stage list - an
    unloaded pipeline (Pilz on the stock xArm launches) comes back as
    0, before anything moves, and must still fall back to OMPL.
    """
    return (
        result.error_code.val in PLANNING_STAGE_ERROR_CODES
        or not result.planned_trajectory.joint_trajectory.points
    )


# How long spin waits run before handing control back to Python, so
# a Ctrl+C (KeyboardInterrupt) is noticed promptly. See
# XArm7Controller._spin_until_done.
SPIN_SLICE_SEC = 0.1

# State-validity requests are sent ONE AT A TIME. With several in
# flight at once, move_group (rmw_fastrtps, Jazzy) intermittently never
# answers some of them - measured: most multi-request checks lost at
# least one after the first. Sequential checks cost ~5 ms per state.
# A request unanswered after VALIDITY_TIMEOUT_SEC is re-sent, up to
# VALIDITY_ATTEMPTS times, before the state counts as invalid.
VALIDITY_TIMEOUT_SEC = 1.0
VALIDITY_ATTEMPTS = 3


class XArm7Controller(Node):
    """
    Unified xArm7 motion controller.

    Provides:
        1. Current joint-state access
        2. Joint-space MoveIt motion
        3. Relative J7-only rotation
        4. Straight Cartesian tool-Z motion
        5. Straight tool-Z motion + explicit J7 twist

    Driven from the terminal by `laundry move ...` (cli.py), and
    shared by the scan, grasp and pipeline stages.
    """

    JOINT_NAMES = list(config.JOINT_NAMES)

    def __init__(
        self,
        group_name=config.GROUP_NAME,
        base_frame=config.BASE_FRAME,
        flange_link=config.FLANGE_LINK,
        joint_state_topic=config.JOINT_STATE_TOPIC,
        plan_from_observed_state=False,
        pad_obstacles=True,
    ):
        """
        Connect to MoveIt (blocks until its interfaces are up).

        pad_obstacles:
            Make sure move_group has the padded bucket/table (see
            arm/scene.py) before anything moves. On by default: without
            it MoveIt has no bucket or table at all.

        plan_from_observed_state:
            Give every Cartesian plan an explicit start state - the
            latest /joint_states this node has seen - instead of
            leaving it empty, which makes MoveIt plan from its own
            planning-scene copy of the robot state.

            That copy can lag behind the stroke that just finished.
            On the MoveIt fake controller this reliably breaks the
            scan: back-to-back twisted strokes are rejected with
            "start point deviates from current robot state more than
            0.01 at joint 'joint7'" (measured: 3/3 full scans failed
            with it off, 2/2 completed with it on). Off by default
            because the real rig has not been tested with it yet -
            see HARDWARE_TESTS.md.
        """
        super().__init__('xarm7_controller')

        self.plan_from_observed_state = plan_from_observed_state

        # Created on first use: only the planner-free joint paths
        # (move_joints_linear, execute_joint_path, and baking the end
        # scan) need them.
        self._ik_client = None
        self._validity_client = None

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
            '/move_action',
        )

        # =====================================================
        # Cartesian path service
        # =====================================================

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            '/compute_cartesian_path',
        )

        # =====================================================
        # Trajectory execution
        # =====================================================

        self.execute_client = ActionClient(
            self,
            ExecuteTrajectory,
            '/execute_trajectory',
        )

        # =====================================================
        # Emergency cancel of the trajectory controller's goals,
        # for Ctrl+C (see _stop_after_interrupt). Created now so
        # it is already connected when it is needed.
        # =====================================================

        self._controller_cancel_client = self.create_client(
            CancelGoal,
            config.TRAJECTORY_CONTROLLER_ACTION + '/_action/cancel_goal',
        )

        # =====================================================
        # Wait for MoveIt
        # =====================================================

        self.get_logger().info(
            'Waiting for MoveIt interfaces...'
        )

        self.move_group_client.wait_for_server()

        while not self.cartesian_client.wait_for_service(
            timeout_sec=1.0
        ):
            self.get_logger().info(
                'Waiting for /compute_cartesian_path...'
            )

        self.execute_client.wait_for_server()

        if pad_obstacles:
            from . import scene

            scene.apply(
                self,
                config.OBSTACLE_PADDING_M,
                config.GRIPPER_PADDING_M,
                log=self.get_logger().info,
            )

        self.get_logger().info(
            'XArm7Controller ready.'
        )

    # =========================================================
    # JOINT STATE
    # =========================================================

    def _joint_state_callback(self, msg):
        self._latest_joint_state = msg

    def get_current_joints(self, timeout=2.0):
        """
        Return current joints in JOINT_NAMES order.

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
                    'Timed out waiting for joint states.'
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
                f'Missing joints in joint state: {missing}'
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
        Return transform.

            base_frame -> flange_link
        """
        future = self.tf_buffer.wait_for_transform_async(
            self.base_frame,
            self.flange_link,
            rclpy.time.Time(),
        )

        self._spin_until_done(future)

        return self.tf_buffer.lookup_transform(
            self.base_frame,
            self.flange_link,
            rclpy.time.Time(),
        )

    @staticmethod
    def _duration_to_seconds(duration):

        return (
            float(duration.sec)
            + float(duration.nanosec) * 1e-9
        )

    # =========================================================
    # WAITING, AND STOPPING ON CTRL+C
    #
    # Every wait spins in short slices so Python regains control
    # regularly and a Ctrl+C arrives as KeyboardInterrupt at once.
    # (The `laundry` CLI starts rclpy WITHOUT its own SIGINT handler,
    # which would shut the ROS context down before we could cancel
    # anything - see cli._RosSession.)
    #
    # A MoveIt goal outlives the process that sent it, so exiting on
    # Ctrl+C is not enough: _run_goal stops the arm first (see
    # _stop_after_interrupt).
    # =========================================================

    def _spin_until_done(self, future, timeout_sec=None):
        """Spin until `future` is done (or timeout); True if it is done."""
        deadline = (
            None if timeout_sec is None else time.monotonic() + timeout_sec
        )

        while not future.done() and rclpy.ok():
            if deadline is not None and time.monotonic() >= deadline:
                break

            rclpy.spin_until_future_complete(
                self, future, timeout_sec=SPIN_SLICE_SEC
            )

        return future.done()

    def _run_goal(self, client, goal, what):
        """
        Send an action goal and wait for its result; stop the arm on Ctrl+C.

        Returns (goal_handle, wrapped_result): goal_handle is None if
        MoveIt never answered, wrapped_result None if the goal was
        rejected or returned nothing. KeyboardInterrupt is re-raised
        once the arm has been stopped (see _stop_after_interrupt).
        """
        send_future = client.send_goal_async(goal)
        result_future = None

        try:
            self._spin_until_done(send_future)

            goal_handle = send_future.result()

            if goal_handle is None or not goal_handle.accepted:
                return goal_handle, None

            result_future = goal_handle.get_result_async()

            self._spin_until_done(result_future)

            return goal_handle, result_future.result()

        except KeyboardInterrupt:
            self._stop_after_interrupt(send_future, result_future, what)
            raise

    def _cancel_controller_goals(self):
        """Cancel every goal on the trajectory controller; True if any was."""
        client = self._controller_cancel_client

        if not client.service_is_ready():
            return False

        # An all-zero goal id and stamp means "cancel every goal".
        future = client.call_async(CancelGoal.Request())

        if not self._spin_until_done(future, timeout_sec=1.0):
            return False

        response = future.result()

        return response is not None and len(response.goals_canceling) > 0

    def _stop_after_interrupt(
        self, send_future, result_future, what, timeout_sec=10.0
    ):
        """
        Stop the arm after Ctrl+C, and keep it stopped until MoveIt lets go.

        move_group does not act on a cancel while it is executing (see
        config.TRAJECTORY_CONTROLLER_ACTION), so the motion is stopped
        at the trajectory controller itself, which then holds position.
        The MoveIt goal is cancelled too, and the controller cancel is
        repeated until MoveIt reports the goal finished: a Ctrl+C
        during PLANNING would otherwise let execution start after this
        process has exited.
        """
        self.get_logger().warning(f'Interrupted: stopping {what}...')

        try:
            if self._spin_until_done(send_future, timeout_sec=1.0):
                goal_handle = send_future.result()

                if goal_handle is not None and goal_handle.accepted:
                    goal_handle.cancel_goal_async()

                    if result_future is None:
                        result_future = goal_handle.get_result_async()

            stopped = False
            deadline = time.monotonic() + timeout_sec

            while True:
                stopped = self._cancel_controller_goals() or stopped

                if result_future is None or result_future.done():
                    break

                if time.monotonic() >= deadline:
                    break

                self._spin_until_done(result_future, timeout_sec=0.2)

            if result_future is not None and not result_future.done():
                self.get_logger().error(
                    f'{what}: MoveIt has not released the goal after '
                    f'{timeout_sec:g}s; the arm may still move - use the '
                    'e-stop.'
                )
            elif stopped:
                self.get_logger().warning(
                    'Trajectory controller stopped; the arm is holding '
                    'position.'
                )
            elif not self._controller_cancel_client.service_is_ready():
                self.get_logger().error(
                    f"Cannot reach '{config.TRAJECTORY_CONTROLLER_ACTION}' "
                    'to stop the arm - use the e-stop.'
                )
            else:
                self.get_logger().warning(f'{what} was not moving the arm.')

        except KeyboardInterrupt:
            self.get_logger().error(
                'Interrupted again before the arm was confirmed stopped; '
                'it may still be moving - use the e-stop.'
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
        pipeline_id=PILZ_PIPELINE_ID,
        planner_id=PILZ_PTP_PLANNER_ID,
        fallback_pipeline_id=OMPL_PIPELINE_ID,
        fallback_planner_id=OMPL_PLANNER_ID,
    ):
        """
        Collision-aware joint-space movement.

        joint_angles:
            Seven absolute target angles in RADIANS.

        pipeline_id / planner_id:
            Which MoveIt planning pipeline and planner to try
            FIRST. Defaults to Pilz PTP, which interpolates
            straight from the current joint configuration to the
            target on a trapezoidal profile: same start and goal
            always produce the same trajectory, unlike OMPL's
            RRTConnect, which samples randomly and so takes a
            different route every run.

        fallback_pipeline_id / fallback_planner_id:
            Retried once, automatically, if the first attempt
            fails to PLAN. A failure during execution is never
            retried (see PLANNING_STAGE_ERROR_CODES). Pass
            fallback_pipeline_id=None to disable.

            The fallback exists because PTP does not route AROUND
            anything: it collision-checks its straight-line
            interpolation and reports failure if that hits
            something, where a sampling planner would search for a
            way past. That matters here because the bucket and
            table are real collision geometry (world objects,
            config.OBSTACLES - see arm/scene.py), so a direct
            interpolation out of a pose deep inside the bucket can
            genuinely be blocked. Recorded-pose transitions (INTER,
            DROP, BOTTOM) are clear direct paths and stay on PTP;
            only the awkward ones pay for OMPL, and the log says
            which did, so it's visible rather than guessed at.

            Falling back is safe precisely because Pilz fails
            rather than degrading: a failure is a clean signal, not
            a trajectory that half-works.
        """
        if len(joint_angles) != 7:

            raise ValueError(
                'xArm7 requires exactly 7 joint angles.'
            )

        attempts = [(pipeline_id, planner_id)]

        first_failure_reason = None

        if fallback_pipeline_id is not None:
            attempts.append(
                (fallback_pipeline_id, fallback_planner_id)
            )

        for attempt_index, (attempt_pipeline, attempt_planner) in enumerate(
            attempts
        ):

            if attempt_index > 0:

                self.get_logger().warning(
                    f'{attempts[0][0]}/{attempts[0][1]} failed '
                    f'({first_failure_reason}); retrying with '
                    f'{attempt_pipeline}/{attempt_planner}.'
                )

            succeeded, failure_reason, retryable = self._move_joints_once(
                joint_angles,
                velocity=velocity,
                acceleration=acceleration,
                planning_attempts=planning_attempts,
                planning_time=planning_time,
                tolerance=tolerance,
                replan=replan,
                replan_attempts=replan_attempts,
                pipeline_id=attempt_pipeline,
                planner_id=attempt_planner,
                quiet_failure=attempt_index < len(attempts) - 1,
            )

            if succeeded:
                return True

            if not retryable:
                # Failed during or after execution: the arm may have
                # been stopped part-way on purpose. Never re-command.
                self.get_logger().error(
                    f'Joint movement failed with {attempt_pipeline}/'
                    f'{attempt_planner} after planning: {failure_reason}. '
                    'Not retrying - the arm may have been stopped part-way.'
                )
                return False

            if attempt_index == 0:
                first_failure_reason = failure_reason

        return False

    def _move_joints_once(
        self,
        joint_angles,
        velocity,
        acceleration,
        planning_attempts,
        planning_time,
        tolerance,
        replan,
        replan_attempts,
        pipeline_id,
        planner_id,
        quiet_failure=False,
    ):
        """
        One planning/execution attempt with a specific pipeline.

        Returns (succeeded, reason, retryable): reason is None on
        success, and otherwise a short human-readable description of
        what went wrong, so move_joints() can say WHY a fallback
        happened rather than assuming a cause. retryable is True only
        when the failure happened before anything moved (see
        PLANNING_STAGE_ERROR_CODES) - the only case a fallback
        pipeline may be tried.

        quiet_failure downgrades the failure log to a debug line,
        so a first attempt that is about to be retried on another
        pipeline doesn't look like a hard error in the scan log.
        """
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

        goal.request.pipeline_id = pipeline_id
        goal.request.planner_id = planner_id

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
            f'Joint-space target ({pipeline_id}/{planner_id}):'
        )

        for name, angle in zip(
            self.JOINT_NAMES,
            joint_angles,
        ):

            self.get_logger().info(
                f'  {name}: '
                f'{math.degrees(angle):+.2f} deg'
            )

        goal_handle, wrapped_result = self._run_goal(
            self.move_group_client, goal, 'joint-space move'
        )

        def report_failure(message):

            if quiet_failure:
                self.get_logger().debug(message)
            else:
                self.get_logger().error(message)

        # Nothing has moved in the first two cases: the goal never
        # reached MoveIt, or MoveIt refused it outright.
        if goal_handle is None:

            reason = 'no response from MoveIt'
            report_failure('Failed to communicate with MoveIt.')

            return False, reason, True

        if not goal_handle.accepted:

            reason = (
                f'goal rejected by MoveIt - is the '
                f"'{pipeline_id}' pipeline loaded? (check: ros2 "
                f'param get /move_group planning_pipelines)'
            )
            report_failure(
                f'MoveIt rejected joint-space goal '
                f'({pipeline_id}/{planner_id}): {reason}'
            )

            return False, reason, True

        if wrapped_result is None:

            reason = 'MoveIt returned no result'
            report_failure(reason + '.')

            return False, reason, False

        error_code = (
            wrapped_result.result.error_code.val
        )

        if error_code == 1:

            self.get_logger().info(
                'Joint movement completed successfully.'
            )

            return True, None, False

        reason = describe_moveit_error(error_code)

        retryable = failure_is_retryable(wrapped_result.result)

        if retryable:
            report_failure(
                f'Joint movement failed with '
                f'{pipeline_id}/{planner_id}: {reason}'
            )

        return False, reason, retryable

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
        Rotate ONLY the joint at joint_index relative to its current position.

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
            f'{joint_name}-only relative movement:'
        )

        self.get_logger().info(
            f'  current: '
            f'{math.degrees(old_angle):+.2f} deg'
        )

        self.get_logger().info(
            f'  delta:   '
            f'{delta_deg:+.2f} deg'
        )

        self.get_logger().info(
            f'  target:  '
            f'{math.degrees(target[joint_index]):+.2f} deg'
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

        For example,
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

        For example,
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
        Plan (without executing) a straight Cartesian path to target_pose.

        Generate but DO NOT execute a straight Cartesian
        trajectory from the flange's CURRENT actual position to
        target_pose.

        Return value:
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

        if self.plan_from_observed_state:

            joints = self.get_current_joints()

            if joints is not None:
                request.start_state.joint_state = JointState(
                    name=list(self.JOINT_NAMES),
                    position=list(joints),
                )

        request.max_velocity_scaling_factor = (
            velocity
        )

        request.max_acceleration_scaling_factor = (
            acceleration
        )

        self.get_logger().info(
            'Computing Cartesian path...'
        )

        self.get_logger().info(
            f'  target: '
            f'({target_pose.position.x:.4f}, '
            f'{target_pose.position.y:.4f}, '
            f'{target_pose.position.z:.4f})'
        )

        future = self.cartesian_client.call_async(
            request
        )

        self._spin_until_done(future)

        response = future.result()

        if response is None:

            self.get_logger().error(
                'No Cartesian-path response from MoveIt.'
            )

            return None, 0.0

        self.get_logger().info(
            f'Cartesian path fraction: '
            f'{response.fraction * 100.0:.1f}%'
        )

        if response.fraction < 0.999:

            self.get_logger().error(
                'Complete Cartesian path could not '
                'be generated.'
            )

            return None, response.fraction

        if not response.solution.joint_trajectory.points:

            self.get_logger().error(
                'MoveIt returned an empty trajectory.'
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
            should pass, since config's gripper mounting offsets
            are only valid while the reference orientation (INTER)
            is held fixed.
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
        Plan (without executing) a straight stroke along current tool Z.

        Generate but DO NOT execute a straight Cartesian
        trajectory along the flange's current local Z axis.

        Orientation is kept fixed.

        Return value:
            RobotTrajectory or None
        """
        if abs(distance) <= 1e-12:

            self.get_logger().error(
                '_plan_tool_z() requires non-zero distance.'
            )

            return None

        tf = self.get_flange_transform()

        translation = tf.transform.translation
        rotation = tf.transform.rotation

        zx, zy, zz = tool_z_from_quaternion(rotation)

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

        goal_handle, wrapped_result = self._run_goal(
            self.execute_client, goal, 'trajectory execution'
        )

        if goal_handle is None:

            self.get_logger().error(
                'Failed to communicate with '
                '/execute_trajectory.'
            )

            return False

        if not goal_handle.accepted:

            self.get_logger().error(
                'Trajectory execution rejected.'
            )

            return False

        if wrapped_result is None:

            self.get_logger().error(
                'No trajectory execution result.'
            )

            return False

        error_code = (
            wrapped_result.result.error_code.val
        )

        if error_code == 1:

            self.get_logger().info(
                'Trajectory completed successfully.'
            )

            return True

        self.get_logger().error(
            f'Trajectory execution failed. '
            f'MoveIt error code: {error_code}'
        )

        return False

    # =========================================================
    # PLANNER-FREE JOINT PATHS
    #
    # A sampling planner (OMPL) takes a different route every run,
    # and Pilz PTP is only there if move_group loaded it. These build
    # the motion themselves - straight joint-space lines, minimum-jerk
    # timing (arm/joint_path.py) - and ask MoveIt only to CHECK each
    # densely interpolated state for collisions, so the same request
    # always produces the same motion, or the same refusal.
    # =========================================================

    def _client(self, attribute, service_type, name):
        client = getattr(self, attribute, None)

        if client is None:
            client = self.create_client(service_type, name)

            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {name}...')

            setattr(self, attribute, client)

        return client

    def _call(self, client, request):
        future = client.call_async(request)
        self._spin_until_done(future)
        return future.result()

    def _joint_state(self, joints):
        return JointState(
            name=list(self.JOINT_NAMES),
            position=[float(q) for q in joints],
        )

    def compute_ik(self, pose, seed, avoid_collisions=True, timeout=0.2):
        """
        Solve IK for the flange at `pose` (base_frame), seeded at `seed`.

        Returns 7 joint angles in JOINT_NAMES order, or None. The
        seed matters: the xArm7 is redundant, so the solver returns
        the solution nearest the seed's elbow configuration.
        """
        client = self._client('_ik_client', GetPositionIK, '/compute_ik')

        request = GetPositionIK.Request()
        request.ik_request.group_name = self.group_name
        request.ik_request.ik_link_name = self.flange_link
        request.ik_request.avoid_collisions = avoid_collisions
        request.ik_request.robot_state.joint_state = self._joint_state(seed)
        request.ik_request.timeout = Duration(
            sec=int(timeout), nanosec=int((timeout % 1.0) * 1e9)
        )

        stamped = PoseStamped()
        stamped.header.frame_id = self.base_frame
        stamped.pose = pose
        request.ik_request.pose_stamped = stamped

        response = self._call(client, request)

        if response is None or response.error_code.val != 1:
            return None

        solution = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
            )
        )

        return [solution[name] for name in self.JOINT_NAMES]

    def state_is_valid(self, joints):
        """
        Return True if MoveIt finds this joint state collision-free.

        A state MoveIt never answers for counts as invalid.
        """
        response = self._validity_response(joints)

        return bool(response is not None and response.valid)

    def state_contacts(self, joints):
        """
        Return the colliding (body, body) pairs at a joint state.

        [] if the state is valid; None if MoveIt never answered.
        """
        response = self._validity_response(joints)

        if response is None:
            return None

        if response.valid:
            return []

        return sorted({
            tuple(sorted((c.contact_body_1, c.contact_body_2)))
            for c in response.contacts
        })

    def _validity_response(self, joints):
        client = self._client(
            '_validity_client', GetStateValidity, '/check_state_validity'
        )

        request = GetStateValidity.Request()
        request.group_name = self.group_name
        request.robot_state.joint_state = self._joint_state(joints)

        for _attempt in range(VALIDITY_ATTEMPTS):
            future = client.call_async(request)

            if self._spin_until_done(future, VALIDITY_TIMEOUT_SEC):
                return future.result()

        self.get_logger().warning(
            'MoveIt did not answer a state-validity check; treating the '
            'state as invalid.',
            throttle_duration_sec=5.0,
        )

        return None

    def set_arm_padding(self, padding_m):
        """
        Set the arm links' obstacle padding in move_group (metres).

        The gripper keeps config.GRIPPER_PADDING_M: changing its padding
        makes MoveIt rebuild its large mesh, which takes seconds, while
        arm links take ~0.3 s. See arm/scene.py.
        """
        from . import scene

        scene.set_padding(
            self, padding_m, {'gripper_link': config.GRIPPER_PADDING_M}
        )

    def first_invalid_state(self, waypoints):
        """Return the index of the first colliding densified state, or None."""
        for index, joints in enumerate(densify(waypoints)):
            if not self.state_is_valid(joints):
                return index

        return None

    def execute_joint_path(self, waypoints, times, velocities, time_scale=1.0):
        """
        Execute a fixed joint trajectory exactly as given.

        time_scale < 1 replays it slower (times / time_scale, velocities
        * time_scale) without changing the path - for cautious first
        runs on the real arm. The first waypoint must match the arm's
        current state within MoveIt's start tolerance (0.01 rad);
        callers get there with move_joints_linear() first.
        """
        if not 0.0 < time_scale <= 1.0:
            raise ValueError('time_scale must be in (0, 1].')

        trajectory = JointTrajectory()
        trajectory.joint_names = list(self.JOINT_NAMES)

        for joints, t, qd in zip(waypoints, times, velocities):
            t = float(t) / time_scale

            point = JointTrajectoryPoint()
            point.positions = [float(q) for q in joints]
            point.velocities = [float(v) * time_scale for v in qd]
            nanoseconds = int(round(t * 1e9))
            point.time_from_start = Duration(
                sec=nanoseconds // 1000000000,
                nanosec=nanoseconds % 1000000000,
            )
            trajectory.points.append(point)

        robot_trajectory = RobotTrajectory()
        robot_trajectory.joint_trajectory = trajectory

        return self._execute_trajectory(robot_trajectory)

    def move_joints_linear(
        self,
        target,
        max_velocity_rad_s=config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S,
        time_scale=1.0,
    ):
        """
        Move along a straight joint-space line to `target`, deterministically.

        Every state along the line (1 deg apart) is collision-checked
        BEFORE the arm moves; if any collides, nothing moves and this
        returns False. Timing is minimum-jerk with the fastest joint
        peaking at max_velocity_rad_s.
        """
        current = self.get_current_joints()

        if current is None:
            return False

        waypoints = densify([current, list(target)])

        if np.abs(waypoints[-1] - waypoints[0]).max() < 1e-4:
            return True

        bad = self.first_invalid_state(waypoints)

        if bad is not None:
            self.get_logger().error(
                'Straight joint move would collide at '
                f'{100.0 * bad / (len(waypoints) - 1):.0f}% of the way; '
                'not moving.'
            )
            return False

        times, velocities = time_path(waypoints, max_velocity_rad_s)

        return self.execute_joint_path(
            waypoints, times, velocities, time_scale=time_scale
        )

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
        """Move straight along current tool Z, keeping flange orientation."""
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
    # offset (config.GRIPPER_OFFSET_*) has no X/Y component: since that
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
        Straight Cartesian movement of the flange to an absolute (x, y, z) target in base_frame.

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
                f'move_to_pose: infeasible '
                f'(fraction={fraction:.3f}).'
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
        Probe whether a pose is reachable, planning only.

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
        Add a J7 rotation that tracks the stroke's own progress.

        The final J7 trajectory is

            q7(t) = q7_moveit(t) + twist * p(t)

        where p(t) in [0, 1] is the stroke's progress along its path:
        its cumulative joint-space arc length, normalised. Because
        the stroke is a straight tool-Z line, J7 therefore turns a
        fixed angle per centimetre of insertion - a true helix, which
        is what scan/coverage.py models.

        Velocities and accelerations get the matching derivatives
        (twist * dp/dt, and its time derivative), so positions,
        velocities and accelerations stay consistent. That matters on
        the real arm: its trajectory controller interpolates BETWEEN
        points using the velocities, and the xArm driver streams the
        result to the joints at 150 Hz (servo mode). Editing positions
        alone - what this used to do - left J7's velocity near zero
        at every waypoint, so J7 would stop and start at each 5 mm
        point with peaks well above its average speed. MoveIt's
        profile already starts and ends at rest, so dp/dt does too.

        MoveIt never checks the added twist against J7's limits, so
        this does: if the twisted stroke would exceed
        config.JOINT7_MAX_VELOCITY_RAD_S or
        JOINT7_MAX_ACCELERATION_RAD_S2, the whole stroke is slowed
        uniformly (same path, longer duration) until it fits.
        """
        traj = trajectory.joint_trajectory

        if not traj.points:

            self.get_logger().error(
                'Cannot twist an empty trajectory.'
            )

            return False

        joint_names = list(
            traj.joint_names
        )

        if 'joint7' not in joint_names:

            self.get_logger().error(
                'joint7 not present in trajectory.'
            )

            return False

        j7_index = joint_names.index(
            'joint7'
        )

        times = np.array([
            self._duration_to_seconds(point.time_from_start)
            for point in traj.points
        ])

        if times[-1] <= 0.0 or np.any(np.diff(times) <= 0.0):

            self.get_logger().error(
                'Trajectory timing is invalid; cannot add a twist.'
            )

            return False

        positions = np.array([list(p.positions) for p in traj.points])

        dof = positions.shape[1]

        if all(len(p.velocities) == dof for p in traj.points):
            velocities = np.array([list(p.velocities) for p in traj.points])
        else:
            velocities = np.gradient(positions, times, axis=0)

        has_accelerations = all(
            len(p.accelerations) == dof for p in traj.points
        )

        # Progress along the stroke: normalised joint-space arc
        # length. ds/dt is the joint-space speed, taken from MoveIt's
        # own (consistent) velocities rather than differenced.
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
        )
        length = float(arc[-1])

        if length <= 1e-9:

            self.get_logger().error(
                'Stroke does not move; cannot add a twist.'
            )

            return False

        twist_rad = math.radians(
            twist_deg
        )

        progress = arc / length
        speed = np.linalg.norm(velocities, axis=1)

        q7_added = twist_rad * progress
        v7_added = twist_rad * speed / length
        a7_added = np.gradient(v7_added, times)

        v7_total = velocities[:, j7_index] + v7_added

        if has_accelerations:
            a7_total = (
                np.array([p.accelerations[j7_index] for p in traj.points])
                + a7_added
            )
        else:
            a7_total = a7_added

        # Slowing a trajectory by k divides velocities by k and
        # accelerations by k^2.
        peak_velocity = float(np.abs(v7_total).max())
        peak_acceleration = float(np.abs(a7_total).max())

        slowdown = max(
            1.0,
            peak_velocity / config.JOINT7_MAX_VELOCITY_RAD_S,
            math.sqrt(
                peak_acceleration / config.JOINT7_MAX_ACCELERATION_RAD_S2
            ),
        )

        # Diagnostic baseline
        self.get_logger().info(
            'J7 synchronized twist:'
        )

        self.get_logger().info(
            f'  MoveIt baseline: '
            f'{math.degrees(positions[0, j7_index]):+.2f} -> '
            f'{math.degrees(positions[-1, j7_index]):+.2f} deg'
        )

        self.get_logger().info(
            f'  Added twist: '
            f'{twist_deg:+.2f} deg'
        )

        self.get_logger().info(
            f'  Final target: '
            f'{math.degrees(positions[-1, j7_index] + twist_rad):+.2f} deg'
        )

        self.get_logger().info(
            f'  J7 peak: {math.degrees(peak_velocity / slowdown):.0f} deg/s, '
            f'{peak_acceleration / slowdown ** 2:.1f} rad/s^2 '
            f'over {times[-1] * slowdown:.2f} s'
        )

        if slowdown > 1.0:
            self.get_logger().warning(
                f'J7 twist would peak at {math.degrees(peak_velocity):.0f} '
                f'deg/s / {peak_acceleration:.1f} rad/s^2, over the J7 '
                f'limits; stroke slowed {slowdown:.2f}x to fit '
                '(lower --velocity or --sweep to avoid this).',
                throttle_duration_sec=30.0,
            )

        for index, point in enumerate(traj.points):

            point_positions = list(point.positions)
            point_positions[j7_index] += float(q7_added[index])
            point.positions = point_positions

            point_velocities = list(velocities[index])
            point_velocities[j7_index] = float(v7_total[index])
            point.velocities = [v / slowdown for v in point_velocities]

            if has_accelerations:
                point_accelerations = list(point.accelerations)
                point_accelerations[j7_index] = float(a7_total[index])
                point.accelerations = [
                    a / slowdown ** 2 for a in point_accelerations
                ]

            nanoseconds = int(round(times[index] * slowdown * 1e9))
            point.time_from_start = Duration(
                sec=nanoseconds // 1000000000,
                nanosec=nanoseconds % 1000000000,
            )

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

        For example,
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
                'For J7-only movement use rotate_joint7().'
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
