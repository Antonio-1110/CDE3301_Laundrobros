#!/usr/bin/env python3

"""
Questions MoveIt answers without moving the arm: FK, IK, validity.

MoveItQueries is part of XArm7Controller (arm/controller.py), which
inherits it; it is kept apart because none of it moves anything. The
planner-free joint paths (move_joints_linear, the baked routes and
end scan) are built on these: every state is checked here before the
arm is commanded.

Uses the controller's node (create_client, get_logger), its
_spin_until_done, and base_frame / flange_link / group_name.
"""

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
import numpy as np
from sensor_msgs.msg import JointState

from .joint_path import densify, within_joint_limits
from .. import config

# State-validity requests are sent ONE AT A TIME. With several in
# flight at once, move_group (rmw_fastrtps, Jazzy) intermittently never
# answers some of them - measured: most multi-request checks lost at
# least one after the first. Sequential checks cost ~5 ms per state.
# A request unanswered after VALIDITY_TIMEOUT_SEC is re-sent, up to
# VALIDITY_ATTEMPTS times, before the state counts as invalid.
VALIDITY_TIMEOUT_SEC = 1.0
VALIDITY_ATTEMPTS = 3


class MoveItQueries:
    """FK, IK and state-validity queries (a part of XArm7Controller)."""

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

    def compute_fk(self, joints):
        """
        Return (position, orientation quaternion xyzw) of the flange.

        Pure kinematics from the robot model - independent of where
        the arm actually is - so anything built on it is repeatable.
        Returns None if MoveIt refuses.
        """
        client = self._client('_fk_client', GetPositionFK, '/compute_fk')

        request = GetPositionFK.Request()
        request.header.frame_id = self.base_frame
        request.fk_link_names = [self.flange_link]
        request.robot_state.joint_state = self._joint_state(joints)

        response = self._call(client, request)

        if response is None or response.error_code.val != 1:
            return None

        pose = response.pose_stamped[0].pose

        return (
            np.array([pose.position.x, pose.position.y, pose.position.z]),
            np.array([
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w,
            ]),
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
        Return True if the state is within limits and collision-free.

        Limits are config.JOINT_*_LIMITS_RAD less JOINT_LIMIT_MARGIN_RAD,
        checked here because MoveIt's validity service does not. A
        state MoveIt never answers for counts as invalid.
        """
        if not within_joint_limits(
            joints,
            config.JOINT_LOWER_LIMITS_RAD,
            config.JOINT_UPPER_LIMITS_RAD,
            config.JOINT_LIMIT_MARGIN_RAD,
        ):
            return False

        response = self._validity_response(joints)

        return bool(response is not None and response.valid)

    def state_contacts(self, joints):
        """
        Return the colliding (body, body) pairs at a joint state.

        [] if the state is valid; None if MoveIt never answered. A
        joint outside its limits (see state_is_valid) is reported as
        (joint name, 'joint limit'), without asking MoveIt.
        """
        beyond = [
            (name, 'joint limit')
            for name, q, lo, hi in zip(
                self.JOINT_NAMES,
                joints,
                config.JOINT_LOWER_LIMITS_RAD,
                config.JOINT_UPPER_LIMITS_RAD,
            )
            if not within_joint_limits(
                [q], [lo], [hi], config.JOINT_LIMIT_MARGIN_RAD
            )
        ]

        if beyond:
            return beyond

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
            self, padding_m, {config.GRIPPER_LINK: config.GRIPPER_PADDING_M}
        )

    def first_invalid_state(self, waypoints):
        """Return the index of the first colliding densified state, or None."""
        for index, joints in enumerate(densify(waypoints)):
            if not self.state_is_valid(joints):
                return index

        return None
