#!/usr/bin/env python3

"""The xArm7 as MoveIt and xarm_ros2 see it: frames, launch arguments, limits."""

import math

# =============================================================
# FRAMES AND MOVEIT
# =============================================================

GROUP_NAME = 'xarm7'
BASE_FRAME = 'link_base'
FLANGE_LINK = 'link7'
JOINT_STATE_TOPIC = '/joint_states'

# =============================================================
# ROBOT DESCRIPTION (passed to the UNMODIFIED manufacturer xarm_ros2)
#
# laundry_bringup.launch.py hands these to xarm_moveit_config's own
# launch arguments, so nothing in xarm_ros2 is edited:
#
#   - The gripper is the manufacturer's "other geometry": our STL,
#     attached to link_eef (= link7, no offset), turned GRIPPER_MESH_RPY
#     about link7. Its link is called other_geometry_link, and the
#     stock SRDF already exempts it from colliding with link3/6/7.
#   - XARM_LIMITED False gives the xArm7's true hardware joint ranges
#     (the URDF defaults), including J7 +/-360 deg - the scan and grabs
#     turn J7 past -180 deg (INTER is -155, grabs reach ~-197). The
#     manufacturer's default, limited:=true, narrows J7 to +/-178 deg.
#     It also narrows J2 to -118..+120 deg (was +/-125).
# =============================================================

GRIPPER_LINK = 'other_geometry_link'
GRIPPER_MESH = 'xArm7_Gripper_2Plate_Assembled_RevH_ROS2_Meters.stl'
GRIPPER_MESH_XYZ = (0.0, 0.0, 0.0)
GRIPPER_MESH_RPY = (0.0, 0.0, 1.57)
XARM_LIMITED = False


def xarm_description_arguments():
    """Return the xarm_moveit_config launch arguments for our robot."""
    def triple(values):
        # The xacro takes a quoted "x y z" string.
        return '"' + ' '.join(f'{v:g}' for v in values) + '"'

    return {
        'dof': '7',
        'robot_type': 'xarm',
        'hw_ns': 'xarm',
        'no_gui_ctrl': 'false',
        'limited': 'true' if XARM_LIMITED else 'false',
        'add_other_geometry': 'true',
        'geometry_type': 'mesh',
        'geometry_mesh_filename': (
            f'package://laundry_control/meshes/{GRIPPER_MESH}'
        ),
        'geometry_mesh_origin_xyz': triple(GRIPPER_MESH_XYZ),
        'geometry_mesh_origin_rpy': triple(GRIPPER_MESH_RPY),
    }


# ros2_control update rate on the real arm, in Hz: how often the
# trajectory controller sends the arm its next target. Stock xarm_ros2
# runs 150 Hz (6.7 ms a tick), but every tick waits on the arm twice -
# read() asks for the joint states, write() sends set_servo_angle_j and
# waits for the reply - and the arm takes ~3-4 ms to answer each. So
# while moving, ticks overran 6.7 ms all the time, even on the cable
# (logs Sep 30 - Oct 6): late ticks are skipped and the next target
# jumps, which shows up as jerks. 88% of those slow ticks fit in 10 ms.
# laundry_bringup.launch.py applies it (control_rate_hz:= overrides,
# e.g. 150 to compare); xarm_ros2 itself stays unmodified.
ARM_CONTROL_RATE_HZ = 100

# The ros2_control trajectory controller that actually drives the
# joints - the same name on the real arm and the fake controller
# (xarm_controller/config/xarm7_controllers.yaml). Ctrl+C in `laundry`
# cancels its goals directly: move_group does not act on a cancel
# until the execution it is running has finished (measured on the
# fake controller, MoveIt 2.12), so cancelling only the MoveIt goal
# would let the arm complete the motion.
TRAJECTORY_CONTROLLER_ACTION = '/xarm7_traj_controller/follow_joint_trajectory'

# =============================================================
# JOINTS, PLANNERS AND SPEED LIMITS
# =============================================================

JOINT_NAMES = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
]

# Joint position limits (radians), from xarm_description/urdf/xarm7/
# xarm7.urdf.xacro. MoveIt's state-validity check does not reject
# states outside them, so a baked route could leave them: on
# 2026-09-29 an INTER -> DROP via had J2 at 125 deg (limit 120) and
# the arm stopped with C23 "Joint Angle Exceed Limit". Every state
# checked by XArm7Controller.state_is_valid must also be
# JOINT_LIMIT_MARGIN_RAD inside these.
JOINT_LOWER_LIMITS_RAD = [
    -2.0 * math.pi, -2.059, -2.0 * math.pi, -0.19198,
    -2.0 * math.pi, -1.69297, -2.0 * math.pi,
]
JOINT_UPPER_LIMITS_RAD = [
    2.0 * math.pi, 2.0944, 2.0 * math.pi, 3.927,
    2.0 * math.pi, math.pi, 2.0 * math.pi,
]
JOINT_LIMIT_MARGIN_RAD = math.radians(2.0)

# Planning pipelines for joint-space moves.
#
# Pilz PTP is preferred: it interpolates straight from the current
# joint configuration to the target on a trapezoidal velocity
# profile, so the same start and goal always produce the same
# trajectory. OMPL's RRTConnect samples randomly and takes a
# different route every run, which makes motions hard to predict
# and to review.
#
# OMPL remains the automatic fallback because PTP does not route
# around obstacles - it collision-checks its straight-line
# interpolation and fails if that is blocked. The bucket and table
# ARE collision geometry here (config/scene.py), so that case is
# real, not theoretical.
#
# NOTE: move_group must actually LOAD the pilz pipeline for this to
# work - check with:
#
#     ros2 param get /move_group planning_pipelines
#
# If that lists only ['ompl'] (which is what the stock
# xarm7_moveit_fake.launch.py gives), every PTP attempt fails and
# silently falls back, costing a wasted planning attempt per move.
PILZ_PIPELINE_ID = 'pilz_industrial_motion_planner'
PILZ_PTP_PLANNER_ID = 'PTP'

OMPL_PIPELINE_ID = 'ompl'
OMPL_PLANNER_ID = 'RRTConnect'

# J7's velocity and acceleration limits from xarm_moveit_config/
# config/xarm7/joint_limits.yaml (2.14 rad/s = 123 deg/s, 10 rad/s^2).
# The scan's J7 twist is added to a stroke AFTER MoveIt has
# time-parameterised it, so MoveIt never enforces these on it: at
# scan velocity 0.1 (the default until 2026-09) a 150 deg twist over
# a 0.95s stroke averaged ~157 deg/s; the 0.03 default averages ~56
# deg/s. arm/trajectory.add_joint7_twist checks the twisted
# stroke's peak J7 speed and acceleration against these and slows
# the whole stroke down when either would be exceeded.
JOINT7_MAX_VELOCITY_RAD_S = 2.14
JOINT7_MAX_ACCELERATION_RAD_S2 = 10.0

# Acceleration and jerk limits for replaying baked transfers
# (arm/transfers.py; joint_path.s_curve). Each leg used to be timed
# min-jerk, which only averages 53% of its 45 deg/s peak; an S-curve
# under these limits cruises at the peak instead - INTER -> DROP 7.2
# -> 5.4 s, INTER <-> grab_NN 3.1 -> 2.2 s each way (issue #11). The
# values are the firmest the min-jerk transfers already used: their
# shortest segment (29 deg) peaked at 2.0 rad/s^2 and 17 rad/s^3. So
# no transfer accelerates harder than one already did; the long legs
# just do it as often. MoveIt's xArm7 limit is 10 rad/s^2.
TRANSFER_MAX_ACCELERATION_RAD_S2 = 2.0
TRANSFER_MAX_JERK_RAD_S3 = 17.0

# Peak joint speed for the planner-free straight joint moves
# (XArm7Controller.move_joints_linear) and the baked end scan: 45
# deg/s, well inside every xArm7 joint's limit.
LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S = math.radians(45.0)
