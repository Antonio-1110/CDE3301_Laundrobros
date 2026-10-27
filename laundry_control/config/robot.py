#!/usr/bin/env python3

"""The xArm7 as MoveIt and xarm_ros2 see it: frames, launch arguments, limits."""

import math
import os

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
#   - The gripper is the manufacturer's "other geometry": our mesh,
#     attached to link_eef (= link7, no offset), turned GRIPPER_MESH_RPY
#     about link7. Its link is called other_geometry_link, and the
#     stock SRDF already exempts it from colliding with link3/6/7.
#     The mesh is the arm plate (bolted to the flange, its trays
#     hanging 34 mm down beside the wrist) with the gripper plate
#     stacked on its top face, 14 mm up. Both are CoACD convex
#     decompositions (~11k triangles, was a 403k-triangle STL), so
#     collision checks and padding are cheap; RViz shows the hulls.
#     The full-detail CAD is FINAL ARM/GRIPPER PLATE OBJ.obj (mm).
#   - XARM_LIMITED False gives the xArm7's true hardware joint ranges
#     (the URDF defaults), including J7 +/-360 deg - the scan and grabs
#     turn J7 past -180 deg (INTER is -155, grabs reach ~-197). The
#     manufacturer's default, limited:=true, narrows J7 to +/-178 deg.
#     It also narrows J2 to -118..+120 deg (was +/-125). Automatic
#     motion is kept to JOINT_TASK_RANGES_DEG instead.
# =============================================================

GRIPPER_LINK = 'other_geometry_link'
GRIPPER_MESH = 'xArm7_ArmPlate_Gripper_CoACD.obj'
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

# Task joint ranges (degrees) for AUTOMATIC motion (issue #7). With
# XARM_LIMITED False, J1, J3, J5 and J7 may turn +/-360 deg, so the IK
# solver could hand back a posture with J1 a full turn round or the
# wrist flipped, and OMPL could take wild routes. Instead of filtering
# such answers afterwards, MoveIt's robot model gets these as its joint
# limits (laundry_bringup.launch.py, arm/joint_ranges.
# add_to_moveit_limits; xarm_ros2 stays unmodified): KDL's iterations
# and random restarts, OMPL's sampling and Pilz's checks all stay
# inside them. Our own checks (state_is_valid, straight joint moves,
# baked routes) use the same ranges. Changing them needs a bring-up
# restart.
#
# Each is the range used by every baked motion and named pose on
# 2026-10-08 plus about 30 deg (J7 includes the baseline scans'
# -239 .. -75), rounded; test_joint_ranges checks the baked plans still
# fit. The hard limits less JOINT_LIMIT_MARGIN_RAD still apply on top.
# `laundry move jointN` (a straight move, not planned by MoveIt) keeps
# the full hard limits. A straight move that STARTS outside a range
# (after such a hand move) may stay there or come back in, never go
# further out; `laundry move inter` gets back that way.
JOINT_TASK_RANGES_DEG = (
    (-80.0, 170.0),    # J1   used  -48 .. 138
    (-30.0, 143.0),    # J2   used    0 .. 113
    (-210.0, 30.0),    # J3   used -177 ..   0
    (-30.0, 157.0),    # J4   used    0 .. 127
    (-70.0, 145.0),    # J5   used  -37 .. 115
    (-110.0, 125.0),   # J6   used  -79 ..  95
    (-270.0, 30.0),    # J7   used -239 ..   0
)

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
#
# For benchmarks, LAUNDRY_TRANSFER_ACCEL_RAD_S2 / LAUNDRY_TRANSFER_JERK_RAD_S3
# (and LAUNDRY_TRANSFER_VELOCITY_DEG_S below) override these for one
# process tree; the jerk then defaults to the same ~8.4 x the
# acceleration. Baked timings record their limits, so a plan timed for
# other values is not used (pass_through.current_limits).
TRANSFER_MAX_ACCELERATION_RAD_S2 = float(
    os.environ.get('LAUNDRY_TRANSFER_ACCEL_RAD_S2', 7.0)
)
TRANSFER_MAX_JERK_RAD_S3 = float(os.environ.get(
    'LAUNDRY_TRANSFER_JERK_RAD_S3',
    59.0 if 'LAUNDRY_TRANSFER_ACCEL_RAD_S2' not in os.environ
    else round(TRANSFER_MAX_ACCELERATION_RAD_S2 * 59.0 / 7.0, 1),
))

# Peak speed of any point of the claw during a baked transfer, m/s.
# The xArm7 stops with C24 "Speed Exceeds Limit" over 1 m/s at the
# tool: on 2026-10-22 at 7.0 rad/s^2 the INTER -> HOME leg did it with
# no joint over its limit - J1, J3 and J7 near 60 deg/s together add
# up at the gripper, ~0.6 m from the base (1050 mm/s; 975 at 5.0, only
# 2.5% under). `laundry plan bake timing` keeps the claw under this
# (pass_through.Retimer: the corners of the gripper mesh's box, by
# forward kinematics), and slows any route that stops at its vias by
# the factor it records. 10% under the xArm's limit.
TRANSFER_MAX_TIP_SPEED_M_S = 0.9

# Peak joint speed of baked-transfer replays (and of go_to's straight
# fallback). transfers.yaml records the speed it was baked at, but the
# route is speed-independent, so replays use this instead.
#
# RAISING THESE THREE: one step at a time, on the cable, never over
# WiFi (2026-10-07: a WiFi dropout mid-move aborted INTER -> HOME).
# At each step run HOME <-> INTER and a DROP round trip, first with
# --speed 0.3, then at full speed; check ros2_control_node's log for
# overruns and set_servo_angle_j errors, and watch for shiver. Keep
# the last step that ran clean. Per preplanned grab (4 transfer legs):
#
#   speed deg/s  accel rad/s^2  jerk rad/s^3   seconds
#        45           2.0           17           14.0   (tested)
#        60           2.0           17           12.6
#        60           3.0           25           11.2
#        90           3.0           25           10.1
#
# 2026-10-21: set to 60 deg/s, accel 5.0 rad/s^2 (50% of MoveIt's 10),
# jerk 42 rad/s^3 (MoveIt has no jerk limit; this keeps the ~8.4 ratio of
# the earlier settings). Ran on the arm on 2026-10-22 (the sweep and
# DROP <-> HOME; C24 came only at 7.0, below); the routes were timed for
# these limits (`./rebake.sh timing`). Simulated 24-grab sweep on the fake
# controller (single runs, a few seconds of noise): 45/2.0/17 346 s,
# 60/2.0/17 339 s (-2%: the routes are short, so the acceleration limit,
# not the top speed, sets the time), 60/3.0/25 304 s (-12%, but the two
# DROP routes then had no pass-through timing within the limits), and
# 60/5.0/42 318 s with every route passing through its vias (DROP legs
# 3.5 -> 2.5 s). The sweep total did not fall further in that run; why
# was not investigated (more runs would show whether it is noise).
# 2026-10-26: accel 7.0 rad/s^2 (70% of MoveIt's 10), jerk 59 rad/s^3,
# now with TRANSFER_MAX_TIP_SPEED_M_S capping the claw (the first try
# at 7.0, without it, tripped C24 on INTER -> HOME). Re-timed: INTER <->
# DROP 2.36 s each way (claw <= 428 mm/s), BOTTOM 2.29 s; only HOME is
# over the cap (claw 1202 mm/s stopping at its vias) and is slowed to
# x0.749. Simulated 24-grab sweep 203 s. NOT yet run on the arm; staged
# test as above, HOME <-> INTER first.
# Go back one step if the arm overruns, shivers or throws laundry.
TRANSFER_MAX_VELOCITY_RAD_S = math.radians(
    float(os.environ.get('LAUNDRY_TRANSFER_VELOCITY_DEG_S', 60.0))
)

# Pass through the vias of baked transfers instead of stopping at each
# (arm/pass_through.py): uses the timing `laundry plan bake timing`
# stored for these limits; without one a route stops at its vias. The
# rounded corners are collision-checked before every replay.
TRANSFER_PASS_THROUGH_VIAS = True

# Peak joint speed for the planner-free straight joint moves
# (XArm7Controller.move_joints_linear) and the baked end scan: 45
# deg/s, well inside every xArm7 joint's limit.
LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S = math.radians(45.0)
