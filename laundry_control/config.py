#!/usr/bin/env python3

"""
Every hand-measured number the package depends on, in one place.

Recorded joint poses, frame names, sensor/gripper mounting offsets,
servo angles and on-disk data locations all used to be scattered
across arm_position.py, tof_sensor.py, gripper.py, gripper_node.py
and several hardcoded /home/cde3301a/... paths. They are the values
most likely to need changing when the rig is touched (a remounted
sensor, a re-recorded pose, a different machine), so they live here
and nowhere else.

This module has no ROS dependency and does no I/O at import time, so
anything - tests included - can import it freely.
"""

import os

# =============================================================
# RECORDED JOINT POSES
#
# Seven absolute joint angles in RADIANS, J1..J7, recorded on the
# real arm. `laundry move <name>` accepts any upper-case list here
# by its lower-case name (see named_poses()).
# =============================================================

HOME = [
    1.57,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

# The reference pose everything else is measured from: the scan
# starts and ends here, and the ToF/gripper offsets below were
# measured here. Forward kinematics at INTER (MoveIt fake
# controller, same URDF as the real arm):
#
#   link7 local +Z = (0.01, -1.00, -0.01)  horizontal, along -Y:
#                                           straight into the bucket
#                                           along its axis
#   link7 local +X = (-0.01, 0.01, -1.00)  straight DOWN - this is
#                                           the ToF boresight, so at
#                                           INTER's J7 the sensor
#                                           looks at the bucket floor
#
# Older comments said "+Z points straight down at INTER"; it is the
# sensor's +X that does. The scan's +/-75 deg J7 sweep is therefore
# centred on the floor, which is why the ceiling is never seen.
INTER = [
    1.5105054378509521,
    1.5552058219909668,
    -2.4133317470550537,
    0.17237895727157593,
    0.28598350286483765,
    0.12894250452518463,
    -2.70444223365,
]

# Tilted-up pose used at maximum insertion depth so the wrist-
# mounted sensor can see the closed end past the end effector (see
# scan.pattern's BOTTOM detour). FK: local +Z tilts 42 deg up from
# horizontal, and the boresight (+X) points down and toward the
# closed end, ~48 deg below horizontal.
BOTTOM = [
    -0.6021117568016052,
    1.1247814893722534,
    -1.3280805349349976,
    1.6322089433670044,
    0.7299384474754333,
    -1.0157668590545654,
    -2.638805389404297,
]

# Where retrieved laundry is released.
DROP = [
    0.10072359442710876,
    1.4992231130599976,
    -3.0929934978485107,
    0.29612118005752563,
    -0.11858407407999039,
    0.7183085680007935,
    -2.913362979888916,
]

# Fixed grab poses for the sensorless `laundry preplanned` sweep.
RETRIEVE_0 = [
    -0.16877898573875427,
    0.3103211522102356,
    -1.2240749597549438,
    1.2551662921905518,
    0.23523668944835663,
    0.6844640374183655,
    -3.019961357116699,
]

RETRIEVE_1 = [
    1.0142279863357544,
    0.7750195264816284,
    -2.239295482635498,
    0.6214537024497986,
    0.6356657147407532,
    0.6073369383811951,
    -3.199042797088623,
]

RETRIEVE_2 = [
    0.635716438293457,
    0.1730310469865799,
    -1.9444830417633057,
    1.1769441366195679,
    0.14949719607830048,
    1.2463109493255615,
    -2.922783851623535,
]

RETRIEVE_3 = [
    1.0214934349060059,
    0.7060800194740295,
    -2.34552264213562,
    0.7978871464729309,
    0.47409510612487793,
    1.36742103099823,
    -3.12626576423645,
]

_POSE_NAMES = (
    'HOME',
    'INTER',
    'BOTTOM',
    'DROP',
    'RETRIEVE_0',
    'RETRIEVE_1',
    'RETRIEVE_2',
    'RETRIEVE_3',
)


def named_poses():
    """Return every recorded pose as {lower-case name: joint list}."""
    return {name.lower(): list(globals()[name]) for name in _POSE_NAMES}


def get_named_pose(name):
    """
    Look up a recorded pose by name, case-insensitively.

    Returns a fresh list of 7 joint angles in radians. Raises
    KeyError listing what IS available, so a typo is reported before
    anything waits on MoveIt.
    """
    poses = named_poses()

    key = name.strip().lower()

    if key not in poses:
        raise KeyError(
            f"Unknown arm pose '{name}'. "
            f"Available: {', '.join(sorted(poses))}"
        )

    return poses[key]


# =============================================================
# FRAMES AND MOVEIT
# =============================================================

GROUP_NAME = 'xarm7'
BASE_FRAME = 'link_base'
FLANGE_LINK = 'link7'
JOINT_STATE_TOPIC = '/joint_states'

JOINT_NAMES = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
]

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
# ARE collision geometry here (URDF links on link_base, see
# xarm7.urdf.xacro), so that case is real, not theoretical.
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

# J7's velocity limit from xarm_moveit_config/config/xarm7/
# joint_limits.yaml (2.14 rad/s = 123 deg/s). The scan's J7 twist is
# added to trajectory POSITIONS after MoveIt time-parameterises the
# stroke, so MoveIt never enforces this on it: at the default scan
# velocity (0.1) a 150 deg twist over a 0.95s stroke already runs at
# ~157 deg/s. The controller logs a warning when a twist exceeds it.
JOINT7_MAX_VELOCITY_RAD_S = 2.14

# =============================================================
# TOF SENSOR (VL53L0X)
# =============================================================

TOF_RANGE_TOPIC = 'tof_sensor/range'
TOF_SENSOR_FRAME = 'tof_sensor_link'

# VL53L0X datasheet: ~25 degree field of view, usable range
# ~30mm-2000mm. MAX_RANGE_M is deliberately tightened below the
# sensor's own usable range: readings beyond the bucket's own depth
# aren't useful (they're outside the bucket entirely) and are
# dropped at the source, since scan_recorder_node just filters
# against whatever min_range/max_range the sensor node reports.
TOF_FIELD_OF_VIEW_RAD = 0.436
TOF_MIN_RANGE_M = 0.03
TOF_MAX_RANGE_M = 0.25

TOF_PUBLISH_PERIOD_SEC = 0.05

# Calibration offset added to raw VL53L0X readings, in cm. The node
# default; real_arm_scan.launch.py / laundry_bringup.launch.py pass
# the rig's calibrated value (-8.5) explicitly.
TOF_DEFAULT_OFFSET_CM = -10.0

# Translation from the TCP frame (link7) to the ToF sensor's own
# frame. Axis-aligned with link7 (no rotation offset) - the sensor's
# +X (its boresight, per REP 117 / sensor_msgs/Range) points along
# link7's +X.
#
# Measured by ruler at the INTER pose, where link7's local +Z points
# horizontally INTO the bucket (see INTER above):
#
#   - 7.75 cm along local +X - the sensor's own boresight axis. It
#     looks radially outward from the insertion axis at the bucket
#     wall (straight down at INTER's J7), offset slightly along the
#     same direction it looks.
#   - 2.8 cm along local +Z - forward, further into the bucket.
#
# Because the offset is expressed in link7's own frame, it is
# joint7-invariant: it stays correct as J7 rotates during a scan,
# which is what sweeps the beam around the bucket.
TOF_SENSOR_OFFSET_X = 0.0775
TOF_SENSOR_OFFSET_Y = 0.0
TOF_SENSOR_OFFSET_Z = 0.028

# =============================================================
# GRIPPER
# =============================================================

# Gripper contact-point mounting offset, in link7's own local frame
# (same convention as the ToF offset above).
#
# Unlike the ToF sensor's offset (which has a nonzero local X
# component - its outward-pointing boresight), the gripper's contact
# point sits ENTIRELY along local +Z: it projects straight out from
# the TCP along the wrist's own rotation axis (at INTER, forward into
# the bucket). So rotating J7 sweeps the ToF sensor around a circle,
# but does NOT move the gripper's contact point at all - reaching an
# arbitrary bucket-interior point with the gripper therefore needs
# genuine translation/tilt of the TCP, not just J7 rotation the way
# scanning does.
#
# GRIPPER_OFFSET_Z is a user-reported measurement, not yet verified
# against the physical hardware - confirm it (HARDWARE_TESTS.md)
# before trusting it for an unattended grasp.
GRIPPER_OFFSET_X = 0.0
GRIPPER_OFFSET_Y = 0.0
GRIPPER_OFFSET_Z = 0.15

# The claw is roughly 6cm wide, so it doesn't need pinpoint lateral
# accuracy - being within this tolerance of the intended point is
# still graspable. A documented error budget, not an offset: nothing
# applies it as a coordinate correction.
GRIPPER_LATERAL_TOLERANCE_M = 0.03

# Raw servo angles that open/close the physical claw. Which angle
# does what depends on how the claw is linked to the servo horn;
# override on the node with --ros-args rather than editing code if
# the linkage changes.
GRIPPER_OPEN_ANGLE_DEG = 55.0
GRIPPER_CLOSE_ANGLE_DEG = 100.0

# BCM GPIO pin and pulse range of the hobby servo.
SERVO_PIN = 18
SERVO_MIN_ANGLE_DEG = 0.0
SERVO_MAX_ANGLE_DEG = 180.0
SERVO_MIN_PULSE_WIDTH_S = 0.5 / 1000
SERVO_MAX_PULSE_WIDTH_S = 2.5 / 1000

# =============================================================
# DATA LOCATIONS
#
# Saved scans must live in the SOURCE tree, never under install/ or
# build/: both get wiped by `rm -rf build install log` before a
# clean rebuild, which would silently delete every scan.
#
# The source tree is found from this file's own real path. With
# `colcon build --symlink-install` (what the README prescribes)
# the installed module resolves back to the source checkout, so this
# works on any machine and any workspace name. Set LAUNDRY_DATA_DIR
# to override it (e.g. for a non-symlink install).
# =============================================================

DATA_DIR_ENV = 'LAUNDRY_DATA_DIR'


def repo_root():
    """
    Return the CDE3301_Laundrobros checkout that data lives under.

    Order: $LAUNDRY_DATA_DIR, then the source tree this module was
    loaded from (if it looks like one), then the current directory.
    """
    override = os.environ.get(DATA_DIR_ENV)

    if override:
        return os.path.abspath(os.path.expanduser(override))

    here = os.path.dirname(os.path.realpath(__file__))
    candidate = os.path.dirname(here)

    if os.path.isfile(os.path.join(candidate, 'package.xml')):
        return candidate

    return os.getcwd()


def baseline_dir():
    """Directory of empty-bucket baseline scans the detector models."""
    return os.path.join(repo_root(), 'baseline_scans')


def scan_records_dir():
    """Directory ordinary scans are saved under."""
    return os.path.join(repo_root(), 'scan_records')
