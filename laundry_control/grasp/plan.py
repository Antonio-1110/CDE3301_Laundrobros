#!/usr/bin/env python3

"""
Compute where to put the flange/TCP so the gripper grabs a detected cluster.

The gripper's actual contact point (config.GRIPPER_OFFSET_Z) must
land on a graspable point of a detected laundry cluster (see
perception/detect.py).

The ToF sensor only ever reports the top-most surface it hit, so a
cluster's raw centroid sits ON TOP of the laundry, not inside it -
closing a gripper there would just brush the surface. This module
sinks the target below that surface by a DYNAMIC amount, informed
by two things:

  1. How much vertical room actually exists between the sensed top
     and the bucket wall at that same lateral position, taken from
     the fitted bucket model (bucket_model.BaselineSurface) rather
     than from raw baseline points.

  2. Whether the arm can actually reach a given depth at all,
     checked live via a plan-only dry-run (XArm7Controller.
     check_pose_reachable()) rather than assumed.

Rather than approaching purely along local Z (the INTER-orientation
insertion axis) and requiring the wrist itself to translate over
to the cluster's lateral position, the approach orientation TILTS
so the flange's local Z axis - the same axis GRIPPER_OFFSET_Z
extends along - points from the arm's current position straight at
the grasp point. GetCartesianPath already SLERPs orientation
between the current and target Pose, so this produces a natural
combined translate+tilt reach with no extra planning machinery.

This module is intentionally free of any live ROS node/graph
dependency (the `arm` parameter is duck-typed - it only needs
check_pose_reachable() and get_flange_transform() methods - so
it's unit-testable with a stub in place of a real XArm7Controller).
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from geometry_msgs.msg import Quaternion
import numpy as np
from scipy.spatial.transform import Rotation

from ..arm.geometry import look_at_quaternion
from ..config import GRIPPER_OFFSET_Z
from ..perception.bucket_model import to_cylindrical

__all__ = [
    'GraspTarget',
    'compute_grasp_target',
    'estimate_surface_depth_below',
    'look_at_quaternion',
]

# Walked largest to smallest: starts at the midpoint of a
# reasonable 40-60% range, halves on failure (fast convergence),
# then snaps straight to the guaranteed-safe floor (sink=0, i.e.
# grab right at the sensed surface - the same depth the ToF sensor
# itself already proved reachable) rather than continuing to halve
# into sub-centimetre differences.
DEFAULT_SINK_FRACTIONS = (0.5, 0.25, 0.1, 0.0)

# Never sink drastically further than the sensing resolution that
# produced the cluster in the first place: roughly one scan step
# (0.03m) plus margin, so a noisy depth estimate can't justify an
# unreasonably deep sink.
DEFAULT_MAX_SINK_M = 0.05

# Matches the threshold _plan_to_pose/_plan_tool_z already use to
# decide a Cartesian path is "complete".
DEFAULT_REACHABILITY_THRESHOLD = 0.999

# How far below the sensed top to look for the bucket wall before
# giving up, and how precisely to locate it. 0.5mm is well under
# the sensor's own 2mm noise, so the bisection is never the
# limiting error.
DEFAULT_MAX_DROP_M = 0.6
DEFAULT_DEPTH_TOLERANCE_M = 0.0005


def estimate_surface_depth_below(
    surface,
    xy,
    below_z,
    max_drop_m=DEFAULT_MAX_DROP_M,
    tolerance_m=DEFAULT_DEPTH_TOLERANCE_M,
):
    """
    Find the z at which a vertical line through (x, y) leaves the
    bucket, searching downward from below_z.

    This is what limits how far the gripper may sink into a pile:
    the room between the item's sensed top and whatever the gripper
    would hit first on its way down.

    WHY THE MODEL AND NOT THE BASELINE POINTS
    -----------------------------------------
    This used to take the k nearest baseline points in (x, y) and
    keep the highest one below the item. That inherited the exact
    problem the detector was rewritten to escape: z is not
    single-valued over (x, y) on a bucket lying on its side, so the
    "k nearest in (x, y)" are a mix of floor points and wall points
    standing above them at almost the same place. Picking the
    highest one below the item was a workaround, not a fix.

    It also needed baseline points to be THERE. Measured on the
    real 8-scan baseline set, the scan path only reaches about 30%
    of the bucket's surface cells, so a query over an unsampled
    patch was answered from whatever happened to be nearest, which
    could be some distance away on a differently-oriented piece of
    wall.

    The fitted profile has neither problem: it is defined
    everywhere, and asking where a vertical line crosses it is an
    exact geometric question with one answer. Solved by bisection
    rather than algebraically so it stays correct for the whole
    profile - cone, flat closed end, and the corner between them -
    without a special case per segment. It costs about 70 point
    evaluations, which is nothing next to a single motion plan.

    Returns None when the line does not leave the bucket within
    max_drop_m, or when below_z is already outside it. Both mean
    "no trustworthy estimate of what is underneath", and the
    sensible response - don't sink on a guess - belongs to the
    caller.
    """

    x = float(xy[0])
    y = float(xy[1])

    cone = surface.cone
    profile = surface.profile

    def inside(z):

        point = np.array([[x, y, float(z)]])

        s, _theta, r = to_cylindrical(point, cone)

        if s[0] > cone.s_max:
            # Past the mouth: outside the bucket, even though the
            # cone's radius keeps growing if extrapolated.
            return False

        _u, intrusion = profile.project(s, r)

        return bool(intrusion[0] > 0.0)

    if not inside(below_z):
        return None

    step = 0.01

    z_inside = float(below_z)
    z_outside = None

    z = float(below_z)

    while below_z - z < max_drop_m:

        z -= step

        if not inside(z):
            z_outside = z
            break

        z_inside = z

    if z_outside is None:
        return None

    while z_inside - z_outside > tolerance_m:

        midpoint = 0.5 * (z_inside + z_outside)

        if inside(midpoint):
            z_inside = midpoint
        else:
            z_outside = midpoint

    return 0.5 * (z_inside + z_outside)


@dataclass
class GraspTarget:
    """
    tcp_position:
        Where to move the flange/TCP (hand to
        XArm7Controller.move_to_pose()).

    orientation:
        The Quaternion to pass as move_to_pose()'s `orientation` -
        tilts local +Z to point at grasp_point from the arm's
        position when compute_grasp_target() was called.

    grasp_point:
        The actual gripper contact point this achieves, i.e.
        tcp_position offset by config.GRIPPER_OFFSET_Z along
        `orientation`'s local +Z direction.
    """

    tcp_position: np.ndarray
    orientation: Quaternion
    grasp_point: np.ndarray
    sink_fraction_used: float
    sink_amount_m: float
    gap_m: float
    reachability_fraction: float


def compute_grasp_target(
    cluster,
    surface,
    arm,
    gripper_offset_z=GRIPPER_OFFSET_Z,
    sink_fractions: Sequence[float] = DEFAULT_SINK_FRACTIONS,
    max_sink_m: float = DEFAULT_MAX_SINK_M,
    max_step: float = 0.005,
    velocity: float = 0.1,
    acceleration: float = 0.1,
    reachability_threshold: float = DEFAULT_REACHABILITY_THRESHOLD,
) -> Optional[GraspTarget]:
    """
    Compute where to place the flange/TCP, and how to orient it, so
    the gripper's contact point lands on a graspable point of
    `cluster`, sunk below the sensed top by a dynamic amount,
    verified reachable via a plan-only dry-run against `arm`.

    x, y, and the sensed-top z all come from cluster.centroid (one
    consistent point), rather than e.g. pairing centroid xy with
    highest_point's z, which could describe two different points.
    The sink itself stays purely vertical (digging into the pile is
    a gravity-down notion, independent of approach angle) - only
    the final reach from the TCP to that sunk point is tilted.

    The approach direction is computed from wherever `arm` actually
    is when this is called (via get_flange_transform()) toward the
    (sunk) grasp point, so callers should already have moved `arm`
    to a sensible starting pose (e.g. INTER) first.

    Returns None only if even sink=0.0 (the sensed surface itself)
    is unreachable - that indicates a genuine reach/collision
    problem unrelated to sink depth, which sinking less cannot fix.
    """

    x = float(cluster.centroid[0])
    y = float(cluster.centroid[1])
    sensed_top_z = float(cluster.centroid[2])

    floor_z = estimate_surface_depth_below(
        surface,
        (x, y),
        below_z=sensed_top_z,
    )

    if floor_z is None:

        # A vertical line through the cluster never crosses the
        # modelled bucket wall, so there is no trustworthy estimate
        # of how much room is underneath. Sinking on a guess risks
        # driving the gripper into the bucket, so fall back to
        # grasping at the sensed surface itself - the one depth the
        # ToF sensor has already proved reachable.
        print(
            f"WARNING: no bucket surface found below the cluster at "
            f"({x:.3f}, {y:.3f}, {sensed_top_z:.3f}); grasping at the "
            f"sensed surface without sinking."
        )

        gap = 0.0

    else:

        gap = max(0.0, sensed_top_z - floor_z)

        if gap == 0.0:

            print(
                f"WARNING: bucket surface at ({x:.3f}, {y:.3f}) "
                f"modelled at z={floor_z:.3f}, at or above the "
                f"cluster's sensed top z={sensed_top_z:.3f}; grasping "
                f"at the sensed surface without sinking."
            )

    tf = arm.get_flange_transform()

    current_position = np.array(
        [
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ]
    )

    current_rotation = Rotation.from_quat(
        [
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ]
    )

    current_x_axis = current_rotation.apply([1.0, 0.0, 0.0])

    for fraction in sink_fractions:

        sink_amount = min(fraction * gap, max_sink_m)
        grasp_z = sensed_top_z - sink_amount
        grasp_point = np.array([x, y, grasp_z])

        direction = grasp_point - current_position
        direction_norm = np.linalg.norm(direction)

        if direction_norm < 1e-6:
            # Degenerate: the arm is already sitting at the grasp
            # point. Keep pointing along the current local Z rather
            # than dividing by ~0.
            direction = current_rotation.apply([0.0, 0.0, 1.0])
        else:
            direction = direction / direction_norm

        orientation = look_at_quaternion(direction, current_x_axis)

        # TCP sits back from the grasp point by gripper_offset_z,
        # along the SAME direction the wrist is now tilted to face -
        # this is the general form of the old "tcp_z = grasp_z +
        # gripper_offset_z" shortcut, which only worked because that
        # version assumed the approach was always straight down.
        tcp_position = grasp_point - gripper_offset_z * direction

        reachability = arm.check_pose_reachable(
            tcp_position[0], tcp_position[1], tcp_position[2],
            orientation=orientation,
            max_step=max_step,
            velocity=velocity,
            acceleration=acceleration,
        )

        if reachability >= reachability_threshold:

            return GraspTarget(
                tcp_position=tcp_position,
                orientation=orientation,
                grasp_point=grasp_point,
                sink_fraction_used=fraction,
                sink_amount_m=sink_amount,
                gap_m=gap,
                reachability_fraction=reachability,
            )

    return None
