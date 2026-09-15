#!/usr/bin/env python3

"""
grasp_plan.py

Computes where to position the flange/TCP so the gripper's actual
contact point (see gripper.py) lands on a graspable point of a
detected laundry cluster (see laundry_detect.py).

The ToF sensor only ever reports the top-most surface it hit, so a
cluster's raw centroid sits ON TOP of the laundry, not inside it -
closing a gripper there would just brush the surface. This module
sinks the target below that surface by a DYNAMIC amount, informed
by two things:

  1. How much vertical room actually exists between the sensed top
     and the true bucket floor/wall at that same lateral position,
     estimated from the baseline (empty-bucket) scan.

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

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .gripper import GRIPPER_OFFSET_Z

# scan_move.py's default insertion step is 3cm, so the baseline is
# effectively sampled on a ~3cm grid - averaging this many nearest
# neighbours covers roughly one step-radius patch around a query
# point, smoothing single-point sensor noise without smearing
# across an unrelated part of the bucket.
DEFAULT_BASELINE_K = 5

# Walked largest to smallest: starts at the midpoint of a
# reasonable 40-60% range, halves on failure (fast convergence),
# then snaps straight to the guaranteed-safe floor (sink=0, i.e.
# grab right at the sensed surface - the same depth the ToF sensor
# itself already proved reachable) rather than continuing to halve
# into sub-centimetre differences.
DEFAULT_SINK_FRACTIONS = (0.5, 0.25, 0.1, 0.0)

# Never sink drastically further than the sensing resolution that
# produced the cluster in the first place: roughly one scan step
# (0.03m) plus margin, so a noisy/sparse baseline lookup can't
# justify an unreasonably deep sink.
DEFAULT_MAX_SINK_M = 0.05

# Matches the threshold _plan_to_pose/_plan_tool_z already use to
# decide a Cartesian path is "complete".
DEFAULT_REACHABILITY_THRESHOLD = 0.999


def estimate_baseline_depth(baseline_xyz, xy, k=DEFAULT_BASELINE_K):
    """
    Estimate the empty-bucket baseline surface's z at lateral
    position xy=(x, y), via a 2D (XY-only) cKDTree over
    baseline_xyz and averaging the k nearest baseline points' z.
    """

    if baseline_xyz.shape[0] == 0:

        raise ValueError(
            "baseline_xyz has no points."
        )

    tree = cKDTree(baseline_xyz[:, :2])

    k_eff = min(k, baseline_xyz.shape[0])

    _distances, indices = tree.query(
        np.asarray(xy, dtype=np.float64),
        k=k_eff,
    )

    indices = np.atleast_1d(indices)

    return float(baseline_xyz[indices, 2].mean())


@dataclass
class Quaternion:
    """
    Plain x/y/z/w quaternion, duck-type compatible with
    move.py's _build_pose() (which only ever reads those four
    attributes) without pulling a geometry_msgs dependency into
    this otherwise ROS-graph-free module.
    """

    x: float
    y: float
    z: float
    w: float


def look_at_quaternion(direction, reference_x_axis):
    """
    Build an orientation (as a Quaternion, base_frame_from_link7)
    whose local +Z axis points along `direction`, disambiguating
    the remaining roll about that axis by keeping local +X as
    close as possible to `reference_x_axis` (typically the flange's
    CURRENT local +X axis in base_frame, for a minimal-twist tilt
    rather than an arbitrarily rolled one).

    Both inputs are 3-vectors in base_frame; need not be
    normalized or orthogonal to each other.
    """

    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    reference_x_axis = np.asarray(reference_x_axis, dtype=np.float64)

    # Project out the component of reference_x_axis along
    # direction, leaving a valid (orthogonal) target for local +X
    # once tilted.
    x_axis = reference_x_axis - np.dot(reference_x_axis, direction) * direction
    x_axis_norm = np.linalg.norm(x_axis)

    if x_axis_norm < 1e-6:
        # reference_x_axis is (numerically) parallel to direction -
        # fall back to an arbitrary vector orthogonal to direction.
        fallback = np.array([1.0, 0.0, 0.0])

        if abs(np.dot(fallback, direction)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])

        x_axis = fallback - np.dot(fallback, direction) * direction
        x_axis_norm = np.linalg.norm(x_axis)

    x_axis = x_axis / x_axis_norm

    rotation, _rmsd = Rotation.align_vectors(
        a=[direction, x_axis],
        b=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
    )

    qx, qy, qz, qw = rotation.as_quat()

    return Quaternion(x=qx, y=qy, z=qz, w=qw)


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
        tcp_position offset by gripper.GRIPPER_OFFSET_Z along
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
    baseline_xyz,
    arm,
    gripper_offset_z=GRIPPER_OFFSET_Z,
    sink_fractions: Sequence[float] = DEFAULT_SINK_FRACTIONS,
    max_sink_m: float = DEFAULT_MAX_SINK_M,
    baseline_k: int = DEFAULT_BASELINE_K,
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

    baseline_z = estimate_baseline_depth(
        baseline_xyz,
        (x, y),
        k=baseline_k,
    )

    gap = max(0.0, sensed_top_z - baseline_z)

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
        # version assumed direction was always straight along local
        # Z (down at INTER).
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
