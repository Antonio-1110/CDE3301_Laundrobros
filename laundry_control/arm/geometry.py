#!/usr/bin/env python3

"""
Pure tool-orientation math, shared by the arm, scan, grasp and checks.

No node or graph dependency, so all of it is unit-testable.

The precession functions tilt the tool about an insertion axis: the
end scan's coning sweep (scan/endcap.py, which explains the
construction) and INTER/BOTTOM (arm/bucket_poses.py) are both built
from them.
"""

import math

from geometry_msgs.msg import Pose, Quaternion
import numpy as np
from scipy.spatial.transform import Rotation


def tool_z_from_quaternion(q):
    """
    Return the frame's local +Z axis expressed in the parent frame.

    `q` is anything with .x/.y/.z/.w (e.g. a geometry_msgs
    Quaternion from a base_frame -> flange_link transform). This is
    the third column of the rotation matrix, written out directly
    because it is called on every Cartesian plan.

    The result is renormalised, so a slightly denormalised quaternion
    from TF still gives a unit vector. Raises RuntimeError on a
    degenerate (all-zero) quaternion rather than dividing by zero.
    """
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    zx = 2.0 * (x * z + w * y)
    zy = 2.0 * (y * z - w * x)
    zz = 1.0 - 2.0 * (x * x + y * y)

    length = math.sqrt(zx * zx + zy * zy + zz * zz)

    if length <= 1e-12:
        raise RuntimeError('Invalid flange quaternion.')

    return (zx / length, zy / length, zz / length)


def angle_between_deg(a, b):
    """Return the unsigned angle between two 3-vectors, in degrees."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    cos_angle = float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )

    return math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))


def look_at_quaternion(direction, reference_x_axis):
    """
    Build an orientation whose local +Z axis points along `direction`.

    Returned as a Quaternion (base_frame_from_link7). The remaining
    roll about that axis is disambiguated by keeping local +X as
    close as possible to `reference_x_axis` (typically the flange's
    CURRENT local +X axis in base_frame, for a minimal-twist tilt
    rather than an arbitrarily rolled one).

    Both inputs are 3-vectors in base_frame; need not be normalized
    or orthogonal to each other.
    """
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    reference_x_axis = np.asarray(reference_x_axis, dtype=np.float64)

    # Project out the component of reference_x_axis along direction,
    # leaving a valid (orthogonal) target for local +X once tilted.
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


def reference_axes(tool_z):
    """Return (z0, up, side): the insertion axis and its perpendicular basis."""
    z0 = np.asarray(tool_z, dtype=np.float64)
    z0 = z0 / np.linalg.norm(z0)

    up = np.array([0.0, 0.0, 1.0])
    up = up - (up @ z0) * z0
    up = up / np.linalg.norm(up)

    return z0, up, np.cross(z0, up)


def precession_axes(alpha_deg, phi_deg, z0, up, side):
    """Return (tool_z, boresight) for a tilt alpha in direction phi."""
    alpha = np.deg2rad(alpha_deg)
    phi = np.deg2rad(phi_deg)

    u = np.cos(phi) * up + np.sin(phi) * side

    tool_z = np.cos(alpha) * z0 + np.sin(alpha) * u
    boresight = np.sin(alpha) * z0 - np.cos(alpha) * u

    return tool_z, boresight


def precession_pose(pivot, alpha_deg, phi_deg, z0, up, side):
    """Return the flange Pose (base_frame) for a precession sample."""
    tool_z, boresight = precession_axes(alpha_deg, phi_deg, z0, up, side)

    matrix = np.stack([boresight, np.cross(tool_z, boresight), tool_z], axis=1)
    qx, qy, qz, qw = Rotation.from_matrix(matrix).as_quat()

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, pivot)
    pose.orientation.x = float(qx)
    pose.orientation.y = float(qy)
    pose.orientation.z = float(qz)
    pose.orientation.w = float(qw)

    return pose
