#!/usr/bin/env python3

"""
The bucket's shape, where it is, and its own cylindrical coordinates.

The configured bucket (config.OBSTACLES and the extents of
meshes/bucket.obj) as a truncated cone, and the (s, theta, r)
coordinates every part of the package measures the bucket in:

    s     - how far along the bucket axis a point lies
    theta - where around the axis it lies, zero at the top
    r     - how far it sits from the axis

The arm (INTER/BOTTOM, grab placement), the scan and perception all
work in these, so they live here, below all of them. Fitting the
cone to real scans is perception/bucket_model.py.

Pure numpy, no ROS.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import config

# ---------------------------------------------------------------
# Seed geometry
#
# Taken from the bucket's pose in config.OBSTACLES - the same pose
# MoveIt collision-checks against - and from the extents of
# meshes/bucket.obj. That pose is hand-placed, not ground truth,
# which is precisely why it is only a SEED here. fit_cone() refines
# it against real scan data, and fit_report() reports how far it had
# to move (`laundry scene fit` turns that into a corrected pose),
# making this the first thing in the package capable of telling you
# whether the modelled pose (or the sensor extrinsics) is wrong.
# ---------------------------------------------------------------


def seed_origin() -> np.ndarray:
    """
    Return the configured bucket origin in link_base.

    The mesh origin: the centre of the bucket's CLOSED end, which is
    also where s = 0.
    """
    return np.array(config.OBSTACLES['bucket']['xyz'], dtype=np.float64)


# meshes/bucket.obj extents: a truncated cone, narrow at the closed
# end, opening toward the mouth.
MESH_RADIUS_CLOSED_M = 0.185
MESH_RADIUS_MOUTH_M = 0.24
MESH_DEPTH_M = 0.530


def seed_axis_direction() -> np.ndarray:
    """
    Return the bucket's seed axis direction in link_base.

    The bucket's axis direction in link_base: config.OBSTACLES'
    bucket rpy applied to the mesh's local +Z (its depth axis).

    Points from the closed end toward the mouth (i.e. toward the
    robot), so s increases as you come out of the bucket.
    """
    roll, pitch, yaw = config.OBSTACLES['bucket']['rpy']

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # URDF rpy is fixed-axis (extrinsic) xyz, i.e. R = Rz @ Ry @ Rx.
    # Only the third column is needed (R applied to [0, 0, 1]).
    axis = np.array(
        [
            cy * sp * cr + sy * sr,
            sy * sp * cr - cy * sr,
            cp * cr,
        ]
    )

    return axis / np.linalg.norm(axis)


def bucket_pose_from_fit(model: 'ConeModel', spec=None):
    """
    Return (xyz, rpy) placing the bucket mesh on a fitted cone.

    spec defaults to config.OBSTACLES['bucket']. The mesh origin goes
    to the fitted axis point and its depth axis (+Z) onto the fitted
    axis by the smallest rotation, so the mesh keeps its roll about
    its own axis. The axial position cannot be measured from the wall
    alone, so it stays the configured one (fit_cone moves the axis
    point only perpendicular to the seed axis).
    """
    from scipy.spatial.transform import Rotation

    spec = spec or config.OBSTACLES['bucket']

    rotation = Rotation.from_euler('xyz', spec['rpy'])
    current_axis = rotation.apply([0.0, 0.0, 1.0])

    # Smallest rotation taking the configured axis onto the fitted one.
    fitted_axis = np.asarray(model.axis_dir, dtype=np.float64)
    cross = np.cross(current_axis, fitted_axis)
    angle = np.arctan2(np.linalg.norm(cross), current_axis @ fitted_axis)

    align = Rotation.from_rotvec(
        cross / np.linalg.norm(cross) * angle
        if np.linalg.norm(cross) > 1e-12 else np.zeros(3)
    )

    rpy = (align * rotation).as_euler('xyz')

    return np.asarray(model.axis_point, dtype=np.float64), rpy


def axis_basis(axis_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the (e1, e2) basis perpendicular to the axis, e1 = "up".

    An orthonormal basis (e1, e2) spanning the plane perpendicular
    to axis_dir, used as the zero reference for theta.

    e1 is world +Z projected perpendicular to the axis - i.e. "up"
    within the bucket's cross-section - so theta = 0 always means
    the TOP of the bucket, and theta stays physically anchored even
    if a later fit nudges the axis slightly. Anchoring to an
    arbitrary basis instead would let the residual grid rotate
    between fits, silently invalidating every stored cell.

    Falls back to world +X in the degenerate case of an axis nearly
    parallel to world Z (not this bucket, which lies on its side,
    but the fallback keeps the function total).
    """
    reference = np.array([0.0, 0.0, 1.0])

    if abs(float(np.dot(reference, axis_dir))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0])

    e1 = reference - np.dot(reference, axis_dir) * axis_dir
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(axis_dir, e1)

    return e1, e2


@dataclass
class ConeModel:
    """
    A truncated cone: the empty bucket's idealised surface.

    axis_point:
        A point on the axis, taken to be the centre of the closed
        end. This is where s = 0.

    axis_dir:
        Unit vector along the axis, pointing from the closed end
        toward the mouth.

    r0, taper:
        Radius at s = 0 and its rate of change with s, so
        radius_at(s) = r0 + taper * s. taper > 0 means the bucket
        widens toward the mouth, as this one does.

    s_min, s_max:
        Axial extent actually covered by the data the model was fit
        to. Used as a geometric gate in detection - points outside
        it are not judged against an extrapolated surface.
    """

    axis_point: np.ndarray
    axis_dir: np.ndarray
    r0: float
    taper: float
    s_min: float
    s_max: float

    def radius_at(self, s):
        """Model radius at axial position s (metres)."""
        return self.r0 + self.taper * np.asarray(s, dtype=np.float64)

    @property
    def half_angle_rad(self) -> float:
        """Cone half-angle; 0 would be a perfect cylinder."""
        return float(np.arctan(self.taper))


def seed_cone() -> ConeModel:
    """
    Return the starting guess for fit_cone(), from the config and the mesh.

    This is a strong seed - it is already within centimetres and a
    couple of degrees of the truth - which is why fit_cone() can use
    a plain robust least-squares refinement instead of needing
    RANSAC to find the cone from scratch.
    """
    axis_dir = seed_axis_direction()

    taper = (MESH_RADIUS_MOUTH_M - MESH_RADIUS_CLOSED_M) / MESH_DEPTH_M

    return ConeModel(
        axis_point=seed_origin(),
        axis_dir=axis_dir,
        r0=MESH_RADIUS_CLOSED_M,
        taper=taper,
        s_min=0.0,
        s_max=MESH_DEPTH_M,
    )


def to_cylindrical(
    points_xyz: np.ndarray,
    model: ConeModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Express points in the bucket's own cylindrical coordinates.

    Returns (s, theta, r):

        s     - signed distance along the axis from axis_point
        theta - angle around the axis in [0, 2*pi), zero at the top
        r     - perpendicular distance from the axis

    This is the whole point of the module: unlike z over (x, y),
    r over (s, theta) is single-valued everywhere on the bucket.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if points_xyz.size == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    offsets = points_xyz - model.axis_point

    s = offsets @ model.axis_dir

    radial = offsets - np.outer(s, model.axis_dir)

    r = np.linalg.norm(radial, axis=1)

    e1, e2 = axis_basis(model.axis_dir)

    theta = np.arctan2(radial @ e2, radial @ e1)
    theta = np.mod(theta, 2.0 * np.pi)

    return s, theta, r


def surface_residual(
    points_xyz: np.ndarray,
    model: ConeModel,
) -> np.ndarray:
    """
    Return each point's signed perpendicular distance to the cone.

    Signed perpendicular distance from each point to the cone
    surface. Positive means outside the cone (further from the axis
    than the model wall), negative means inside it.

    The cos(half_angle) factor converts the purely radial
    difference into a true perpendicular distance. It is only a
    0.5% correction on this bucket, but it keeps the fit residual
    an honest metric distance rather than a radial one.
    """
    s, _theta, r = to_cylindrical(points_xyz, model)

    return (r - model.radius_at(s)) * np.cos(model.half_angle_rad)
