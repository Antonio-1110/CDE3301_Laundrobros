#!/usr/bin/env python3

"""
Synthetic laundry injection: put a known object into a real empty scan.

Real laundry scans with ground truth are slow to collect and never
cover every size and placement, so detector tuning needs a way to ask
"would THIS item, HERE, be found?" in bulk. This module answers it by
taking a real EMPTY-bucket scan and re-computing the readings of every
beam a hypothetical object would have intercepted, respecting the
sensor's cone field of view. Everything else about the scan - its
path, its gaps, its noise, its per-scan quirks - stays real.

RAYS
----
A reading is a ray, not a point, but the baseline CSVs predate ray
recording (no raw_range/ox/oy/oz columns), so the ray is rebuilt from
the scan geometry:

  - Strokes: the ToF sensor sits TOF_SENSOR_OFFSET_X = 7.75cm from
    link7's axis and looks radially outward from it (its boresight is
    link7 +X, which J7 sweeps around +Z). During the tool-Z strokes
    that axis is the INSERTION LINE through INTER's flange position
    along INTER's tool +Z, so for a point P: the foot F of P on that
    line gives the radial direction v = (P - F)/|P - F|, the origin
    O = F + 0.0775 v and the range |P - F| - 0.0775.

  - BOTTOM detour: the same construction about BOTTOM's tilted tool
    axis, for points beyond the strokes' axial reach that lie at the
    sensor's 2.8cm forward offset along that axis (measured: median
    2.5cm on the real baselines).

Both lines come from forward kinematics of the recorded poses on the
MoveIt fake controller (same URDF as the rig). Readings taken while
J1-J6 move between the two (the detour's entry/exit tilts) are
approximated by whichever hypothesis fits; they are a few percent of
a scan. When a scan HAS ray columns, they are used directly.

THE OBJECT
----------
A mound draped on the fitted bucket surface: footprint radius `a`
along the surface, height `h` inward along the surface normal, with a
half-ellipsoid profile h * sqrt(1 - (rho/a)^2). Draping on the real
curved wall (rather than a tangent plane) matters: on a 0.2m-radius
bucket the wall rises 4cm above the tangent plane 12cm from the
anchor, which would otherwise swallow a large item's edges.

THE SENSOR MODEL
----------------
The VL53L0X does not return the boresight range. It integrates the
return from its whole ~25 degree cone (config.TOF_FIELD_OF_VIEW_RAD)
and reports a single distance. With two targets in the cone - the
object and the wall behind it - the reading lands in between, pulled
toward whichever returns more signal. Two models:

  'mixed'   (default, realistic): reported range is the return-
            weighted mean distance over the cone. Each object surface
            patch contributes solid angle x beam profile (Gaussian,
            sigma = half the half-angle) x reflectivity / d^2; the rest
            of the cone sees the wall at the original reading. A
            partially covered cone shortens only partially, so thin
            or small items lose signal - which is the real failure
            mode.

  'nearest' (optimistic bound): the nearest object patch anywhere in
            the cone wins outright. Blooms every item by the cone's
            footprint (4cm across at 10cm range).

Neither captures multipath or specular fabrics; real-cloth scans
(HARDWARE_TESTS.md) remain the final word. What this CAN do is rank
detector settings against each other on identical, known inputs.

Readings that fall below TOF_MIN_RANGE_M are dropped, as the sensor
node would drop them. Beams the original scan never recorded (beyond
TOF_MAX_RANGE_M, or simply never pointed there) cannot be restored,
so an object in a region the scan path never reaches is invisible
here exactly as it would be for real - which is how scan-coverage
gaps show up in the results.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bucket_model import _axis_basis, BucketProfile, to_cylindrical
from ..config import (
    TOF_FIELD_OF_VIEW_RAD,
    TOF_MIN_RANGE_M,
    TOF_SENSOR_OFFSET_X,
    TOF_SENSOR_OFFSET_Z,
)

# Forward kinematics of the recorded poses (config.INTER, config.BOTTOM)
# on xarm7_moveit_fake: flange position and link7 +Z in link_base.
INTER_FLANGE_POSITION = np.array([0.158, -0.138, 0.474])
INTER_TOOL_Z = np.array([0.010, -1.000, -0.009])

BOTTOM_FLANGE_POSITION = np.array([0.145, -0.562, 0.462])
BOTTOM_TOOL_Z = np.array([0.001, -0.739, 0.673])

# Default scan depth (scan.pattern.DEFAULT_DEPTH_M) plus the sensor's
# forward offset, plus a margin: sensor axial positions the strokes
# can actually reach along the insertion line.
STROKE_AXIAL_REACH_M = 0.42 + TOF_SENSOR_OFFSET_Z + 0.02

# How close to the sensor's forward offset (along BOTTOM's axis) a
# point beyond stroke reach must be to be explained by the detour.
BOTTOM_AXIAL_TOLERANCE_M = 0.03

# Sensor noise added to re-computed readings, matching the pooled
# per-cell sigma measured on the real baselines (2.06mm).
DEFAULT_RANGE_NOISE_M = 0.002

# Object surface sampling pitch. Well under the cone footprint (~2cm
# at the shortest ranges), so the solid-angle sums are smooth.
DEFAULT_SAMPLE_PITCH_M = 0.003


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


@dataclass
class Rays:
    """Per-reading beam geometry, index-aligned with the scan's points."""

    origin: np.ndarray
    direction: np.ndarray
    range_m: np.ndarray
    from_bottom: np.ndarray

    @property
    def endpoint(self):
        """Return origin + range * direction, i.e. the scan points."""
        return self.origin + self.direction * self.range_m[:, None]


def _rays_about_line(points, line_point, line_dir):
    """Rebuild radial beams leaving a circle of the sensor's offset about a line."""
    line_dir = _unit(line_dir)

    axial = (points - line_point) @ line_dir
    foot = line_point + np.outer(axial, line_dir)

    radial = points - foot
    distance = np.linalg.norm(radial, axis=1)

    direction = radial / np.maximum(distance, 1e-9)[:, None]
    origin = foot + TOF_SENSOR_OFFSET_X * direction

    return origin, direction, distance - TOF_SENSOR_OFFSET_X, axial


def reconstruct_rays(points_xyz, columns=None):
    """
    Return the Rays behind a scan's points.

    columns, if given, is scan.cloud_io.load_scan_csv() output; its
    recorded ray origins are used directly when present. Otherwise
    the rays are rebuilt from the scan geometry (see module docs).
    """
    points = np.asarray(points_xyz, dtype=np.float64)

    if columns is not None and all(
        name in columns for name in ('ox', 'oy', 'oz')
    ):
        origin = np.stack(
            [columns['ox'], columns['oy'], columns['oz']], axis=1
        )

        if np.all(np.isfinite(origin)):
            offset = points - origin
            range_m = np.linalg.norm(offset, axis=1)

            return Rays(
                origin=origin,
                direction=offset / np.maximum(range_m, 1e-9)[:, None],
                range_m=range_m,
                from_bottom=np.zeros(points.shape[0], dtype=bool),
            )

    origin, direction, range_m, axial = _rays_about_line(
        points, INTER_FLANGE_POSITION, INTER_TOOL_Z
    )

    b_origin, b_direction, b_range, b_axial = _rays_about_line(
        points, BOTTOM_FLANGE_POSITION, BOTTOM_TOOL_Z
    )

    from_bottom = (axial > STROKE_AXIAL_REACH_M) & (
        np.abs(b_axial - TOF_SENSOR_OFFSET_Z) < BOTTOM_AXIAL_TOLERANCE_M
    )

    origin[from_bottom] = b_origin[from_bottom]
    direction[from_bottom] = b_direction[from_bottom]
    range_m[from_bottom] = b_range[from_bottom]

    return Rays(
        origin=origin,
        direction=direction,
        range_m=range_m,
        from_bottom=from_bottom,
    )


@dataclass
class Mound:
    """
    A synthetic laundry item draped on the bucket surface.

    anchor/normal are the footprint centre on the wall and the inward
    surface normal there; surface_points/normals sample the object's
    exposed top surface; apex is the tip, i.e. where a gripper aimed
    at "the item" should go.
    """

    anchor: np.ndarray
    normal: np.ndarray
    footprint_radius_m: float
    height_m: float
    surface_points: np.ndarray
    surface_normals: np.ndarray
    patch_area_m2: float
    on_cap: bool

    @property
    def apex(self):
        """Return the tip of the mound."""
        return self.anchor + self.height_m * self.normal


def _dome_height(rho, a, h):
    inside = np.clip(1.0 - (rho / a) ** 2, 0.0, None)
    return h * np.sqrt(inside)


def make_mound(
    profile: BucketProfile,
    footprint_radius_m,
    height_m,
    s=None,
    theta=None,
    cap_radius=None,
    cap_angle=0.0,
    pitch_m=DEFAULT_SAMPLE_PITCH_M,
):
    """
    Build a Mound on the lateral wall at (s, theta), or on the cap.

    Wall placement: s is the axial position, theta the angle around
    the axis (0 = top of the bucket, pi = floor). Cap placement: pass
    cap_radius (distance from the cap centre) and cap_angle instead;
    requires profile.has_cap.

    The footprint is laid out on a grid in the local tangent plane,
    and every grid point is then projected onto the real surface
    before being lifted by the dome height, so the item drapes over
    the wall's curvature.
    """
    cone = profile.cone
    axis_dir = cone.axis_dir
    e1, e2 = _axis_basis(axis_dir)

    a = float(footprint_radius_m)
    h = float(height_m)

    steps = np.arange(-a, a + 1e-12, pitch_m)
    gu, gv = np.meshgrid(steps, steps, indexing='ij')
    keep = gu ** 2 + gv ** 2 <= a ** 2
    gu = gu[keep]
    gv = gv[keep]
    rho = np.hypot(gu, gv)
    lift = _dome_height(rho, a, h)

    on_cap = cap_radius is not None

    if on_cap:
        if not profile.has_cap:
            raise ValueError('Profile has no closed end to place an item on.')

        s_cap = float(profile.vertices[0][0])
        centre = cone.axis_point + s_cap * axis_dir

        radial = np.cos(cap_angle) * e1 + np.sin(cap_angle) * e2
        tangent = np.cross(axis_dir, radial)

        anchor = centre + cap_radius * radial
        normal = axis_dir.copy()

        wall = anchor + np.outer(gu, radial) + np.outer(gv, tangent)
        normals = np.tile(normal, (wall.shape[0], 1))

    else:
        radial = np.cos(theta) * e1 + np.sin(theta) * e2
        anchor = cone.axis_point + s * axis_dir + cone.radius_at(s) * radial
        normal = -radial

        tangent = np.cross(axis_dir, radial)

        # Tangent-plane grid, then projected back onto the cone at
        # each grid point's own (s, theta).
        planar = anchor + np.outer(gu, axis_dir) + np.outer(gv, tangent)
        ps, ptheta, _pr = to_cylindrical(planar, cone)

        pradial = (
            np.cos(ptheta)[:, None] * e1 + np.sin(ptheta)[:, None] * e2
        )

        wall = (
            cone.axis_point
            + np.outer(ps, axis_dir)
            + cone.radius_at(ps)[:, None] * pradial
        )
        normals = -pradial

    points = wall + lift[:, None] * normals

    # Surface normal of the dome itself (for the incidence term):
    # the wall normal tilted outward by the dome's local slope,
    # |dz/drho| = (h/a)^2 * rho / z for a half-ellipsoid.
    slope = np.where(
        lift > 1e-9,
        (h / a) ** 2 * rho / np.maximum(lift, 1e-9),
        50.0,
    )

    tilt = np.arctan(np.clip(slope, 0.0, 50.0))
    outward = np.stack([gu, gv], axis=1) / np.maximum(rho, 1e-9)[:, None]

    if on_cap:
        in_plane = outward[:, :1] * radial + outward[:, 1:] * tangent
    else:
        in_plane = outward[:, :1] * axis_dir + outward[:, 1:] * tangent

    surface_normals = (
        np.cos(tilt)[:, None] * normals + np.sin(tilt)[:, None] * in_plane
    )
    surface_normals /= np.linalg.norm(surface_normals, axis=1)[:, None]

    return Mound(
        anchor=anchor,
        normal=normal,
        footprint_radius_m=a,
        height_m=h,
        surface_points=points,
        surface_normals=surface_normals,
        patch_area_m2=pitch_m ** 2,
        on_cap=on_cap,
    )


# Sub-rays the 'mixed' model splits each cone into. Equal solid angle
# each, so the beam profile alone sets their relative weight.
DEFAULT_SUB_RAYS = 64


def _cone_sub_rays(half_angle_rad, count=DEFAULT_SUB_RAYS):
    """
    Return (unit directions in the beam's own frame, angle off boresight).

    Uniform in solid angle over the cap (cos(angle) uniform, golden-
    angle azimuth), boresight along local +Z.
    """
    k = np.arange(count) + 0.5
    cos_angle = 1.0 - (k / count) * (1.0 - np.cos(half_angle_rad))
    sin_angle = np.sqrt(1.0 - cos_angle ** 2)
    azimuth = k * np.pi * (3.0 - np.sqrt(5.0))

    directions = np.stack(
        [sin_angle * np.cos(azimuth), sin_angle * np.sin(azimuth), cos_angle],
        axis=1,
    )

    return directions, np.arccos(cos_angle)


def _beam_basis(direction):
    helper = np.array([0.0, 0.0, 1.0])

    if abs(direction @ helper) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])

    b1 = _unit(np.cross(direction, helper))
    b2 = np.cross(direction, b1)

    return b1, b2


def inject(
    points_xyz,
    rays: Rays,
    mound: Mound,
    model='mixed',
    reflectivity=1.0,
    noise_m=DEFAULT_RANGE_NOISE_M,
    rng: Optional[np.random.Generator] = None,
    half_angle_rad=TOF_FIELD_OF_VIEW_RAD / 2.0,
):
    """
    Return (new_points, affected_mask) with the mound placed in the scan.

    new_points has the same row order as points_xyz minus any readings
    that fell under the sensor's minimum range (dropped, as the sensor
    node would). affected_mask marks, in the ORIGINAL indexing, which
    readings changed.

    reflectivity scales the object's return relative to the bucket
    wall (1.0 = as bright; ~0.3 is a fair stand-in for dark fabric,
    which absorbs IR). Only used by the 'mixed' model.
    """
    if model not in ('mixed', 'nearest'):
        raise ValueError(f'Unknown sensor model {model!r}.')

    if rng is None:
        rng = np.random.default_rng(0)

    points = np.asarray(points_xyz, dtype=np.float64)
    n = points.shape[0]

    new_range = rays.range_m.copy()
    affected = np.zeros(n, dtype=bool)

    # Cheap prefilter: only beams whose cone can reach the mound's
    # bounding sphere need the full per-patch treatment.
    centre = mound.anchor + 0.5 * mound.height_m * mound.normal
    radius = mound.footprint_radius_m + mound.height_m

    to_centre = centre - rays.origin
    along = np.einsum('ij,ij->i', to_centre, rays.direction)
    across = np.linalg.norm(
        to_centre - along[:, None] * rays.direction, axis=1
    )

    reach = along * np.tan(half_angle_rad) + radius
    candidates = np.flatnonzero((along > 0.0) & (across <= reach))

    local_sub_rays, sub_angle = _cone_sub_rays(half_angle_rad)

    # Gaussian emitter profile (sigma = half the half-angle): the
    # centre of the cone carries most of the return.
    sub_weight = np.exp(-0.5 * (sub_angle / (half_angle_rad / 2.0)) ** 2)

    cos_half = np.cos(half_angle_rad)

    for index in candidates:
        boresight = rays.direction[index]
        wall_range = rays.range_m[index]

        offset = mound.surface_points - rays.origin[index]
        distance = np.linalg.norm(offset, axis=1)
        unit = offset / distance[:, None]

        in_cone = unit @ boresight >= cos_half

        # Patches facing away from the sensor (the far side of the
        # dome) contribute nothing, and neither does any patch beyond
        # the wall this beam actually hit - it is behind the surface.
        facing = -np.einsum('ij,ij->i', unit, mound.surface_normals)
        in_cone &= (facing > 0.0) & (distance < wall_range + 0.005)

        if not in_cone.any():
            continue

        d = distance[in_cone]

        if model == 'nearest':
            reading = float(d.min())

        else:
            b1, b2 = _beam_basis(boresight)
            sub_rays = (
                local_sub_rays[:, :1] * b1
                + local_sub_rays[:, 1:2] * b2
                + local_sub_rays[:, 2:] * boresight
            )

            # Each in-cone patch belongs to its nearest sub-ray; each
            # sub-ray sees the nearest patch it owns, else the wall.
            owner = np.argmax(unit[in_cone] @ sub_rays.T, axis=1)

            sub_range = np.full(sub_rays.shape[0], wall_range)
            sub_reflect = np.ones(sub_rays.shape[0])

            np.minimum.at(sub_range, owner, d)

            hit = sub_range < wall_range
            sub_reflect[hit] = reflectivity

            # Return from each sub-ray: profile x reflectivity / d^2
            # (equal solid angle each, so that factor cancels).
            weight = sub_weight * sub_reflect / sub_range ** 2

            reading = float((weight * sub_range).sum() / weight.sum())

        if reading < wall_range - 1e-6:
            new_range[index] = reading + rng.normal(0.0, noise_m)
            affected[index] = True

    keep = new_range >= TOF_MIN_RANGE_M

    new_points = rays.origin + rays.direction * new_range[:, None]
    new_points[~affected] = points[~affected]

    return new_points[keep], affected
