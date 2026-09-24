#!/usr/bin/env python3

"""
bucket_model.py

A fitted geometric model of the empty bucket, plus the learned
residual/noise field that turns it into a detection reference.

WHY THIS EXISTS
---------------
laundry_detect.py used to diff a scan against a single baseline
scan by asking "is this point higher in z than the tallest baseline
point nearby?". That question is ill-posed on this bucket: it lies
on its SIDE (axis ~[0.00, 0.996, 0.089] in link_base), so its
cross-sections are circles in the XZ plane and every (x, y) carries
TWO valid z values - the lower wall and the upper wall. Near the
silhouette the surface is vertical, so z as a function of (x, y)
has infinite slope. That is the root cause of the 13cm-of-z-inside-
one-3cm-cell problem documented in laundry_detect.py, and it is
also why the old test was one-sided in z and therefore only ever
valid on the bucket FLOOR: laundry resting against a side wall
displaces x, not z, and laundry on the ceiling lowers z.

In the bucket's OWN cylindrical coordinates the problem disappears.
Parameterise a point by

    s     - how far along the bucket axis it lies
    theta - where around the axis it lies
    r     - how far it sits from the axis

and the empty bucket's surface is r = f(s, theta), which is
single-valued EVERYWHERE, with no degenerate regions. Laundry can
only ever intercept the beam before it reaches the wall, so the
detection test becomes

    intrusion = r_expected(s, theta) - r_measured > 0

which is one-sided by construction and equally valid on the floor,
the walls and the ceiling.

WHY THE MODEL IS A HYBRID
-------------------------
r_expected is NOT the bare fitted cone. It is

    fitted cone  +  mean residual measured over N empty scans

because the VL53L0X does not measure the geometric surface. It has
a ~25 degree cone field of view (tof_sensor.py), so its reading is
roughly the nearest strong reflector anywhere in that cone, not the
range along the boresight - on an obliquely-viewed curved wall it
reads systematically SHORT by an amount that depends on incidence
angle. The mounting extrinsics are also ruler-measured with zero
rotation (tof_sensor.py), so any boresight error shows up as a
theta-dependent distortion.

Both of those are deterministic given a fixed scan trajectory, and
the detection scan always runs the same path as the baselines, so
they appear identically in both and cancel. The learned residual
field is what captures them. A bare parametric cone would leave
them in the signal instead, where they look exactly like laundry.

What the cone buys us on top of the old CSV diff is DENSE, gap-free
coverage: the old code assigned -inf to any candidate point with no
baseline neighbour within 3cm and could never flag it, which is a
real blind spot given the scan path is known to leave gaps.

WHAT MULTIPLE BASELINES DO AND DO NOT BUY
-----------------------------------------
Averaging N baselines removes ZERO-MEAN noise only. Extrinsic
error, incidence-angle bias and motion skew are identical in every
baseline and survive averaging completely untouched. So do not read
a shrinking spread as licence to lower the detection threshold.

The real prize from multiple baselines is the per-cell SIGMA map:
it makes the threshold a calibrated statistic (k sigma, with a
measurable false-positive rate) instead of the single global 1.5cm
constant that was tuned on one empty-vs-empty pair.

This module has no ROS dependency - it is pure numpy/scipy over
already-loaded point arrays, so it can be unit-tested and used
offline without a running ROS system.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

# ---------------------------------------------------------------
# Seed geometry
#
# Taken from the fixed base_to_bucket_joint in
# xarm_description/urdf/xarm7/xarm7.urdf.xacro and from the extents
# of meshes/bucket.obj. Both READMEs caveat that this pose is
# approximate and not ground truth - which is precisely why it is
# only a SEED here. fit_cone() refines it against real scan data,
# and fit_report() reports how far it had to move, making this the
# first thing in the package capable of telling you whether the
# URDF pose (or the sensor extrinsics) is wrong.
# ---------------------------------------------------------------

# base_to_bucket_joint origin xyz - the centre of the bucket's
# CLOSED end, which is also where s = 0.
URDF_BUCKET_ORIGIN = np.array([0.163, -0.72, 0.42])

# base_to_bucket_joint origin rpy, applied to the mesh's local +Z
# (the bucket's depth axis) to get the axis direction in link_base.
URDF_BUCKET_RPY = (1.66, 3.14, -3.14)

# meshes/bucket.obj extents: a truncated cone, narrow at the closed
# end, opening toward the mouth.
MESH_RADIUS_CLOSED_M = 0.185
MESH_RADIUS_MOUTH_M = 0.24
MESH_DEPTH_M = 0.530

# Huber transition for fit_cone(). Residuals under this are treated
# quadratically, beyond it linearly. Set just above the sensor's
# own per-point noise so genuine surface scatter is fitted normally
# while stray returns (a glimpse of the gripper, a specular flyer)
# cannot drag the whole model.
DEFAULT_HUBER_SCALE_M = 0.01

# Grid resolution for the offset/sigma field, in arc length along
# the bucket profile.
#
# Measured on real hardware (8 baselines, 24 Sep 2026): a default
# scan_move run is ~50s at 20Hz, so about 1000 points per scan and
# 8000 pooled. On this grid that gives a median of 16 points per
# occupied cell and a 2.06mm per-cell sigma, which is the sensor's
# genuine noise floor - so the resolution is right.
#
# What it does NOT fix is coverage: the scan path only touches
# about 30% of the cells, and because the trajectory is
# deterministic (Pilz PTP, chosen for repeatability) extra
# baselines deepen the same cells rather than reaching new ones -
# coverage went 19.8% for one scan to 29.5% for eight, with the
# eighth adding 0.8%. Coarser bins trade that away badly: 6cm x
# 30deg lifts confident cells 22% -> 48% but inflates sigma
# 2.06 -> 3.03mm, pushing the smallest detectable intrusion from
# 8.2mm to 12.1mm. Empty cells are handled by neighbour-fill and an
# inflated sigma instead, which costs nothing when they are right.
#
# If you want real coverage, make the scan denser (smaller --step,
# lower --velocity), not the bins coarser.
#
# Always check the occupancy histogram against new hardware -
# build_baseline_surface() returns it for exactly that purpose.
DEFAULT_ARC_BIN_M = 0.02
DEFAULT_ANGULAR_BIN_RAD = np.deg2rad(10.0)

# A cell needs at least this many samples before its OWN standard
# deviation is trusted; below it the cell falls back to the pooled
# sigma and is marked low-confidence.
DEFAULT_MIN_CELL_SAMPLES = 5

# Empty cells (no baseline coverage at all) get the pooled sigma
# multiplied by this. They are not excluded - excluding them is
# what created the old -inf blind spot - but they are made harder
# to trip and are always flagged low-confidence.
EMPTY_CELL_SIGMA_INFLATION = 1.5


def seed_axis_direction() -> np.ndarray:
    """
    The bucket's axis direction in link_base, from the URDF joint's
    rpy applied to the mesh's local +Z.

    Points from the closed end toward the mouth (i.e. toward the
    robot), so s increases as you come out of the bucket.
    """

    roll, pitch, yaw = URDF_BUCKET_RPY

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


def _axis_basis(axis_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
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
    The starting guess for fit_cone(), straight from the URDF joint
    and the mesh extents.

    This is a strong seed - it is already within centimetres and a
    couple of degrees of the truth - which is why fit_cone() can use
    a plain robust least-squares refinement instead of needing
    RANSAC to find the cone from scratch.
    """

    axis_dir = seed_axis_direction()

    taper = (MESH_RADIUS_MOUTH_M - MESH_RADIUS_CLOSED_M) / MESH_DEPTH_M

    return ConeModel(
        axis_point=URDF_BUCKET_ORIGIN.copy(),
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

    e1, e2 = _axis_basis(model.axis_dir)

    theta = np.arctan2(radial @ e2, radial @ e1)
    theta = np.mod(theta, 2.0 * np.pi)

    return s, theta, r


def surface_residual(
    points_xyz: np.ndarray,
    model: ConeModel,
) -> np.ndarray:
    """
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


def _unpack_params(params, seed: ConeModel) -> ConeModel:
    """
    Turn the 6 free fit parameters into a ConeModel.

    The parameterisation is deliberately relative to the seed:

      - The axis point moves only PERPENDICULAR to the seed axis
        (2 DOF, via the seed's own e1/e2). Allowing it to slide
        along the axis as well would be a pure gauge freedom - the
        same infinite line with a different anchor - which would
        leave the Jacobian rank-deficient and s = 0 meaningless.

      - The axis direction is perturbed in the same two directions
        and renormalised (2 DOF), rather than fitting a free
        3-vector, for the same reason: a free 3-vector carries a
        redundant length DOF.

      - r0 and taper are absolute (2 DOF).
    """

    a, b, c, d, r0, taper = params

    e1, e2 = _axis_basis(seed.axis_dir)

    axis_dir = seed.axis_dir + c * e1 + d * e2
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    axis_point = seed.axis_point + a * e1 + b * e2

    return ConeModel(
        axis_point=axis_point,
        axis_dir=axis_dir,
        r0=float(r0),
        taper=float(taper),
        s_min=seed.s_min,
        s_max=seed.s_max,
    )


def fit_cone(
    points_xyz: np.ndarray,
    seed: Optional[ConeModel] = None,
    huber_scale_m: float = DEFAULT_HUBER_SCALE_M,
    extent_percentile: float = 1.0,
) -> ConeModel:
    """
    Robustly fit a truncated cone to empty-bucket scan points.

    Uses a Huber loss rather than plain least squares: the
    baselines are nominally empty, but a stray return off the
    gripper or a specular flyer would otherwise pull the whole
    surface toward itself, and a distorted surface is far more
    damaging than a few ignored points - it biases EVERY subsequent
    detection, not just the region it came from.

    s_min/s_max are taken from the data's own axial extent (trimmed
    by extent_percentile at each end to shrug off flyers), not from
    the mesh, because they are used downstream as a gate on where
    the model may be trusted - and that is bounded by where the
    scan actually reached, which is less than the full 0.53m depth
    (the end effector blocks the far end; see scan_move.py's BOTTOM
    detour).
    """

    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(
            f"points_xyz must be (N, 3); got {points_xyz.shape}."
        )

    if points_xyz.shape[0] < 6:
        raise ValueError(
            f"Need at least 6 points to fit 6 cone parameters; "
            f"got {points_xyz.shape[0]}."
        )

    if seed is None:
        seed = seed_cone()

    def residuals(params):
        return surface_residual(points_xyz, _unpack_params(params, seed))

    initial = np.array([0.0, 0.0, 0.0, 0.0, seed.r0, seed.taper])

    solution = least_squares(
        residuals,
        initial,
        loss="huber",
        f_scale=huber_scale_m,
    )

    model = _unpack_params(solution.x, seed)

    s, _theta, _r = to_cylindrical(points_xyz, model)

    model.s_min = float(np.percentile(s, extent_percentile))
    model.s_max = float(np.percentile(s, 100.0 - extent_percentile))

    return model


def fit_report(model: ConeModel, seed: Optional[ConeModel] = None) -> str:
    """
    Human-readable comparison of a fitted cone against the seed.

    A fit that had to move the axis by centimetres, or that
    disagrees with the mesh radii, means either the URDF bucket
    pose is wrong or the ToF mounting extrinsics are. Print it and
    look at it; do not skip to detection on an unexamined fit.

    IMPORTANT - what this report CANNOT tell you apart:

        A lateral sensor-extrinsic/boresight error produces a
        radial bias proportional to cos(theta), and a cos(theta)
        radial bias is EXACTLY a lateral displacement of the axis.
        It is perfectly degenerate with axis_point. Verified on
        synthetic data: injecting a 4mm cos(theta) bias moves the
        fitted axis 4.01mm and leaves nothing at all in the
        residual field, while a cos(2theta) bias of the same size
        moves the axis 0.03mm and lands in the residual field in
        full.

        So a displaced axis here means "the bucket pose and/or the
        sensor extrinsics are off by this much, combined" - this
        report cannot attribute it. Separating the two needs the
        residual-vs-joint-7 analysis (Phase 6), because an
        extrinsic error tracks J7 in the sensor's own rotating
        frame while a bucket-pose error does not.

        For DETECTION this degeneracy is harmless: the fitted model
        matches what the sensor actually reports along the fixed
        scan trajectory, which is all the detector needs. It only
        matters when you want to trust axis_point as the bucket's
        true physical pose.
    """

    if seed is None:
        seed = seed_cone()

    axis_shift = np.linalg.norm(model.axis_point - seed.axis_point)

    cos_angle = float(
        np.clip(np.dot(model.axis_dir, seed.axis_dir), -1.0, 1.0)
    )
    axis_tilt_deg = np.degrees(np.arccos(cos_angle))

    r_mouth = float(model.radius_at(model.s_max))
    r_closed = float(model.radius_at(model.s_min))

    return (
        "Cone fit vs. URDF/mesh seed:\n"
        f"  axis point   : {np.round(model.axis_point, 4)} "
        f"(moved {axis_shift * 100:.2f} cm)\n"
        f"  axis dir     : {np.round(model.axis_dir, 4)} "
        f"(tilted {axis_tilt_deg:.2f} deg)\n"
        f"  r0           : {model.r0:.4f} m "
        f"(seed {seed.r0:.4f} m)\n"
        f"  taper        : {model.taper:.4f} "
        f"(seed {seed.taper:.4f}, half-angle "
        f"{np.degrees(model.half_angle_rad):.2f} deg)\n"
        f"  wall extent  : s = {model.s_min:.4f} .. {model.s_max:.4f} m "
        f"(mesh depth {MESH_DEPTH_M:.3f} m)\n"
        f"  radius range : {r_closed:.4f} .. {r_mouth:.4f} m "
        f"(mesh {MESH_RADIUS_CLOSED_M:.3f} .. "
        f"{MESH_RADIUS_MOUTH_M:.3f} m)"
    )


def _robust_sigma(values: np.ndarray) -> float:
    """
    Median-absolute-deviation estimate of the standard deviation,
    scaled to match sigma for a Gaussian.

    Used for the pooled fallback sigma rather than np.std, because
    that pooled value is what thinly-sampled and empty cells rely
    on - so it must not itself be inflated by the handful of
    outliers robust fitting already decided to discount.
    """

    if values.size == 0:
        return 0.0

    mad = float(np.median(np.abs(values - np.median(values))))

    return 1.4826 * mad


# A point must be at least this far inside the wall radius before
# it is considered "not on the wall" when looking for the closed
# end. Comfortably above the sensor's own scatter, so ordinary wall
# noise is never mistaken for cap geometry.
DEFAULT_CAP_GAP_M = 0.02

# ...and there must be at least this many such points, concentrated
# in s, before a cap is modelled at all. A scan that never reached
# the closed end should get a wall-only profile rather than a cap
# invented out of a handful of stray returns.
DEFAULT_MIN_CAP_POINTS = 50
DEFAULT_MAX_CAP_SPREAD_M = 0.03

# Band either side of the cap/wall corner excluded from BOTH the
# cone fit and the cap-position estimate. Points in it genuinely
# belong to neither primitive cleanly, and forcing them into one
# biases that one.
DEFAULT_CORNER_MARGIN_M = 0.02


@dataclass
class BucketProfile:
    """
    The bucket as its MERIDIAN PROFILE: a polyline in (s, r) that is
    swept around the axis.

    WHY NOT JUST THE CONE
    ---------------------
    The bucket's closed end is a flat disc, not part of the cone's
    lateral surface. In (s, theta) that disc collapses into a single
    axial row carrying every radius from 0 to R at once - which is
    the same "surface is not single-valued in these coordinates"
    failure that made z over (x, y) unusable, just relocated.

    Measured consequences of ignoring it: sigma in the closed-end
    rows inflates about 20-fold (2mm -> 20-45mm) and bleeds into the
    neighbouring wall rows, so a 2.5cm item within ~5cm of the
    closed end is only partly flagged or missed outright; and the
    disc's points drag the cone fit itself (r0 low by 9mm, taper
    0.104 -> 0.137 when a fifth of the points are cap). The closed
    end is where laundry collects in a bucket lying on its side, so
    that is the worst possible place to be weakest.

    THE PROFILE FIXES BOTH
    ----------------------
    Every point is described by

        u     - arc length along the profile, from the CENTRE of the
                closed end, outward across the cap, around the
                corner, then along the wall toward the mouth
        theta - angle around the axis, as before

    and the surface is a single curve in that coordinate, so it is
    single-valued across the cap, the corner AND the wall. Intrusion
    becomes the perpendicular distance to the profile, signed by
    whether the point lies inside the bucket volume - one definition
    that works everywhere, with no special cases at the corner.

    vertices are (s, r) pairs, ordered by increasing u:

        (s_cap, 0)      centre of the closed end        u = 0
        (s_cap, r_cap)  corner where cap meets wall     u = r_cap
        (s_max, r_max)  the mouth                       u = r_cap + slant

    A wall-only profile (no closed end reached by the scan) simply
    omits the first vertex.
    """

    cone: ConeModel
    vertices: np.ndarray
    has_cap: bool

    @property
    def u_breaks(self) -> np.ndarray:
        """Cumulative arc length at each vertex."""

        segments = np.linalg.norm(np.diff(self.vertices, axis=0), axis=1)

        return np.concatenate([[0.0], np.cumsum(segments)])

    @property
    def u_max(self) -> float:
        return float(self.u_breaks[-1])

    def radius_at_u(self, u):
        """
        Radius of the swept surface at arc position u.

        This is the local circumference factor: it is what converts
        an angular bin into a real area, and it correctly goes to
        zero at the centre of the closed end, where a whole ring of
        theta collapses to a single point.
        """

        return np.interp(
            np.asarray(u, dtype=np.float64),
            self.u_breaks,
            self.vertices[:, 1],
        )

    def project(self, s, r):
        """
        Project points onto the profile.

        Returns (u, intrusion):

            u         - arc length of the nearest point on the
                        profile
            intrusion - perpendicular distance to the profile,
                        POSITIVE when the point lies inside the
                        bucket volume, i.e. the beam stopped short
                        of the wall, which is the only thing a real
                        object can cause

        The sign comes from an explicit inside test rather than from
        which side of a segment the point falls on, so it stays
        correct in the corner region where the nearest segment is
        ambiguous.

        KNOWN BEHAVIOUR IN THE CORNER: the bucket's interior is
        convex, so distance-to-the-nearest-surface is the right
        definition of "how far inside" - but it means a point
        tucked into the concave corner reads SHORT. Something 2.5cm
        clear of the wall but only 1cm clear of the closed end
        scores 1cm, because 1cm is genuinely all the room there is.
        Measured effect: an item straddling the corner has roughly
        70% of its points flagged rather than 100%, while items a
        few centimetres clear are flagged in full. The cluster is
        still found comfortably; this only trims its edges.
        """

        s = np.asarray(s, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)

        point = np.stack([s, r], axis=1)

        breaks = self.u_breaks

        best_distance = np.full(s.shape, np.inf)
        best_u = np.zeros(s.shape)

        for index in range(len(self.vertices) - 1):

            start = self.vertices[index]
            end = self.vertices[index + 1]

            edge = end - start
            length_sq = float(edge @ edge)

            if length_sq == 0.0:
                continue

            t = ((point - start) @ edge) / length_sq
            t = np.clip(t, 0.0, 1.0)

            closest = start + t[:, None] * edge

            distance = np.linalg.norm(point - closest, axis=1)

            closer = distance < best_distance

            best_distance = np.where(closer, distance, best_distance)
            best_u = np.where(
                closer, breaks[index] + t * np.sqrt(length_sq), best_u
            )

        inside = r < self.cone.radius_at(s)

        if self.has_cap:
            inside = inside & (s > float(self.vertices[0][0]))

        return best_u, np.where(inside, best_distance, -best_distance)


def fit_bucket_profile(
    points_xyz: np.ndarray,
    cone: Optional[ConeModel] = None,
    cap_gap_m: float = DEFAULT_CAP_GAP_M,
    min_cap_points: int = DEFAULT_MIN_CAP_POINTS,
    max_cap_spread_m: float = DEFAULT_MAX_CAP_SPREAD_M,
    corner_margin_m: float = DEFAULT_CORNER_MARGIN_M,
) -> BucketProfile:
    """
    Fit the full bucket profile: the lateral cone, plus the flat
    closed end if the scan actually reached it.

    The cone is re-fitted on WALL points only. Cap points sit far
    inside the wall radius, so to a lateral-surface fit they are
    gross outliers; the Huber loss limits the damage but does not
    remove it, and a distorted cone biases every later detection
    across the whole bucket rather than only near the cap.

    A cap is only modelled when there is real evidence for one: a
    decent number of points well inside the wall radius AND tightly
    concentrated in s. Otherwise the profile is wall-only, which is
    the right answer for a scan that stopped short of the closed
    end.
    """

    points_xyz = np.asarray(points_xyz, dtype=np.float64)

    if cone is None:
        cone = fit_cone(points_xyz)

    s, _theta, r = to_cylindrical(points_xyz, cone)

    off_wall = (cone.radius_at(s) - r) > cap_gap_m

    has_cap = False
    s_cap = cone.s_min

    if int(off_wall.sum()) >= min_cap_points:

        candidate_s = s[off_wall]
        spread = _robust_sigma(candidate_s)

        if spread <= max_cap_spread_m:
            has_cap = True
            s_cap = float(np.median(candidate_s))

    if has_cap:

        # Refine by PROJECTION, not by the radial gap that found the
        # cap in the first place. That gap test cannot classify the
        # corner: points within cap_gap_m of the rim are only
        # shallowly inside the wall radius, so they read as wall
        # points and keep dragging the lateral fit. On this bucket
        # that is about a fifth of the disc's area - measured at 20%
        # cap points it left r0 out by 2.2mm even after one refit.
        #
        # Projecting onto a provisional profile asks the right
        # question (which surface is this point actually nearest?),
        # and a margin either side of the corner drops the genuinely
        # ambiguous band from BOTH fits rather than forcing it into
        # one.
        for _iteration in range(2):

            r_cap = float(cone.radius_at(s_cap))

            provisional = BucketProfile(
                cone=cone,
                vertices=np.array(
                    [
                        [s_cap, 0.0],
                        [s_cap, r_cap],
                        [cone.s_max, float(cone.radius_at(cone.s_max))],
                    ]
                ),
                has_cap=True,
            )

            u, _intrusion = provisional.project(s, r)
            u_corner = float(provisional.u_breaks[1])

            is_wall = u > u_corner + corner_margin_m
            is_cap = u < u_corner - corner_margin_m

            if int(is_wall.sum()) >= 6:
                cone = fit_cone(points_xyz[is_wall], seed=cone)
                s, _theta, r = to_cylindrical(points_xyz, cone)

            if int(is_cap.sum()) >= min_cap_points:
                s_cap = float(np.median(s[is_cap]))

        # The wall model should start at the corner, not inside the
        # cap, so the wall grid never sees cap points at all.
        cone.s_min = max(cone.s_min, s_cap)

        r_cap = float(cone.radius_at(s_cap))

        vertices = np.array(
            [
                [s_cap, 0.0],
                [s_cap, r_cap],
                [cone.s_max, float(cone.radius_at(cone.s_max))],
            ]
        )

    else:

        vertices = np.array(
            [
                [cone.s_min, float(cone.radius_at(cone.s_min))],
                [cone.s_max, float(cone.radius_at(cone.s_max))],
            ]
        )

    return BucketProfile(cone=cone, vertices=vertices, has_cap=has_cap)


@dataclass
class BaselineSurface:
    """
    The full detection reference: a fitted bucket profile plus
    per-cell learned offset and noise, on a (u, theta) grid.

    u is arc length along the bucket's meridian profile (see
    BucketProfile), not axial position, so the grid covers the flat
    closed end, the corner and the lateral wall in one continuous
    chart with no degenerate region.

    offset_mean[i, j]:
        Mean signed distance from the bare profile in that cell,
        over all baseline scans, positive meaning "inside the
        bucket". Captures the ToF's incidence-angle bias and any
        extrinsic error - deterministic effects that the fixed scan
        trajectory reproduces identically in the detection scan.

    offset_sigma[i, j]:
        Spread of that distance, i.e. the local noise floor. This
        is what makes the detection threshold a calibrated
        statistic instead of a global constant.

    count[i, j]:
        Number of baseline points that landed in the cell. Kept so
        occupancy can be audited rather than assumed.

    confident[i, j]:
        False where the cell was empty or too thinly sampled for
        its own sigma. Such cells are NOT excluded from detection -
        excluding them is exactly the -inf blind spot the old
        implementation had - but they are harder to trip and their
        verdicts are reported as low-confidence.
    """

    profile: BucketProfile
    offset_mean: np.ndarray
    offset_sigma: np.ndarray
    count: np.ndarray
    confident: np.ndarray
    arc_bin_m: float
    angular_bin_rad: float
    pooled_sigma_m: float
    min_cell_samples: int

    @property
    def cone(self) -> ConeModel:
        """The lateral wall model, for callers that only need it."""
        return self.profile.cone

    @property
    def n_arc(self) -> int:
        return self.offset_mean.shape[0]

    @property
    def n_angular(self) -> int:
        return self.offset_mean.shape[1]

    def cell_indices(self, u, theta):
        """
        Map (u, theta) to grid indices.

        u is clipped into range: a point just past the profile's
        ends should be judged against the nearest modelled cell
        rather than crash or silently vanish. Detection applies its
        own explicit bounds gate before trusting the result.

        theta wraps, since it is periodic by nature.
        """

        u = np.asarray(u, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)

        i = np.floor(u / self.arc_bin_m).astype(int)
        i = np.clip(i, 0, self.n_arc - 1)

        j = np.floor(np.mod(theta, 2.0 * np.pi) / self.angular_bin_rad)
        j = np.mod(j.astype(int), self.n_angular)

        return i, j

    def expected_offset(self, u, theta):
        """
        The expected signed distance from the bare profile, its
        local noise sigma, and whether the cell is well-sampled.

        Returns (offset, sigma, confident), all aligned with the
        inputs.
        """

        i, j = self.cell_indices(u, theta)

        return (
            self.offset_mean[i, j],
            self.offset_sigma[i, j],
            self.confident[i, j],
        )

    def cell_area(self):
        """
        Area of each grid cell on the bucket surface, as an
        (n_arc, 1) column broadcastable over theta.

        The angular extent of a cell is arc_bin_m by
        angular_bin_rad * radius_at_u, and that radius correctly
        shrinks to zero at the centre of the closed end - where a
        whole ring of theta really does collapse to a point, so
        those cells genuinely carry almost no area.
        """

        u_centre = (np.arange(self.n_arc) + 0.5) * self.arc_bin_m

        return (
            self.arc_bin_m
            * self.angular_bin_rad
            * self.profile.radius_at_u(u_centre)
        )[:, None]


def _fill_empty_cells(
    offset_mean: np.ndarray,
    count: np.ndarray,
) -> np.ndarray:
    """
    Give empty cells the mean offset of their occupied immediate
    neighbours, wrapping in theta.

    A cell with no baseline coverage still has to be judged - the
    alternative is the old -inf blind spot, where a gap in the scan
    path became a region laundry could never be detected in. Its
    immediate neighbours are a much better estimate of the local
    sensor bias than the bare profile is, since that bias varies
    smoothly with incidence angle.

    Cells with no occupied neighbour either fall back to the bare
    profile (offset 0). Every filled cell is marked low-confidence
    by the caller regardless of which branch it took.
    """

    occupied = count > 0

    if occupied.all() or not occupied.any():
        return offset_mean

    weighted_sum = np.zeros_like(offset_mean)
    weight = np.zeros_like(offset_mean)

    masked = np.where(occupied, offset_mean, 0.0)

    for dt in (-1, 0, 1):

        # theta wraps around the bucket; u does not, so its shifts
        # are taken as slices rather than rolls.
        rolled_values = np.roll(masked, dt, axis=1)
        rolled_occupied = np.roll(occupied, dt, axis=1)

        for ds in (-1, 0, 1):

            if ds == 0:
                shifted_values = rolled_values
                shifted_occupied = rolled_occupied

            else:
                shifted_values = np.zeros_like(rolled_values)
                shifted_occupied = np.zeros_like(rolled_occupied)

                if ds == -1:
                    shifted_values[:-1] = rolled_values[1:]
                    shifted_occupied[:-1] = rolled_occupied[1:]
                else:
                    shifted_values[1:] = rolled_values[:-1]
                    shifted_occupied[1:] = rolled_occupied[:-1]

            weighted_sum += shifted_values
            weight += shifted_occupied

    filled = offset_mean.copy()

    fillable = (~occupied) & (weight > 0)
    filled[fillable] = weighted_sum[fillable] / weight[fillable]

    return filled


def build_baseline_surface(
    baseline_scans: Sequence[np.ndarray],
    profile: Optional[BucketProfile] = None,
    cone: Optional[ConeModel] = None,
    arc_bin_m: float = DEFAULT_ARC_BIN_M,
    angular_bin_rad: float = DEFAULT_ANGULAR_BIN_RAD,
    min_cell_samples: int = DEFAULT_MIN_CELL_SAMPLES,
) -> BaselineSurface:
    """
    Build the detection reference from several empty-bucket scans.

    baseline_scans:
        One (N_i, 3) array per empty-bucket scan, all in base_frame.
        Pass 8-10 of them: fewer leaves the per-cell sigma too
        thinly sampled to be worth having, which is the whole
        reason for taking multiple baselines in the first place.

    profile:
        A pre-fitted BucketProfile. If omitted, one is fitted here
        from all the scans pooled together. Pass it explicitly when
        doing leave-one-out validation, so the held-out scan cannot
        influence the model it is being judged against.

    cone:
        Legacy convenience - a pre-fitted lateral cone to seed the
        profile fit from. Ignored when `profile` is given.

    Check the occupancy report (see occupancy_summary) before
    trusting the result: if the median cell count is much under
    min_cell_samples, widen the bins rather than relying on sigma
    estimated from two or three points.
    """

    if len(baseline_scans) == 0:
        raise ValueError("Need at least one baseline scan.")

    pooled_points = np.concatenate(
        [np.asarray(scan, dtype=np.float64) for scan in baseline_scans],
        axis=0,
    )

    if profile is None:
        profile = fit_bucket_profile(pooled_points, cone=cone)

    s, theta, r = to_cylindrical(pooled_points, profile.cone)

    u, offset = profile.project(s, r)

    n_arc = max(1, int(np.ceil(profile.u_max / arc_bin_m)))
    n_angular = max(1, int(np.round(2.0 * np.pi / angular_bin_rad)))

    # Only points inside the profile's own extent contribute to the
    # model. Beyond it the surface is extrapolation, so letting such
    # points define cells would bake extrapolation error into the
    # reference itself.
    inside = (s <= profile.cone.s_max) & (u < profile.u_max)

    i = np.clip(np.floor(u[inside] / arc_bin_m).astype(int), 0, n_arc - 1)
    j = np.mod(
        np.floor(theta[inside] / angular_bin_rad).astype(int),
        n_angular,
    )

    flat = i * n_angular + j
    n_cells = n_arc * n_angular

    count = np.bincount(flat, minlength=n_cells).astype(np.float64)

    total = np.bincount(flat, weights=offset[inside], minlength=n_cells)
    total_sq = np.bincount(
        flat, weights=offset[inside] ** 2, minlength=n_cells
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), 0.0)
        variance = np.where(
            count > 1,
            total_sq / np.maximum(count, 1) - mean**2,
            0.0,
        )

    # Floating-point cancellation in the sum-of-squares form can
    # push a genuinely-zero variance slightly negative.
    variance = np.maximum(variance, 0.0)

    count = count.reshape(n_arc, n_angular)
    offset_mean = mean.reshape(n_arc, n_angular)
    cell_sigma = np.sqrt(variance).reshape(n_arc, n_angular)

    # Spread about each cell's OWN mean, not about the bare profile.
    # The raw offset still carries the deterministic incidence/
    # extrinsic bias that offset_mean exists to absorb, so pooling
    # it directly would fold that bias into the noise estimate -
    # inflating the fallback sigma used by exactly the empty and
    # thinly-sampled cells that most need an honest one. Measured on
    # synthetic data with 2mm noise and a 3mm bias, the raw form
    # reported 3.1mm.
    pooled_sigma = _robust_sigma(offset[inside] - mean[flat])

    well_sampled = count >= min_cell_samples

    offset_sigma = np.where(well_sampled, cell_sigma, pooled_sigma)
    offset_sigma = np.where(
        count > 0,
        offset_sigma,
        pooled_sigma * EMPTY_CELL_SIGMA_INFLATION,
    )

    offset_mean = _fill_empty_cells(offset_mean, count)

    return BaselineSurface(
        profile=profile,
        offset_mean=offset_mean,
        offset_sigma=offset_sigma,
        count=count,
        confident=well_sampled,
        arc_bin_m=arc_bin_m,
        angular_bin_rad=angular_bin_rad,
        pooled_sigma_m=float(pooled_sigma),
        min_cell_samples=min_cell_samples,
    )


def occupancy_summary(surface: BaselineSurface) -> str:
    """
    Report how well the (s, theta) grid is actually covered.

    Per the plan's Phase 2 verification step, this must be looked at
    before per-cell sigma is trusted: a median cell count much under
    min_cell_samples means the sigma map is mostly the pooled
    fallback wearing a per-cell costume, and the bins should be
    widened instead.
    """

    counts = surface.count.ravel()
    occupied = counts[counts > 0]

    empty_pct = 100.0 * (counts.size - occupied.size) / counts.size
    confident_pct = 100.0 * surface.confident.sum() / surface.confident.size

    if occupied.size == 0:
        return "Occupancy: grid is entirely empty - check the cone fit."

    cap_note = (
        "  closed end     : modelled as a flat cap\n"
        if surface.profile.has_cap
        else "  closed end     : NOT modelled - the scan never reached it\n"
    )

    header = (
        f"Occupancy over {surface.n_arc} x {surface.n_angular} "
        f"= {counts.size} cells:\n"
    )

    body = (
        f"  empty cells    : {empty_pct:.1f}%\n"
        f"  confident cells: {confident_pct:.1f}% "
        f"(>= {surface.min_cell_samples} samples)\n"
        f"  count per cell : "
        f"min {occupied.min():.0f}, "
        f"median {np.median(occupied):.0f}, "
        f"mean {occupied.mean():.1f}, "
        f"max {occupied.max():.0f}\n"
        f"  pooled sigma   : {surface.pooled_sigma_m * 1000:.2f} mm"
    )

    return header + cap_note + body
