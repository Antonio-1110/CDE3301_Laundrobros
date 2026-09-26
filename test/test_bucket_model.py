"""
Unit tests for bucket_model.py.

Everything here runs on synthetic points generated from a known
geometry, so the fit can be checked against ground truth rather
than against another estimate. No ROS, no hardware, no CSVs.
"""

from laundry_control.perception.bucket_model import (
    _axis_basis,
    build_baseline_surface,
    ConeModel,
    fit_bucket_profile,
    fit_cone,
    fit_report,
    occupancy_summary,
    seed_axis_direction,
    seed_cone,
    surface_residual,
    to_cylindrical,
)
import numpy as np
import pytest

S_CAP = 0.02


def sample_cone(model, n=4000, seed=0, noise_m=0.002, bias=None):
    """
    Sample points over a cone's lateral surface, optionally biased.

    Points scattered over a cone's LATERAL surface, optionally with
    a deterministic theta-dependent radial bias standing in for the
    ToF's incidence-angle error.
    """
    rng = np.random.default_rng(seed)

    s = rng.uniform(model.s_min + 0.02, model.s_max - 0.02, n)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)

    r = model.radius_at(s) + rng.normal(0.0, noise_m, n)

    if bias is not None:
        r = r + bias(theta)

    e1, e2 = _axis_basis(model.axis_dir)

    return (
        model.axis_point
        + s[:, None] * model.axis_dir
        + (r * np.cos(theta))[:, None] * e1
        + (r * np.sin(theta))[:, None] * e2
    )


def sample_with_cap(model, n=5000, seed=0, cap_fraction=0.1, noise_m=0.002):
    """
    Sample the lateral wall plus the flat closed end.

    Lateral wall PLUS the flat closed end - the real bucket's
    geometry, and the case a bare cone cannot represent.
    """
    rng = np.random.default_rng(seed)
    e1, e2 = _axis_basis(model.axis_dir)

    n_cap = int(n * cap_fraction)
    n_wall = n - n_cap

    s = rng.uniform(S_CAP, model.s_max - 0.02, n_wall)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_wall)
    r = model.radius_at(s) + rng.normal(0.0, noise_m, n_wall)

    # Uniform over the disc's AREA, so the centre is not oversampled.
    r_cap = model.radius_at(S_CAP) * np.sqrt(rng.uniform(0.0, 1.0, n_cap))
    theta_cap = rng.uniform(0.0, 2.0 * np.pi, n_cap)
    s_cap = np.full(n_cap, S_CAP) + rng.normal(0.0, noise_m, n_cap)

    s_all = np.concatenate([s, s_cap])
    theta_all = np.concatenate([theta, theta_cap])
    r_all = np.concatenate([r, r_cap])

    return (
        model.axis_point
        + s_all[:, None] * model.axis_dir
        + (r_all * np.cos(theta_all))[:, None] * e1
        + (r_all * np.sin(theta_all))[:, None] * e2
    )


# ---------------------------------------------------------------
# seed geometry
# ---------------------------------------------------------------

def test_seed_is_the_configured_bucket_pose():
    # The same pose MoveIt collision-checks against: the mesh's local
    # +Z (depth axis) under config.OBSTACLES' URDF-convention rpy.
    from laundry_control import config
    from scipy.spatial.transform import Rotation

    spec = config.OBSTACLES['bucket']
    axis = seed_axis_direction()

    assert np.allclose(
        axis, Rotation.from_euler('xyz', spec['rpy']).apply([0, 0, 1])
    )
    assert np.allclose(seed_cone().axis_point, spec['xyz'])
    # The bucket lies on its side, so its axis is nearly +Y in link_base.
    assert axis[1] > 0.98


def test_bucket_pose_from_fit_moves_the_mesh_onto_the_cone():
    from laundry_control import config
    from laundry_control.perception.bucket_model import bucket_pose_from_fit
    from scipy.spatial.transform import Rotation

    seed = seed_cone()
    e1, e2 = _axis_basis(seed.axis_dir)
    fitted = ConeModel(
        axis_point=seed.axis_point + 0.03 * e1,
        axis_dir=(seed.axis_dir + 0.03 * e2) / np.linalg.norm(
            seed.axis_dir + 0.03 * e2
        ),
        r0=seed.r0, taper=seed.taper, s_min=0.0, s_max=0.5,
    )

    xyz, rpy = bucket_pose_from_fit(fitted)

    assert np.allclose(xyz, fitted.axis_point)
    new = Rotation.from_euler('xyz', rpy)
    assert np.allclose(new.apply([0, 0, 1]), fitted.axis_dir)
    # Smallest rotation: the mesh turns by just the axis tilt.
    old = Rotation.from_euler('xyz', config.OBSTACLES['bucket']['rpy'])
    assert (new * old.inv()).magnitude() == pytest.approx(
        np.arccos(fitted.axis_dir @ seed.axis_dir), abs=1e-9
    )
    # Unchanged when the fit agrees with the configuration.
    same_xyz, same_rpy = bucket_pose_from_fit(seed)
    assert np.allclose(same_xyz, seed.axis_point)
    assert (Rotation.from_euler('xyz', same_rpy) * old.inv()).magnitude() < 1e-9


def test_seed_cone_widens_toward_the_mouth():
    cone = seed_cone()

    assert cone.taper > 0.0
    assert cone.radius_at(cone.s_max) > cone.radius_at(cone.s_min)


# ---------------------------------------------------------------
# to_cylindrical
# ---------------------------------------------------------------

def test_to_cylindrical_round_trips():
    cone = seed_cone()
    e1, e2 = _axis_basis(cone.axis_dir)

    s_in = np.array([0.05, 0.20, 0.40])
    theta_in = np.array([0.3, 2.0, 5.5])
    r_in = np.array([0.19, 0.21, 0.23])

    points = (
        cone.axis_point
        + s_in[:, None] * cone.axis_dir
        + (r_in * np.cos(theta_in))[:, None] * e1
        + (r_in * np.sin(theta_in))[:, None] * e2
    )

    s, theta, r = to_cylindrical(points, cone)

    assert np.allclose(s, s_in)
    assert np.allclose(theta, theta_in)
    assert np.allclose(r, r_in)


def test_theta_zero_is_the_top_of_the_bucket():
    """Theta is anchored to world +Z so the grid stays stable."""
    cone = seed_cone()
    e1, _e2 = _axis_basis(cone.axis_dir)

    assert e1[2] > 0.9


def test_to_cylindrical_handles_empty_input():
    s, theta, r = to_cylindrical(np.empty((0, 3)), seed_cone())

    assert s.size == 0 and theta.size == 0 and r.size == 0


# ---------------------------------------------------------------
# fit_cone
# ---------------------------------------------------------------

def test_fit_cone_recovers_known_parameters():
    seed = seed_cone()
    e1, e2 = _axis_basis(seed.axis_dir)

    axis_dir = seed.axis_dir + 0.02 * e1 - 0.015 * e2
    axis_dir = axis_dir / np.linalg.norm(axis_dir)

    truth = ConeModel(
        axis_point=seed.axis_point + 0.012 * e1 - 0.008 * e2,
        axis_dir=axis_dir,
        r0=0.1925,
        taper=0.092,
        s_min=0.0,
        s_max=0.44,
    )

    fitted = fit_cone(sample_cone(truth, n=4000))

    assert np.linalg.norm(fitted.axis_point - truth.axis_point) < 0.002
    assert abs(fitted.r0 - truth.r0) < 0.002
    assert abs(fitted.taper - truth.taper) < 0.01

    tilt = np.degrees(
        np.arccos(np.clip(abs(fitted.axis_dir @ truth.axis_dir), -1.0, 1.0))
    )
    assert tilt < 0.5


def test_fit_cone_is_robust_to_gross_outliers():
    """
    A few stray returns must not distort the fitted surface.

    The Huber loss is there so a handful of stray returns cannot
    distort the surface. A distorted surface biases EVERY later
    detection, not just the region the outliers came from.
    """
    truth = seed_cone()
    points = sample_cone(truth, n=4000, seed=1)

    rng = np.random.default_rng(7)
    corrupt = rng.choice(len(points), int(0.10 * len(points)), replace=False)
    points[corrupt] += rng.normal(0.0, 0.06, (corrupt.size, 3))

    fitted = fit_cone(points)

    assert np.linalg.norm(fitted.axis_point - truth.axis_point) < 0.003
    assert abs(fitted.r0 - truth.r0) < 0.003


def test_fit_cone_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_cone(np.zeros((3, 3)))


def test_fit_cone_rejects_wrong_shape():
    with pytest.raises(ValueError):
        fit_cone(np.zeros((10, 2)))


def test_surface_residual_is_zero_on_the_model_surface():
    cone = seed_cone()
    points = sample_cone(cone, n=500, noise_m=0.0)

    assert np.allclose(surface_residual(points, cone), 0.0, atol=1e-9)


def test_fit_report_mentions_the_seed_comparison():
    report = fit_report(fit_cone(sample_cone(seed_cone(), n=1000)))

    assert 'axis point' in report
    assert 'taper' in report


# ---------------------------------------------------------------
# BucketProfile - the flat closed end
# ---------------------------------------------------------------

def test_profile_finds_the_closed_end():
    profile = fit_bucket_profile(sample_with_cap(seed_cone(), seed=0))

    assert profile.has_cap
    assert profile.vertices[0][0] == pytest.approx(S_CAP, abs=0.005)
    assert profile.vertices[0][1] == pytest.approx(0.0)


def test_profile_omits_a_cap_the_scan_never_reached():
    """A wall-only scan must not get a closed end invented for it."""
    profile = fit_bucket_profile(sample_cone(seed_cone(), n=5000))

    assert not profile.has_cap
    assert len(profile.vertices) == 2


def test_cap_points_do_not_drag_the_cone_fit():
    """
    Cap points must not drag the lateral cone fit.

    Regression: the flat disc sits far inside the wall radius, so a
    lateral-surface fit sees it as a mass of outliers. Measured
    before the profile split, with a fifth of the points on the
    cap, r0 came out 9mm low and the taper went 0.104 -> 0.137 -
    a distortion applied to the WHOLE bucket, not just the cap.
    """
    truth = seed_cone()

    naive = fit_cone(sample_with_cap(truth, n=5000, cap_fraction=0.20))
    profile = fit_bucket_profile(
        sample_with_cap(truth, n=5000, cap_fraction=0.20)
    )

    naive_error = abs(naive.r0 - truth.r0)
    profile_error = abs(profile.cone.r0 - truth.r0)

    assert naive_error > 0.005
    assert profile_error < 0.002
    assert abs(profile.cone.taper - truth.taper) < 0.01


def test_profile_arc_runs_from_cap_centre_to_mouth():
    profile = fit_bucket_profile(sample_with_cap(seed_cone(), seed=0))

    breaks = profile.u_breaks

    assert breaks[0] == 0.0
    assert np.all(np.diff(breaks) > 0.0)

    # u = 0 is the cap's centre, where a whole ring of theta really
    # does collapse to a single point.
    assert profile.radius_at_u(0.0) == pytest.approx(0.0, abs=1e-9)
    assert profile.radius_at_u(profile.u_max) > 0.2


def test_profile_project_signs_intrusion_by_being_inside():
    cone = seed_cone()
    profile = fit_bucket_profile(sample_with_cap(cone, seed=0))

    s_cap = float(profile.vertices[0][0])

    # On the wall, just inside -> positive; just outside -> negative.
    s = np.array([0.20, 0.20])
    r = np.array([cone.radius_at(0.20) - 0.01, cone.radius_at(0.20) + 0.01])

    _u, intrusion = profile.project(s, r)

    assert intrusion[0] > 0.0
    assert intrusion[1] < 0.0

    # Same at the closed end, where "inside" means nearer the mouth.
    s = np.array([s_cap + 0.01, s_cap - 0.01])
    r = np.array([0.05, 0.05])

    _u, intrusion = profile.project(s, r)

    assert intrusion[0] > 0.0
    assert intrusion[1] < 0.0


# ---------------------------------------------------------------
# BaselineSurface
# ---------------------------------------------------------------

def test_baseline_surface_recovers_the_noise_level():
    truth = seed_cone()
    scans = [
        sample_cone(truth, n=5000, seed=i, noise_m=0.002)
        for i in range(10)
    ]

    surface = build_baseline_surface(scans)

    assert np.isclose(np.median(surface.offset_sigma), 0.002, atol=0.0005)

    # Not all() - the last arc bin is only partially spanned by the
    # data (the fitted extent runs to the 99th percentile of s, and
    # the bin count is a ceil), so a handful of edge cells are
    # legitimately thin.
    assert surface.confident.mean() > 0.95


def test_closed_end_does_not_inflate_sigma():
    """
    Regression, and the reason the profile exists.

    With the cap folded into an (s, theta) grid, every radius from 0
    to R landed in one axial row, so sigma there measured the disc's
    own extent rather than sensor noise: 20-45mm against a 2mm noise
    floor, bleeding into the neighbouring wall rows. On the profile
    grid the cap is spread along u like any other surface, so its
    sigma is just noise again.
    """
    truth = seed_cone()
    scans = [
        sample_with_cap(truth, n=5000, seed=i, cap_fraction=0.1)
        for i in range(10)
    ]

    surface = build_baseline_surface(scans)

    assert surface.profile.has_cap

    occupied = surface.count > 0
    worst = surface.offset_sigma[occupied].max()

    assert worst < 0.006, f'worst cell sigma {worst * 1000:.1f}mm'
    assert surface.pooled_sigma_m < 0.004


def test_baseline_surface_absorbs_a_deterministic_bias():
    """
    A cos(2*theta) bias must land in the residual field in full.

    A cos(2*theta) radial bias is exactly the kind of systematic
    sensor artefact the offset field exists to soak up - it is not
    in the cone's own span, so the cone cannot absorb it.
    """
    truth = seed_cone()

    def bias(theta):
        return 0.004 * np.cos(2.0 * theta)

    scans = [
        sample_cone(truth, n=5000, seed=i, noise_m=0.002, bias=bias)
        for i in range(10)
    ]

    surface = build_baseline_surface(scans)

    assert surface.offset_mean.max() > 0.003
    assert surface.offset_mean.min() < -0.003

    # ...and having absorbed it, a held-out empty scan reads flat.
    held_out = sample_cone(truth, n=5000, seed=99, noise_m=0.002, bias=bias)
    s, theta, r = to_cylindrical(held_out, surface.cone)
    u, raw = surface.profile.project(s, r)
    offset, _sigma, _confident = surface.expected_offset(u, theta)

    assert abs(np.mean(raw - offset)) < 0.0005


def test_cos_theta_bias_is_degenerate_with_the_axis_position():
    """
    A cos(theta) bias is absorbed by the axis, not the residual field.

    Documents a real limitation rather than a behaviour anyone
    chose: a cos(theta) radial bias - the signature of a lateral
    boresight/extrinsic error - IS a lateral shift of the axis, so
    the fit absorbs it completely and the offset field never sees
    it.

    Consequence: a displaced fitted axis cannot be attributed to
    the bucket pose or to the sensor mounting from the fit alone.
    See bucket_model.fit_report.
    """
    truth = seed_cone()

    def bias(theta):
        return 0.004 * np.cos(theta)

    scans = [
        sample_cone(truth, n=5000, seed=i, noise_m=0.002, bias=bias)
        for i in range(10)
    ]

    surface = build_baseline_surface(scans)

    shift = np.linalg.norm(surface.cone.axis_point - truth.axis_point)

    # The axis soaked up essentially the whole 4mm...
    assert 0.003 < shift < 0.005

    # ...leaving almost nothing behind in the offset field. Taken
    # at p99 rather than max: the handful of thinly-sampled edge
    # cells carry ordinary sampling noise, which says nothing about
    # whether the bias was absorbed.
    assert np.percentile(np.abs(surface.offset_mean), 99) < 0.002


def test_empty_cells_are_still_judgeable():
    """
    Cells with no baseline coverage must still be judged.

    The old detector gave points with no baseline neighbour -inf
    and could never flag them, so every gap in the scan path was a
    place laundry could hide. Empty cells must still return a
    usable expectation - just a wider sigma and a low-confidence
    flag.
    """
    truth = seed_cone()

    points = sample_cone(truth, n=4000)
    _s, theta, _r = to_cylindrical(points, truth)
    partial = points[theta < np.pi]

    surface = build_baseline_surface([partial])

    assert (surface.count == 0).any()

    offset, sigma, confident = surface.expected_offset(
        np.array([0.2]), np.array([1.5 * np.pi])
    )

    assert np.isfinite(offset).all()
    assert (sigma > 0).all()
    assert not confident.any()


def test_build_baseline_surface_requires_a_scan():
    with pytest.raises(ValueError):
        build_baseline_surface([])


def test_occupancy_summary_reports_counts_and_cap_state():
    surface = build_baseline_surface(
        [sample_with_cap(seed_cone(), n=5000, seed=i) for i in range(3)]
    )

    summary = occupancy_summary(surface)

    assert 'Occupancy' in summary
    assert 'closed end' in summary


def test_cell_area_vanishes_at_the_cap_centre():
    """
    Cells at the closed-end centre must carry almost no area.

    At u = 0 every theta is the same physical point, so those cells
    carry essentially no area and must not contribute volume as if
    they were full-width rings.
    """
    surface = build_baseline_surface(
        [sample_with_cap(seed_cone(), n=5000, seed=i) for i in range(3)]
    )

    area = surface.cell_area().ravel()

    assert area[0] < 0.1 * area[-1]
    assert area[0] >= 0.0
