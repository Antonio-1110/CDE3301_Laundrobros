"""
Unit tests for laundry_detect.py.

Synthetic points are generated on a known bucket - lateral wall
PLUS the flat closed end - and then pushed INWARD over a patch to
stand in for a laundry item, which is what a real item does to a
ToF reading: it intercepts the beam before it reaches the wall.
"""

import numpy as np
import pytest
from builtin_interfaces.msg import Time

from laundry_control.bucket_model import (
    _axis_basis,
    build_baseline_surface,
    seed_cone,
    to_cylindrical,
)
from laundry_control.laundry_detect import (
    cluster_points,
    cluster_volume_m3,
    compute_intrusion,
    detect_laundry,
    load_baseline_scans,
    summarize_clusters,
)
from laundry_control.scan_cloud_util import save_xyz_csv

CONE = seed_cone()
E1, E2 = _axis_basis(CONE.axis_dir)
S_CAP = 0.02


def _to_xyz(s, theta, r):
    return (
        CONE.axis_point
        + s[:, None] * CONE.axis_dir
        + (r * np.cos(theta))[:, None] * E1
        + (r * np.sin(theta))[:, None] * E2
    )


def empty_scan(n=5000, seed=0, noise_m=0.002, cap_fraction=0.1):
    """An empty bucket: lateral wall plus the flat closed end."""

    rng = np.random.default_rng(seed)

    n_cap = int(n * cap_fraction)
    n_wall = n - n_cap

    s = rng.uniform(S_CAP, 0.42, n_wall)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_wall)
    r = CONE.radius_at(s) + rng.normal(0.0, noise_m, n_wall)

    r_cap = CONE.radius_at(S_CAP) * np.sqrt(rng.uniform(0.0, 1.0, n_cap))
    theta_cap = rng.uniform(0.0, 2.0 * np.pi, n_cap)
    s_cap = np.full(n_cap, S_CAP) + rng.normal(0.0, noise_m, n_cap)

    return _to_xyz(
        np.concatenate([s, s_cap]),
        np.concatenate([theta, theta_cap]),
        np.concatenate([r, r_cap]),
    )


def plant_on_wall(scan_xyz, s0, theta0, depth_m=0.025, ds=0.035, dtheta=0.22):
    """Push a patch of wall points inward, as a real item would."""

    s, theta, r = to_cylindrical(scan_xyz, CONE)

    angular_gap = np.abs((theta - theta0 + np.pi) % (2.0 * np.pi) - np.pi)
    on_wall = r > CONE.radius_at(s) - 0.02

    hit = (np.abs(s - s0) < ds) & (angular_gap < dtheta) & on_wall

    r_new = r.copy()
    r_new[hit] -= depth_m

    return _to_xyz(s, theta, r_new), int(hit.sum())


def plant_on_cap(scan_xyz, depth_m=0.03, r_max=0.08):
    """
    A towel lying against the CLOSED END. It intercepts the beam
    early, so it reads nearer the mouth - a displacement in s, not
    in r, which is why a purely radial model cannot see it.
    """

    s, theta, r = to_cylindrical(scan_xyz, CONE)

    hit = (np.abs(s - S_CAP) < 0.01) & (r < r_max)

    s_new = s.copy()
    s_new[hit] += depth_m

    return _to_xyz(s_new, theta, r), int(hit.sum())


@pytest.fixture(scope="module")
def surface():
    return build_baseline_surface([empty_scan(seed=i) for i in range(10)])


def _write_csv(path, xyz):
    stamp = Time()
    save_xyz_csv(str(path), [(x, y, z, stamp) for x, y, z in xyz])


def _summaries(scan, surface, radius=0.04, min_size=3):
    result = compute_intrusion(scan, surface)

    if not result.mask.any():
        return [], result

    flagged = scan[result.mask]
    clusters = cluster_points(flagged, radius, min_size)

    return (
        summarize_clusters(
            clusters,
            flagged,
            result.intrusion_m[result.mask],
            u=result.u[result.mask],
            theta=result.theta[result.mask],
            surface=surface,
            confident=result.confident[result.mask],
        ),
        result,
    )


# ---------------------------------------------------------------
# compute_intrusion
# ---------------------------------------------------------------

def test_empty_bucket_produces_no_detections(surface):
    result = compute_intrusion(empty_scan(seed=99), surface)

    assert not result.mask.any()


@pytest.mark.parametrize(
    "name,theta0",
    [
        ("floor", np.pi),
        ("side wall", 0.5 * np.pi),
        ("ceiling", 0.0),
    ],
)
def test_detects_an_item_anywhere_around_the_bucket(surface, name, theta0):
    """
    The regression that motivated the rewrite.

    The old test asked whether a point stood proud in z, which is
    only meaningful on the bucket FLOOR: an item against a side
    wall displaces x rather than z, and one on the ceiling LOWERS
    z. Both were structurally invisible. Working perpendicular to
    the fitted surface makes all three the same question.
    """

    scan, planted = plant_on_wall(empty_scan(seed=500), s0=0.20, theta0=theta0)

    result = compute_intrusion(scan, surface)

    assert planted > 0
    assert result.mask.sum() >= 0.9 * planted, name


def test_detects_an_item_against_the_closed_end(surface):
    """
    Laundry lying against the flat far end of the bucket.

    This is where laundry collects in a bucket on its side, and it
    is the case a cone-only model handles worst: the disc packs
    every radius into one axial row, inflating sigma there about
    20-fold. Modelling the closed end as its own profile segment is
    what makes this detectable.
    """

    assert surface.profile.has_cap

    scan, planted = plant_on_cap(empty_scan(seed=501), depth_m=0.03)

    result = compute_intrusion(scan, surface)

    assert planted > 0
    assert result.mask.sum() >= 0.8 * planted


def test_no_blind_band_next_to_the_closed_end(surface):
    """
    Regression with a measured before/after.

    With the cap folded into the wall grid, its 20-45mm sigma bled
    into the neighbouring rows: a 2.5cm item 3.5cm from the closed
    end came back 25/66 points, and at 5cm 33/73 - partial or
    missed, in the very region laundry gathers. On the profile grid
    the cap has its own cells, so the wall stays at its normal 2mm
    noise floor right up to the corner.
    """

    for s0 in (0.035, 0.05, 0.07, 0.10):

        scan, planted = plant_on_wall(
            empty_scan(seed=502), s0=s0, theta0=np.pi
        )

        summaries, result = _summaries(scan, surface)

        assert planted > 0

        # The item must come back as one solid cluster, not as a
        # scatter that the gates then throw away.
        assert len(summaries) == 1, f"s0={s0} gave {len(summaries)} clusters"
        assert summaries[0].volume_m3 > 5e-5, f"s0={s0} volume too small"

        # Point-level recall is checked loosely on purpose. Right in
        # the corner it is genuinely lower - see
        # BucketProfile.project: for a point tucked into a concave
        # corner the distance to the NEAREST surface is smaller than
        # the distance to the wall alone, so its intrusion reads
        # short. That is correct geometry, not a defect, and the
        # cluster as a whole still carries plenty of signal.
        found = int(result.mask.sum())
        assert found >= 0.6 * planted, (
            f"only {found}/{planted} found at s0={s0}"
        )


def test_points_behind_the_wall_are_discarded(surface):
    """Physically impossible, so a stray return rather than laundry."""

    s = np.full(20, 0.2)
    theta = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    outside = _to_xyz(s, theta, CONE.radius_at(s) + 0.10)

    result = compute_intrusion(outside, surface)

    assert not result.in_bounds.any()
    assert not result.mask.any()


def test_points_beyond_the_mouth_are_not_judged(surface):
    """Past the fitted extent the surface is extrapolation."""

    s = np.full(20, surface.cone.s_max + 0.15)
    theta = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    beyond = _to_xyz(s, theta, CONE.radius_at(s) - 0.05)

    result = compute_intrusion(beyond, surface)

    assert not result.in_bounds.any()
    assert not result.mask.any()


def test_points_behind_the_closed_end_are_discarded(surface):
    """
    Beyond the cap plane is outside the bucket entirely. Without
    the cap the cone alone would happily call such a point "inside
    the wall radius" and flag it.
    """

    s = np.full(20, S_CAP - 0.08)
    theta = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
    behind = _to_xyz(s, theta, np.full(20, 0.05))

    result = compute_intrusion(behind, surface)

    assert not result.mask.any()


def test_intrusion_is_signed_toward_the_surface(surface):
    scan, _planted = plant_on_wall(empty_scan(seed=7), s0=0.20, theta0=np.pi)

    result = compute_intrusion(scan, surface)

    assert result.intrusion_m[result.mask].min() > 0.0


def test_compute_intrusion_empty_input(surface):
    result = compute_intrusion(np.empty((0, 3)), surface)

    assert result.mask.size == 0
    assert result.intrusion_m.size == 0
    assert result.u.size == 0


def test_compute_intrusion_rejects_wrong_shape(surface):
    with pytest.raises(ValueError):
        compute_intrusion(np.zeros((5, 2)), surface)


# ---------------------------------------------------------------
# cluster_points
# ---------------------------------------------------------------

def _tight_cluster(center, n_points=10, spread=0.005, seed=0):
    rng = np.random.default_rng(seed)
    return np.asarray(center) + rng.uniform(-spread, spread, (n_points, 3))


def test_cluster_points_splits_two_separate_blobs():
    cluster_a = _tight_cluster((0.0, 0.0, 0.0), n_points=10, seed=1)
    cluster_b = _tight_cluster((0.5, 0.5, 0.5), n_points=10, seed=2)
    noise = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])

    points = np.concatenate([cluster_a, cluster_b, noise], axis=0)

    clusters = cluster_points(points, radius_m=0.04, min_cluster_size=4)

    assert len(clusters) == 2
    assert {c.shape[0] for c in clusters} == {10, 10}

    clustered = set(np.concatenate(clusters).tolist())
    assert clustered.isdisjoint({20, 21})


def test_cluster_points_merges_nearby_points_within_radius():
    # A -> B -> C chain: A-B and B-C are each within radius, but
    # A-C alone would not be. Connectivity must still be transitive.
    points = np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.06, 0.0, 0.0]])

    clusters = cluster_points(points, radius_m=0.04, min_cluster_size=1)

    assert len(clusters) == 1
    assert clusters[0].shape[0] == 3


def test_cluster_points_empty_input():
    assert cluster_points(np.empty((0, 3))) == []


def test_cluster_points_all_isolated_dropped_by_min_size():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    assert cluster_points(points, radius_m=0.01, min_cluster_size=2) == []


def test_ceiling_item_is_a_single_cluster(surface):
    """
    theta = 0 is the top of the bucket. An earlier version clustered
    on the unwrapped surface, whose seam ran exactly there, and
    split every ceiling item in two - each half then at risk of
    failing the volume gate. 3D clustering has no seam to split on.
    """

    scan, planted = plant_on_wall(empty_scan(seed=500), s0=0.20, theta0=0.0)

    summaries, _result = _summaries(scan, surface)

    assert len(summaries) == 1
    assert summaries[0].size >= 0.9 * planted


def test_extent_is_not_inflated_at_the_ceiling(surface):
    """
    Same seam, different symptom: extent was measured on unwrapped
    coordinates, so a ceiling item reported the bucket's whole
    1.31m circumference for a ~0.12m sock.
    """

    scan, _planted = plant_on_wall(empty_scan(seed=500), s0=0.20, theta0=0.0)

    summaries, _result = _summaries(scan, surface)

    assert 0.05 < summaries[0].surface_extent_m < 0.25


def test_separate_items_stay_separate(surface):
    scan, _ = plant_on_wall(empty_scan(seed=500), s0=0.10, theta0=0.0)
    scan, _ = plant_on_wall(scan, s0=0.35, theta0=np.pi)

    summaries, _result = _summaries(scan, surface)

    assert len(summaries) == 2


# ---------------------------------------------------------------
# volume
# ---------------------------------------------------------------

def test_volume_is_far_less_density_dependent_than_point_count(surface):
    """
    The reason the size gate is volume rather than point count: the
    helical scan samples some parts of the bucket several times
    more densely than others, so a point count partly measures
    where an item happened to land rather than how much fabric is
    there.

    Volume is NOT perfectly invariant - a thinly sampled item still
    under-reads, because cells its footprint covers but no point
    landed in contribute nothing (see cluster_volume_m3). The claim
    this test pins down is the one the design rests on: volume
    tracks the item across a 4x density change far more tightly
    than the point count does.
    """

    volumes = []
    counts = []

    for n in (3000, 12000):
        scan, _ = plant_on_wall(
            empty_scan(n=n, seed=3), s0=0.20, theta0=np.pi
        )
        result = compute_intrusion(scan, surface)

        counts.append(int(result.mask.sum()))
        volumes.append(
            cluster_volume_m3(
                result.u[result.mask],
                result.theta[result.mask],
                result.intrusion_m[result.mask],
                surface,
            )
        )

    volume_ratio = max(volumes) / min(volumes)
    count_ratio = max(counts) / min(counts)

    # Compared on departure from 1.0, since 1.0 - not 0 - is what
    # perfect invariance would look like.
    assert volume_ratio < 1.5
    assert (volume_ratio - 1.0) < 0.2 * (count_ratio - 1.0)


def test_volume_is_stable_once_sampling_is_adequate(surface):
    """
    Above roughly 6000 points per scan the residual density
    dependence has largely gone. Worth pinning separately, because
    it is what makes DEFAULT_MIN_VOLUME_M3 a usable fixed gate at
    the real scan's point count rather than something that has to
    be retuned per scan length.
    """

    volumes = []

    for n in (6000, 24000):
        scan, _ = plant_on_wall(
            empty_scan(n=n, seed=3), s0=0.20, theta0=np.pi
        )
        result = compute_intrusion(scan, surface)

        volumes.append(
            cluster_volume_m3(
                result.u[result.mask],
                result.theta[result.mask],
                result.intrusion_m[result.mask],
                surface,
            )
        )

    assert max(volumes) / min(volumes) < 1.15


# ---------------------------------------------------------------
# summarize_clusters
# ---------------------------------------------------------------

def test_summarize_clusters_computes_correct_centroid_and_bbox():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
        ]
    )
    intrusion = np.array([0.02, 0.03, 0.04, 0.05])

    summaries = summarize_clusters([np.array([0, 1, 2, 3])], points, intrusion)

    assert len(summaries) == 1
    summary = summaries[0]

    assert summary.size == 4
    assert np.allclose(summary.centroid, [0.025, 0.05, 0.075])
    assert np.allclose(summary.bbox_min, [0.0, 0.0, 0.0])
    assert np.allclose(summary.bbox_max, [0.1, 0.2, 0.3])
    assert np.allclose(summary.extent, [0.1, 0.2, 0.3])
    assert np.allclose(summary.highest_point, [0.0, 0.0, 0.3])
    assert summary.mean_deviation_m == pytest.approx(0.035)
    assert summary.max_intrusion_m == pytest.approx(0.05)


def test_summarize_clusters_without_intrusion_is_nan():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    summaries = summarize_clusters([np.array([0, 1])], points)

    assert np.isnan(summaries[0].mean_deviation_m)
    assert np.isnan(summaries[0].volume_m3)


def test_cluster_is_low_confidence_when_mostly_unsampled():
    points = np.zeros((4, 3))
    intrusion = np.full(4, 0.02)
    confident = np.array([True, False, False, False])

    summaries = summarize_clusters(
        [np.array([0, 1, 2, 3])], points, intrusion, confident=confident
    )

    assert not summaries[0].confident


# ---------------------------------------------------------------
# detect_laundry, end to end via CSV
# ---------------------------------------------------------------

def test_detect_laundry_end_to_end(tmp_path):
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()

    for i in range(10):
        _write_csv(baseline_dir / f"empty_{i}.csv", empty_scan(seed=i))

    scan, _ = plant_on_wall(empty_scan(seed=500), s0=0.20, theta0=np.pi)
    candidate = tmp_path / "candidate.csv"
    _write_csv(candidate, scan)

    clusters = detect_laundry(
        baseline_csv=str(baseline_dir),
        candidate_csv=str(candidate),
    )

    assert len(clusters) == 1
    assert clusters[0].volume_m3 > 0.0
    assert clusters[0].max_intrusion_m > 0.015


def test_detect_laundry_on_an_empty_bucket_finds_nothing(tmp_path):
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()

    for i in range(10):
        _write_csv(baseline_dir / f"empty_{i}.csv", empty_scan(seed=i))

    candidate = tmp_path / "candidate.csv"
    _write_csv(candidate, empty_scan(seed=99))

    clusters = detect_laundry(
        baseline_csv=str(baseline_dir),
        candidate_csv=str(candidate),
    )

    assert clusters == []


def test_detect_laundry_sorts_by_volume(tmp_path, surface):
    scan, _ = plant_on_wall(
        empty_scan(seed=500), s0=0.12, theta0=np.pi,
        depth_m=0.02, dtheta=0.12,
    )
    scan, _ = plant_on_wall(
        scan, s0=0.34, theta0=0.5 * np.pi, depth_m=0.05, dtheta=0.30
    )

    candidate = tmp_path / "candidate.csv"
    _write_csv(candidate, scan)

    clusters = detect_laundry(
        baseline_csv="unused",
        candidate_csv=str(candidate),
        surface=surface,
    )

    assert len(clusters) == 2
    assert clusters[0].volume_m3 > clusters[1].volume_m3


def test_volume_gate_rejects_a_speckle(tmp_path, surface):
    """
    A handful of adjacent noise points should not survive as a
    detection just because they happened to land near each other.
    """

    scan, _ = plant_on_wall(
        empty_scan(seed=500), s0=0.20, theta0=np.pi,
        depth_m=0.02, ds=0.004, dtheta=0.02,
    )

    candidate = tmp_path / "candidate.csv"
    _write_csv(candidate, scan)

    clusters = detect_laundry(
        baseline_csv="unused",
        candidate_csv=str(candidate),
        surface=surface,
        min_volume_m3=1e-4,
    )

    assert clusters == []


# ---------------------------------------------------------------
# baseline loading
# ---------------------------------------------------------------

def test_load_baseline_scans_from_directory(tmp_path):
    for i in range(3):
        _write_csv(tmp_path / f"empty_{i}.csv", empty_scan(n=100, seed=i))

    assert len(load_baseline_scans(str(tmp_path))) == 3


def test_load_baseline_scans_from_single_file(tmp_path):
    path = tmp_path / "one.csv"
    _write_csv(path, empty_scan(n=100))

    assert len(load_baseline_scans(str(path))) == 1


def test_load_baseline_scans_from_list(tmp_path):
    paths = []

    for i in range(2):
        path = tmp_path / f"empty_{i}.csv"
        _write_csv(path, empty_scan(n=100, seed=i))
        paths.append(str(path))

    assert len(load_baseline_scans(paths)) == 2


def test_load_baseline_scans_rejects_empty_directory(tmp_path):
    with pytest.raises(ValueError):
        load_baseline_scans(str(tmp_path))
