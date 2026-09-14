import numpy as np
import pytest
from builtin_interfaces.msg import Time

from laundry_control.laundry_detect import (
    cluster_points,
    compute_deviation,
    detect_laundry,
    summarize_clusters,
)
from laundry_control.scan_cloud_util import save_xyz_csv


def _make_grid(n_side=10, spacing=0.05, z=0.0):
    xs = np.arange(n_side) * spacing
    ys = np.arange(n_side) * spacing
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, z)
    return np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)


def _make_tight_cluster(center, n_points=10, spread=0.005, seed=0):
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-spread, spread, size=(n_points, 3))
    return np.asarray(center) + offsets


# ---------------------------------------------------------------
# compute_deviation
# ---------------------------------------------------------------

def test_compute_deviation_identical_clouds():
    points = _make_grid()

    result = compute_deviation(points, points, threshold_m=0.02)

    assert np.allclose(result.distances, 0.0)
    assert not result.mask.any()


def test_compute_deviation_flags_injected_offset():
    baseline = _make_grid()

    # Lift a handful of points 5 cm up, simulating laundry sitting
    # above the empty-bucket-bottom baseline surface.
    injected_indices = np.array([3, 4, 5, 13, 14])
    candidate = baseline.copy()
    candidate[injected_indices, 2] += 0.05

    result = compute_deviation(baseline, candidate, threshold_m=0.02)

    expected_mask = np.zeros(baseline.shape[0], dtype=bool)
    expected_mask[injected_indices] = True

    assert np.array_equal(result.mask, expected_mask)


def test_compute_deviation_empty_baseline_raises():
    baseline = np.empty((0, 3))
    candidate = _make_grid()

    with pytest.raises(ValueError):
        compute_deviation(baseline, candidate)


# ---------------------------------------------------------------
# cluster_points
# ---------------------------------------------------------------

def test_cluster_points_splits_two_separate_blobs():
    cluster_a = _make_tight_cluster((0.0, 0.0, 0.0), n_points=10, seed=1)
    cluster_b = _make_tight_cluster((0.5, 0.5, 0.5), n_points=10, seed=2)
    noise = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])

    points = np.concatenate([cluster_a, cluster_b, noise], axis=0)

    clusters = cluster_points(points, radius_m=0.04, min_cluster_size=4)

    assert len(clusters) == 2
    assert {c.shape[0] for c in clusters} == {10, 10}

    # Noise points must not appear in any surviving cluster.
    clustered_indices = set(np.concatenate(clusters).tolist())
    noise_indices = {20, 21}
    assert clustered_indices.isdisjoint(noise_indices)


def test_cluster_points_merges_nearby_points_within_radius():
    # A -> B -> C chain: A-B and B-C are each within radius, but
    # A-C alone would not be. Connectivity must still be transitive.
    radius = 0.04
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.06, 0.0, 0.0],
        ]
    )

    clusters = cluster_points(points, radius_m=radius, min_cluster_size=1)

    assert len(clusters) == 1
    assert clusters[0].shape[0] == 3


def test_cluster_points_empty_input():
    points = np.empty((0, 3))

    clusters = cluster_points(points)

    assert clusters == []


def test_cluster_points_all_isolated_dropped_by_min_size():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )

    clusters = cluster_points(points, radius_m=0.01, min_cluster_size=2)

    assert clusters == []


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
    deviation_distances = np.array([0.02, 0.03, 0.04, 0.05])

    summaries = summarize_clusters(
        [np.array([0, 1, 2, 3])],
        points,
        deviation_distances,
    )

    assert len(summaries) == 1
    summary = summaries[0]

    assert summary.size == 4
    assert np.allclose(summary.centroid, [0.025, 0.05, 0.075])
    assert np.allclose(summary.bbox_min, [0.0, 0.0, 0.0])
    assert np.allclose(summary.bbox_max, [0.1, 0.2, 0.3])
    assert np.allclose(summary.extent, [0.1, 0.2, 0.3])
    assert np.allclose(summary.highest_point, [0.0, 0.0, 0.3])
    assert summary.mean_deviation_m == pytest.approx(0.035)


def test_summarize_clusters_without_deviation_distances_is_nan():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    summaries = summarize_clusters([np.array([0, 1])], points)

    assert np.isnan(summaries[0].mean_deviation_m)


# ---------------------------------------------------------------
# detect_laundry (end-to-end via CSV round-trip)
# ---------------------------------------------------------------

def _write_csv(path, xyz):
    stamp = Time()
    save_xyz_csv(str(path), [(x, y, z, stamp) for x, y, z in xyz])


def test_detect_laundry_end_to_end_with_tmp_csvs(tmp_path):
    baseline_xyz = _make_grid()

    laundry_cluster = _make_tight_cluster(
        (0.2, 0.2, 0.1), n_points=12, spread=0.005, seed=3
    )
    candidate_xyz = np.concatenate([baseline_xyz, laundry_cluster], axis=0)

    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"

    _write_csv(baseline_csv, baseline_xyz)
    _write_csv(candidate_csv, candidate_xyz)

    clusters = detect_laundry(
        baseline_csv=str(baseline_csv),
        candidate_csv=str(candidate_csv),
        threshold_m=0.02,
        cluster_radius_m=0.04,
        min_cluster_size=4,
    )

    assert len(clusters) == 1
    assert clusters[0].size == 12
    assert np.allclose(clusters[0].centroid, [0.2, 0.2, 0.1], atol=0.01)


def test_detect_laundry_no_deviations_returns_empty_list(tmp_path):
    baseline_xyz = _make_grid()

    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"

    _write_csv(baseline_csv, baseline_xyz)
    _write_csv(candidate_csv, baseline_xyz)

    clusters = detect_laundry(
        baseline_csv=str(baseline_csv),
        candidate_csv=str(candidate_csv),
    )

    assert clusters == []


def test_detect_laundry_all_points_deviate_and_cluster(tmp_path):
    # Grid spacing tight enough that the fully-deviating candidate
    # points still link into cluster(s), unlike the coarser-spacing
    # case below.
    baseline_xyz = _make_grid(spacing=0.02, z=0.0)
    candidate_xyz = _make_grid(spacing=0.02, z=1.0)

    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"

    _write_csv(baseline_csv, baseline_xyz)
    _write_csv(candidate_csv, candidate_xyz)

    clusters = detect_laundry(
        baseline_csv=str(baseline_csv),
        candidate_csv=str(candidate_csv),
    )

    assert sum(c.size for c in clusters) == candidate_xyz.shape[0]


def test_detect_laundry_all_points_deviate_does_not_crash(tmp_path):
    baseline_xyz = _make_grid(z=0.0)
    # Same grid, shifted up by 1m: every point deviates from the
    # baseline, but grid spacing (0.05m) exceeds the default cluster
    # radius (0.04m), so each shifted point is its own singleton
    # component and none survive min_cluster_size - the point here
    # is just that this doesn't crash, not that anything clusters.
    candidate_xyz = _make_grid(z=1.0)

    baseline_csv = tmp_path / "baseline.csv"
    candidate_csv = tmp_path / "candidate.csv"

    _write_csv(baseline_csv, baseline_xyz)
    _write_csv(candidate_csv, candidate_xyz)

    deviation = compute_deviation(baseline_xyz, candidate_xyz)
    assert deviation.mask.all()

    clusters = detect_laundry(
        baseline_csv=str(baseline_csv),
        candidate_csv=str(candidate_csv),
    )

    assert clusters == []
