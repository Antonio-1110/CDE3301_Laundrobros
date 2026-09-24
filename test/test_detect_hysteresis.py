"""Tests for hysteresis growth, the low-confidence gate and grasp points."""

from laundry_control.perception.bucket_model import build_baseline_surface
from laundry_control.perception.detect import (
    DEFAULT_ABS_FLOOR_M,
    detect_on_points,
)
import numpy as np
import pytest

from test_detect import _to_xyz, CONE, empty_scan, plant_on_wall


@pytest.fixture(scope='module')
def surface():
    return build_baseline_surface(
        [empty_scan(seed=seed) for seed in range(1, 6)]
    )


def _graded_mound(scan, s0, theta0, peak_m, radius_s=0.05, radius_theta=0.3):
    """Push wall points inward by a dome profile: tall centre, thin edges."""
    from laundry_control.perception.bucket_model import to_cylindrical

    s, theta, r = to_cylindrical(scan, CONE)
    gap = np.abs((theta - theta0 + np.pi) % (2.0 * np.pi) - np.pi)
    rho2 = ((s - s0) / radius_s) ** 2 + (gap / radius_theta) ** 2
    on_wall = r > CONE.radius_at(s) - 0.02
    lift = np.where(
        on_wall & (rho2 < 1.0), peak_m * np.sqrt(np.clip(1.0 - rho2, 0.0, None)), 0.0
    )

    return _to_xyz(s, theta, r - lift)


def test_legacy_settings_reproduce_the_seed_only_detector(surface):
    scan, _hits = plant_on_wall(empty_scan(seed=9), 0.25, np.pi)

    legacy, _ = detect_on_points(
        scan, surface, grow_k_sigma=None, low_confidence_min_peak_sigma=0.0
    )

    assert len(legacy) == 1


def test_growth_recovers_an_items_thin_edges(surface):
    scan = _graded_mound(empty_scan(seed=9), 0.25, np.pi, peak_m=0.015)

    legacy, _ = detect_on_points(
        scan, surface, grow_k_sigma=None, low_confidence_min_peak_sigma=0.0
    )
    grown, _ = detect_on_points(scan, surface)

    assert grown, 'the dome must be detected'
    assert legacy, 'its peak alone clears the seed threshold'
    # Growth adds the edges that sit between 2.5 and 4 sigma.
    assert grown[0].size > legacy[0].size
    assert grown[0].surface_extent_m > legacy[0].surface_extent_m


def test_every_grown_cluster_contains_a_seed(surface):
    # A shallow item: most of it sits between the growth and seed bars.
    scan = _graded_mound(
        empty_scan(seed=9), 0.25, np.pi, peak_m=DEFAULT_ABS_FLOOR_M * 1.2
    )

    clusters, result = detect_on_points(scan, surface)

    seed_bar = np.maximum(4.0 * result.sigma_m, DEFAULT_ABS_FLOOR_M)
    assert result.mask.any()

    for cluster in clusters:
        # Hysteresis only ever GROWS a seeded cluster: at least one
        # member must clear the seed bar on its own.
        assert cluster.point_intrusion_m.max() >= seed_bar.min()


def test_low_confidence_gate_needs_a_strong_peak(surface):
    scan, _hits = plant_on_wall(empty_scan(seed=9), 0.25, np.pi)

    clusters, _ = detect_on_points(scan, surface)
    assert clusters

    peak = clusters[0].peak_sigma
    assert np.isfinite(peak) and peak > 4.0

    # Force every point to count as low-confidence: a gate above the
    # item's peak removes it, one below keeps it.
    import laundry_control.perception.detect as detect

    real = detect.compute_intrusion

    def all_low_confidence(*args, **kwargs):
        result = real(*args, **kwargs)
        result.confident[:] = False
        return result

    detect.compute_intrusion = all_low_confidence
    try:
        kept, _ = detect.detect_on_points(
            scan, surface, low_confidence_min_peak_sigma=peak - 0.1
        )
        dropped, _ = detect.detect_on_points(
            scan, surface, low_confidence_min_peak_sigma=peak + 0.1
        )
    finally:
        detect.compute_intrusion = real

    assert kept and not dropped


def test_grasp_point_sits_nearer_the_peak_than_the_centroid(surface):
    s0, theta0, peak = 0.25, np.pi, 0.04
    scan = _graded_mound(empty_scan(seed=9), s0, theta0, peak_m=peak)

    clusters, _ = detect_on_points(scan, surface)
    assert clusters

    apex = _to_xyz(
        np.array([s0]), np.array([theta0]),
        np.array([CONE.radius_at(s0) - peak]),
    )[0]

    cluster = clusters[0]
    assert cluster.grasp_point is not None
    assert (
        np.linalg.norm(cluster.target_point - apex)
        < np.linalg.norm(cluster.centroid - apex)
    )
