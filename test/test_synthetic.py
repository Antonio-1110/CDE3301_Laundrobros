"""Tests for synthetic injection (perception/synthetic.py) and coverage."""

from laundry_control.perception.bucket_model import build_baseline_surface
from laundry_control.perception.synthetic import (
    inject,
    make_mound,
    Rays,
    reconstruct_rays,
)
from laundry_control.perception.synthetic_eval import run_campaign, summarize
from laundry_control.scan import coverage
import numpy as np
import pytest

from test_detect import empty_scan


@pytest.fixture(scope='module')
def surface():
    return build_baseline_surface(
        [empty_scan(seed=seed) for seed in range(1, 4)]
    )


@pytest.fixture(scope='module')
def simulated_scan(surface):
    """Beams of the current scan path cast against the synthetic bucket."""
    return coverage.simulate_path(coverage.ScanPath(), surface.profile)


def test_reconstructed_rays_end_on_the_points(simulated_scan):
    points = simulated_scan.endpoint

    rays = reconstruct_rays(points)

    assert np.allclose(rays.endpoint, points)
    assert np.all(rays.range_m > 0.0)


def test_recorded_ray_columns_are_used_directly():
    points = np.array([[0.0, 0.0, 0.1], [0.0, 0.2, 0.0]])
    columns = {'ox': [0.0, 0.0], 'oy': [0.0, 0.0], 'oz': [0.0, 0.0]}

    rays = reconstruct_rays(points, columns)

    assert np.allclose(rays.range_m, [0.1, 0.2])
    assert np.allclose(rays.direction[0], [0.0, 0.0, 1.0])


def _floor_mound(surface, a=0.06, h=0.04):
    return make_mound(surface.profile, a, h, s=0.25, theta=np.pi)


def test_mound_apex_is_inside_the_bucket(surface):
    mound = _floor_mound(surface)

    assert np.allclose(mound.normal @ mound.normal, 1.0)
    # The apex is h inward of the wall along the inward normal.
    assert np.linalg.norm(mound.apex - mound.anchor) == pytest.approx(0.04)


def test_injection_only_shortens_beams(surface, simulated_scan):
    points = simulated_scan.endpoint
    rays = reconstruct_rays(points)

    new_points, affected = inject(
        points, rays, _floor_mound(surface), noise_m=0.0
    )

    assert affected.any()
    # Nothing here reads under the sensor's minimum range, so rows
    # stay index-aligned.
    assert new_points.shape == points.shape

    assert np.array_equal(new_points[~affected], points[~affected])

    new_range = np.linalg.norm(new_points - rays.origin, axis=1)
    assert np.all(new_range[affected] < rays.range_m[affected])


def test_nearest_model_bounds_the_mixed_model(surface, simulated_scan):
    points = simulated_scan.endpoint
    rays = reconstruct_rays(points)
    mound = _floor_mound(surface)

    mixed, mixed_hit = inject(points, rays, mound, noise_m=0.0)
    nearest, nearest_hit = inject(
        points, rays, mound, model='nearest', noise_m=0.0
    )

    # Same beams reach the object either way; the optimistic model
    # can only read at least as short.
    assert np.array_equal(mixed_hit, nearest_hit)

    both = np.flatnonzero(mixed_hit)
    mixed_range = np.linalg.norm(mixed[both] - rays.origin[both], axis=1)
    nearest_range = np.linalg.norm(nearest[both] - rays.origin[both], axis=1)

    assert np.all(nearest_range <= mixed_range + 1e-9)


def test_darker_items_read_less_short(surface, simulated_scan):
    points = simulated_scan.endpoint
    rays = reconstruct_rays(points)
    mound = _floor_mound(surface)

    bright, hit = inject(points, rays, mound, noise_m=0.0)
    dark, _ = inject(points, rays, mound, reflectivity=0.3, noise_m=0.0)

    idx = np.flatnonzero(hit)
    assert np.all(
        np.linalg.norm(dark[idx] - rays.origin[idx], axis=1)
        >= np.linalg.norm(bright[idx] - rays.origin[idx], axis=1) - 1e-9
    )


def test_item_out_of_every_beam_changes_nothing(surface, simulated_scan):
    points = simulated_scan.endpoint
    rays = reconstruct_rays(points)

    # The ceiling: the current 150 deg sweep never looks there.
    mound = make_mound(surface.profile, 0.04, 0.02, s=0.25, theta=0.0)

    new_points, affected = inject(points, rays, mound)

    assert not affected.any()
    assert np.array_equal(new_points, points)


def test_wider_sweep_covers_more_of_the_bucket(surface):
    narrow = coverage.coverage_by_region(
        surface.profile, coverage.simulate_path(coverage.ScanPath(), surface.profile)
    )
    wide = coverage.coverage_by_region(
        surface.profile,
        coverage.simulate_path(coverage.path_for_sweep(360.0), surface.profile),
    )

    assert narrow['ceiling'] < 0.05
    assert wide['ceiling'] > 0.5
    assert wide['whole_bucket'] > narrow['whole_bucket']


def test_campaign_scores_detections(surface, simulated_scan):
    scans = [
        simulated_scan.endpoint + np.random.default_rng(i).normal(0, 0.002, (1, 3))
        for i in range(3)
    ]

    trials = run_campaign(
        scans, sizes=['towel'], regions=['floor'], per_region=2
    )

    headline = summarize(trials)

    assert headline['trials'] == 6
    assert headline['observable'] == 6
    assert headline['recall_observable'] > 0.5


def test_rays_endpoint_property():
    rays = Rays(
        origin=np.zeros((1, 3)),
        direction=np.array([[1.0, 0.0, 0.0]]),
        range_m=np.array([0.5]),
        from_bottom=np.zeros(1, dtype=bool),
    )

    assert np.allclose(rays.endpoint, [[0.5, 0.0, 0.0]])
