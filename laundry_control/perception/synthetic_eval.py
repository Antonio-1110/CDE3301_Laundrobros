#!/usr/bin/env python3

"""
Measure detection rate and localisation error with synthetic laundry.

`laundry evaluate --synthetic` runs this. For each leave-one-out fold
(model from N-1 empty baselines, the held-out one as the scan), known
items are injected (perception/synthetic.py) at sampled placements in
each region of the bucket, the detector is run, and every trial is
scored:

    detected      - a reported cluster's centroid lies within
                    MATCH_MARGIN_M of the item's footprint (footprint
                    radius + margin from the apex).
    error_*       - distance from the matched cluster's grasp-point
                    estimate to the item's apex, for each estimator in
                    GRASP_ESTIMATORS, so they can be compared on the
                    same trials.
    stray         - reported clusters that match nothing (false
                    positives in a scan that does contain an item).
    observable    - whether any beam was affected at all. An item no
                    beam reaches is a SCAN-COVERAGE miss, not a
                    detector miss, and the two are reported apart
                    because they have different fixes.

The held-out scan never contributes to the model that judges it, so
these numbers are honest in the same way the empty-bucket leave-one-
out is.
"""

import csv
from dataclasses import asdict, dataclass

import numpy as np

from .bucket_model import build_baseline_surface
from .detect import detect_on_points
from .synthetic import inject, make_mound, reconstruct_rays

# Item sizes: (footprint radius a, height h) in metres. Chosen to span
# a flat cloth at the noise floor up to a bundled towel, in a bucket
# of ~0.2m radius.
ITEM_SIZES = {
    'flat_cloth': (0.08, 0.008),
    'sock': (0.04, 0.02),
    'shirt': (0.07, 0.04),
    'towel': (0.10, 0.06),
    'bundle': (0.13, 0.09),
}

# Regions by theta (0 = top of the bucket, pi = floor), symmetric
# about the vertical, plus the flat closed end.
REGIONS = {
    'floor': (150.0, 180.0),
    'lower_wall': (90.0, 150.0),
    'upper_wall': (30.0, 90.0),
    'ceiling': (0.0, 30.0),
    'closed_end': None,
}

MATCH_MARGIN_M = 0.03


def _estimate_centroid(cluster):
    return cluster.centroid


def _estimate_peak(cluster):
    if cluster.point_intrusion_m is None:
        return cluster.centroid
    return cluster.points[int(np.argmax(cluster.point_intrusion_m))]


def _estimate_weighted(cluster):
    if cluster.point_intrusion_m is None:
        return cluster.centroid
    weights = np.clip(cluster.point_intrusion_m, 0.0, None) ** 2
    if weights.sum() <= 0.0:
        return cluster.centroid
    return (cluster.points * weights[:, None]).sum(axis=0) / weights.sum()


def _estimate_top_quartile(cluster):
    if cluster.point_intrusion_m is None:
        return cluster.centroid
    cut = np.percentile(cluster.point_intrusion_m, 75.0)
    members = cluster.points[cluster.point_intrusion_m >= cut]
    return members.mean(axis=0)


def _estimate_grasp_point(cluster):
    return cluster.target_point


GRASP_ESTIMATORS = {
    'grasp_point': _estimate_grasp_point,
    'centroid': _estimate_centroid,
    'peak': _estimate_peak,
    'weighted': _estimate_weighted,
    'top_quartile': _estimate_top_quartile,
}


@dataclass
class Trial:
    """One injected item and how the detector did on it."""

    fold: int
    size: str
    region: str
    s: float
    theta_deg: float
    observable: bool
    affected: int
    detected: bool
    stray: int
    rank: int
    error_grasp_point: float
    error_centroid: float
    error_peak: float
    error_weighted: float
    error_top_quartile: float


def _sample_placements(profile, region, count, a, rng):
    """Yield make_mound() keyword arguments for `count` placements."""
    cone = profile.cone

    if REGIONS[region] is None:
        if not profile.has_cap:
            return

        r_cap = float(profile.vertices[1][1])
        limit = max(r_cap - a - 0.01, 0.0)

        for _ in range(count):
            yield {
                'cap_radius': float(np.sqrt(rng.uniform(0.0, 1.0)) * limit),
                'cap_angle': float(rng.uniform(0.0, 2.0 * np.pi)),
            }
        return

    low, high = REGIONS[region]

    s_low = cone.s_min + a + 0.02
    s_high = cone.s_max - a - 0.02

    for _ in range(count):
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        theta = np.pi + side * (np.pi - np.deg2rad(rng.uniform(low, high)))

        yield {
            's': float(rng.uniform(s_low, s_high)),
            'theta': float(np.mod(theta, 2.0 * np.pi)),
        }


def run_campaign(
    baseline_scans,
    detect_kwargs=None,
    sizes=None,
    regions=None,
    per_region=6,
    model='mixed',
    reflectivity=1.0,
    seed=0,
    surfaces=None,
):
    """
    Run the injection campaign; returns a list of Trial.

    surfaces may pass pre-built leave-one-out surfaces (one per fold,
    built without that fold's scan) so several detector settings can
    be compared on identical models without refitting.
    """
    detect_kwargs = dict(detect_kwargs or {})
    sizes = sizes or list(ITEM_SIZES)
    regions = regions or list(REGIONS)

    rng = np.random.default_rng(seed)

    trials = []

    for fold, scan in enumerate(baseline_scans):
        if surfaces is not None:
            surface = surfaces[fold]
        else:
            surface = build_baseline_surface(
                [other for i, other in enumerate(baseline_scans) if i != fold]
            )

        rays = reconstruct_rays(scan)

        for size in sizes:
            a, h = ITEM_SIZES[size]

            for region in regions:
                for placement in _sample_placements(
                    surface.profile, region, per_region, a, rng
                ):
                    mound = make_mound(surface.profile, a, h, **placement)

                    points, affected = inject(
                        scan,
                        rays,
                        mound,
                        model=model,
                        reflectivity=reflectivity,
                        rng=rng,
                    )

                    clusters, _result = detect_on_points(
                        points, surface, **detect_kwargs
                    )

                    trials.append(
                        _score(
                            fold, size, region, placement, mound,
                            affected, clusters,
                        )
                    )

    return trials


def _score(fold, size, region, placement, mound, affected, clusters):
    apex = mound.apex
    reach = mound.footprint_radius_m + MATCH_MARGIN_M

    distances = [
        float(np.linalg.norm(cluster.centroid - apex)) for cluster in clusters
    ]

    matched = [i for i, d in enumerate(distances) if d <= reach]

    errors = {name: float('nan') for name in GRASP_ESTIMATORS}

    rank = -1

    if matched:
        # The cluster the grasp stage would pick among the matches is
        # the highest-ranked one (clusters arrive sorted by volume).
        rank = matched[0]
        best = clusters[rank]

        for name, estimator in GRASP_ESTIMATORS.items():
            errors[name] = float(np.linalg.norm(estimator(best) - apex))

    s = placement.get('s', float('nan'))
    theta = placement.get('theta', float('nan'))

    return Trial(
        fold=fold,
        size=size,
        region=region,
        s=s,
        theta_deg=float(np.degrees(theta)) if np.isfinite(theta) else float('nan'),
        observable=bool(affected.any()),
        affected=int(affected.sum()),
        detected=bool(matched),
        stray=len(clusters) - len(matched),
        rank=rank,
        error_grasp_point=errors['grasp_point'],
        error_centroid=errors['centroid'],
        error_peak=errors['peak'],
        error_weighted=errors['weighted'],
        error_top_quartile=errors['top_quartile'],
    )


def summarize(trials, estimator='grasp_point'):
    """Return the campaign's headline numbers as a dict."""
    observable = [t for t in trials if t.observable]
    detected = [t for t in observable if t.detected]

    errors = np.array([getattr(t, f'error_{estimator}') for t in detected])

    return {
        'trials': len(trials),
        'observable': len(observable),
        'detected': len(detected),
        'recall_observable': len(detected) / max(len(observable), 1),
        'recall_all': len(detected) / max(len(trials), 1),
        'stray_per_trial': sum(t.stray for t in trials) / max(len(trials), 1),
        'median_error_m': float(np.median(errors)) if errors.size else float('nan'),
        'p90_error_m': (
            float(np.percentile(errors, 90)) if errors.size else float('nan')
        ),
    }


def print_tables(trials, estimator='grasp_point'):
    """Print recall by size x region, and localisation by estimator."""
    sizes = list(dict.fromkeys(t.size for t in trials))
    regions = list(dict.fromkeys(t.region for t in trials))

    print('Recall on OBSERVABLE items (detected / reached by >= 1 beam),')
    print('with the share of items no beam reached at all in brackets:')
    print()
    header = f"  {'':12s}" + ''.join(f'{r:>17s}' for r in regions)
    print(header)

    for size in sizes:
        row = f'  {size:12s}'

        for region in regions:
            cell = [t for t in trials if t.size == size and t.region == region]
            seen = [t for t in cell if t.observable]
            hits = sum(t.detected for t in seen)
            unseen = len(cell) - len(seen)

            if not cell:
                row += f"{'-':>17s}"
            elif not seen:
                row += f"{'unseen':>11s} ({unseen:>2d})"
            else:
                row += (
                    f'{100.0 * hits / len(seen):>9.0f}% '
                    f'({100.0 * unseen / len(cell):>3.0f}%)'
                )

        print(row)

    print()
    print('Localisation error to the item apex, detected items, by grasp-')
    print('point estimator (median / p90, cm):')
    print()

    for name in GRASP_ESTIMATORS:
        values = np.array([
            getattr(t, f'error_{name}') for t in trials if t.detected
        ])

        if values.size:
            print(
                f'  {name:13s} {100 * np.median(values):5.1f} / '
                f'{100 * np.percentile(values, 90):5.1f}'
            )

    headline = summarize(trials, estimator)

    print()
    print(
        f"  {headline['detected']}/{headline['observable']} observable items "
        f"detected ({100 * headline['recall_observable']:.1f}%); "
        f"{headline['trials'] - headline['observable']} of "
        f"{headline['trials']} never reached by any beam; "
        f"{headline['stray_per_trial']:.2f} stray cluster(s) per trial."
    )


def write_csv(path, trials):
    """Write one row per trial, for plotting or further analysis."""
    rows = [asdict(t) for t in trials]

    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
