#!/usr/bin/env python3

r"""
Measure how well the detector actually works: `laundry evaluate`.

Picks the detector's operating point from data rather than by eye.

WHY THIS IS NOT OPTIONAL
------------------------
The previous detection threshold was a single constant chosen from
ONE empty-vs-empty scan pair. That gives no false-positive rate, no
recall figure, and no idea of the smallest item that can be found -
so there was no way to tell a working detector from a lucky one.

Two measurements fix that:

    FALSE POSITIVES (--baseline, always run):
        Leave-one-out over the empty baseline scans. For each fold,
        the model is built from the other N-1 scans and then run
        against the held-out one. The bucket was empty in every
        scan, so EVERY cluster reported is a false positive, and N
        folds give N independent looks instead of one.

        Crucially the held-out scan never contributes to the model
        that judges it. Scoring a scan against a model it helped
        build measures memorisation, not detection, and would make
        any threshold look far better than it is.

    RECALL (--laundry, optional but strongly recommended):
        Scans with a known item in a known place. Every miss is a
        false negative. Include the HARD placements deliberately -
        the bucket mouth, the upper wall, the closed end - because
        those are where the model is weakest and where an item is
        most likely to be missed, so a recall number built only
        from easy central placements is worthless.

        Vary the fabric too. Dark fabric absorbs IR, so a black
        sock is a genuinely harder target than a white towel, and
        the dropout counts reported here are the evidence for
        whether missing returns need handling as a signal of their
        own.

CHOOSING AN OPERATING POINT
---------------------------
--sweep walks k_sigma and reports both sides together. Pick the
point that keeps false positives at zero across all folds while
still catching the hardest placement you care about - and when
those two conflict, favour RECALL. A false positive costs one
wasted look; a false negative leaves laundry in the bucket, which
is the failure the machine exists to prevent.

Record whatever you pick, and the scans it came from, in
perception/detect.py's threshold comments. A threshold with no
provenance is how the last one ended up unfalsifiable.

PER-POINT VIEW (--repeat)
-------------------------
The cluster-level numbers above say what the detector DOES. The
per-point view says WHY: where the noise tail ends, where the signal
tail begins, and whether there is any gap between them at all. When
validation comes back bad, this is what shows whether the threshold
is wrong or whether the underlying separation was never there.

Intrusions there are reported in units of the LOCAL sigma, because
that is what the detector thresholds on (k * sigma). Reporting raw
metres would hide the thing that makes the threshold work: noise
varies several-fold across the bucket with incidence angle and
coverage, so a single metre value means very different things in
different places.

Re-run all of this whenever the sensor, the scan path, or the
bucket placement changes.

Usage:
    laundry evaluate
    laundry evaluate --sweep
    laundry evaluate \\
        --laundry scan_records/sock_mouth.csv \\
        --laundry scan_records/towel_ceiling.csv
    laundry evaluate --repeat scan_records/another_empty.csv \\
        --laundry scan_records/sock_mouth.csv
"""

import os

import numpy as np

from .bucket_model import (
    build_baseline_surface,
    fit_report,
    occupancy_summary,
)
from .detect import (
    compute_intrusion,
    DEFAULT_ABS_FLOOR_M,
    DEFAULT_K_SIGMA,
    DEFAULT_MIN_EXTENT_M,
    DEFAULT_MIN_VOLUME_M3,
    detect_on_points,
    load_baseline_scans,
    load_points_xyz,
)
from .. import config

# k_sigma values walked by --sweep. Spans "trigger-happy" to
# "conservative" so the false-positive knee is visible rather than
# guessed at.
SWEEP_K_SIGMA = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)


def leave_one_out(baseline_scans, **detect_kwargs):
    """
    Build the model from every baseline but one, run it against the one held out, and repeat.

    Returns a list of (fold_index, clusters, intrusion_result). Any
    cluster in any fold is a false positive by construction.
    """
    folds = []

    for held_out in range(len(baseline_scans)):

        others = [
            scan
            for index, scan in enumerate(baseline_scans)
            if index != held_out
        ]

        surface = build_baseline_surface(others)

        clusters, result = detect_on_points(
            baseline_scans[held_out],
            surface,
            **detect_kwargs,
        )

        folds.append((held_out, clusters, result))

    return folds


def _dropout_note(result):
    """
    How much of the scan fell outside the model's trusted region.

    A large figure is not a detection result, it is a warning: it
    means the cone fit, the bucket pose or the sensor extrinsics
    disagree with where the points actually landed, and every
    number alongside it should be distrusted until that is sorted
    out.
    """
    total = result.in_bounds.size

    if total == 0:
        return 'no points'

    out_of_bounds = total - int(result.in_bounds.sum())

    return (
        f'{out_of_bounds}/{total} pts out of bounds '
        f'({100.0 * out_of_bounds / total:.1f}%)'
    )


def report_false_positives(baseline_scans, **detect_kwargs):

    print('=' * 64)
    print('FALSE POSITIVES - leave-one-out over empty baselines')
    print('=' * 64)

    folds = leave_one_out(baseline_scans, **detect_kwargs)

    total_clusters = 0

    for held_out, clusters, result in folds:

        total_clusters += len(clusters)

        flagged = int(result.mask.sum())

        detail = ''

        if clusters:
            biggest = max(cluster.volume_m3 for cluster in clusters)
            detail = f'  largest {biggest * 1e6:.1f}cm3'

        print(
            f'  fold {held_out:2d}: {len(clusters)} cluster(s), '
            f'{flagged} flagged pt(s), {_dropout_note(result)}{detail}'
        )

    n_folds = len(folds)

    print()
    print(
        f'  TOTAL: {total_clusters} false cluster(s) over '
        f'{n_folds} fold(s) '
        f'= {total_clusters / max(n_folds, 1):.2f} per empty scan'
    )
    print()

    return total_clusters


def report_recall(surface, laundry_csvs, **detect_kwargs):

    print('=' * 64)
    print('RECALL - scans with known laundry present')
    print('=' * 64)

    found = 0

    for path in laundry_csvs:

        candidate_xyz = load_points_xyz(path)

        clusters, result = detect_on_points(
            candidate_xyz, surface, **detect_kwargs
        )

        name = os.path.basename(path)

        if clusters:
            found += 1
            biggest = clusters[0]
            confidence = '' if biggest.confident else ' [low confidence]'

            print(
                f'  {name}: FOUND {len(clusters)} cluster(s), '
                f'largest {biggest.volume_m3 * 1e6:.1f}cm3 '
                f'({biggest.size} pts, max intrusion '
                f'{biggest.max_intrusion_m * 100:.1f}cm){confidence}'
            )

        else:
            print(
                f'  {name}: MISSED - nothing cleared the gates '
                f'({int(result.mask.sum())} pt(s) flagged, '
                f'{_dropout_note(result)})'
            )

    print()
    print(f'  TOTAL: {found}/{len(laundry_csvs)} scan(s) detected')
    print()

    return found


def sweep(baseline_scans, laundry_csvs, base_kwargs):
    """
    Walk k_sigma and show both error rates side by side.

    Printed as one table on purpose: the choice is a trade-off, and
    seeing false positives without recall (or the reverse) is how
    you end up with a threshold that is excellent at one and
    useless at the other.
    """
    print('=' * 64)
    print('THRESHOLD SWEEP')
    print('=' * 64)
    print(
        f"  {'k_sigma':>8}  {'false clusters':>15}  "
        f"{'recall':>12}"
    )

    full_surface = build_baseline_surface(baseline_scans)

    for k_sigma in SWEEP_K_SIGMA:

        kwargs = dict(base_kwargs, k_sigma=k_sigma)

        folds = leave_one_out(baseline_scans, **kwargs)
        false_clusters = sum(len(clusters) for _, clusters, _ in folds)

        if laundry_csvs:

            found = 0

            for path in laundry_csvs:
                clusters, _ = detect_on_points(
                    load_points_xyz(path), full_surface, **kwargs
                )
                found += bool(clusters)

            recall = f'{found}/{len(laundry_csvs)}'

        else:
            recall = 'n/a'

        print(
            f'  {k_sigma:>8.1f}  {false_clusters:>15d}  {recall:>12}'
        )

    print()
    print(
        '  Favour recall where the two conflict: a false positive '
        'costs one wasted look, a false negative leaves laundry in '
        'the bucket.'
    )
    print()


NOISE_PERCENTILES = (50, 90, 95, 99, 99.9, 100)
SIGNAL_PERCENTILES = (50, 90, 99, 99.9, 100)


def _percentile_table(label, values, percentiles):

    if values.size == 0:
        print(f'{label}: no points to report.')
        return

    print(f'{label} (n={values.size}):')

    for percentile in percentiles:
        print(
            f'    p{percentile:<5} = '
            f'{np.percentile(values, percentile):7.2f} sigma'
        )

    print(f'    mean    = {values.mean():7.2f} sigma')


def _normalised_intrusion(candidate_xyz, surface):
    """
    Return in-bounds positive intrusions in units of local sigma.

    Per-point intrusion divided by that point's local sigma,
    restricted to points inside the model's trusted region.

    Only the POSITIVE tail is physically meaningful: an object can
    only intercept the beam early, so negative intrusion is the
    bucket wall reading slightly further out than modelled, which
    is noise by definition and cannot be evidence of laundry.
    """
    result = compute_intrusion(candidate_xyz, surface)

    usable = result.in_bounds & (result.sigma_m > 0.0)

    normalised = result.intrusion_m[usable] / result.sigma_m[usable]

    return normalised[normalised > 0.0], result


def report_point_separation(surface, repeat_csv, laundry_csv=None):
    """
    Print the per-point noise-vs-signal separation, in local sigmas.

    repeat_csv is a FURTHER empty-bucket scan over the same path
    that is not among the baselines; laundry_csv optionally a scan
    with known laundry present.
    """
    repeat_xyz = load_points_xyz(repeat_csv)
    print(f'Loaded repeat  : {repeat_csv} ({repeat_xyz.shape[0]} pts)')
    print()

    print('=' * 64)
    print('NOISE FLOOR (empty bucket, scan not used to build the model)')
    print('=' * 64)

    noise, _ = _normalised_intrusion(repeat_xyz, surface)
    _percentile_table('positive intrusion', noise, NOISE_PERCENTILES)

    if laundry_csv is None:
        print()
        print(
            'No --laundry scan given. The noise tail above is a '
            'lower bound on k_sigma, but on its own it cannot tell '
            'you whether anything is still detectable above it - '
            're-run with --laundry, and run the cluster-level results above '
            'for the cluster-level numbers that actually decide the '
            'threshold.'
        )
        return

    laundry_xyz = load_points_xyz(laundry_csv)

    print()
    print(f'Loaded laundry : {laundry_csv} ({laundry_xyz.shape[0]} pts)')
    print()

    print('=' * 64)
    print('SIGNAL (known laundry in the bucket)')
    print('=' * 64)

    signal, _ = _normalised_intrusion(laundry_xyz, surface)
    _percentile_table('positive intrusion', signal, SIGNAL_PERCENTILES)

    print()

    if noise.size and signal.size:

        noise_ceiling = np.percentile(noise, 99.9)
        signal_peak = signal.max()

        print(
            f'Separation: noise p99.9 = {noise_ceiling:.2f} sigma, '
            f'signal max = {signal_peak:.2f} sigma.'
        )

        if signal_peak <= noise_ceiling:
            print(
                '  NO GAP. The item never rose above the noise '
                'tail, so no threshold can separate them - the '
                'problem is upstream (model fit, coverage, or the '
                'item being too small for this sensor), not the '
                'threshold.'
            )
        else:
            print(
                '  Most points in a laundry scan still hit bare '
                'bucket, so the bulk of this distribution SHOULD '
                'look like noise. Only the upper tail is the item. '
                'Judge the threshold on the cluster-level results '
                'from the cluster-level results above, not on the percentiles '
                'above.'
            )


# Detector keyword arguments that reproduce the pre-hysteresis
# detector exactly (see perception.detect.detect_on_points).
LEGACY_DETECTOR = {
    'grow_k_sigma': None,
    'low_confidence_min_peak_sigma': 0.0,
}


def report_synthetic(baseline_scans, detect_kwargs, args):
    """Run and print the synthetic-injection campaign."""
    from . import synthetic_eval

    print('=' * 64)
    print(
        f'SYNTHETIC LAUNDRY - {args.sensor_model} sensor model, '
        f'reflectivity {args.reflectivity:g}, leave-one-out'
    )
    print('=' * 64)

    trials = synthetic_eval.run_campaign(
        baseline_scans,
        detect_kwargs=detect_kwargs,
        per_region=args.per_region,
        model=args.sensor_model,
        reflectivity=args.reflectivity,
        seed=args.seed,
    )

    synthetic_eval.print_tables(trials)

    if args.csv:
        synthetic_eval.write_csv(args.csv, trials)
        print(f'  per-trial results: {args.csv}')

    print()


def report_coverage(surface, baseline_scans):
    """Print measured vs simulated coverage, and candidate scan paths."""
    from ..scan import coverage
    from .synthetic import reconstruct_rays

    def line(label, result, seconds=None):
        cells = '  '.join(
            f'{name}={100 * value:3.0f}%' for name, value in result.items()
        )
        tail = f'  ~{seconds:.0f}s' if seconds else ''
        print(f'  {label:30s} {cells}{tail}')

    print('=' * 64)
    print('SCAN COVERAGE - share of the bucket surface inside a beam footprint')
    print('=' * 64)

    line(
        'measured, one scan',
        coverage.coverage_by_region(
            surface.profile, reconstruct_rays(baseline_scans[0])
        ),
    )
    line(
        f'measured, {len(baseline_scans)} scans pooled',
        coverage.measured_coverage(surface.profile, baseline_scans),
    )

    print()
    print('  Simulated (validate: "current" should match "one scan" above):')

    for label, path in coverage.CANDIDATE_PATHS.items():
        rays = coverage.simulate_path(path, surface.profile)
        line(
            label,
            coverage.coverage_by_region(surface.profile, rays),
            coverage.estimated_duration_s(path),
        )

    print()


def add_evaluate_arguments(parser):
    """Add `laundry evaluate` options to a parser."""
    parser.add_argument(
        '--baseline',
        type=str,
        default=None,
        help=(
            'Directory of empty-bucket baseline scan CSVs (default: '
            '<repo>/baseline_scans). Leave-one-out needs at least two.'
        ),
    )

    parser.add_argument(
        '--laundry',
        type=str,
        action='append',
        default=[],
        help=(
            'Scan CSV with known laundry present. Repeat the flag '
            'for several. Include hard placements: bucket mouth, '
            'upper wall, closed end, dark fabric.'
        ),
    )

    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Walk k_sigma and tabulate both error rates.',
    )

    parser.add_argument(
        '--repeat',
        type=str,
        default=None,
        help=(
            'A FURTHER empty-bucket scan, not among the baselines: '
            'prints the per-point noise-vs-signal separation.'
        ),
    )

    parser.add_argument(
        '--synthetic',
        action='store_true',
        help=(
            'Inject synthetic laundry into the held-out empty scans and '
            'report detection rate / localisation by size and region.'
        ),
    )

    parser.add_argument(
        '--per-region',
        type=int,
        default=6,
        help='Synthetic placements per size, region and fold (default: 6).',
    )

    parser.add_argument(
        '--sensor-model',
        choices=('mixed', 'nearest'),
        default='mixed',
        help=(
            "Synthetic ToF model: 'mixed' (return-weighted over the cone, "
            "default) or 'nearest' (optimistic bound)."
        ),
    )

    parser.add_argument(
        '--reflectivity',
        type=float,
        default=1.0,
        help=(
            'Synthetic item brightness relative to the bucket wall '
            '(default 1.0; ~0.3 for dark fabric).'
        ),
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for synthetic placements (default: 0).',
    )

    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='With --synthetic: write one row per trial to this CSV.',
    )

    parser.add_argument(
        '--coverage',
        action='store_true',
        help=(
            'Report measured bucket coverage, the simulated current scan '
            'path next to it, and candidate path changes.'
        ),
    )

    parser.add_argument(
        '--legacy',
        action='store_true',
        help=(
            'Evaluate the detector as it was before hysteresis and the '
            'low-confidence gate, for before/after comparisons.'
        ),
    )

    parser.add_argument(
        '--k-sigma',
        type=float,
        default=DEFAULT_K_SIGMA,
        help=f'Detection threshold in sigmas (default: {DEFAULT_K_SIGMA}).',
    )

    parser.add_argument(
        '--abs-floor',
        type=float,
        default=DEFAULT_ABS_FLOOR_M,
        help=(
            'Absolute intrusion floor in metres '
            f'(default: {DEFAULT_ABS_FLOOR_M}).'
        ),
    )

    parser.add_argument(
        '--min-volume',
        type=float,
        default=DEFAULT_MIN_VOLUME_M3,
        help=(
            'Minimum cluster volume in cubic metres '
            f'(default: {DEFAULT_MIN_VOLUME_M3}).'
        ),
    )

    parser.add_argument(
        '--min-extent',
        type=float,
        default=DEFAULT_MIN_EXTENT_M,
        help=(
            'Minimum cluster footprint in metres '
            f'(default: {DEFAULT_MIN_EXTENT_M}).'
        ),
    )


def run_evaluate(args):
    """Run `laundry evaluate`; returns a process exit code."""
    baseline = args.baseline or config.baseline_dir()

    baseline_scans = load_baseline_scans(baseline)

    print(
        f'Loaded {len(baseline_scans)} baseline scan(s) from '
        f'{baseline} '
        f'({sum(scan.shape[0] for scan in baseline_scans)} pts)'
    )
    print()

    if len(baseline_scans) < 2:
        print(
            'Leave-one-out needs at least 2 baseline scans; '
            f'found {len(baseline_scans)}. Point --baseline at a '
            'directory of empty-bucket scans (8-10 is the target).'
        )
        return 1

    if len(baseline_scans) < 5:
        print(
            f'WARNING: {len(baseline_scans)} baseline scans is thin. '
            'Per-cell sigma needs ~8-10 to be meaningful, and with '
            'fewer folds the false-positive rate below carries a '
            'wide error bar of its own.'
        )
        print()

    full_surface = build_baseline_surface(baseline_scans)

    print(fit_report(full_surface.cone))
    print()
    print(occupancy_summary(full_surface))
    print()

    detect_kwargs = {
        'abs_floor_m': args.abs_floor,
        'min_volume_m3': args.min_volume,
        'min_extent_m': args.min_extent,
    }

    if args.legacy:
        detect_kwargs.update(LEGACY_DETECTOR)
        print('Evaluating the LEGACY detector (no hysteresis, no gate).')
        print()

    if args.sweep:
        sweep(baseline_scans, args.laundry, detect_kwargs)

    report_false_positives(
        baseline_scans, k_sigma=args.k_sigma, **detect_kwargs
    )

    if args.laundry:
        report_recall(
            full_surface,
            args.laundry,
            k_sigma=args.k_sigma,
            **detect_kwargs,
        )
    else:
        print(
            'No --laundry scans given, so only the false-positive '
            'side was measured. A threshold validated on empty '
            'scans alone is only half-validated: it says nothing '
            'about what the detector can still find.'
        )

    if args.synthetic:
        print()
        report_synthetic(baseline_scans, dict(detect_kwargs, k_sigma=args.k_sigma), args)

    if args.coverage:
        print()
        report_coverage(full_surface, baseline_scans)

    if args.repeat:
        print()
        report_point_separation(
            full_surface,
            args.repeat,
            args.laundry[0] if args.laundry else None,
        )

    return 0
