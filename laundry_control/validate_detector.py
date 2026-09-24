#!/usr/bin/env python3

r"""
validate_detector.py

Measure how well the detector actually works, and pick its
operating point from data rather than by eye.

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
laundry_detect.py's threshold comments. A threshold with no
provenance is how the last one ended up unfalsifiable.

Usage:
    validate_detector --baseline baseline_scans
    validate_detector --baseline baseline_scans --sweep
    validate_detector --baseline baseline_scans \\
        --laundry scan_records/sock_mouth.csv \\
        --laundry scan_records/towel_ceiling.csv
"""

import argparse
import os

from .bucket_model import (
    build_baseline_surface,
    fit_report,
    occupancy_summary,
)
from .laundry_detect import (
    DEFAULT_ABS_FLOOR_M,
    DEFAULT_CLUSTER_RADIUS_M,
    DEFAULT_K_SIGMA,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_EXTENT_M,
    DEFAULT_MIN_VOLUME_M3,
    cluster_points,
    compute_intrusion,
    load_baseline_scans,
    load_points_xyz,
    summarize_clusters,
)

# k_sigma values walked by --sweep. Spans "trigger-happy" to
# "conservative" so the false-positive knee is visible rather than
# guessed at.
SWEEP_K_SIGMA = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)


def detect_on_points(
    candidate_xyz,
    surface,
    k_sigma=DEFAULT_K_SIGMA,
    abs_floor_m=DEFAULT_ABS_FLOOR_M,
    cluster_radius_m=DEFAULT_CLUSTER_RADIUS_M,
    min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
    min_extent_m=DEFAULT_MIN_EXTENT_M,
    min_volume_m3=DEFAULT_MIN_VOLUME_M3,
):
    """
    Run the detector over an already-loaded point array.

    Mirrors laundry_detect.detect_laundry() but skips the CSV load,
    because validation re-runs the same points against many
    thresholds and re-reading the file each time would dominate the
    runtime.
    """

    result = compute_intrusion(
        candidate_xyz,
        surface,
        k_sigma=k_sigma,
        abs_floor_m=abs_floor_m,
    )

    if not result.mask.any():
        return [], result

    flagged_xyz = candidate_xyz[result.mask]

    cluster_indices = cluster_points(
        flagged_xyz,
        radius_m=cluster_radius_m,
        min_cluster_size=min_cluster_size,
    )

    summaries = summarize_clusters(
        cluster_indices,
        flagged_xyz,
        result.intrusion_m[result.mask],
        u=result.u[result.mask],
        theta=result.theta[result.mask],
        surface=surface,
        confident=result.confident[result.mask],
    )

    kept = [
        summary
        for summary in summaries
        if summary.surface_extent_m >= min_extent_m
        and summary.volume_m3 >= min_volume_m3
    ]

    kept.sort(key=lambda summary: summary.volume_m3, reverse=True)

    return kept, result


def leave_one_out(baseline_scans, **detect_kwargs):
    """
    Build the model from every baseline but one, run it against the
    one held out, and repeat.

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
        return "no points"

    out_of_bounds = total - int(result.in_bounds.sum())

    return (
        f"{out_of_bounds}/{total} pts out of bounds "
        f"({100.0 * out_of_bounds / total:.1f}%)"
    )


def report_false_positives(baseline_scans, **detect_kwargs):

    print("=" * 64)
    print("FALSE POSITIVES - leave-one-out over empty baselines")
    print("=" * 64)

    folds = leave_one_out(baseline_scans, **detect_kwargs)

    total_clusters = 0

    for held_out, clusters, result in folds:

        total_clusters += len(clusters)

        flagged = int(result.mask.sum())

        detail = ""

        if clusters:
            biggest = max(cluster.volume_m3 for cluster in clusters)
            detail = f"  largest {biggest * 1e6:.1f}cm3"

        print(
            f"  fold {held_out:2d}: {len(clusters)} cluster(s), "
            f"{flagged} flagged pt(s), {_dropout_note(result)}{detail}"
        )

    n_folds = len(folds)

    print()
    print(
        f"  TOTAL: {total_clusters} false cluster(s) over "
        f"{n_folds} fold(s) "
        f"= {total_clusters / max(n_folds, 1):.2f} per empty scan"
    )
    print()

    return total_clusters


def report_recall(surface, laundry_csvs, **detect_kwargs):

    print("=" * 64)
    print("RECALL - scans with known laundry present")
    print("=" * 64)

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
            confidence = "" if biggest.confident else " [low confidence]"

            print(
                f"  {name}: FOUND {len(clusters)} cluster(s), "
                f"largest {biggest.volume_m3 * 1e6:.1f}cm3 "
                f"({biggest.size} pts, max intrusion "
                f"{biggest.max_intrusion_m * 100:.1f}cm){confidence}"
            )

        else:
            print(
                f"  {name}: MISSED - nothing cleared the gates "
                f"({int(result.mask.sum())} pt(s) flagged, "
                f"{_dropout_note(result)})"
            )

    print()
    print(f"  TOTAL: {found}/{len(laundry_csvs)} scan(s) detected")
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

    print("=" * 64)
    print("THRESHOLD SWEEP")
    print("=" * 64)
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

            recall = f"{found}/{len(laundry_csvs)}"

        else:
            recall = "n/a"

        print(
            f"  {k_sigma:>8.1f}  {false_clusters:>15d}  {recall:>12}"
        )

    print()
    print(
        "  Favour recall where the two conflict: a false positive "
        "costs one wasted look, a false negative leaves laundry in "
        "the bucket."
    )
    print()


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Measure the detector's false-positive rate (leave-one-"
            "out over empty baselines) and its recall (known-laundry "
            "scans), and sweep the threshold."
        )
    )

    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help=(
            "Directory of empty-bucket baseline scan CSVs (or a "
            "single CSV, though leave-one-out needs at least two)."
        ),
    )

    parser.add_argument(
        "--laundry",
        type=str,
        action="append",
        default=[],
        help=(
            "Scan CSV with known laundry present. Repeat the flag "
            "for several. Include hard placements: bucket mouth, "
            "upper wall, closed end, dark fabric."
        ),
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Walk k_sigma and tabulate both error rates.",
    )

    parser.add_argument(
        "--k-sigma",
        type=float,
        default=DEFAULT_K_SIGMA,
        help=f"Detection threshold in sigmas (default: {DEFAULT_K_SIGMA}).",
    )

    parser.add_argument(
        "--abs-floor",
        type=float,
        default=DEFAULT_ABS_FLOOR_M,
        help=(
            "Absolute intrusion floor in metres "
            f"(default: {DEFAULT_ABS_FLOOR_M})."
        ),
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=DEFAULT_MIN_VOLUME_M3,
        help=(
            "Minimum cluster volume in cubic metres "
            f"(default: {DEFAULT_MIN_VOLUME_M3})."
        ),
    )

    parser.add_argument(
        "--min-extent",
        type=float,
        default=DEFAULT_MIN_EXTENT_M,
        help=(
            "Minimum cluster footprint in metres "
            f"(default: {DEFAULT_MIN_EXTENT_M})."
        ),
    )

    return parser


def main():

    args = build_parser().parse_args()

    baseline_scans = load_baseline_scans(args.baseline)

    print(
        f"Loaded {len(baseline_scans)} baseline scan(s) from "
        f"{args.baseline} "
        f"({sum(scan.shape[0] for scan in baseline_scans)} pts)"
    )
    print()

    if len(baseline_scans) < 2:
        raise SystemExit(
            "Leave-one-out needs at least 2 baseline scans; "
            f"found {len(baseline_scans)}. Point --baseline at a "
            "directory of empty-bucket scans (8-10 is the target)."
        )

    if len(baseline_scans) < 5:
        print(
            f"WARNING: {len(baseline_scans)} baseline scans is thin. "
            "Per-cell sigma needs ~8-10 to be meaningful, and with "
            "fewer folds the false-positive rate below carries a "
            "wide error bar of its own."
        )
        print()

    full_surface = build_baseline_surface(baseline_scans)

    print(fit_report(full_surface.cone))
    print()
    print(occupancy_summary(full_surface))
    print()

    detect_kwargs = {
        "abs_floor_m": args.abs_floor,
        "min_volume_m3": args.min_volume,
        "min_extent_m": args.min_extent,
    }

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
            "No --laundry scans given, so only the false-positive "
            "side was measured. A threshold validated on empty "
            "scans alone is only half-validated: it says nothing "
            "about what the detector can still find."
        )


if __name__ == "__main__":
    main()
