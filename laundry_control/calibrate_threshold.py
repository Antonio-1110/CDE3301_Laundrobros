#!/usr/bin/env python3

r"""
calibrate_threshold.py

Per-POINT view of the noise-vs-signal separation behind
laundry_detect.py's thresholds.

This is the companion to validate_detector.py, not a replacement
for it. They answer different questions and you want both:

    validate_detector.py works at the CLUSTER level and tells you
    what the detector actually does - how many false clusters per
    empty scan, how many known items it finds. That is the number
    that matters, and it is the one to choose an operating point
    from.

    this script works at the POINT level and tells you WHY - where
    the noise tail ends, where the signal tail begins, and whether
    there is any gap between them at all. When validation comes
    back bad, this is what shows whether the threshold is wrong or
    whether the underlying separation was never there.

Intrusions are reported in units of the LOCAL sigma, because that
is what the detector thresholds on (k * sigma). Reporting raw
metres would hide the thing that makes the new threshold work:
noise varies several-fold across the bucket with incidence angle
and coverage, so a single metre value means very different things
in different places.

Re-run this whenever the sensor, the scan path, or the bucket
placement changes.

Usage:
    calibrate_threshold \\
        --baseline baseline_scans \\
        --repeat scan_records/scan_<second_empty_bucket_run>.csv

    calibrate_threshold \\
        --baseline baseline_scans \\
        --repeat scan_records/scan_<second_empty_bucket_run>.csv \\
        --laundry scan_records/scan_<known_laundry_run>.csv
"""

import argparse

import numpy as np

from .bucket_model import build_baseline_surface, fit_report, occupancy_summary
from .laundry_detect import (
    compute_intrusion,
    load_baseline_scans,
    load_points_xyz,
)

NOISE_PERCENTILES = (50, 90, 95, 99, 99.9, 100)
SIGNAL_PERCENTILES = (50, 90, 99, 99.9, 100)


def _percentile_table(label, values, percentiles):

    if values.size == 0:
        print(f"{label}: no points to report.")
        return

    print(f"{label} (n={values.size}):")

    for percentile in percentiles:
        print(
            f"    p{percentile:<5} = "
            f"{np.percentile(values, percentile):7.2f} sigma"
        )

    print(f"    mean    = {values.mean():7.2f} sigma")


def _normalised_intrusion(candidate_xyz, surface):
    """
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


def run(baseline, repeat_csv, laundry_csv=None):

    baseline_scans = load_baseline_scans(baseline)

    print(
        f"Loaded {len(baseline_scans)} baseline scan(s) from {baseline} "
        f"({sum(scan.shape[0] for scan in baseline_scans)} pts)"
    )

    surface = build_baseline_surface(baseline_scans)

    print()
    print(fit_report(surface.cone))
    print()
    print(occupancy_summary(surface))
    print()

    repeat_xyz = load_points_xyz(repeat_csv)
    print(f"Loaded repeat  : {repeat_csv} ({repeat_xyz.shape[0]} pts)")
    print()

    print("=" * 64)
    print("NOISE FLOOR (empty bucket, scan not used to build the model)")
    print("=" * 64)

    noise, _ = _normalised_intrusion(repeat_xyz, surface)
    _percentile_table("positive intrusion", noise, NOISE_PERCENTILES)

    if laundry_csv is None:
        print()
        print(
            "No --laundry scan given. The noise tail above is a "
            "lower bound on k_sigma, but on its own it cannot tell "
            "you whether anything is still detectable above it - "
            "re-run with --laundry, and run validate_detector.py "
            "for the cluster-level numbers that actually decide the "
            "threshold."
        )
        return

    laundry_xyz = load_points_xyz(laundry_csv)

    print()
    print(f"Loaded laundry : {laundry_csv} ({laundry_xyz.shape[0]} pts)")
    print()

    print("=" * 64)
    print("SIGNAL (known laundry in the bucket)")
    print("=" * 64)

    signal, _ = _normalised_intrusion(laundry_xyz, surface)
    _percentile_table("positive intrusion", signal, SIGNAL_PERCENTILES)

    print()

    if noise.size and signal.size:

        noise_ceiling = np.percentile(noise, 99.9)
        signal_peak = signal.max()

        print(
            f"Separation: noise p99.9 = {noise_ceiling:.2f} sigma, "
            f"signal max = {signal_peak:.2f} sigma."
        )

        if signal_peak <= noise_ceiling:
            print(
                "  NO GAP. The item never rose above the noise "
                "tail, so no threshold can separate them - the "
                "problem is upstream (model fit, coverage, or the "
                "item being too small for this sensor), not the "
                "threshold."
            )
        else:
            print(
                "  Most points in a laundry scan still hit bare "
                "bucket, so the bulk of this distribution SHOULD "
                "look like noise. Only the upper tail is the item. "
                "Judge the threshold on the cluster-level results "
                "from validate_detector.py, not on the percentiles "
                "above."
            )


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Report per-point intrusion distributions, in units of "
            "local sigma, for an empty repeat scan and (optionally) "
            "a known-laundry scan."
        )
    )

    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help=(
            "Directory of empty-bucket baseline scan CSVs, or a "
            "single CSV."
        ),
    )

    parser.add_argument(
        "--repeat",
        type=str,
        required=True,
        help=(
            "A FURTHER empty-bucket scan CSV, over the same "
            "physical path, not among the baselines. Used to "
            "measure the noise floor."
        ),
    )

    parser.add_argument(
        "--laundry",
        type=str,
        default=None,
        help=(
            "Optional scan CSV taken with known laundry in the "
            "bucket, used to measure the detection signal."
        ),
    )

    return parser


def main():

    args = build_parser().parse_args()

    run(
        baseline=args.baseline,
        repeat_csv=args.repeat,
        laundry_csv=args.laundry,
    )


if __name__ == "__main__":
    main()
