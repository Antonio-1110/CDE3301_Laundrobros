#!/usr/bin/env python3

"""
calibrate_threshold.py

Offline calibration helper for laundry_detect.py's thresholds.

Re-derives the "noise vs. signal" percentile gap described in
laundry_detect.py's module comment (DEFAULT_THRESHOLD_M etc.).
Re-run it against fresh scans whenever the sensor, the scan path,
or the bucket placement changes - the current defaults were picked
from one such run and are only as good as the scans behind them.

Two comparisons are made, both with the SAME one-sided
signed-z logic the detector itself uses:

    noise (--repeat):
        baseline vs. a SECOND empty-bucket scan, run over the same
        physical path (see scan_move.py). Any signed_z > 0 here is
        pure registration/timing noise - the bucket was empty in
        both scans - so its distribution is the noise floor the
        threshold must sit above.

    signal (--laundry, optional):
        baseline vs. a scan with known laundry actually in the
        bucket. Its signed_z distribution (restricted to the region
        actually covering the item, if you know it) is the real
        detection signal the threshold must sit below.

A good threshold sits in the gap between the noise distribution's
upper tail (e.g. p95/p99) and the signal distribution's lower tail
(e.g. p5/p10). If --laundry is omitted, only the noise-side
percentiles are reported - useful on its own for sanity-checking
how tight the one-sided noise floor is before recording a "known
laundry" reference scan.

This module does no ROS/graph work - like laundry_detect.py, it
only reads saved CSVs - so it works as a bare offline script.

Usage:
    calibrate_threshold \\
        --baseline baseline_scans/baseline.csv \\
        --repeat scan_records/scan_<second_empty_bucket_run>.csv

    calibrate_threshold \\
        --baseline baseline_scans/baseline.csv \\
        --repeat scan_records/scan_<second_empty_bucket_run>.csv \\
        --laundry scan_records/scan_<known_laundry_run>.csv
"""

import argparse

import numpy as np

from .laundry_detect import (
    compute_deviation,
    load_points_xyz,
)

PERCENTILES = (50, 90, 95, 99, 100)


def _percentile_table(label, values):

    if values.size == 0:
        print(f"{label}: no points to report.")
        return

    print(f"{label} (n={values.size}):")

    for p in PERCENTILES:
        print(f"    p{p:<3d} = {np.percentile(values, p):.4f} m")

    print(f"    mean  = {values.mean():.4f} m")


def _positive(signed_z):
    # Only the one-sided "candidate above baseline" tail is
    # physically meaningful for a threshold decision - see this
    # module's docstring and laundry_detect.compute_deviation().
    return signed_z[signed_z > 0.0]


def run(
    baseline_csv,
    repeat_csv,
    laundry_csv=None,
):

    baseline_xyz = load_points_xyz(baseline_csv)
    repeat_xyz = load_points_xyz(repeat_csv)

    print(f"Loaded baseline: {baseline_csv} ({baseline_xyz.shape[0]} pts)")
    print(f"Loaded repeat  : {repeat_csv} ({repeat_xyz.shape[0]} pts)")
    print()

    print("=" * 60)
    print("NOISE FLOOR (empty bucket vs. empty bucket)")
    print("=" * 60)

    point_noise = compute_deviation(baseline_xyz, repeat_xyz, threshold_m=0.0)
    _percentile_table(
        "signed_z > 0 only",
        _positive(point_noise.signed_z),
    )

    if laundry_csv is None:
        print()
        print(
            "No --laundry scan given - pick a threshold above the "
            "noise p95/p99 above, then re-run with --laundry once "
            "you have a known-laundry scan to confirm the signal "
            "side clears it."
        )
        return

    laundry_xyz = load_points_xyz(laundry_csv)

    print()
    print(f"Loaded laundry : {laundry_csv} ({laundry_xyz.shape[0]} pts)")
    print()

    print("=" * 60)
    print("SIGNAL (empty bucket vs. known laundry)")
    print("=" * 60)

    point_signal = compute_deviation(baseline_xyz, laundry_xyz, threshold_m=0.0)
    _percentile_table(
        "signed_z > 0 only",
        _positive(point_signal.signed_z),
    )

    print()
    print(
        "Pick a threshold in the gap between the NOISE p95/p99 "
        "above and the SIGNAL p5/p10 (i.e. re-run this print with "
        "np.percentile(..., [5, 10]) on the signal arrays if the "
        "gap isn't obvious from p50/p90/p95/p99 alone)."
    )


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Calibrate laundry_detect.py's deviation thresholds by "
            "comparing empty-bucket repeatability noise against "
            "known-laundry signal, using the same one-sided "
            "signed-z logic the detector uses."
        )
    )

    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to the empty-bucket baseline scan CSV.",
    )

    parser.add_argument(
        "--repeat",
        type=str,
        required=True,
        help=(
            "Path to a SECOND empty-bucket scan CSV, taken over "
            "the same physical path, used to measure the noise "
            "floor (baseline vs. baseline)."
        ),
    )

    parser.add_argument(
        "--laundry",
        type=str,
        default=None,
        help=(
            "Optional path to a scan CSV taken with known laundry "
            "in the bucket, used to measure the detection signal "
            "(baseline vs. laundry)."
        ),
    )

    return parser


def main():

    args = build_parser().parse_args()

    run(
        baseline_csv=args.baseline,
        repeat_csv=args.repeat,
        laundry_csv=args.laundry,
    )


if __name__ == "__main__":
    main()
