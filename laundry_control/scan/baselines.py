#!/usr/bin/env python3

"""
Build the empty-bucket baseline set the detector models the bucket from.

Two ways in:

    laundry baseline collect   - run N empty-bucket scans in a row
    laundry baseline promote   - add an already-saved scan after the fact

WHY A COMMAND AND NOT JUST RUNNING `laundry scan` EIGHT TIMES
-------------------------------------------------------------
Two things have to be right, and both are easy to get wrong by hand:

  - Each run must land in its own file, named for which baseline it
    is. Each run is a `laundry scan --save <path>`, which pins
    scan_recorder_node's csv_path for that one scan and puts it
    back afterwards (in a finally, so it survives Ctrl-C too). If it
    were left pinned at a baseline path, the next ordinary scan
    would silently overwrite a baseline with a scan that has laundry
    in it - poisoning the model with no visible symptom.

  - A scan that produced nothing must not pass silently. A ToF node
    that died, or a sensor that returned nothing but out-of-range
    readings, yields a valid-looking CSV with no points in it. Each
    run is checked for a plausible point count before it counts
    toward the total, and a failed one is moved aside as .rejected.

Each scan is a SUBPROCESS rather than an in-process call: a scan
that wedges can be killed without taking the collection with it,
and every run starts from a fresh rclpy context.

BEFORE COLLECTING
-----------------
  1. The bucket must be EMPTY, and stay empty for every run.
  2. The gripper must be OPEN, on every run. A closed gripper
     changes the geometry the sensor sees and shows up later as
     laundry that is not there.
  3. Nothing may move - not the bucket, not the robot base. Every
     baseline has to describe the same physical scene.
  4. `ros2 launch laundry_control laundry_bringup.launch.py` (or
     real_arm_scan.launch.py) must already be running.

Anything after -- is forwarded to `laundry scan`, so the baselines
can be captured with the same non-default scan settings the real
detection scans will use:

    laundry baseline collect --count 10 -- --depth 0.40 --sweep 150

HARDWARE ONLY (collect). promote is a plain file copy.
"""

import datetime
import os
import shutil
import subprocess
import sys
import time

from .. import config

DEFAULT_COUNT = 8

# A scan that comes back with fewer points than this did not really
# happen - the ToF node is down, the sensor is out of range, or the
# arm never moved. Far below a healthy scan (several thousand), so
# it only catches genuine failures.
DEFAULT_MIN_POINTS = 200


def count_points(csv_path):
    """Rows in a scan CSV, not counting the header."""

    if not os.path.isfile(csv_path):
        return 0

    with open(csv_path) as handle:
        return max(0, sum(1 for line in handle if line.strip()) - 1)


def run_one_scan(csv_path, scan_args, timeout_sec):
    """Run one `laundry scan --save csv_path`; True if it reported success."""
    command = [
        sys.executable,
        '-m',
        'laundry_control.cli',
        'scan',
        '--save',
        csv_path,
    ] + list(scan_args)

    try:
        result = subprocess.run(command, timeout=timeout_sec)

    except subprocess.TimeoutExpired:
        print(f'  ! scan exceeded {timeout_sec:.0f}s and was killed')
        return False

    return result.returncode == 0


def confirm_setup(count, dest):

    print("=" * 62)
    print(f"About to record {count} EMPTY-bucket baseline scans into")
    print(f"  {dest}")
    print("=" * 62)
    print()
    print("Check all of these before continuing:")
    print("  [ ] the bucket is EMPTY")
    print("  [ ] the gripper is OPEN")
    print("  [ ] the bucket and robot base will not be moved")
    print("  [ ] laundry_bringup.launch.py is running")
    print()

    answer = input("Type 'yes' to start: ").strip().lower()

    return answer == "yes"


def collect(
    count,
    dest,
    scan_args,
    min_points,
    keep_going,
    timeout_sec,
    settle_sec,
):

    os.makedirs(dest, exist_ok=True)

    session = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    written = []
    failed = []

    started = time.time()

    try:

        for index in range(1, count + 1):

            csv_path = os.path.join(
                dest, f"baseline_{session}_{index:02d}.csv"
            )

            elapsed = time.time() - started

            if written:
                per_scan = elapsed / len(written)
                remaining = per_scan * (count - index + 1)
                eta = f", ~{remaining / 60:.0f} min left"
            else:
                eta = ""

            print()
            print("-" * 62)
            print(
                f"Scan {index}/{count} -> {os.path.basename(csv_path)}"
                f"{eta}"
            )
            print("-" * 62)

            scan_started = time.time()
            ok = run_one_scan(csv_path, scan_args, timeout_sec)
            took = time.time() - scan_started

            points = count_points(csv_path)

            if not ok:
                reason = "laundry scan reported failure"

            elif points < min_points:
                # A CSV with almost nothing in it is worse than no
                # CSV: it would quietly drag the model toward
                # whatever handful of points it does contain.
                reason = f"only {points} points (expected >= {min_points})"

            else:
                reason = None

            if reason is None:
                written.append(csv_path)
                print(
                    f"  ok - {points} points in {took / 60:.1f} min"
                )

            else:
                failed.append((index, reason))
                print(f"  FAILED - {reason}")

                if os.path.isfile(csv_path):
                    os.replace(csv_path, csv_path + ".rejected")
                    print(
                        "  moved aside as "
                        f"{os.path.basename(csv_path)}.rejected so it "
                        "cannot be picked up as a baseline"
                    )

                if not keep_going:
                    print("\nStopping after a failed scan.")
                    break

            if index < count and settle_sec > 0:
                print(f"  settling {settle_sec:.0f}s...")
                time.sleep(settle_sec)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    return written, failed


def report(written, failed, dest):

    print()
    print("=" * 62)
    print(f"Collected {len(written)} baseline scan(s) into {dest}")
    print("=" * 62)

    for path in written:
        print(f"  {os.path.basename(path)}  ({count_points(path)} pts)")

    if failed:
        print()
        print(f"{len(failed)} scan(s) failed:")

        for index, reason in failed:
            print(f"  scan {index}: {reason}")

    total = len(
        [
            name
            for name in os.listdir(dest)
            if name.endswith(".csv")
        ]
    )

    print()
    print(f"{dest} now holds {total} baseline CSV(s) in total.")

    if total < 5:
        print(
            "\nThat is thin. The per-cell noise map needs about 8-10 "
            "empty scans before the detection threshold means much; "
            "below 5 most cells fall back to a pooled sigma. Run "
            "this again to top up - it adds to the set rather than "
            "replacing it."
        )

    print()
    print("Next, check the model the detector builds from these:")
    print()
    print(f"  laundry evaluate --baseline {dest}")
    print()
    print("Look for all of:")
    print("  - 'closed end : modelled as a flat cap'")
    print("  - the fitted cone within a few mm of the URDF seed")
    print("  - median cell count >= 5, empty cells near 0%")
    print("  - 0 false clusters across the leave-one-out folds")


def promote(src_path, dest=None, force=False):
    """
    Copy an already-saved empty-bucket scan into the baseline set.

    For when a scan is only decided to be a good empty-bucket
    reference after the fact. A baseline file is a completely
    ordinary scan CSV - nothing tags it as special on disk, so it
    stays viewable with `laundry replay`; being "a baseline" is
    purely a matter of which directory it sits in.

    `dest` is normally a DIRECTORY (default: config.baseline_dir());
    the scan keeps its own filename inside it, so repeated
    promotions accumulate into the 8-10 empty scans the model wants.
    A fixed destination filename would silently replace the set
    instead, leaving one scan and a sigma map that is entirely
    pooled fallback. An explicit .csv path still works for one-off
    use.

    Returns the path actually written.
    """
    if dest is None:
        dest = config.baseline_dir()

    if not os.path.isfile(src_path):
        raise FileNotFoundError(f'No such scan CSV: {src_path!r}')

    treat_as_dir = os.path.isdir(dest) or not dest.endswith('.csv')

    if treat_as_dir:
        dest_path = os.path.join(dest, os.path.basename(src_path))
    else:
        dest_path = dest

    if os.path.exists(dest_path) and not force:
        raise FileExistsError(
            f'{dest_path!r} already exists. Pass --force to overwrite it.'
        )

    dest_dir = os.path.dirname(dest_path)

    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    shutil.copyfile(src_path, dest_path)

    return dest_path


def add_collect_arguments(parser):
    """Add `laundry baseline collect` options to a parser."""
    parser.add_argument(
        '--count',
        type=int,
        default=DEFAULT_COUNT,
        help=f'How many empty scans to record (default: {DEFAULT_COUNT}).',
    )

    parser.add_argument(
        '--dest',
        type=str,
        default=None,
        help='Where to put them (default: <repo>/baseline_scans).',
    )

    parser.add_argument(
        '--min-points',
        type=int,
        default=DEFAULT_MIN_POINTS,
        help=(
            'Reject a scan that produced fewer points than this '
            f'(default: {DEFAULT_MIN_POINTS}).'
        ),
    )

    parser.add_argument(
        '--keep-going',
        action='store_true',
        help=(
            'Carry on after a failed scan instead of stopping. Off '
            'by default: a failure usually means something is wrong '
            'with the setup, and seven more scans will not fix it.'
        ),
    )

    parser.add_argument(
        '--settle',
        type=float,
        default=3.0,
        help=(
            'Seconds to wait between scans, so the arm comes fully '
            'to rest (default: 3).'
        ),
    )

    parser.add_argument(
        '--timeout',
        type=float,
        default=900.0,
        help=(
            'Give up on a single scan after this many seconds '
            '(default: 900).'
        ),
    )

    parser.add_argument(
        '--yes',
        action='store_true',
        help='Skip the empty-bucket confirmation prompt.',
    )


def run_collect(args, scan_args):
    """Run `laundry baseline collect`; returns a process exit code."""
    dest = args.dest or config.baseline_dir()

    if args.count < 1:
        print('--count must be at least 1.', file=sys.stderr)
        return 2

    if not args.yes and not confirm_setup(args.count, dest):
        print('Aborted.')
        return 1

    if scan_args:
        print(f"\nForwarding to laundry scan: {' '.join(scan_args)}")

    written, failed = collect(
        count=args.count,
        dest=dest,
        scan_args=scan_args,
        min_points=args.min_points,
        keep_going=args.keep_going,
        timeout_sec=args.timeout,
        settle_sec=args.settle,
    )

    report(written, failed, dest)

    return 0 if written and not failed else 1
