#!/usr/bin/env python3

"""
collect_baselines.py

Run scan_move.py N times over an EMPTY bucket and collect the
results into baseline_scans/, ready for the detector to build its
model from.

WHY A SCRIPT AND NOT JUST RUNNING scan_move EIGHT TIMES
-------------------------------------------------------
Three things have to be right, and all three are easy to get wrong
by hand:

  - Each run must land in its own file. scan_recorder_node
    auto-names by timestamp only when csv_path is empty, so this
    pins csv_path explicitly per run and gives each scan a name
    that says which baseline it is.

  - csv_path must be put BACK afterwards. If it is left pinned at a
    baseline path, the next ordinary scan silently overwrites a
    baseline with a scan that has laundry in it - which would
    poison the model with no visible symptom. That reset happens in
    a finally block, so it survives Ctrl-C too.

  - A scan that produced nothing must not pass silently. A ToF node
    that died, or a sensor that returned nothing but out-of-range
    readings, yields a valid-looking CSV with no points in it. Each
    run is checked for a plausible point count before it counts
    toward the total.

scan_move.py is invoked as a SUBPROCESS rather than imported. It
calls rclpy.init()/shutdown() around each run and exits via
SystemExit, so one process per scan is the clean way to repeat it -
and it means a scan that wedges can be killed without taking this
script with it.

BEFORE RUNNING
--------------
  1. The bucket must be EMPTY, and stay empty for every run.
  2. The gripper must be OPEN, on every run. A closed gripper
     changes the geometry the sensor sees and shows up later as
     laundry that is not there.
  3. Nothing may move - not the bucket, not the robot base. Every
     baseline has to describe the same physical scene.
  4. real_arm_scan.launch.py must already be running.

Usage (from this directory - scan_move.py uses bare imports and
only resolves them when run from here):

    python3 collect_baselines.py
    python3 collect_baselines.py --count 10
    python3 collect_baselines.py --count 3 --keep-going

Anything after -- is forwarded to scan_move.py, so the baselines
can be captured with the same non-default scan settings the real
detection scans will use:

    python3 collect_baselines.py -- --depth 0.40 --sweep 150
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

SCAN_SCRIPT = "scan_move.py"

SCAN_RECORDER_NODE = "/scan_recorder_node"

DEFAULT_COUNT = 8

DEFAULT_DEST = (
    "/home/cde3301a/ros2_ws/src/CDE3301_Laundrobros/baseline_scans"
)

# A scan that comes back with fewer points than this did not really
# happen - the ToF node is down, the sensor is out of range, or the
# arm never moved. Far below a healthy scan (several thousand), so
# it only catches genuine failures.
DEFAULT_MIN_POINTS = 200


def set_csv_path(value, timeout_sec=15.0):
    """
    Point scan_recorder_node at an exact output file (or hand it
    back its auto-naming behaviour, by passing "").

    Uses the ros2 CLI rather than the SetParameters service so this
    stays a plain script with no rclpy node of its own - one less
    thing to initialise, and one less thing to leave running if a
    scan is interrupted.
    """

    result = subprocess.run(
        [
            "ros2",
            "param",
            "set",
            SCAN_RECORDER_NODE,
            "csv_path",
            value,
        ],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )

    if result.returncode != 0:
        print(
            f"  ! could not set csv_path: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
        return False

    return True


def count_points(csv_path):
    """Rows in a scan CSV, not counting the header."""

    if not os.path.isfile(csv_path):
        return 0

    with open(csv_path) as handle:
        return max(0, sum(1 for line in handle if line.strip()) - 1)


def run_one_scan(scan_args, timeout_sec):
    """
    Run scan_move.py once. Returns True if it reported success.

    cwd is this file's own directory because scan_move.py imports
    arm_position/move/scan_record as top-level modules, which only
    resolve when it runs from here.
    """

    here = os.path.dirname(os.path.abspath(__file__))

    try:
        result = subprocess.run(
            [sys.executable, SCAN_SCRIPT] + list(scan_args),
            cwd=here,
            timeout=timeout_sec,
        )

    except subprocess.TimeoutExpired:
        print(f"  ! scan exceeded {timeout_sec:.0f}s and was killed")
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
    print("  [ ] real_arm_scan.launch.py is running")
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

    # Bound before the loop: the finally block below names it while
    # restoring the recorder's default, and an interrupt can land
    # before the first iteration assigns it.
    csv_path = None

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

            if not set_csv_path(csv_path):
                failed.append((index, "could not set csv_path"))

                if not keep_going:
                    print(
                        "\nStopping. Is real_arm_scan.launch.py "
                        "running?"
                    )
                    break

                continue

            scan_started = time.time()
            ok = run_one_scan(scan_args, timeout_sec)
            took = time.time() - scan_started

            points = count_points(csv_path)

            if not ok:
                reason = "scan_move.py reported failure"

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

    finally:

        # Always hand scan_recorder_node its auto-naming back. If it
        # were left pinned at a baseline path, the next ordinary
        # scan would overwrite a baseline with a scan that has
        # laundry in it, and nothing would say so.
        print("\nRestoring scan_recorder_node's default csv_path...")

        if not set_csv_path(""):

            clash = (
                f" or it will overwrite {os.path.basename(csv_path)}"
                if csv_path
                else ""
            )

            print(
                "  ! FAILED to reset csv_path. Do it by hand before "
                f"the next scan{clash}:\n"
                f"    ros2 param set {SCAN_RECORDER_NODE} csv_path \"\""
            )

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
    print("  ros2 run laundry_control validate_detector \\")
    print(f"    --baseline {dest}")
    print()
    print("Look for all of:")
    print("  - 'closed end : modelled as a flat cap'")
    print("  - the fitted cone within a few mm of the URDF seed")
    print("  - median cell count >= 5, empty cells near 0%")
    print("  - 0 false clusters across the leave-one-out folds")


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Record several empty-bucket scans into baseline_scans/ "
            "for the laundry detector to model the bucket from."
        ),
        epilog=(
            "Arguments after -- are passed through to scan_move.py, "
            "e.g. collect_baselines.py -- --depth 0.40"
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=(
            "How many empty scans to record "
            f"(default: {DEFAULT_COUNT})."
        ),
    )

    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_DEST,
        help=f"Where to put them (default: {DEFAULT_DEST}).",
    )

    parser.add_argument(
        "--min-points",
        type=int,
        default=DEFAULT_MIN_POINTS,
        help=(
            "Reject a scan that produced fewer points than this "
            f"(default: {DEFAULT_MIN_POINTS})."
        ),
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "Carry on after a failed scan instead of stopping. Off "
            "by default: a failure usually means something is wrong "
            "with the setup, and seven more scans will not fix it."
        ),
    )

    parser.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help=(
            "Seconds to wait between scans, so the arm comes fully "
            "to rest (default: 3)."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help=(
            "Give up on a single scan after this many seconds "
            "(default: 900)."
        ),
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the empty-bucket confirmation prompt.",
    )

    return parser


def main():

    argv = sys.argv[1:]

    if "--" in argv:
        split = argv.index("--")
        own_args, scan_args = argv[:split], argv[split + 1:]
    else:
        own_args, scan_args = argv, []

    args = build_parser().parse_args(own_args)

    if args.count < 1:
        raise SystemExit("--count must be at least 1.")

    if not args.yes and not confirm_setup(args.count, args.dest):
        raise SystemExit("Aborted.")

    if scan_args:
        print(f"\nForwarding to scan_move.py: {' '.join(scan_args)}")

    written, failed = collect(
        count=args.count,
        dest=args.dest,
        scan_args=scan_args,
        min_points=args.min_points,
        keep_going=args.keep_going,
        timeout_sec=args.timeout,
        settle_sec=args.settle,
    )

    report(written, failed, args.dest)

    raise SystemExit(0 if written and not failed else 1)


if __name__ == "__main__":
    main()
