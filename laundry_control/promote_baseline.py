#!/usr/bin/env python3

"""
promote_baseline.py

Copies an already-saved scan CSV to the well-known baseline path
(baseline_scans/baseline.csv by default - kept in its own
directory, separate from scan_records/, so it never gets mixed up
with or accidentally treated as just another recorded scan), for
when a scan is only decided to be a good empty-bucket reference
after the fact.

To capture a baseline directly instead of promoting one after the
fact, run scan_recorder_node with an explicit output path:

    ros2 run laundry_control scan_recorder_node \\
        --ros-args -p csv_path:=baseline_scans/baseline.csv

then run scan_move.py once with the bucket empty.

A baseline file is a completely ordinary scan CSV (same
x,y,z,stamp_sec,stamp_nanosec format) - nothing tags it as special
on disk, so it stays viewable via scan_replay.py unmodified; being
"the baseline" is purely a matter of which path laundry_detect_cli
is pointed at via --baseline.
"""

import argparse
import os
import shutil

DEFAULT_DEST_PATH = "baseline_scans/baseline.csv"


def promote(
    src_path: str,
    dest_path: str = DEFAULT_DEST_PATH,
    force: bool = False,
) -> None:

    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"No such scan CSV: {src_path!r}")

    if os.path.exists(dest_path) and not force:

        raise FileExistsError(
            f"{dest_path!r} already exists. Pass --force to "
            "overwrite it."
        )

    dest_dir = os.path.dirname(dest_path)

    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    shutil.copyfile(src_path, dest_path)


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Promote an already-saved scan CSV to the well-known "
            "baseline path."
        )
    )

    parser.add_argument(
        "src_path",
        type=str,
        help=(
            "Path to the scan CSV to promote "
            "(should be an empty-bucket scan)."
        ),
    )

    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_DEST_PATH,
        help=f"Destination path (default: {DEFAULT_DEST_PATH}).",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing baseline file.",
    )

    return parser


def main():

    args = build_parser().parse_args()

    promote(
        src_path=args.src_path,
        dest_path=args.dest,
        force=args.force,
    )

    print(f"Promoted {args.src_path!r} -> {args.dest!r}")


if __name__ == "__main__":
    main()
