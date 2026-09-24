#!/usr/bin/env python3

"""
The stages chained together: scan -> detect -> grasp -> drop.

Each stage can also run on its own from the CLI, handing off through
files (scan CSV -> targets JSON -> grasp); `laundry run` chains them
here in one process and one arm connection instead:

    open gripper -> scan -> detect -> INTER -> approach + close
    (grasp) -> retract to INTER -> DROP -> open (release) -> INTER

The gripper is open for the whole run except while carrying the
item. Scanning open (not just idling open) is deliberate: the end
effector occludes part of the bucket, so the scan has to be taken in
the same gripper state the baselines were recorded in, or the
difference shows up as laundry that isn't there.

Also here: the sensorless `laundry preplanned` sweep.
"""

from datetime import datetime
import os

from . import config
from .grasp.execute import grasp_best
from .perception.bucket_model import build_baseline_surface
from .perception.detect import detect_on_points, load_baseline_scans
from .perception.detect import load_points_xyz
from .perception.report import print_model_report, print_report
from .scan.pattern import scan


def timestamped_scan_path(prefix='scan'):
    """Return a fresh <repo>/scan_records/<prefix>_<timestamp>.csv path."""
    return os.path.join(
        config.scan_records_dir(),
        f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )


def run_scan(arm, recorder, csv_path, scan_kwargs):
    """
    Scan the bucket into exactly csv_path; True on success.

    scan_recorder_node normally auto-generates a timestamped
    filename, which the caller would have no way to know in advance -
    so its csv_path parameter is pinned first, and ALWAYS handed back
    ("", auto-naming) afterwards, in a finally so it survives Ctrl-C.
    Left pinned, the next ordinary scan would silently overwrite this
    one.

    Old points are cleared before the arm moves (blocking is safe
    then), or they would be appended to this scan's.
    """
    csv_path = os.path.abspath(csv_path)

    directory = os.path.dirname(csv_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    print(f'Scan will be saved to {csv_path}')

    if not recorder.set_csv_path(csv_path):
        print("Failed to pin the recorder's csv_path; aborting.")
        return False

    try:
        recorder.clear_blocking()

        return scan(arm=arm, recorder=recorder, **scan_kwargs)

    finally:
        if not recorder.set_csv_path(''):
            print(
                "WARNING: could not restore scan_recorder_node's "
                'auto-naming. Do it by hand before the next scan:\n'
                '    ros2 param set /scan_recorder_node csv_path ""'
            )


def detect_scan(csv_path, baseline, detect_params):
    """
    Build the bucket model, report it, and detect laundry in csv_path.

    Returns (clusters, surface). The surface is returned because the
    grasp stage must plan against the SAME model the detection was
    judged against - fitting it twice would waste the expensive step
    and risk the two disagreeing.
    """
    baseline_scans = load_baseline_scans(baseline)
    surface = build_baseline_surface(baseline_scans)

    print_model_report(surface, baseline_scans, baseline)

    candidate_xyz = load_points_xyz(csv_path)

    clusters, _result = detect_on_points(
        candidate_xyz, surface, **detect_params
    )

    print_report(csv_path, candidate_xyz.shape[0], clusters, detect_params)
    print()

    return clusters, surface


def run_full(
    arm,
    recorder,
    gripper,
    csv_path,
    baseline,
    scan_kwargs,
    detect_params,
    dry_run=False,
):
    """Run scan -> detect -> grasp -> drop; True on success."""
    print('========== SCAN ==========')

    print('Opening gripper so the scan matches the baseline geometry...')

    if not gripper.open_blocking():
        print(
            'WARNING: could not confirm the gripper opened. If it is '
            "closed, this scan's end-effector occlusion differs from "
            "the baselines' and may produce phantom detections."
        )

    if not run_scan(arm, recorder, csv_path, scan_kwargs):
        print('Scan failed; aborting.')
        return False

    print('========== DETECT ==========')

    clusters, surface = detect_scan(csv_path, baseline, detect_params)

    if not clusters:
        print('No laundry detected; nothing to retrieve.')
        return True

    print('========== GRASP ==========')

    # scan() already returns to INTER at its own end; grasp_best()
    # moves there again explicitly so reachability probing always
    # starts from the known reference orientation regardless.
    return grasp_best(
        arm,
        gripper,
        clusters,
        surface,
        drop=True,
        dry_run=dry_run,
    )


# Highest RETRIEVE_n first, descending to lowest.
PREPLANNED_SEQUENCE = (
    'RETRIEVE_3',
    'RETRIEVE_2',
    'RETRIEVE_1',
    'RETRIEVE_0',
)


def run_preplanned(arm, gripper):
    """
    Sensorless "clear the bucket" sweep over fixed recorded poses.

    No sensor data and no detection at all - just visits the
    recorded RETRIEVE_n poses, closing the gripper at each one (as if
    grabbing something there) and opening it at DROP in between:

        INTER
        -> RETRIEVE_3 -> close -> DROP -> open
        -> RETRIEVE_2 -> close -> DROP -> open
        -> RETRIEVE_1 -> close -> DROP -> open
        -> RETRIEVE_0 -> close -> DROP -> open
        -> INTER

    Returns True if every motion succeeded.
    """
    print('Moving to INTER...')

    if not arm.move_joints(config.INTER):
        print('Failed to reach INTER; aborting.')
        return False

    for name in PREPLANNED_SEQUENCE:
        print(f'Moving to {name}...')

        if not arm.move_joints(config.get_named_pose(name)):
            print(f'Failed to reach {name}; aborting.')
            return False

        print('Closing gripper...')

        gripper.close_blocking()

        print('Moving to DROP...')

        if not arm.move_joints(config.DROP):
            print('Failed to reach DROP; aborting.')
            return False

        print('Opening gripper...')

        gripper.open_blocking()

    print('Returning to INTER...')

    return arm.move_joints(config.INTER)
