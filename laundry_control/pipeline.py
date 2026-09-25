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

Also here: the sensorless `laundry preplanned` sweep, and `laundry
clear`, which empties the bucket with both (run_clear).
"""

from datetime import datetime
import os

from . import config
from .arm.transfers import go_to
from .grasp.execute import describe_plan, execute_plan, grasp_best, plan_grasp
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


def build_model(baseline):
    """Fit the empty-bucket model from the baselines, and report it."""
    baseline_scans = load_baseline_scans(baseline)
    surface = build_baseline_surface(baseline_scans)

    print_model_report(surface, baseline_scans, baseline)

    return surface


def detect_scan(csv_path, baseline, detect_params, surface=None):
    """
    Build the bucket model, report it, and detect laundry in csv_path.

    Returns (clusters, surface). The surface is returned because the
    grasp stage must plan against the SAME model the detection was
    judged against - fitting it twice would waste the expensive step
    and risk the two disagreeing. Pass an already-built surface to
    skip the fit (run_clear detects many scans against one model).
    """
    if surface is None:
        surface = build_model(baseline)

    candidate_xyz = load_points_xyz(csv_path)

    clusters, _result = detect_on_points(
        candidate_xyz, surface, **detect_params
    )

    print_report(csv_path, candidate_xyz.shape[0], clusters, detect_params)
    print()

    return clusters, surface


def open_for_scan(gripper, dry_run=False):
    """Open the gripper before a scan; False if that must abort the run."""
    print('Opening gripper so the scan matches the baseline geometry...')

    if gripper.open_blocking():
        return True

    if not dry_run:
        print(
            'Could not confirm the gripper opened (is gripper_node '
            'running?). The grasp needs it; aborting before the scan.'
        )
        return False

    print(
        'WARNING: could not confirm the gripper opened. If it is '
        "closed, this scan's end-effector occlusion differs from "
        "the baselines' and may produce phantom detections."
    )

    return True


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

    if not open_for_scan(gripper, dry_run):
        return False

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


def run_preplanned(arm, gripper, recorded=False, limit=None, time_scale=1.0):
    """
    Sensorless "clear the bucket" sweep over fixed poses.

    Uses the generated grab grid (grasp/retrieve_grid.py,
    scan_plans/retrieve.yaml) when it has been baked, unless
    recorded=True; limit keeps only the first N grabs. Otherwise, the
    recorded poses below.

    RECORDED POSES

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
    from .arm import scene
    from .grasp import retrieve_grid

    grabs, stamp = ([], '') if recorded else retrieve_grid.load()

    if grabs:
        stale = scene.stale_plan_message(
            stamp, 'scan_plans/retrieve.yaml', 'laundry plan bake retrieve'
        )

        if stale:
            print(f'WARNING: {stale}')

        changed = retrieve_grid.grid_mismatch()

        if changed:
            print(f'WARNING: {changed}')

        grabs = grabs[:limit] if limit else grabs

        print(f'Sensorless sweep over {len(grabs)} generated grab(s).')

        if not go_to(arm, 'inter', time_scale=time_scale):
            print('Failed to reach INTER; aborting.')
            return False

        return retrieve_grid.run(
            arm, gripper, grabs, go_to, time_scale=time_scale
        )

    if not recorded:
        print('No generated grab grid (laundry plan bake retrieve); using '
              'the recorded RETRIEVE poses.')

    sequence = PREPLANNED_SEQUENCE[:limit] if limit else PREPLANNED_SEQUENCE

    print('Opening gripper...')

    if not gripper.open_blocking():
        print('Gripper did not confirm it opened; aborting before any grab.')
        return False

    print('Moving to INTER...')

    if not go_to(arm, 'inter', time_scale=time_scale):
        print('Failed to reach INTER; aborting.')
        return False

    for name in sequence:
        print(f'Moving to {name}...')

        if not go_to(arm, name, time_scale=time_scale):
            print(f'Failed to reach {name}; aborting.')
            return False

        print('Closing gripper...')

        if not gripper.close_blocking():
            print('Gripper did not confirm it closed; returning to INTER '
                  'and aborting.')
            go_to(arm, 'inter', time_scale=time_scale)
            return False

        print('Moving to DROP...')

        if not go_to(arm, 'drop', time_scale=time_scale):
            print('Failed to reach DROP; aborting.')
            return False

        print('Opening gripper...')

        if not gripper.open_blocking():
            print('Gripper did not confirm it opened at DROP; returning to '
                  'INTER and aborting.')
            go_to(arm, 'inter', time_scale=time_scale)
            return False

    print('Returning to INTER...')

    return go_to(arm, 'inter', time_scale=time_scale)


# `laundry clear` gives up after this many scan -> grasp rounds, or
# after this many grasps in a row that failed (an item nothing can
# reach, or a gripper problem). One failed grasp is worth a retry: the
# attempt itself may have moved the pile.
DEFAULT_CLEAR_MAX_ROUNDS = 15
DEFAULT_CLEAR_MAX_FAILED = 2


def run_clear(
    arm,
    recorder,
    gripper,
    baseline,
    scan_kwargs,
    detect_params,
    sweep=True,
    sweep_limit=None,
    max_rounds=DEFAULT_CLEAR_MAX_ROUNDS,
    max_failed=DEFAULT_CLEAR_MAX_FAILED,
    scan_path=None,
    time_scale=1.0,
):
    """
    Empty the bucket: the sensorless grab sweep, then scan until clear.

        1. the grab-grid sweep (run_preplanned; sweep=False skips it) -
           laundry is expected at the start, so no scan is needed yet;
        2. open -> scan -> detect -> grasp the best reachable item ->
           DROP, repeated until a scan finds nothing.

    The bucket model is fitted once and every scan is judged against
    it. Stops after max_rounds, or max_failed failed grasps in a row.
    scan_path(round) names each round's scan CSV. Returns True only if
    the last scan found the bucket clear.
    """
    scan_path = scan_path or (
        lambda index: timestamped_scan_path(f'clear_{index:02d}')
    )

    print('========== BUCKET MODEL ==========')

    surface = build_model(baseline)

    if sweep:
        print('========== GRAB SWEEP ==========')

        if not run_preplanned(
            arm, gripper, limit=sweep_limit, time_scale=time_scale
        ):
            print('The grab sweep failed; stopping before any scan.')
            return False

    retrieved = 0
    failed_in_a_row = 0

    for index in range(1, max_rounds + 1):
        print(f'========== ROUND {index}: SCAN ==========')

        csv_path = scan_path(index)

        if not open_for_scan(gripper):
            return False

        if not run_scan(arm, recorder, csv_path, scan_kwargs):
            print('Scan failed; stopping.')
            return False

        clusters, _surface = detect_scan(
            csv_path, baseline, detect_params, surface=surface
        )

        if not clusters:
            print(
                f'The bucket is clear: {retrieved} item(s) retrieved after '
                f'the sweep, in {index} scan(s).'
            )
            return True

        print(f'========== ROUND {index}: GRASP ==========')

        if not go_to(arm, 'inter'):
            print('Failed to reach INTER; stopping.')
            return False

        plan = plan_grasp(arm, clusters, surface)

        if plan is None:
            # Nothing moved, so a rescan would see the same pile.
            print(
                f'Laundry detected ({len(clusters)} cluster(s)) but none '
                f'is reachable; stopping. Last scan: {csv_path}'
            )
            return False

        print(describe_plan(plan))

        if execute_plan(arm, gripper, plan, drop=True):
            retrieved += 1
            failed_in_a_row = 0
            continue

        failed_in_a_row += 1

        if failed_in_a_row >= max_failed:
            print(
                f'{failed_in_a_row} grasps failed in a row; stopping with '
                'laundry still detected. Check the last scan: '
                f'{csv_path}'
            )
            return False

        print('Grasp failed; rescanning and trying again.')

    print(
        f'Stopped after {max_rounds} rounds ({retrieved} item(s) retrieved) '
        'with laundry still detected.'
    )

    return False
