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
from .scan.segments import STROKES


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

        # The recorder records only during a scan; never leave it on
        # (a failed or interrupted scan would), or every later motion
        # piles points into the RViz cloud.
        if not recorder.set_recording(False):
            print(
                "WARNING: could not stop scan_recorder_node's recording. "
                'Do it by hand:\n'
                '    ros2 param set /scan_recorder_node recording false'
            )


def build_model(baseline, segments=None):
    """
    Fit the empty-bucket model from the baselines, and report it.

    segments: model only these scan segments of the baselines (scan/
    segments.py) - (STROKES,) for quick scans; None for full scans.
    """
    baseline_scans = load_baseline_scans(baseline, segments)
    surface = build_baseline_surface(baseline_scans)

    print_model_report(surface, baseline_scans, baseline)

    return surface


def detect_scan(csv_path, baseline, detect_params, surface=None,
                segments=None):
    """
    Build the bucket model, report it, and detect laundry in csv_path.

    Returns (clusters, surface). The surface is returned because the
    grasp stage must plan against the SAME model the detection was
    judged against - fitting it twice would waste the expensive step
    and risk the two disagreeing. Pass an already-built surface to
    skip the fit (run_clear detects many scans against one model).
    """
    if surface is None:
        surface = build_model(baseline, segments)

    if segments is None:
        candidate_xyz = load_points_xyz(csv_path)
    else:
        # Judge like with like: only the scan's readings of the
        # segments the model was built from (a quick scan has no
        # others anyway).
        from .scan.segments import load_points

        candidate_xyz = load_points(csv_path, segments)

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


def bucket_models(baseline):
    """
    Return model(kind): the bucket model for 'strokes' or 'full' scans.

    Each is fitted from the same baselines on first use (all of each
    scan for 'full', its strokes for 'strokes' - scan/segments.py) and
    reused after, so `clear` fits each at most once.
    """
    cache = {}

    def model(kind):
        if kind not in cache:
            print(f'========== BUCKET MODEL ({kind}) ==========')
            cache[kind] = build_model(
                baseline, (STROKES,) if kind == 'strokes' else None
            )

        return cache[kind]

    return model


def run_end_scan_only(arm, recorder, csv_path, scan_kwargs):
    """
    Run the end scan alone into exactly csv_path; True on success.

    Unlike run_scan, the recorder is NOT cleared: the quick scan just
    recorded is kept, so csv_path gets its strokes plus the end scan -
    a full scan's readings (scan.pattern.end_scan_only).
    """
    from .scan.pattern import end_scan_only

    csv_path = os.path.abspath(csv_path)

    if not recorder.set_csv_path(csv_path):
        print("Failed to pin the recorder's csv_path; aborting.")
        return False

    try:
        if not end_scan_only(arm=arm, recorder=recorder, **scan_kwargs):
            return False

        return recorder.save_blocking()

    finally:
        recorder.set_csv_path('')
        recorder.set_recording(False)


def scan_and_detect(
    arm, recorder, csv_path, baseline, scan_kwargs, detect_params, model
):
    """
    Scan and detect; returns (clusters, surface, scan CSV), or None.

    A full scan (end_scan 'precession') is detected against the full
    model. A QUICK scan (end_scan 'none') against the strokes-only
    model; if it finds nothing, the end scan runs on its own (no
    second pass of strokes) and the quick scan plus that pass - a full
    scan's readings, saved to <csv_path>_end.csv - is detected against
    the full model, so laundry against the closed end is still found.
    None if a motion failed.
    """
    quick = scan_kwargs.get('end_scan') == 'none'

    if not run_scan(arm, recorder, csv_path, scan_kwargs):
        print('Scan failed.')
        return None

    print('========== DETECT ==========')

    clusters, surface = detect_scan(
        csv_path, baseline, detect_params,
        surface=model('strokes' if quick else 'full'),
        segments=(STROKES,) if quick else None,
    )

    if clusters or not quick:
        return clusters, surface, csv_path

    if scan_kwargs.get('end_plan') is None:
        print('The quick scan found nothing; no baked end scan to check '
              'the closed end with.')
        return clusters, surface, csv_path

    print('========== THE QUICK SCAN FOUND NOTHING: END SCAN ONLY ==========')

    end_csv = os.path.splitext(csv_path)[0] + '_end.csv'

    if not run_end_scan_only(arm, recorder, end_csv, scan_kwargs):
        print('The end-scan pass failed.')
        return None

    print('========== DETECT (quick scan + end scan) ==========')

    clusters, surface = detect_scan(
        end_csv, baseline, detect_params, surface=model('full')
    )

    return clusters, surface, end_csv


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
    """
    Run scan -> detect -> grasp -> drop; True on success.

    With a quick scan that finds nothing, the end scan runs on its own
    before giving up (scan_and_detect).
    """
    print('========== SCAN ==========')

    if not open_for_scan(gripper, dry_run):
        return False

    result = scan_and_detect(
        arm, recorder, csv_path, baseline, scan_kwargs, detect_params,
        bucket_models(baseline),
    )

    if result is None:
        return False

    clusters, surface, _csv = result

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


def run_preplanned(arm, gripper, limit=None, time_scale=1.0):
    """
    Sensorless "clear the bucket" sweep over the generated grab grid.

    No sensor data and no detection: visits each baked grab
    (grasp/retrieve_grid.py, scan_plans/retrieve.yaml), closing the
    gripper there as if grabbing something, and opens it at DROP;
    limit keeps only the first N grabs. Refuses, without moving, if
    the grid has not been baked.

    Returns True if every motion succeeded.
    """
    from .arm import scene
    from .grasp import retrieve_grid

    grabs, stamp = retrieve_grid.load()

    if not grabs:
        print('No generated grab grid (scan_plans/retrieve.yaml). Bake it '
              'on the fake controller: ./rebake.sh retrieve')
        return False

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

    return retrieve_grid.run(arm, gripper, grabs, go_to, time_scale=time_scale)


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

    With quick scans (end_scan 'none', the default for `laundry
    clear`) each round is scan_and_detect's: a quick scan, and only
    when that finds nothing the end scan on its own. So the bucket is
    emptied with quick scans - which never swing the arm low - and the
    tilting end scan runs only once they find nothing, when little is
    left for it to disturb; the bucket is clear when that finds
    nothing too.

    Each bucket model is fitted once and every scan is judged against
    it. Stops after max_rounds, or max_failed failed grasps in a row.
    scan_path(round) names each round's scan CSV. Returns True only if
    the last scan found the bucket clear.
    """
    scan_path = scan_path or (
        lambda index: timestamped_scan_path(f'clear_{index:02d}')
    )

    model = bucket_models(baseline)

    # Fit the model the first scan needs before anything moves: a
    # baseline problem should stop the run here, not mid-sweep.
    model('strokes' if scan_kwargs.get('end_scan') == 'none' else 'full')

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

        result = scan_and_detect(
            arm, recorder, csv_path, baseline, scan_kwargs, detect_params,
            model,
        )

        if result is None:
            print('Scan failed; stopping.')
            return False

        clusters, surface, csv_path = result

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
