#!/usr/bin/env python3

"""
The stages chained together: scan -> detect -> grasp -> drop.

Each stage can also run on its own from the CLI, handing off through
files (scan CSV -> targets JSON -> grasp); `laundry run` chains them
here in one process and one arm connection instead:

    open gripper -> scan -> detect -> INTER -> approach + close
    (grasp) -> retract to INTER -> DROP -> open (release) -> INTER

The scan searches coarse first (scan_and_detect): a quick 12 cm
pass, then the 3 cm one only if that found nothing, then the end scan
alone - detecting while the arm is still in the bucket, at the end
of each pass (and, on the 3 cm pass, already at the bottom of its
inward strokes), and going for the laundry as soon as one finds it.

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
import time

from . import config
from .arm.transfers import go_to, prepare_go_to
from .grasp.execute import describe_plan, execute_plan, grasp_best, plan_grasp
from .perception.bucket_model import build_baseline_surface
from .perception.detect import detect_on_points, load_baseline_scans
from .perception.detect import load_points_xyz
from .perception.report import print_model_report, print_report
from .scan.pattern import (
    CONFIDENT_MIN_PEAK_SIGMA, CONFIDENT_MIN_VOLUME_M3, DEFAULT_COARSE_STEPS_M,
    DEFAULT_SEARCH, DEFAULT_STEP_M, scan, SEARCHES,
)
from .scan.segments import STROKES


def timestamped_scan_path(prefix='scan'):
    """Return a fresh <repo>/scan_records/<prefix>_<timestamp>.csv path."""
    return os.path.join(
        config.scan_records_dir(),
        f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )


def run_scan(arm, recorder, csv_path, scan_kwargs, check=None):
    """
    Scan the bucket into exactly csv_path; True on success.

    check: scan()'s, called at the deepest point with the inward
    strokes already saved to csv_path.

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

        return scan(arm=arm, recorder=recorder, check=check, **scan_kwargs)

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


def run_end_scan_only(arm, recorder, csv_path, scan_kwargs,
                      start_inside=False):
    """
    Run the end scan alone into exactly csv_path; True on success.

    Unlike run_scan, the recorder is NOT cleared: the quick scan just
    recorded is kept, so csv_path gets its strokes plus the end scan -
    a full scan's readings (scan.pattern.end_scan_only). start_inside:
    from the entry pose that quick scan stayed at.
    """
    from .scan.pattern import end_scan_only

    csv_path = os.path.abspath(csv_path)

    if not recorder.set_csv_path(csv_path):
        print("Failed to pin the recorder's csv_path; aborting.")
        return False

    try:
        if not end_scan_only(
            arm=arm, recorder=recorder, start_inside=start_inside,
            **scan_kwargs
        ):
            return False

        return recorder.save_blocking()

    finally:
        recorder.set_csv_path('')
        recorder.set_recording(False)


def coarse_pass_kwargs(scan_kwargs, step):
    """
    Return scan_kwargs for a pass at a coarser step.

    The stroke velocity/acceleration scale with the step from the
    scan's own (as scan.pattern.scaled_stroke_speed does from the
    default), so every stroke takes about as long and J7 sweeps at the
    baselines' speed.
    """
    base = scan_kwargs.get('step', DEFAULT_STEP_M)
    kwargs = dict(scan_kwargs, step=step)

    for key in ('velocity', 'acceleration'):
        if key in scan_kwargs:
            kwargs[key] = min(1.0, scan_kwargs[key] * step / base)

    return kwargs


def coarse_pass_path(csv_path, step):
    """Return <csv_path>_<step>cm.csv, a coarse pass's own scan CSV."""
    root, ext = os.path.splitext(csv_path)
    return f'{root}_{step * 100:g}cm{ext}'


def _out_of_bucket(arm, result):
    """Leave the entry pose for INTER; result, or None if that failed."""
    print('========== OUT OF THE BUCKET ==========')

    if not go_to(arm, 'inter'):
        print('Could not move out to INTER (the arm is still in the '
              'bucket, at the entry pose): laundry move inter')
        return None

    return result


def confident_find(cluster):
    """
    Return True if a cluster is a confident find (in-out search).

    Big, strong and in well-sampled cells: scan.pattern.
    CONFIDENT_MIN_VOLUME_M3 / CONFIDENT_MIN_PEAK_SIGMA.
    """
    return bool(
        getattr(cluster, 'confident', False)
        and cluster.volume_m3 >= CONFIDENT_MIN_VOLUME_M3
        and cluster.peak_sigma >= CONFIDENT_MIN_PEAK_SIGMA
    )


def _search_in_out(
    arm, recorder, csv_path, baseline, scan_kwargs, detect_params, model,
    coarse_steps, bottom_check,
):
    """
    Search in one trip in and out (scan.pattern.SEARCHES); result or None.

    The result is (clusters, surface, scan CSV).

    The arm is back at INTER when this returns a result.
    """
    fine = scan_kwargs.get('step', DEFAULT_STEP_M)
    coarse = coarse_steps[0] if coarse_steps else fine
    end_plan = scan_kwargs.get('end_plan')

    kwargs = dict(
        coarse_pass_kwargs(scan_kwargs, coarse),
        end_scan='precession' if end_plan is not None else 'none',
        outward_step=fine,
        outward_velocity=scan_kwargs.get('velocity'),
        outward_acceleration=scan_kwargs.get('acceleration'),
        outward_when_found=True,
        start_inside=False,
        stay_inside=False,
    )

    found = []
    strokes_model = model('strokes')

    def check():
        print('========== DETECT (coarse inward strokes, at the bottom) '
              '==========')

        clusters, _ = detect_scan(
            csv_path, baseline, detect_params, surface=strokes_model,
            segments=(STROKES,),
        )
        found.extend(clusters)

        if not clusters:
            print('Nothing yet: end scan here, then the '
                  f'{fine * 100:g} cm strokes out.')
            return False

        if any(confident_find(cluster) for cluster in clusters):
            print('Confident find: no end scan; coming out on the '
                  f'{coarse * 100:g} cm strokes.')
            return coarse

        print('Weak find: no end scan; coming out on the '
              f'{fine * 100:g} cm strokes to confirm and aim.')
        return True

    print(f'========== SCAN IN AT {coarse * 100:g} CM, OUT AT '
          f'{fine * 100:g} CM ==========')

    started = time.monotonic()

    if not run_scan(
        arm, recorder, csv_path, kwargs,
        check=check if bottom_check else None,
    ):
        print('Scan failed.')
        return None

    print(f'Searching took {time.monotonic() - started:.1f} s.')

    # The end scan ran only when nothing was found at the bottom.
    full = end_plan is not None and not found
    surface = model('full') if full else strokes_model

    print('========== DETECT (' + (
        'inward and outward strokes, and the end scan' if full
        else 'inward and outward strokes'
    ) + ') ==========')

    clusters, _ = detect_scan(
        csv_path, baseline, detect_params, surface=surface,
        segments=None if full else (STROKES,),
    )

    if not clusters and not full and end_plan is None:
        print('No baked end scan: the closed end was not checked.')

    return clusters, surface, csv_path


def scan_and_detect(
    arm, recorder, csv_path, baseline, scan_kwargs, detect_params, model,
    coarse_steps=DEFAULT_COARSE_STEPS_M, bottom_check=True,
    search=DEFAULT_SEARCH,
):
    """
    Scan and detect; returns (clusters, surface, scan CSV), or None.

    The arm is back at INTER whenever this returns a result. None if
    a motion failed.

    A full scan (end_scan 'precession') is one pass, detected against
    the full model. QUICK scans (end_scan 'none') search as `search`
    says (SEARCHES): 'in-out' in one trip (_search_in_out), or 'ladder',
    coarse first, each pass detected against the strokes-only model:

        1. a pass at each of coarse_steps, coarsest first (saved to
           <csv_path>_<step>cm.csv), then one at the scan's own step
           (csv_path). Each pass is checked once complete, at the
           entry pose, and the arm stays in the bucket between
           passes. With bottom_check, the LAST pass is also checked
           at the deepest point on its inward strokes, and laundry
           found there ends it (straight out, no outward strokes).
           Coarse passes always run whole: their outward strokes are
           few (~9 s at 12 cm, against ~15 s to leave from the
           bottom), and they sharpen a coarse scan's poor aim.
        2. If none finds anything, the end scan runs on its own (no
           further strokes) and the last pass plus that - a full
           scan's readings, saved to <csv_path>_end.csv - is detected
           against the full model, so laundry against the closed end
           is still found.

    The bucket is only ever called empty after the scan's own step and
    the end scan: a coarse pass that finds nothing proves nothing.
    """
    if scan_kwargs.get('end_scan') != 'none':
        if not run_scan(arm, recorder, csv_path, scan_kwargs):
            print('Scan failed.')
            return None

        print('========== DETECT ==========')

        clusters, surface = detect_scan(
            csv_path, baseline, detect_params, surface=model('full')
        )

        return clusters, surface, csv_path

    if search == 'in-out':
        return _search_in_out(
            arm, recorder, csv_path, baseline, scan_kwargs, detect_params,
            model, coarse_steps, bottom_check,
        )

    if search != 'ladder':
        raise ValueError(f'Unknown search {search!r}; one of {SEARCHES}.')

    surface = model('strokes')
    started = time.monotonic()

    passes = [
        (step, coarse_pass_kwargs(scan_kwargs, step),
         coarse_pass_path(csv_path, step))
        for step in coarse_steps
    ] + [(scan_kwargs.get('step', DEFAULT_STEP_M), scan_kwargs, csv_path)]

    inside = False

    for index, (step, kwargs, path) in enumerate(passes):
        label = f'{step * 100:g} cm'
        last = index == len(passes) - 1
        found = []

        def check(path=path, found=found):
            print('========== DETECT (inward strokes, at the bottom) '
                  '==========')

            clusters, _ = detect_scan(
                path, baseline, detect_params, surface=surface,
                segments=(STROKES,),
            )
            found.extend(clusters)

            return bool(clusters)

        print(f'========== QUICK SCAN, {label} ==========')

        pass_started = time.monotonic()

        if not run_scan(
            arm, recorder, path,
            dict(kwargs, start_inside=inside, stay_inside=True),
            check=check if bottom_check and last else None,
        ):
            print('Scan failed.' + (
                ' The arm may still be in the bucket: laundry move inter'
                if inside else ''
            ))
            return None

        print(f'{label} pass: {time.monotonic() - pass_started:.1f} s '
              f'({time.monotonic() - started:.1f} s searching)')

        if found:
            # The scan stopped at the bottom and went out to INTER.
            print(f'Laundry found at the bottom of the {label} pass.')
            return found, surface, path

        inside = True

        print('========== DETECT ==========')

        clusters, _ = detect_scan(
            path, baseline, detect_params, surface=surface,
            segments=(STROKES,),
        )

        if clusters:
            return _out_of_bucket(arm, (clusters, surface, path))

    if scan_kwargs.get('end_plan') is None:
        print('The quick scans found nothing; no baked end scan to check '
              'the closed end with.')
        return _out_of_bucket(arm, ([], surface, csv_path))

    print('========== THE QUICK SCANS FOUND NOTHING: END SCAN ONLY '
          '==========')

    end_csv = os.path.splitext(csv_path)[0] + '_end.csv'

    if not run_end_scan_only(
        arm, recorder, end_csv, scan_kwargs, start_inside=True
    ):
        print('The end-scan pass failed.')
        return None

    print(f'Searching took {time.monotonic() - started:.1f} s.')

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
    coarse_steps=DEFAULT_COARSE_STEPS_M,
    bottom_check=True,
    search=DEFAULT_SEARCH,
):
    """
    Run scan -> detect -> grasp -> drop; True on success.

    Quick scans search coarse first, and the end scan runs on its own
    before giving up (scan_and_detect, with coarse_steps and
    bottom_check).
    """
    print('========== SCAN ==========')

    if not open_for_scan(gripper, dry_run):
        return False

    result = scan_and_detect(
        arm, recorder, csv_path, baseline, scan_kwargs, detect_params,
        bucket_models(baseline), coarse_steps, bottom_check, search,
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


def run_preplanned(arm, gripper, limit=None, time_scale=1.0, fill=None):
    """
    Sensorless "clear the bucket" sweep over the generated grab grid.

    No sensor data and no detection: visits each baked grab
    (grasp/retrieve_grid.py, scan_plans/retrieve.yaml), closing the
    gripper there as if grabbing something, and opens it at DROP;
    fill: start at that layer and sweep every one under it (grasp/
    pile_layers.select_layers); None, every layer. limit keeps only the
    first N grabs of those. Refuses, without moving, if the grid has
    not been baked.

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

    from .grasp import pile_layers

    grabs = pile_layers.select_layers(grabs, fill)

    if fill is not None and not any('layer' in grab for grab in grabs):
        print('WARNING: the grab list has no layers (placed by hand); '
              '--fill is ignored and every grab is swept.')

    grabs = grabs[:limit] if limit else grabs

    layers = f' (layers {fill} down to 1)' if fill is not None else ''
    print(f'Sensorless sweep over {len(grabs)} generated grab(s){layers}.')

    if not go_to(arm, 'inter', time_scale=time_scale):
        print('Failed to reach INTER; aborting.')
        return False

    return retrieve_grid.run(
        arm, gripper, grabs, go_to, prepare_go_to, time_scale=time_scale
    )


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
    sweep_fill=None,
    max_rounds=DEFAULT_CLEAR_MAX_ROUNDS,
    max_failed=DEFAULT_CLEAR_MAX_FAILED,
    scan_path=None,
    time_scale=1.0,
    coarse_steps=DEFAULT_COARSE_STEPS_M,
    bottom_check=True,
    search=DEFAULT_SEARCH,
):
    """
    Empty the bucket: the sensorless grab sweep, then scan until clear.

        1. the grab-grid sweep (run_preplanned, from layer sweep_fill
           down; sweep=False skips it) - laundry is expected at the
           start, so no scan is needed yet;
        2. open -> scan -> detect -> grasp the best reachable item ->
           DROP, repeated until a scan finds nothing.

    With quick scans (end_scan 'none', the default for `laundry
    clear`) each round is scan_and_detect's: coarse quick passes first
    (coarse_steps), then one at the scan's step (checked at the bottom
    with bottom_check), and only when those find nothing the end scan
    on its own. So the bucket is
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
            arm, gripper, limit=sweep_limit, time_scale=time_scale,
            fill=sweep_fill,
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
            model, coarse_steps, bottom_check, search,
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
