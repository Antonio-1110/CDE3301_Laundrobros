#!/usr/bin/env python3

"""
The main jobs: `laundry run`, `laundry clear`, `laundry preplanned`.

Each chains the stages (cli/stages.py) in one process - see
pipeline.py for what they do.
"""

from .common import (
    add_detector_arguments,
    add_fake_arguments,
    baseline_path,
    baselines_ready,
    detect_params,
    make_gripper,
    make_recorder,
    RosSession,
)


def cmd_run(args):
    """Run `laundry run`: scan -> detect -> grasp -> drop."""
    from ..pipeline import run_full, timestamped_scan_path
    from ..scan.pattern import scan_kwargs_from_args

    csv_path = args.save or timestamped_scan_path('run')

    if not baselines_ready(args):
        return 2

    with RosSession(args=args) as arm:
        recorder = make_recorder(arm, args)
        gripper = make_gripper(arm, args.fake_hardware)

        ok = run_full(
            arm,
            recorder,
            gripper,
            csv_path=csv_path,
            baseline=baseline_path(args),
            scan_kwargs=scan_kwargs_from_args(args),
            detect_params=detect_params(args),
            dry_run=args.dry_run,
        )

    return 0 if ok else 1


def cmd_clear(args):
    """Run `laundry clear`: grab sweep, then scan -> grasp until empty."""
    from ..pipeline import run_clear
    from ..scan.pattern import scan_kwargs_from_args

    if not baselines_ready(args):
        return 2

    with RosSession(args=args) as arm:
        ok = run_clear(
            arm,
            make_recorder(arm, args),
            make_gripper(arm, args.fake_hardware),
            baseline=baseline_path(args),
            scan_kwargs=scan_kwargs_from_args(args),
            detect_params=detect_params(args),
            sweep=not args.no_grabs,
            sweep_limit=args.grab_limit,
            max_rounds=args.max_rounds,
            max_failed=args.max_failed,
            time_scale=args.speed,
        )

    return 0 if ok else 1


def cmd_preplanned(args):
    """Run `laundry preplanned`."""
    from ..pipeline import run_preplanned

    with RosSession(args=args) as arm:
        ok = run_preplanned(
            arm,
            make_gripper(arm, args.fake_hardware),
            limit=args.limit,
            time_scale=args.speed,
        )

    return 0 if ok else 1


def add_run_parser(subparsers):
    """Add `laundry run`."""
    from ..scan.pattern import add_scan_arguments

    run = subparsers.add_parser(
        'run',
        help=(
            'Full pipeline: scan -> detect -> grasp -> drop. Quick scan by '
            'default, and if it finds nothing the end scan alone; --full '
            'for a full scan.'
        ),
    )
    run.add_argument(
        '--save', type=str, default=None,
        help='Where to save the scan (default: scan_records/run_<time>.csv).',
    )
    run.add_argument(
        '--dry-run', action='store_true',
        help='Stop after printing the grasp target.',
    )
    # The quick scan: it leaves the laundry where it lies (the full
    # scan's tilting end scan swings the arm low - scan.pattern).
    add_scan_arguments(run, end_scan_default='none')
    add_detector_arguments(run)
    add_fake_arguments(run, with_scan_from=True)
    run.set_defaults(func=cmd_run)


def add_clear_parser(subparsers):
    """Add `laundry clear`."""
    from ..scan.pattern import add_scan_arguments
    from ..pipeline import DEFAULT_CLEAR_MAX_FAILED, DEFAULT_CLEAR_MAX_ROUNDS

    clear = subparsers.add_parser(
        'clear',
        help=(
            'Empty the bucket: the grab-grid sweep, then scan -> detect -> '
            'grasp -> drop until a scan finds nothing. Quick scans by '
            'default, then the end scan alone once they find nothing; '
            '--full for full scans throughout.'
        ),
    )
    clear.add_argument(
        '--no-grabs', action='store_true',
        help='Skip the sensorless grab-grid pass; start with a scan.',
    )
    clear.add_argument(
        '--grab-limit', type=int, default=None,
        help='Only the first N grabs of the grid (they run mouth-first).',
    )
    clear.add_argument(
        '--max-rounds', type=int, default=DEFAULT_CLEAR_MAX_ROUNDS,
        help=(
            'Give up after this many scan -> grasp rounds '
            f'(default: {DEFAULT_CLEAR_MAX_ROUNDS}).'
        ),
    )
    clear.add_argument(
        '--max-failed', type=int, default=DEFAULT_CLEAR_MAX_FAILED,
        help=(
            'Give up after this many failed grasps in a row '
            f'(default: {DEFAULT_CLEAR_MAX_FAILED}).'
        ),
    )
    clear.add_argument(
        '--speed', type=float, default=1.0,
        help='Fraction of the baked speed for the grabs, (0, 1] (default: 1).',
    )
    # Quick scans, then the end scan alone once they find nothing
    # (pipeline.run_clear).
    add_scan_arguments(clear, end_scan_default='none')
    add_detector_arguments(clear)
    add_fake_arguments(clear, with_scan_from=True)
    clear.set_defaults(func=cmd_clear)


def add_preplanned_parser(subparsers):
    """Add `laundry preplanned`."""
    preplanned = subparsers.add_parser(
        'preplanned',
        help=(
            'Sensorless sweep: grab at each generated grab pose '
            '(scan_plans/retrieve.yaml) and drop.'
        ),
    )
    preplanned.add_argument(
        '--limit', type=int, default=None,
        help='Only the first N grabs (they run mouth-first).',
    )
    preplanned.add_argument(
        '--speed', type=float, default=1.0,
        help='Fraction of the baked speed, (0, 1] (default: 1).',
    )
    add_fake_arguments(preplanned)
    preplanned.set_defaults(func=cmd_preplanned)
