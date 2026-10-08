#!/usr/bin/env python3

"""
One stage at a time: `laundry scan`, `detect`, `grasp`, `replay`.

Stages hand off through files (a scan CSV, then a targets JSON), so
each can be run alone, rerun offline, or inspected in between.
"""

from .common import (
    add_detector_arguments,
    add_fake_arguments,
    baseline_path,
    detect_params,
    make_gripper,
    make_recorder,
    RosSession,
)


def cmd_scan(args):
    """Run `laundry scan`."""
    from ..pipeline import run_scan, timestamped_scan_path
    from ..scan.pattern import scan_kwargs_from_args

    csv_path = args.save or timestamped_scan_path('scan')

    with RosSession(args=args) as arm:
        recorder = make_recorder(arm, args)

        ok = run_scan(arm, recorder, csv_path, scan_kwargs_from_args(args))

    if ok:
        print(f'Scan saved: {csv_path}')

    return 0 if ok else 1


def cmd_detect(args):
    """Run `laundry detect` (offline)."""
    from ..pipeline import detect_scan

    baseline = baseline_path(args)
    params = detect_params(args)

    from ..scan.segments import end_scan_of, for_end_scan

    end_scan = args.end_scan

    if end_scan is None:
        end_scan = end_scan_of(args.scan_csv)
        if end_scan == 'none':
            print(
                'No end-scan readings: judging it as a QUICK scan, '
                "against the baselines' strokes."
            )
        else:
            print('Judging it as a FULL scan.')

    clusters, _surface = detect_scan(
        args.scan_csv, baseline, params,
        segments=for_end_scan(end_scan),
    )

    if args.output:
        from ..grasp.targets_io import save_targets

        save_targets(args.output, clusters, args.scan_csv, baseline, params)

        print(f'Wrote {len(clusters)} target(s) to {args.output}')

    if args.publish:
        from ..perception.report import publish_clusters

        publish_clusters(clusters, topic=args.topic)

    return 0


def cmd_grasp(args):
    """Run `laundry grasp targets.json`."""
    import os

    from ..grasp.execute import grasp_best
    from ..grasp.targets_io import load_targets
    from ..perception.bucket_model import build_baseline_surface
    from ..perception.detect import load_baseline_scans

    clusters, document = load_targets(args.targets_json)

    if not clusters:
        print('Targets file has no clusters; nothing to grasp.')
        return 0

    baseline = args.baseline or document.get('baseline')

    if args.baseline and document.get('baseline') and (
        os.path.abspath(args.baseline)
        != os.path.abspath(document['baseline'])
    ):
        print(
            f"WARNING: targets were detected against {document['baseline']} "
            f'but the grasp will be planned against {args.baseline}. '
            'Sink depth comes from the bucket model, so the two should '
            'normally match.'
        )

    surface = build_baseline_surface(load_baseline_scans(baseline))

    with RosSession(args=args) as arm:
        gripper = make_gripper(arm, args.fake_hardware)

        ok = grasp_best(
            arm,
            gripper,
            clusters,
            surface,
            drop=args.drop,
            dry_run=args.dry_run,
        )

    return 0 if ok else 1


def cmd_replay(args):
    """Run `laundry replay scan.csv`."""
    from ..scan import replay

    replay.main([args.scan_csv, '--topic', args.topic, '--frame', args.frame])

    return 0


def add_scan_parser(subparsers):
    """Add `laundry scan`."""
    from ..scan.pattern import add_scan_arguments

    scan = subparsers.add_parser('scan', help='Scan the bucket into a CSV.')
    scan.add_argument(
        '--save', type=str, default=None,
        help='Where to save the scan (default: scan_records/scan_<time>.csv).',
    )
    add_scan_arguments(scan)
    add_fake_arguments(scan, with_scan_from=True)
    scan.set_defaults(func=cmd_scan)


def add_detect_parser(subparsers):
    """Add `laundry detect`."""
    from ..perception.report import DEFAULT_TOPIC

    detect = subparsers.add_parser(
        'detect', help='Find laundry in a saved scan (offline).'
    )
    detect.add_argument('scan_csv', help='Scan CSV to search.')
    detect.add_argument(
        '-o', '--output', type=str, default=None,
        help='Write the ranked targets to this JSON file.',
    )
    detect.add_argument(
        '--publish', action='store_true',
        help='Publish the clusters as a PointCloud2 for RViz (needs ROS).',
    )
    detect.add_argument(
        '--topic', type=str, default=DEFAULT_TOPIC,
        help=f'Topic for --publish (default: {DEFAULT_TOPIC}).',
    )
    detect.add_argument(
        '--end-scan', choices=('precession', 'none', 'bottom'),
        default=None,
        help=(
            "The kind of scan the CSV is: 'none' for a quick scan (modelled "
            "from the baselines' strokes alone), else a full scan. Default: "
            'read from the CSV (full if it has end-scan readings).'
        ),
    )
    from ..scan.pattern import add_quick_full_flags

    add_quick_full_flags(detect)
    add_detector_arguments(detect)
    detect.set_defaults(func=cmd_detect)


def add_grasp_parser(subparsers):
    """Add `laundry grasp`."""
    grasp = subparsers.add_parser(
        'grasp', help='Grasp the best reachable target from a targets JSON.'
    )
    grasp.add_argument('targets_json', help='Output of `laundry detect -o`.')
    grasp.add_argument(
        '--baseline', type=str, default=None,
        help='Baseline set for the bucket model (default: the one in the JSON).',
    )
    grasp.add_argument(
        '--drop', action='store_true',
        help='Carry the item to DROP and release it (default: stop at INTER).',
    )
    grasp.add_argument(
        '--dry-run', action='store_true',
        help=(
            'Move to INTER and print the grasp target, but never '
            'approach, grip or drop.'
        ),
    )
    add_fake_arguments(grasp)
    grasp.set_defaults(func=cmd_grasp)


def add_replay_parser(subparsers):
    """Add `laundry replay`."""
    replay = subparsers.add_parser(
        'replay', help='Republish a saved scan for RViz.'
    )
    replay.add_argument('scan_csv', help='Scan CSV to replay.')
    replay.add_argument('--topic', type=str, default='scan_record/points')
    replay.add_argument('--frame', type=str, default='link_base')
    replay.set_defaults(func=cmd_replay)
