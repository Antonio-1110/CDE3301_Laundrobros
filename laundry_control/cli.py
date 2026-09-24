#!/usr/bin/env python3

"""
`laundry`: one command for every stage of the pipeline.

    laundry move home|inter|bottom|drop|retrieve_0..3
    laundry move joints J1 .. J7 [--degrees]
    laundry move joint6 DEG | joint7 DEG
    laundry move linear M | twist M DEG
    laundry scan [--save scan.csv]
    laundry detect scan.csv [-o targets.json] [--publish]   # offline
    laundry grasp targets.json [--drop] [--dry-run]
    laundry gripper open|close|ANGLE
    laundry baseline collect [--count N] [-- <scan options>]
    laundry baseline promote scan.csv
    laundry replay scan.csv
    laundry check-flange
    laundry evaluate [--sweep] [--synthetic] [--laundry scan.csv ...]
    laundry run [--dry-run]                                 # everything
    laundry preplanned

Stages hand off through files - a scan CSV, then a targets JSON - so
each one can run alone, be rerun offline, or be inspected in
between. `laundry run` chains them in memory in one process.

WHAT NEEDS WHAT
---------------
    detect, evaluate, baseline promote   nothing: plain files, no ROS
                                         graph, no arm
    replay                               a ROS graph (for RViz)
    move, check-flange                   MoveIt (real or fake)
    scan, grasp, run, preplanned         MoveIt + scan_recorder_node/
                                         tof_sensor/gripper_node - or
                                         --fake-hardware
    gripper open|close                   gripper_node
    gripper ANGLE                        the servo on this machine's GPIO

--fake-hardware swaps the gripper and the ToF recorder for stand-ins
(hardware/fake.py) while every arm motion still goes through MoveIt,
so the whole pipeline runs against the fake controller:

    ros2 launch laundry_control laundry_bringup.launch.py fake:=true
    laundry run --fake-hardware --scan-from baseline_scans/<one>.csv

Heavy imports (rclpy, MoveIt messages) happen inside each command, so
`laundry --help` and the offline commands stay fast.
"""

import argparse
import math
import sys

# =============================================================
# SHARED OPTION GROUPS
# =============================================================


def _add_detector_arguments(parser):
    from .perception.detect import (
        DEFAULT_ABS_FLOOR_M,
        DEFAULT_CLUSTER_RADIUS_M,
        DEFAULT_K_SIGMA,
        DEFAULT_MIN_CLUSTER_SIZE,
        DEFAULT_MIN_EXTENT_M,
        DEFAULT_MIN_VOLUME_M3,
    )

    group = parser.add_argument_group('detector')

    group.add_argument(
        '--baseline',
        type=str,
        default=None,
        help=(
            'Empty-bucket baseline scan CSV, or a directory of them '
            '(default: <repo>/baseline_scans).'
        ),
    )

    group.add_argument(
        '--k-sigma',
        type=float,
        default=DEFAULT_K_SIGMA,
        help=(
            'How many local noise sigmas a point must intrude past '
            f'the modelled wall to be flagged (default: {DEFAULT_K_SIGMA}).'
        ),
    )

    group.add_argument(
        '--abs-floor',
        type=float,
        default=DEFAULT_ABS_FLOOR_M,
        help=(
            'Absolute minimum intrusion (metres) regardless of sigma '
            f'(default: {DEFAULT_ABS_FLOOR_M}).'
        ),
    )

    group.add_argument(
        '--cluster-radius',
        type=float,
        default=DEFAULT_CLUSTER_RADIUS_M,
        help=(
            'Radius (metres) within which intruding points are linked '
            f'into one cluster (default: {DEFAULT_CLUSTER_RADIUS_M}).'
        ),
    )

    group.add_argument(
        '--min-cluster-size',
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=(
            'Pre-filter on raw point count; the real gates are '
            f'--min-extent and --min-volume (default: {DEFAULT_MIN_CLUSTER_SIZE}).'
        ),
    )

    group.add_argument(
        '--min-extent',
        type=float,
        default=DEFAULT_MIN_EXTENT_M,
        help=(
            'Minimum cluster footprint (metres) '
            f'(default: {DEFAULT_MIN_EXTENT_M}).'
        ),
    )

    group.add_argument(
        '--min-volume',
        type=float,
        default=DEFAULT_MIN_VOLUME_M3,
        help=(
            'Minimum integrated intrusion volume (cubic metres) '
            f'(default: {DEFAULT_MIN_VOLUME_M3}).'
        ),
    )


def _detect_params(args):
    return {
        'k_sigma': args.k_sigma,
        'abs_floor_m': args.abs_floor,
        'cluster_radius_m': args.cluster_radius,
        'min_cluster_size': args.min_cluster_size,
        'min_extent_m': args.min_extent,
        'min_volume_m3': args.min_volume,
    }


def _baseline_path(args):
    from . import config

    return args.baseline or config.baseline_dir()


def _add_observed_state_argument(parser):
    parser.add_argument(
        '--observed-start-state',
        action='store_true',
        help=(
            'Plan Cartesian strokes from the observed /joint_states '
            'instead of MoveIt\'s planning-scene state (always on with '
            '--fake-hardware; see HARDWARE_TESTS.md before using it on '
            'the real rig).'
        ),
    )


def _add_fake_arguments(parser, with_scan_from=False):
    _add_observed_state_argument(parser)

    parser.add_argument(
        '--fake-hardware',
        action='store_true',
        help=(
            'Replace the gripper (and the ToF recorder) with stand-ins; '
            'arm motions still go through MoveIt, so run the MoveIt '
            'fake controller (laundry_bringup.launch.py fake:=true).'
        ),
    )

    if with_scan_from:
        parser.add_argument(
            '--scan-from',
            type=str,
            default=None,
            help=(
                'With --fake-hardware: a saved scan CSV that stands in '
                "for the scan's output (the scan motion still runs)."
            ),
        )


# =============================================================
# ROS / DEVICE HELPERS
# =============================================================


class _RosSession:
    """rclpy.init() + an XArm7Controller, shut down cleanly on exit."""

    def __init__(self, need_arm=True, node_name='laundry', args=None):
        self.need_arm = need_arm
        self.node_name = node_name
        self.node = None

        # See XArm7Controller's plan_from_observed_state: required on
        # the fake controller, opt-in on the real rig until tested.
        self.plan_from_observed_state = bool(
            args is not None
            and (
                getattr(args, 'fake_hardware', False)
                or getattr(args, 'observed_start_state', False)
            )
        )

    def __enter__(self):
        import rclpy

        rclpy.init()

        if self.need_arm:
            from .arm.controller import XArm7Controller

            self.node = XArm7Controller(
                plan_from_observed_state=self.plan_from_observed_state
            )
        else:
            from rclpy.node import Node

            self.node = Node(self.node_name)

        return self.node

    def __exit__(self, exc_type, exc, traceback):
        import rclpy

        if self.node is not None:
            self.node.destroy_node()

        rclpy.shutdown()

        return False


def _make_gripper(node, fake):
    if fake:
        from .hardware.fake import FakeGripper

        return FakeGripper(node)

    from .hardware.gripper_client import GripperClient

    return GripperClient(node)


def _make_recorder(node, args):
    if args.fake_hardware:
        if not args.scan_from:
            raise SystemExit(
                '--fake-hardware has no ToF sensor: pass --scan-from '
                'a saved scan CSV to stand in for the scan output.'
            )

        from .hardware.fake import FakeRecorder

        return FakeRecorder(node, args.scan_from)

    from .scan.recorder_client import ScanRecorderClient

    return ScanRecorderClient(node)


# =============================================================
# COMMANDS
# =============================================================


def cmd_move(args):
    """Run `laundry move ...`."""
    from . import config

    # Resolve everything that can fail before touching ROS/MoveIt,
    # so a typo is reported immediately instead of after waiting on
    # MoveIt interfaces to come up.
    kind = args.move_kind

    joint_speed = (
        args.velocity if args.velocity is not None else 0.3,
        args.acceleration if args.acceleration is not None else 0.3,
    )

    cartesian_speed = (
        args.velocity if args.velocity is not None else 0.1,
        args.acceleration if args.acceleration is not None else 0.1,
    )

    if kind == 'joints':
        target = list(args.angles)

        if args.degrees:
            target = [math.radians(x) for x in target]

    elif kind in config.named_poses():
        target = config.get_named_pose(kind)

    with _RosSession(args=args) as arm:
        if kind == 'joint6':
            velocity, acceleration = joint_speed
            ok = arm.rotate_joint6(
                args.angle, velocity=velocity, acceleration=acceleration
            )

        elif kind == 'joint7':
            velocity, acceleration = joint_speed
            ok = arm.rotate_joint7(
                args.angle, velocity=velocity, acceleration=acceleration
            )

        elif kind == 'linear':
            velocity, acceleration = cartesian_speed
            ok = arm.move_tool_z(
                args.distance,
                max_step=args.step,
                velocity=velocity,
                acceleration=acceleration,
            )

        elif kind == 'twist':
            velocity, acceleration = cartesian_speed
            ok = arm.move_tool_z_with_twist(
                args.distance,
                args.angle,
                max_step=args.step,
                velocity=velocity,
                acceleration=acceleration,
            )

        else:
            velocity, acceleration = joint_speed
            ok = arm.move_joints(
                target, velocity=velocity, acceleration=acceleration
            )

    return 0 if ok else 1


def cmd_scan(args):
    """Run `laundry scan`."""
    from .pipeline import run_scan, timestamped_scan_path
    from .scan.pattern import scan_kwargs_from_args

    csv_path = args.save or timestamped_scan_path('scan')

    with _RosSession(args=args) as arm:
        recorder = _make_recorder(arm, args)

        ok = run_scan(arm, recorder, csv_path, scan_kwargs_from_args(args))

    if ok:
        print(f'Scan saved: {csv_path}')

    return 0 if ok else 1


def cmd_detect(args):
    """Run `laundry detect` (offline)."""
    from .pipeline import detect_scan

    baseline = _baseline_path(args)
    params = _detect_params(args)

    clusters, _surface = detect_scan(args.scan_csv, baseline, params)

    if args.output:
        from .grasp.targets_io import save_targets

        save_targets(args.output, clusters, args.scan_csv, baseline, params)

        print(f'Wrote {len(clusters)} target(s) to {args.output}')

    if args.publish:
        from .perception.report import publish_clusters

        publish_clusters(clusters, topic=args.topic)

    return 0


def cmd_grasp(args):
    """Run `laundry grasp targets.json`."""
    import os

    from .grasp.execute import grasp_best
    from .grasp.targets_io import load_targets
    from .perception.bucket_model import build_baseline_surface
    from .perception.detect import load_baseline_scans

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

    with _RosSession(args=args) as arm:
        gripper = _make_gripper(arm, args.fake_hardware)

        ok = grasp_best(
            arm,
            gripper,
            clusters,
            surface,
            drop=args.drop,
            dry_run=args.dry_run,
        )

    return 0 if ok else 1


def cmd_gripper(args):
    """Run `laundry gripper open|close|ANGLE`."""
    action = args.action.strip().lower()

    if action not in ('open', 'close'):
        try:
            angle = float(action)
        except ValueError:
            print(
                f"Expected 'open', 'close' or an angle; got {args.action!r}",
                file=sys.stderr,
            )
            return 2

        from .hardware import servo

        # Validate before touching GPIO, so a typo never moves it.
        servo.validate_angle(angle)

        if args.fake_hardware:
            print(f'[fake gripper] servo -> {angle:.1f} deg')
            return 0

        # Drives the servo directly, not through gripper_node: there
        # is no set-angle service, and this is for calibrating the
        # open/close angles in the first place. Don't run it while
        # the pipeline is actively using the gripper.
        servo.set_servo_angle(angle)
        return 0

    with _RosSession(need_arm=False, node_name='laundry_gripper') as node:
        gripper = _make_gripper(node, args.fake_hardware)

        if action == 'open':
            ok = gripper.open_blocking()
        else:
            ok = gripper.close_blocking()

    return 0 if ok else 1


def cmd_baseline(args):
    """Run `laundry baseline collect|promote`."""
    from .scan import baselines

    if args.baseline_action == 'collect':
        return baselines.run_collect(args, args.forwarded_scan_args)

    written = baselines.promote(args.src_path, dest=args.dest, force=args.force)

    print(f'Promoted {args.src_path!r} -> {written!r}')

    return 0


def cmd_replay(args):
    """Run `laundry replay scan.csv`."""
    from .scan import replay

    replay.main([args.scan_csv, '--topic', args.topic, '--frame', args.frame])

    return 0


def cmd_check_flange(args):
    """Run `laundry check-flange`."""
    from .arm import flange_check

    flange_check.main()

    return 0


def cmd_evaluate(args):
    """Run `laundry evaluate`."""
    from .perception import evaluate

    return evaluate.run_evaluate(args)


def cmd_run(args):
    """Run `laundry run`: scan -> detect -> grasp -> drop."""
    from .pipeline import run_full, timestamped_scan_path
    from .scan.pattern import scan_kwargs_from_args

    csv_path = args.save or timestamped_scan_path('run')

    with _RosSession(args=args) as arm:
        recorder = _make_recorder(arm, args)
        gripper = _make_gripper(arm, args.fake_hardware)

        ok = run_full(
            arm,
            recorder,
            gripper,
            csv_path=csv_path,
            baseline=_baseline_path(args),
            scan_kwargs=scan_kwargs_from_args(args),
            detect_params=_detect_params(args),
            dry_run=args.dry_run,
        )

    return 0 if ok else 1


def cmd_preplanned(args):
    """Run `laundry preplanned`."""
    from .pipeline import run_preplanned

    with _RosSession(args=args) as arm:
        ok = run_preplanned(arm, _make_gripper(arm, args.fake_hardware))

    return 0 if ok else 1


# =============================================================
# PARSER
# =============================================================


def _build_move_parser(subparsers):
    from . import config

    move = subparsers.add_parser(
        'move',
        help='Move the arm (named pose, joints, J6/J7, linear, twist).',
    )

    move.add_argument(
        '--velocity',
        type=float,
        default=None,
        help='MoveIt velocity scaling (default 0.3 joint, 0.1 Cartesian).',
    )

    move.add_argument(
        '--acceleration',
        type=float,
        default=None,
        help='MoveIt acceleration scaling (default 0.3 joint, 0.1 Cartesian).',
    )

    _add_observed_state_argument(move)

    kinds = move.add_subparsers(dest='move_kind', required=True)

    for name in config.named_poses():
        kinds.add_parser(name, help=f'Move to the recorded {name.upper()} pose.')

    joints = kinds.add_parser('joints', help='Move all seven joints.')
    joints.add_argument(
        'angles', type=float, nargs=7, metavar='J',
        help='Seven absolute target joint angles (radians).',
    )
    joints.add_argument(
        '--degrees', action='store_true',
        help='Interpret angles as degrees.',
    )

    for joint in ('joint6', 'joint7'):
        parser = kinds.add_parser(
            joint, help=f'Rotate {joint.upper()} relative to where it is.'
        )
        parser.add_argument(
            'angle', type=float, help='Relative rotation in degrees.'
        )

    linear = kinds.add_parser('linear', help='Move along current tool Z.')
    linear.add_argument('distance', type=float, help='Distance in metres.')
    linear.add_argument(
        '--step', type=float, default=0.005,
        help='Cartesian interpolation step (default: 0.005).',
    )

    twist = kinds.add_parser(
        'twist', help='Move along tool Z while rotating J7.'
    )
    twist.add_argument('distance', type=float, help='Distance in metres.')
    twist.add_argument(
        'angle', type=float, help='Additional J7 rotation in degrees.'
    )
    twist.add_argument(
        '--step', type=float, default=0.005,
        help='Cartesian interpolation step (default: 0.005).',
    )

    move.set_defaults(func=cmd_move)


def build_parser():
    """Build the full `laundry` argument parser."""
    from .perception.evaluate import add_evaluate_arguments
    from .perception.report import DEFAULT_TOPIC
    from .scan.baselines import add_collect_arguments
    from .scan.pattern import add_scan_arguments

    parser = argparse.ArgumentParser(
        prog='laundry',
        description='Scan the bucket, find laundry, and pull it out.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Run `laundry <command> --help` for each command.',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    _build_move_parser(subparsers)

    scan = subparsers.add_parser('scan', help='Scan the bucket into a CSV.')
    scan.add_argument(
        '--save', type=str, default=None,
        help='Where to save the scan (default: scan_records/scan_<time>.csv).',
    )
    add_scan_arguments(scan)
    _add_fake_arguments(scan, with_scan_from=True)
    scan.set_defaults(func=cmd_scan)

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
    _add_detector_arguments(detect)
    detect.set_defaults(func=cmd_detect)

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
    _add_fake_arguments(grasp)
    grasp.set_defaults(func=cmd_grasp)

    gripper = subparsers.add_parser(
        'gripper', help='Open/close the gripper, or set a raw servo angle.'
    )
    gripper.add_argument(
        'action', help="'open', 'close' (via gripper_node) or ANGLE in degrees."
    )
    _add_fake_arguments(gripper)
    gripper.set_defaults(func=cmd_gripper)

    baseline = subparsers.add_parser(
        'baseline', help='Collect or promote empty-bucket baseline scans.'
    )
    baseline_actions = baseline.add_subparsers(
        dest='baseline_action', required=True
    )
    collect = baseline_actions.add_parser(
        'collect',
        help='Record N empty-bucket scans (options after -- go to `laundry scan`).',
    )
    add_collect_arguments(collect)
    promote = baseline_actions.add_parser(
        'promote', help='Copy a saved empty-bucket scan into the baseline set.'
    )
    promote.add_argument('src_path', help='The empty-bucket scan CSV.')
    promote.add_argument(
        '--dest', type=str, default=None,
        help='Baseline directory (default: <repo>/baseline_scans).',
    )
    promote.add_argument(
        '--force', action='store_true', help='Overwrite an existing file.'
    )
    baseline.set_defaults(func=cmd_baseline)

    replay = subparsers.add_parser(
        'replay', help='Republish a saved scan for RViz.'
    )
    replay.add_argument('scan_csv', help='Scan CSV to replay.')
    replay.add_argument('--topic', type=str, default='scan_record/points')
    replay.add_argument('--frame', type=str, default='link_base')
    replay.set_defaults(func=cmd_replay)

    check = subparsers.add_parser(
        'check-flange',
        help="Report the insertion axis's alignment with the bucket.",
    )
    check.set_defaults(func=cmd_check_flange)

    evaluate = subparsers.add_parser(
        'evaluate', help='Measure false positives / recall of the detector.'
    )
    add_evaluate_arguments(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    run = subparsers.add_parser(
        'run', help='Full pipeline: scan -> detect -> grasp -> drop.'
    )
    run.add_argument(
        '--save', type=str, default=None,
        help='Where to save the scan (default: scan_records/run_<time>.csv).',
    )
    run.add_argument(
        '--dry-run', action='store_true',
        help='Stop after printing the grasp target.',
    )
    add_scan_arguments(run)
    _add_detector_arguments(run)
    _add_fake_arguments(run, with_scan_from=True)
    run.set_defaults(func=cmd_run)

    preplanned = subparsers.add_parser(
        'preplanned', help='Sensorless sweep over the recorded RETRIEVE poses.'
    )
    _add_fake_arguments(preplanned)
    preplanned.set_defaults(func=cmd_preplanned)

    return parser


def _split_forwarded(argv):
    """Split `laundry baseline collect ... -- <scan args>` at the --."""
    if 'baseline' in argv and '--' in argv:
        index = argv.index('--')
        return argv[:index], argv[index + 1:]

    return argv, []


def main(argv=None):
    """Entry point for the `laundry` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)

    own_argv, forwarded = _split_forwarded(argv)

    args = build_parser().parse_args(own_argv)
    args.forwarded_scan_args = forwarded

    try:
        code = args.func(args)
    except KeyboardInterrupt:
        print('Interrupted.', file=sys.stderr)
        code = 130
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f'laundry {args.command}: {exc}', file=sys.stderr)
        code = 1

    raise SystemExit(code)


if __name__ == '__main__':
    main()
