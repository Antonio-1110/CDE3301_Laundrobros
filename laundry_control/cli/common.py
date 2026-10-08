#!/usr/bin/env python3

"""What every `laundry` command shares: option groups, ROS session, devices."""

import os
import sys


# =============================================================
# SHARED OPTION GROUPS
# =============================================================


def add_detector_arguments(parser):
    """Add the detector thresholds (perception/detect.py) to a command."""
    from ..perception.detect import (
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


def detect_params(args):
    """Return the detector thresholds from add_detector_arguments as kwargs."""
    return {
        'k_sigma': args.k_sigma,
        'abs_floor_m': args.abs_floor,
        'cluster_radius_m': args.cluster_radius,
        'min_cluster_size': args.min_cluster_size,
        'min_extent_m': args.min_extent,
        'min_volume_m3': args.min_volume,
    }


def baseline_path(args):
    """Return --baseline, else the baseline set for this kind of scan."""
    from .. import config

    return args.baseline or config.baseline_dir()


def baselines_ready(args):
    """
    Return True if this scan's baseline set exists; say what to do if not.

    Checked before the arm moves: a scan detection cannot use (e.g. a
    quick scan with no quick baselines yet) only fails after scanning.
    """
    import glob

    path = baseline_path(args)

    if os.path.isfile(path) or glob.glob(os.path.join(path, '*.csv')):
        return True

    print(
        f'No baseline scans in {path}. Collect them first, bucket empty '
        '(full scans; quick scans use their strokes):\n'
        '    laundry baseline collect',
        file=sys.stderr,
    )
    return False


def add_observed_state_argument(parser):
    """Add --observed-start-state (see XArm7Controller)."""
    parser.add_argument(
        '--observed-start-state',
        action='store_true',
        help=(
            'Plan Cartesian strokes from the observed /joint_states '
            "instead of MoveIt's planning-scene state (always on with "
            '--fake-hardware; see HARDWARE_TESTS.md before using it on '
            'the real rig).'
        ),
    )


def add_fake_arguments(parser, with_scan_from=False):
    """Add --fake-hardware (and --scan-from) for commands that use devices."""
    add_observed_state_argument(parser)

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
            nargs='+',
            default=None,
            help=(
                'With --fake-hardware: a saved scan CSV that stands in '
                "for the scan's output (the scan motion still runs). "
                'Several are used one per scan, in order, the last '
                'repeating.'
            ),
        )


# =============================================================
# ROS / DEVICE HELPERS
# =============================================================


class RosSession:
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
        from rclpy.signals import SignalHandlerOptions

        # No rclpy SIGINT handler: it shuts the ROS context down the
        # moment Ctrl+C is pressed, so the MoveIt goal in flight could
        # no longer be cancelled - and a goal outlives the process
        # that sent it, so the arm would carry on moving. Python's own
        # handler raises KeyboardInterrupt instead, and
        # XArm7Controller cancels the goal before it propagates.
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

        if self.need_arm:
            from ..arm.controller import XArm7Controller

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


def make_gripper(node, fake):
    """Return the gripper: gripper_node's client, or the stand-in."""
    if fake:
        from ..hardware.fake import FakeGripper

        return FakeGripper(node)

    from ..hardware.gripper_client import GripperClient

    return GripperClient(node)


def make_recorder(node, args):
    """Return the ToF scan recorder's client, or the --scan-from stand-in."""
    if args.fake_hardware:
        if not args.scan_from:
            raise SystemExit(
                '--fake-hardware has no ToF sensor: pass --scan-from '
                'a saved scan CSV to stand in for the scan output.'
            )

        from ..hardware.fake import FakeRecorder

        return FakeRecorder(node, args.scan_from)

    from ..scan.recorder_client import ScanRecorderClient

    return ScanRecorderClient(node)
