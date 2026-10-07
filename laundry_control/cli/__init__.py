#!/usr/bin/env python3

"""
`laundry`: one command for every stage of the pipeline.

HOW TO READ THE USAGE LINES
---------------------------
    [--opt]     optional
    a|b|c       pick one of these
    N, M, DEG   a number you supply (count, metres, degrees)
    ...         more of the same

Every command has `--help` with the full list of options; below are
only the ones you normally need. Commands marked * MOVE THE ARM.

THE MAIN JOBS
-------------
  * laundry clear [--no-grabs] [--max-rounds N]
        Empty the bucket: blind grabs over the grab grid, then
        scan -> grasp rounds until a scan finds nothing.
  * laundry run [--dry-run]
        One round: scan -> detect -> grasp the best item -> DROP.
  * laundry preplanned [--limit N] [--speed 0.3]
        Blind grabs only: every grab_NN in turn, no sensing.

ONE STAGE AT A TIME
-------------------
Stages hand off through files (a scan CSV, then a targets JSON), so
each can be run alone, rerun offline, or inspected in between.
`run` and `clear` chain the same stages in memory.

  * laundry scan [--save scan.csv] [--full]
        Sweep the ToF sensor through the bucket into a CSV.
    laundry detect scan.csv [-o targets.json] [--publish]
        Find laundry in a scan by comparing it with the baselines.
  * laundry grasp targets.json [--drop] [--dry-run]
        Grasp the best reachable target from `detect -o`.
    laundry replay scan.csv
        Publish a saved scan to RViz.

MOVING AND CHECKING THE ARM
---------------------------
  * laundry move home|inter|bottom|drop|grab_NN
        Go to a named pose (baked route if there is one).
  * laundry move joints J1 .. J7 [--degrees]
  * laundry move joint6 DEG | joint7 DEG       turn one joint by DEG
  * laundry move linear M | twist M DEG        along the tool axis by M
                                               metres (twist: also J7)
    laundry gripper open|close|ANGLE
    laundry check-flange
        Is the tool axis at INTER lined up with the bucket axis?

THE BUCKET, OBSTACLES AND BAKED MOTIONS
---------------------------------------
    laundry scene apply|check|fit
        apply: put the padded bucket and table into MoveIt.
        check: collision-check every named pose and baked route.
        fit:   measure where the bucket really is, from the baselines.
    laundry plan edit-grabs [--fill 0.67]
        Place the sweep's grabs by dragging them in RViz; CHECK
        their reach, SAVE to scan_plans/grab_targets.yaml.
  * laundry plan bake [poses|retrieve|transfers|endcap|all]
        Solve and save the fixed motions in scan_plans/ (INTER/BOTTOM,
        the sweep's grabs, routes from INTER, end scan). Run it
        through ./rebake.sh, on the fake controller.
  * laundry plan replay [--speed 0.3]
        Run only the end scan, INTER to INTER.

BASELINES AND DETECTOR TUNING
-----------------------------
  * laundry baseline collect [--count N] [--archive]
        Record N empty-bucket scans. --archive replaces the current
        set (the old one moves to baseline_scans/archive/).
    laundry baseline promote scan.csv|dir ... [--move] [--archive]
    laundry baseline archive | list | restore LABEL
    laundry evaluate [--sweep] [--synthetic] [--laundry scan.csv ...]
        Measure the detector: false positives on the baselines, and
        recall on scans with known laundry.

FULL AND QUICK SCANS (--full / --quick)
---------------------------------------
    quick  (--quick, --end-scan none)
           Strokes in and out of the bucket only. Doesn't disturb the
           laundry, but never sees the closed end.
    full   (--full, --end-scan precession)
           The strokes plus the end scan: the tool tilts in a cone at
           the deepest point - the only view of the closed end, but it
           swings the arm low near the walls.

    Which one each command does by default:
        scan, run, clear     quick. run/clear add the end scan on its
                             own when a quick scan finds nothing,
                             before saying the bucket is empty.
        baseline collect     always full.
        evaluate             full.
        detect               reads it from the CSV (were end-scan
                             readings recorded?); --quick/--full
                             override.

    There is one baseline set, baseline_scans/, of full scans. A quick
    scan is compared with their stroke readings alone
    (scan/segments.py).

WHAT EACH COMMAND NEEDS RUNNING
-------------------------------
    detect, evaluate, scene fit,   nothing: plain files
    baseline promote|archive|list|restore
    replay, plan edit-grabs        a ROS graph (RViz to look at it;
                                   edit-grabs' CHECK: MoveIt too)
    move, check-flange,            MoveIt, real or fake
    scene apply|check, plan
    scan, grasp, run, clear,       the full bring-up: MoveIt plus the
    preplanned, baseline collect   ToF, recorder and gripper nodes
    gripper open|close             gripper_node
    gripper ANGLE                  the servo on this machine's GPIO,
                                   or the ESP32's (GRIPPER_BACKEND)

    Without the hardware, --fake-hardware swaps the gripper and the
    ToF recorder for stand-ins (hardware/fake.py); the arm still goes
    through MoveIt, on the fake controller:

        ros2 launch laundry_control laundry_bringup.launch.py fake:=true
        laundry run --fake-hardware --scan-from baseline_scans/<one>.csv

WHERE EACH COMMAND LIVES
------------------------
    cli/jobs.py      run, clear, preplanned
    cli/stages.py    scan, detect, grasp, replay
    cli/arm.py       move, gripper, check-flange
    cli/scene.py     scene apply|check|fit
    cli/plan.py      plan bake|replay|edit-grabs
    cli/baseline.py  baseline ..., evaluate
    cli/common.py    the option groups and ROS helpers they share

Each module holds a command's handler (cmd_*) next to the parser that
reaches it (add_*_parser). Heavy imports (rclpy, MoveIt messages)
happen inside each command, so `laundry --help` and the offline
commands stay fast.
"""

import argparse
import sys

from . import arm, baseline, jobs, plan, scene, stages


def build_parser():
    """Build the full `laundry` argument parser."""
    parser = argparse.ArgumentParser(
        prog='laundry',
        description='Scan the bucket, find laundry, and pull it out.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Run `laundry <command> --help` for each command.',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    # In the order of the overview above.
    for add_parser in (
        jobs.add_run_parser,
        jobs.add_clear_parser,
        jobs.add_preplanned_parser,
        stages.add_scan_parser,
        stages.add_detect_parser,
        stages.add_grasp_parser,
        stages.add_replay_parser,
        arm.add_move_parser,
        arm.add_gripper_parser,
        arm.add_check_flange_parser,
        scene.add_scene_parser,
        plan.add_plan_parser,
        baseline.add_baseline_parser,
        baseline.add_evaluate_parser,
    ):
        add_parser(subparsers)

    return parser


def split_forwarded(argv):
    """Split `laundry baseline collect ... -- <scan args>` at the --."""
    if 'baseline' in argv and '--' in argv:
        index = argv.index('--')
        return argv[:index], argv[index + 1:]

    return argv, []


def main(argv=None):
    """Entry point for the `laundry` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # Launched as a ROS node (laundry_bringup.launch.py's `scene apply`),
    # launch may append ROS arguments; they are not ours.
    if '--ros-args' in argv:
        argv = argv[:argv.index('--ros-args')]

    own_argv, forwarded = split_forwarded(argv)

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
