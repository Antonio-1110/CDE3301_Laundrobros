#!/usr/bin/env python3

"""
Baked motions: `laundry plan bake|replay|edit-grabs`.

`plan bake` solves the fixed motions once (INTER/BOTTOM, the grabs,
the routes from INTER, the end scan) and saves them in scan_plans/,
so every run replays the same motion. Run it through ./rebake.sh.
"""

import math
import sys

from .common import add_fake_arguments, add_observed_state_argument, RosSession


def cmd_plan(args):
    """Run `laundry plan bake [poses|endcap|transfers|retrieve|all]`."""
    import socket

    from .. import config
    from ..arm import bucket_poses, transfers
    from ..grasp import retrieve_grid
    from ..scan import endcap

    speed = math.radians(args.max_joint_speed)
    status = 0

    with RosSession(args=args) as arm:
        if args.which in ('poses', 'all'):
            # First: everything else is baked from INTER.
            print(
                'Deriving INTER and BOTTOM from the bucket '
                '(config.BUCKET_POSES)...'
            )
            poses, failures = bucket_poses.solve(arm, log=print)

            if poses:
                # Keep a previously derived pose that failed this time
                # out, rather than falling back to the recorded one.
                kept = {
                    name: joints
                    for name, joints in config.derived_bucket_poses().items()
                    if name in failures
                }
                bucket_poses.save(
                    dict(kept, **poses), baked_on=socket.gethostname()
                )
                print(f'Saved {bucket_poses.plan_path()}')

            if failures:
                print(
                    'Not derived: ' + ', '.join(n.upper() for n in failures)
                    + '. Check config.OBSTACLES and config.BUCKET_POSES.',
                    file=sys.stderr,
                )
                status = 1

                if args.which == 'all' and 'inter' in failures:
                    print(
                        'Not baking the rest: it would be baked from an INTER '
                        'that does not fit the bucket.',
                        file=sys.stderr,
                    )
                    return status

        if args.which in ('retrieve', 'all'):
            from ..grasp import grab_targets

            placed = grab_targets.load()

            if not placed:
                print(
                    f'No {grab_targets.path()}: place the grabs first '
                    '(laundry plan edit-grabs, SAVE). Grabs not baked.',
                    file=sys.stderr,
                )
                status = 1

                if args.which == 'retrieve':
                    return status

            else:
                print(
                    f'Solving the {len(placed)} grabs of '
                    'scan_plans/grab_targets.yaml exactly as placed...'
                )
                grabs, misses = retrieve_grid.solve(arm, placed, log=print)
                retrieve_grid.save(
                    grabs, baked_on=socket.gethostname(), targets=placed
                )
                print(
                    f'Saved {retrieve_grid.plan_path()} ({len(grabs)} grabs)'
                )

                if misses:
                    print(
                        f'{len(misses)} grab(s) unreachable and left out; '
                        'see above. Move them: laundry plan edit-grabs.',
                        file=sys.stderr,
                    )
                    status = 1

        if args.which in ('transfers', 'retrieve', 'all'):
            # `retrieve` re-bakes only the grab routes and keeps the
            # rest of transfers.yaml (if it matches the current scene).
            targets = (
                tuple(config.generated_grab_poses())
                if args.which == 'retrieve' else None
            )
            routes, clearances, failures = transfers.bake(
                arm, targets=targets, log=print
            )

            path = transfers.default_plan_path()
            transfers.save(
                path,
                routes,
                speed,
                baked_on=socket.gethostname(),
                clearances=clearances,
                keep=transfers.routes_to_keep(path, routes),
            )
            print(f'Saved {path} ({", ".join(routes) or "no routes"})')

            if failures:
                for name, reason in failures.items():
                    print(f'No route to {name.upper()}: {reason}', file=sys.stderr)

                print(
                    'Moves to those poses fall back to a straight or planned '
                    'move, not a baked one.',
                    file=sys.stderr,
                )
                status = 1

        if args.which in ('endcap', 'all'):
            try:
                plan = endcap.bake(
                    arm, args.depth, max_velocity_rad_s=speed, log=print
                )
            except endcap.BakeError as exc:
                print(f'End-scan bake failed: {exc}', file=sys.stderr)
                return 1

            plan.baked_on = socket.gethostname()
            output = args.output or endcap.default_plan_path()
            plan.save(output)

            print(f'Saved {output}')

            for alpha, low, high in plan.rings:
                print(f'  ring alpha={alpha:g} deg: phi {low:+g} .. {high:+g} deg')

            print(f'  {plan.duration_s:.1f} s, {len(plan.waypoints)} points.')

    print('Commit scan_plans/ so every run replays the same motion.')

    return status


def cmd_plan_edit_grabs(args):
    """Run `laundry plan edit-grabs`: drag the sweep's grabs in RViz."""
    from ..grasp import grab_editor

    argv = []

    if args.fill is not None:
        argv += ['--fill', str(args.fill)]

    if args.file:
        argv += ['--file', args.file]

    return grab_editor.main(argv)


def cmd_plan_replay(args):
    """Run `laundry plan replay`: the end scan alone, INTER to INTER."""
    from ..scan import endcap

    plan = endcap.EndcapPlan.load(args.end_plan or endcap.default_plan_path())

    from ..arm.transfers import go_to

    with RosSession(args=args) as arm:
        ok = (
            go_to(arm, 'inter')
            and arm.move_tool_z(plan.depth_m)
            and endcap.run_plan(arm, plan, plan.depth_m, time_scale=args.speed)
        )

        # Always try to come back out, even after a failure part-way.
        arm.move_joints_linear(plan.start, time_scale=args.speed)
        arm.move_tool_z(-plan.depth_m)
        go_to(arm, 'inter')

    return 0 if ok else 1


def add_plan_parser(subparsers):
    """Add `laundry plan` and its actions."""
    plan = subparsers.add_parser(
        'plan', help='Bake planner-free motions (the precession end scan).'
    )
    plan_actions = plan.add_subparsers(dest='plan_action', required=True)
    bake = plan_actions.add_parser(
        'bake',
        help=(
            'Solve, collision-check and save INTER/BOTTOM from the bucket, '
            "the sweep's grabs, the transfers from INTER to every named pose "
            'and the end-scan trajectory. The end scan MOVES THE ARM.'
        ),
    )
    bake.add_argument(
        'which', nargs='?',
        choices=('poses', 'endcap', 'transfers', 'retrieve', 'all'),
        default='all',
        help=(
            'What to bake (default: all, in the order poses, retrieve, '
            'transfers, endcap). poses: derive INTER and BOTTOM from the '
            'bucket (config.BUCKET_POSES); no motion. retrieve: solve the '
            "sweep's grabs exactly as placed (scan_plans/grab_targets.yaml, "
            '`laundry plan edit-grabs`) and bake routes to them.'
        ),
    )
    from ..scan.pattern import DEFAULT_DEPTH_M

    bake.add_argument(
        '--depth', type=float, default=DEFAULT_DEPTH_M,
        help=f'Scan depth the plan is for (default: {DEFAULT_DEPTH_M}).',
    )
    bake.add_argument(
        '-o', '--output', type=str, default=None,
        help='Where to save the end scan (default: <repo>/scan_plans/endcap.yaml).',
    )
    bake.add_argument(
        '--max-joint-speed', type=float, default=45.0,
        help='Peak joint speed along the plan, deg/s (default: 45).',
    )
    add_observed_state_argument(bake)
    bake.set_defaults(func=cmd_plan)

    replay_plan = plan_actions.add_parser(
        'replay',
        help=(
            'Run only the end scan: INTER, in to the plan depth, replay, '
            'back out. For checking clearance on the rig.'
        ),
    )
    replay_plan.add_argument(
        '--speed', type=float, default=0.3,
        help='Fraction of the baked speed, (0, 1] (default: 0.3).',
    )
    replay_plan.add_argument(
        '--end-plan', type=str, default=None,
        help='Plan file (default: <repo>/scan_plans/endcap.yaml).',
    )
    add_fake_arguments(replay_plan)
    replay_plan.set_defaults(func=cmd_plan_replay)

    edit_grabs = plan_actions.add_parser(
        'edit-grabs',
        help=(
            "Place the sweep's grabs by dragging them in RViz, CHECK "
            'their reachability (needs MoveIt; nothing moves) and SAVE '
            'them to scan_plans/grab_targets.yaml.'
        ),
    )
    edit_grabs.add_argument(
        '--fill', type=float, default=None,
        help='How full the drum is shown, as a fraction of its height '
             '(default: 2/3).',
    )
    edit_grabs.add_argument(
        '--file', type=str, default=None,
        help='Grab targets file (default: <repo>/scan_plans/grab_targets.yaml).',
    )
    edit_grabs.set_defaults(func=cmd_plan_edit_grabs)
