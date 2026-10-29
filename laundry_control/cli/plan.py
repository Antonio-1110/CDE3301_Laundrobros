#!/usr/bin/env python3

"""
Baked motions: `laundry plan bake|replay|edit-grabs`.

`plan bake` solves the fixed motions once (INTER/BOTTOM, the grabs,
the routes from INTER, the end scan) and saves them in scan_plans/,
so every run replays the same motion. Run it through ./rebake.sh.
"""

import math
import sys

import numpy as np

from .common import add_fake_arguments, add_observed_state_argument, RosSession


def _bake_timing(arm):
    """
    Time every baked transfer with vias to pass through them; save them.

    Route geometry is not changed. Routes without vias keep the
    stop-at-each timing, which is already as fast for a single leg.
    Each timing's curve is collision-checked with the route's own
    gripper clearance, as the route itself was baked.
    """
    from .. import config
    from ..arm import pass_through, transfers
    from ..arm.joint_path import time_stop_at_each

    routes, _speed = transfers.load()

    if not routes:
        print('No baked transfers to time (scan_plans/transfers.yaml).')
        return 1

    v_max = config.TRANSFER_MAX_VELOCITY_RAD_S
    a_max = config.TRANSFER_MAX_ACCELERATION_RAD_S2
    j_max = config.TRANSFER_MAX_JERK_RAD_S3
    limits = pass_through.current_limits()

    with_vias = {
        name: waypoints for name, waypoints in routes.items()
        if len(waypoints) > 2
    }
    # Routes without vias keep the stop-at-each timing.
    timings = {name: None for name in routes if name not in with_vias}
    status = 0

    print(
        f'Timing {len(with_vias)} transfer(s) with vias to pass through them '
        f'(limits {math.degrees(v_max):g} deg/s, {a_max:g} rad/s^2, '
        f'{j_max:g} rad/s^3); TOTG runs in its own MoveIt (~25 s)...',
        flush=True,
    )

    chains = transfers.chain_pairs(routes)
    chain_waypoints = {
        name: transfers.chain_waypoints(routes, back, out)
        for name, (back, out) in chains.items()
    }

    if chains:
        print(
            f'... and {len(chains)} grab <-> DROP move(s) in one piece, '
            'passing through INTER.',
            flush=True,
        )

    directs = transfers.load_direct()

    if directs:
        print(
            f'... and {len(directs)} direct grab <-> DROP route(s), not '
            'through INTER.',
            flush=True,
        )

    tip_speed = config.TRANSFER_MAX_TIP_SPEED_M_S
    print(
        f'... the claw kept under {tip_speed * 1000:.0f} mm/s on every '
        'route (C24 on the xArm over 1000 mm/s).',
        flush=True,
    )

    # Every route goes to the child, those without vias too: each gets
    # its stop-at-each claw check; only the ones with vias keep a
    # pass-through timing.
    timed, stop_scales = pass_through.time_routes(
        dict(routes, **chain_waypoints, **directs), v_max, a_max, j_max,
        tip_speed=tip_speed, corners=pass_through.gripper_corners(),
    )

    def padding_of(name):
        return (
            transfers.route_arm_padding(name),
            transfers.expected_gripper_clearance(name),
        )

    # Every curve - routes, chains, direct routes - is checked in ONE
    # pass, grouped by padding: each padding change makes move_group
    # rebuild its padded meshes and grow, and twelve changes in one
    # bake grew it to 3.5 GB and got it OOM-killed (2026-10-21, and
    # 2026-10-08 before that).
    jobs = {}

    for name in sorted(with_vias):
        if timed.get(name) is None:
            continue

        times, positions, velocities, _settings = timed[name]
        jobs.setdefault(padding_of(name), []).append(
            (name, pass_through.dense_states(times, positions, velocities))
        )

    chain_states = _chain_jobs(routes, chains, timed, padding_of, jobs)
    _direct_jobs(directs, timed, padding_of, jobs)

    failed = _grouped_checks(arm, jobs, 'rounded corners collide')

    for name in sorted(with_vias):
        waypoints = with_vias[name]
        stop = time_stop_at_each(
            waypoints, v_max, max_acceleration=a_max, max_jerk=j_max
        )[1][-1]

        if timed.get(name) is None or name in failed:
            print(
                f'  {name:<11} '
                + ('no timing within the limits' if name not in failed
                   else 'rounded corners collide')
                + '; stops at vias'
            )
            timings[name] = None
            status = 1
            continue

        times, positions, velocities, settings = timed[name]
        timings[name] = pass_through.to_document(
            times, positions, velocities, limits, settings
        )
        v, a, j = settings['peaks']
        print(
            f'  {name:<11} {len(waypoints) - 2} via(s): {stop:.2f} s '
            f'stopping -> {times[-1]:.2f} s passing through (peaks '
            f'{math.degrees(v):.0f} deg/s, {a:.2f} rad/s^2, {j:.1f} '
            f'rad/s^3, claw {settings.get("tip_peak_m_s", 0.0) * 1000:.0f} '
            f'mm/s; checked with {padding_of(name)[1] * 100:g} cm '
            'gripper clearance)',
            flush=True,
        )

    chain_timings = _keep_chains(
        routes, chains, timed, limits, chain_states, failed,
        v_max, a_max, j_max, stop=stop_scales,
    )
    direct_timings = _keep_direct(
        routes, directs, timed, chain_timings, limits, failed,
        v_max, a_max, j_max, stop=stop_scales,
    )

    slowed = {
        name: value for name, value in sorted(stop_scales.items())
        if value[0] < 1.0 and (name in routes or name in directs)
    }
    print(
        f'Stopping at its vias, {len(slowed)} route(s) would take the claw '
        f'over {tip_speed * 1000:.0f} mm/s and are slowed for it'
        + (':' if slowed else '.')
    )
    for name, (scale, peak) in slowed.items():
        print(f'  {name:<20} claw {peak * 1000:.0f} mm/s -> x{scale:.3f} speed')

    path = transfers.default_plan_path()
    transfers.save_timings(
        path, timings, chains=chain_timings, direct=direct_timings,
        stop_scales={
            name: value for name, value in stop_scales.items()
            if name in routes or name in directs
        },
    )
    print(f'Saved the timings in {path}')

    return status


def _grouped_checks(arm, jobs, what):
    """
    Collision-check states under their padding; return the names that collide.

    jobs: {(arm padding, gripper clearance): [(name, states)]}. One
    padding change per group: every change makes move_group rebuild
    its padded meshes (see _bake_timing).
    """
    from .. import config
    from ..arm import scene
    from ..arm.joint_path import densify

    failed = set()

    try:
        for (arm_padding, gripper_clearance), checks in sorted(jobs.items()):
            scene.set_padding(
                arm, arm_padding, {config.GRIPPER_LINK: gripper_clearance}
            )

            for name, states in checks:
                if name in failed:
                    continue

                bad = arm.first_invalid_state(states)

                if bad is not None:
                    # bad indexes the densified states, not states.
                    print(
                        f'  {name:<20} {what} '
                        f'({arm._why_invalid(densify(states)[bad])})'
                    )
                    failed.add(name)
    finally:
        scene.set_padding(
            arm,
            config.OBSTACLE_PADDING_M,
            {config.GRIPPER_LINK: config.GRIPPER_PADDING_M},
        )

    return failed


def _leg_time(routes, timed, name, v_max, a_max, j_max, stop=None):
    """
    Return a route's replay time: its pass-through timing, else stopping.

    stop: {route: (time scale, claw peak)}; a route stopping at its
    vias is replayed slower by its scale, to keep the claw's speed.
    """
    from ..arm.joint_path import time_stop_at_each

    timing = timed.get(name) if len(routes[name]) > 2 else None

    if timing is not None:
        return float(timing[0][-1])

    scale = (stop or {}).get(name, (1.0,))[0]

    return float(time_stop_at_each(
        routes[name], v_max, max_acceleration=a_max, max_jerk=j_max
    )[1][-1]) / scale


def _direct_jobs(directs, timed, padding_of, jobs):
    """
    Add the direct routes' curve checks (arm/direct.py) to jobs.

    The curve up to the exit is checked under the grab route's
    padding, the rest under the direct route's own.
    """
    from ..arm import direct, pass_through, transfers

    exits = {
        name: int(entry['exit'])
        for name, entry in (
            transfers._read(None) or {}
        ).get('direct', {}).items()
    }

    for name, waypoints in directs.items():
        if timed.get(name) is None:
            continue

        grab, join = direct.split(name)
        times, positions, velocities, _settings = timed[name]
        states = pass_through.dense_states(times, positions, velocities)
        # Where the curve passes the exit (direct.py).
        exit_pose = waypoints[exits.get(name, 1)]
        turn = int(np.argmin(np.abs(states - exit_pose).max(axis=1)))

        jobs.setdefault(padding_of(grab), []).append((name, states[:turn + 1]))
        jobs.setdefault(
            (transfers.route_arm_padding(join),
             transfers.expected_gripper_clearance(join)),
            [],
        ).append((name, states[turn:]))


def _keep_direct(
    routes, directs, timed, chain_timings, limits, failed, v_max, a_max, j_max,
    stop=None,
):
    """
    Return the direct routes to keep: {name: timing document or None}.

    For save_timings; None replays it stopping at its vias (its
    rounded corners collide, or TOTG found no timing). A route is kept
    only where it beats the way through INTER (the chain timing if
    there is one, else the two routes).
    """
    from ..arm import direct, pass_through
    from ..arm.joint_path import time_stop_at_each

    keep = {}

    for name in sorted(directs):
        grab, join = direct.split(name)
        chain = (
            f'{grab}..{join}' if join == 'drop' else f'{join}..{grab}'
        )

        if chain in chain_timings:
            via_inter = chain_timings[chain]['timed']['times'][-1]
        else:
            via_inter = sum(
                _leg_time(routes, timed, leg, v_max, a_max, j_max, stop)
                for leg in (grab, join)
            )

        passing = timed.get(name) if name not in failed else None
        seconds = (
            float(passing[0][-1]) if passing is not None
            else float(time_stop_at_each(
                directs[name], v_max, max_acceleration=a_max, max_jerk=j_max
            )[1][-1]) / (stop or {}).get(name, (1.0,))[0]
        )

        if seconds >= via_inter:
            print(
                f'  {name:<20} {seconds:.2f} s, via INTER '
                f'{via_inter:.2f} s: not faster, not kept'
            )
            continue

        keep[name] = None if passing is None else pass_through.to_document(
            *passing[:3], limits, passing[3]
        )
        print(
            f'  {name:<20} {seconds:.2f} s direct (via INTER '
            f'{via_inter:.2f} s)',
            flush=True,
        )

    print(f'{len(keep)} of {len(directs)} direct route(s) kept.')

    return keep


def _chain_jobs(routes, chains, timed, padding_of, jobs):
    """
    Add the chain timings' curve checks (transfers.chain_pairs) to jobs.

    Each half of a chain's curve - up to where it passes INTER, and
    after - is checked under its own route's padding, as that route
    was baked: the way to DROP keeps the laundry clear of the rim,
    the way into the bucket may bring the gripper to the floor.
    Returns the names checked.
    """
    from ..arm import pass_through

    checked = set()

    for name, (back, out) in chains.items():
        if timed.get(name) is None:
            print(f'  {name:<20} no timing within the limits; stops at INTER')
            continue

        times, positions, velocities, _settings = timed[name]
        states = pass_through.dense_states(times, positions, velocities)
        inter = routes[out][0]
        turn = int(np.argmin(np.abs(states - inter).max(axis=1)))
        checked.add(name)

        jobs.setdefault(padding_of(back), []).append((name, states[:turn + 1]))
        jobs.setdefault(padding_of(out), []).append((name, states[turn:]))

    return checked


def _keep_chains(
    routes, chains, timed, limits, checked, failed, v_max, a_max, j_max,
    stop=None,
):
    """
    Return the chains to keep, {name: {'back', 'out', 'timed'}}.

    For save_timings: collision-free, and faster than stopping at INTER.
    """
    from ..arm import pass_through

    def leg_time(name):
        return _leg_time(routes, timed, name, v_max, a_max, j_max, stop)

    good = {}

    for name in sorted(checked - failed):
        back, out = chains[name]
        times, positions, velocities, settings = timed[name]
        stopping = leg_time(back) + leg_time(out)

        # Most wrist joints reverse at INTER (into the bucket and back
        # out), so rounding it saves ~0.1 s at best, and TOTG's slower
        # settings over the longer path can lose that: keep a chain
        # only when it is faster.
        if times[-1] >= stopping:
            print(
                f'  {name:<20} {stopping:.2f} s stopping at INTER -> '
                f'{times[-1]:.2f} s passing through: not faster, not kept'
            )
            continue

        good[name] = {
            'back': back,
            'out': out,
            'timed': pass_through.to_document(
                times, positions, velocities, limits, settings
            ),
        }
        print(
            f'  {name:<20} {stopping:.2f} s stopping at INTER -> '
            f'{times[-1]:.2f} s passing through',
            flush=True,
        )

    print(
        f'{len(good)} of {len(chains)} grab <-> DROP move(s) pass INTER '
        '(faster that way, and collision-free).'
    )

    return good


def cmd_plan(args):
    """Run `laundry plan bake [poses|endcap|transfers|retrieve|direct|timing|all]`."""
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
            from ..grasp import grab_targets, pile_layers

            placed = grab_targets.load()
            generated = grab_targets.generated_for()

            # Generate the list from config.PILE when asked, when there
            # is none, or when it was generated for another PILE;
            # a hand-placed list (edit-grabs) is kept.
            if (
                args.regenerate
                or not placed
                or pile_layers.pile_changed(generated)
            ):
                heights = config.PILE['layer_heights_m']
                print(
                    'Generating the sweep from config.PILE (layers '
                    + ', '.join(f'{h * 100:.0f}' for h in heights)
                    + ' cm above the floor)...'
                )
                placed, gen_misses, layers = pile_layers.generate(
                    arm, log=print
                )

                if not placed:
                    print(
                        "No sweep grabs: nothing in config.PILE's layers "
                        'was reachable.',
                        file=sys.stderr,
                    )
                    return 1

                grab_targets.save(
                    placed,
                    generated={
                        'pile': pile_layers.pile_signature(),
                        'layers': len(layers),
                    },
                )
                print(
                    f'Saved {grab_targets.path()} ({len(placed)} grabs in '
                    f'{len(layers)} layer(s); {len(gen_misses)} spot(s) '
                    'unreachable).'
                )

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

            if args.only:
                targets = tuple(name.lower() for name in args.only)

            routes, clearances, failures = transfers.bake(
                arm, targets=targets, log=print
            )

            # Routes re-baked for replay time, not least travel.
            from ..arm import fast_route

            for name in sorted(set(routes) & set(fast_route.TIME_ROUTES)):
                routes[name], _seconds = fast_route.faster(
                    arm, name, routes[name], log=print
                )

            path = transfers.default_plan_path()
            # A partial bake keeps the chains and direct routes; their
            # digests drop any whose routes changed (load_timings).
            before = transfers._read(path) if args.only else None
            transfers.save(
                path,
                routes,
                speed,
                baked_on=socket.gethostname(),
                clearances=clearances,
                keep=transfers.routes_to_keep(path, routes),
            )

            if before:
                document = transfers._read(path)
                for section in ('chains', 'direct'):
                    if before.get(section):
                        document[section] = before[section]
                transfers._write(path, document)

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

        if args.which == 'direct':
            from ..arm import direct

            routes, _speed = transfers.load()
            path = transfers.default_plan_path()
            transfers.save_direct(path, direct.bake(arm, routes, log=print))
            print(
                f'Saved the direct routes in {path}. Now time them, in a '
                'fresh fake bring-up: ./rebake.sh timing'
            )

        if args.which in ('timing', 'transfers', 'retrieve', 'all'):
            status = max(status, _bake_timing(arm))

        if args.which in ('transfers', 'retrieve', 'all'):
            # Not in the same bake: each search changes the gripper
            # padding, and move_group grows with every change (it was
            # OOM-killed at 3.5 GB, 2026-10-21).
            print(
                'Direct grab <-> DROP routes are baked on their own: '
                './rebake.sh direct, then ./rebake.sh timing.'
            )

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
        choices=(
            'poses', 'endcap', 'transfers', 'retrieve', 'direct', 'timing',
            'all',
        ),
        default='all',
        help=(
            'What to bake (default: all, in the order poses, retrieve, '
            'transfers, timing, endcap). poses: derive INTER and BOTTOM '
            'from the bucket (config.BUCKET_POSES); no motion. retrieve: '
            "solve the sweep's grabs exactly as placed "
            '(scan_plans/grab_targets.yaml, `laundry plan edit-grabs`) and '
            'bake routes to them. timing: time the baked transfers to pass '
            'through their vias (also run after transfers/retrieve); needs '
            'moveit_py. Re-run it after changing config.TRANSFER_MAX_*. '
            'direct: grab <-> DROP routes that skip INTER (arm/direct.py); '
            'run timing after it, in a fresh bring-up.'
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
    bake.add_argument(
        '--only', action='append', metavar='NAME', default=None,
        help=(
            'With transfers: bake only this route (repeat for more), keeping '
            'the others. HOME is searched for replay time '
            '(arm/fast_route.py).'
        ),
    )
    bake.add_argument(
        '--regenerate', action='store_true',
        help=(
            'With retrieve/all: generate the sweep from config.PILE even '
            'when a grab list exists (a hand-placed one is otherwise '
            'kept; a generated one regenerates by itself when PILE '
            'changes). Every layer is generated; `--fill` picks the '
            'starting one at run time.'
        ),
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
