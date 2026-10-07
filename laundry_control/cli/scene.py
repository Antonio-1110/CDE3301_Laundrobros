#!/usr/bin/env python3

"""The obstacles in MoveIt: `laundry scene apply|check|fit`."""

from .common import RosSession


def _describe_contacts(contacts):
    if contacts is None:
        return 'MoveIt did not answer'

    return ', '.join(f'{a} <-> {b}' for a, b in contacts) or 'collides'


def _check_path(arm, waypoints):
    """Return None if a joint path is collision-free, else what it hits."""
    from ..arm.joint_path import densify

    dense = densify(waypoints)

    for index, joints in enumerate(dense):
        contacts = arm.state_contacts(joints)

        if contacts != []:
            return (
                f'collides at state {index}/{len(dense) - 1}: '
                f'{_describe_contacts(contacts)}'
            )

    return None


def cmd_scene_fit(args):
    """Run `laundry scene fit`: the bucket pose the baseline scans measure."""
    import numpy as np

    from .. import config
    from ..bucket import bucket_pose_from_fit, seed_axis_direction, seed_origin
    from ..perception.bucket_model import fit_cone
    from ..perception.detect import load_baseline_scans

    directory = args.baseline or config.baseline_dir()
    scans = load_baseline_scans(directory)
    model = fit_cone(np.concatenate(scans))

    xyz, rpy = bucket_pose_from_fit(model)

    axis = seed_axis_direction()

    print(
        f'Fitted the bucket wall to {len(scans)} baseline scan(s) in '
        f'{directory}.\nOffset of the scanned bucket axis from '
        'config.OBSTACLES, cm (x, y, z in link_base):'
    )

    for s in (0.0, 0.4):
        d = model.axis_point + s * model.axis_dir - seed_origin()
        d -= np.dot(d, axis) * axis
        print(
            f'  {s * 100:3.0f} cm from the closed end: '
            f'{np.round(d * 100, 1)}'
        )

    tilt = np.degrees(np.arccos(np.clip(model.axis_dir @ axis, -1.0, 1.0)))
    print(f'  axis tilt: {tilt:.1f} deg')

    print(
        '\nThe fit relies on the ToF extrinsics (config.TOF_SENSOR_OFFSET_*),'
        '\nso an error there moves it too: check it against a tape measure.'
        '\nIf it is right, the bucket entry in config.OBSTACLES becomes:\n'
    )
    print(f"        'xyz': [{', '.join(f'{v:.4f}' for v in xyz)}],")
    print(f"        'rpy': [{', '.join(f'{v:.4f}' for v in rpy)}],")
    print('\nThen: laundry scene check, and laundry plan bake.')

    return 0


def _fake_start_at_home(arm):
    """
    FAKE CONTROLLER ONLY: move the arm from all-zeros to HOME.

    The mock hardware starts every joint at 0, and there the modelled
    gripper sits in the table, so MoveIt refuses to plan out of it. The
    move goes straight to the trajectory controller (no planning, so no
    start-state collision check). Refuses unless the arm reads exactly
    all-zeros - a real arm never does, it starts where it was left.
    """
    from .. import config
    from ..arm.joint_path import time_stop_at_each

    current = arm.get_current_joints()

    if current is None or max(abs(q) for q in current) > 1e-6:
        print('--fake-start-home: the arm is not at the mock all-zeros '
              'start; not moving it.')
        return current is not None

    waypoints, times, velocities = time_stop_at_each(
        [current, config.HOME], config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S
    )

    ok = arm.execute_joint_path(waypoints, times, velocities)
    print('Fake arm moved to HOME.' if ok else 'Could not move the fake arm.')

    return ok


def cmd_scene(args):
    """Run `laundry scene apply|check|fit`: the obstacles in MoveIt."""
    import os

    if args.scene_action == 'fit':
        return cmd_scene_fit(args)

    from .. import config
    from ..arm import scene, transfers
    from ..scan import endcap

    current = scene.signature()

    print(f'Obstacles (config.OBSTACLES, scene {current}):')

    for name, spec in config.OBSTACLES.items():
        print(
            f'  {name:<8} {spec["mesh"]:<12} xyz {spec["xyz"]}  '
            f'rpy {spec["rpy"]}'
        )

    print(
        f'Padding: arm links {config.OBSTACLE_PADDING_M * 100:g} cm, gripper '
        f'{config.GRIPPER_PADDING_M * 100:g} cm.'
    )

    # Connecting applies the scene (XArm7Controller(pad_obstacles=True)).
    with RosSession(args=args) as arm:
        if args.scene_action == 'apply':
            print('move_group has the obstacles.')

            if args.fake_start_home:
                return 0 if _fake_start_at_home(arm) else 1

            return 0

        problems = 0

        from ..arm import bucket_poses

        print('\nINTER and BOTTOM:')

        for name, where in bucket_poses.status().items():
            print(f'  {name:<11} {where}')

        if config.derived_bucket_poses():
            stale = scene.stale_plan_message(
                bucket_poses.baked_scene(), '  bucket_poses.yaml',
                'laundry plan bake',
            )

            if stale:
                problems += 1
                print(stale)

            changed = bucket_poses.spec_mismatch()

            if changed:
                problems += 1
                print(f'  {changed}')

        print('\nNamed poses (arm link padding as live):')

        for name, joints in config.named_poses().items():
            contacts = arm.state_contacts(joints)
            verdict = 'ok' if contacts == [] else (
                'COLLIDES: ' + _describe_contacts(contacts)
            )
            problems += contacts != []
            print(f'  {name:<11} {verdict}')

        routes, _speed = transfers.load()
        stamp = transfers.baked_scene()
        mismatches = transfers.padding_mismatches()

        print('\nBaked transfers from INTER (scan_plans/transfers.yaml):')

        stale = scene.stale_plan_message(
            stamp, '  transfers.yaml', 'laundry plan bake transfers'
        )

        if stale:
            print(stale)

        moved = transfers.from_other_inter(routes)

        for name in transfers.transfer_targets():
            route = routes.get(name)

            if name in moved:
                print(
                    f'  {name:<11} BAKED FROM ANOTHER INTER: not used; '
                    're-bake: laundry plan bake transfers'
                )
                problems += 1
                continue

            if route is None:
                print(
                    f'  {name:<11} NO ROUTE: moves there go straight if that '
                    'is clear, else through the planner'
                )
                problems += 1
                continue

            padding = transfers.route_arm_padding(name)
            arm.set_arm_padding(padding)

            try:
                hit = _check_path(arm, route)
            finally:
                arm.set_arm_padding(config.OBSTACLE_PADDING_M)

            problems += hit is not None
            print(
                f'  {name:<11} {"ok" if hit is None else "COLLIDES " + hit} '
                f'({len(route) - 2} via(s), arm links {padding * 100:g} cm)'
            )

            if name in mismatches:
                problems += 1
                print(f'              {mismatches[name]}')

        from ..grasp import retrieve_grid

        grabs, grab_stamp = retrieve_grid.load()

        if grabs:
            print(
                '\nGrab descents (scan_plans/retrieve.yaml, approach -> grab):'
            )

            stale = scene.stale_plan_message(
                grab_stamp, '  retrieve.yaml', 'laundry plan bake retrieve'
            )

            if stale:
                print(stale)

            changed = retrieve_grid.grid_mismatch()

            if changed:
                problems += 1
                print(f'  {changed}')

            for grab in grabs:
                hit = _check_path(arm, [grab['approach'], grab['grab']])
                problems += hit is not None
                print(
                    f'  {grab["name"]:<11} '
                    f'{"ok" if hit is None else "COLLIDES " + hit}'
                )

        plan_path = endcap.default_plan_path()

        if os.path.isfile(plan_path):
            plan = endcap.EndcapPlan.load(plan_path)

            print(
                f'\nEnd scan (scan_plans/endcap.yaml, arm link padding '
                f'{config.ENDCAP_PADDING_M * 100:g} cm):'
            )

            stale = scene.stale_plan_message(
                plan.scene, '  endcap.yaml',
                f'laundry plan bake endcap --depth {plan.depth_m:.3f}',
            )

            if stale:
                print(stale)

            moved = endcap.inter_mismatch(arm, plan)

            if moved:
                problems += 1
                print(f'  {moved}')

            if abs(plan.padding_m - config.ENDCAP_PADDING_M) > 1e-9:
                problems += 1
                print(
                    f'  baked with {plan.padding_m * 100:g} cm arm padding, '
                    f'config says {config.ENDCAP_PADDING_M * 100:g} cm; '
                    're-bake: laundry plan bake endcap'
                )

            arm.set_arm_padding(config.ENDCAP_PADDING_M)

            try:
                bad = arm.first_invalid_state(plan.waypoints)
            finally:
                arm.set_arm_padding(config.OBSTACLE_PADDING_M)

            problems += bad is not None
            print(
                '  ok' if bad is None
                else f'  COLLIDES at checked state {bad}'
            )

    print(
        '\nAll clear.' if not problems else
        f'\n{problems} problem(s). Fix config.OBSTACLES or re-record the '
        'pose, then `laundry plan bake`.'
    )

    return 1 if problems else 0


def add_scene_parser(subparsers):
    """Add `laundry scene`."""
    scene_parser = subparsers.add_parser(
        'scene',
        help='Put the obstacles (config.OBSTACLES) into MoveIt, and check them.',
    )
    scene_parser.add_argument(
        'scene_action', choices=('apply', 'check', 'fit'),
        help=(
            'apply: add the padded bucket/table to move_group (bring-up '
            'does this). check: also collision-check every recorded pose '
            'and baked route against them. Neither moves the arm. fit '
            '(offline): the bucket pose the baseline scans measure, as a '
            'config.OBSTACLES entry.'
        ),
    )
    scene_parser.add_argument(
        '--fake-start-home', action='store_true',
        help=(
            'apply, FAKE CONTROLLER ONLY: then move the arm from the mock '
            "hardware's all-zeros start (in the table) to HOME. Bring-up "
            'passes it with fake:=true.'
        ),
    )
    scene_parser.add_argument(
        '--baseline', type=str, default=None,
        help='fit: directory of empty-bucket scans (default: <repo>/baseline_scans).',
    )
    scene_parser.set_defaults(func=cmd_scene)
