#!/usr/bin/env python3

"""
Direct routes between a sweep grab and DROP, not through INTER.

WHY
---
Every baked route starts at INTER (transfers.py), so grab_NN -> DROP
went back along grab_NN's route to INTER and out along DROP's: ~3.6 s
each way at 60 deg/s, against ~1.5 s for the straight line - which
hits the bucket on all 24 grabs (2026-10-21). Most of the detour is
that INTER is a fixed pose: the arm climbs all the way back to it,
and then DROP's route starts by backing out of the bucket again.

WHAT
----
A direct route leaves the grab back along grab_NN's own route - a
line already checked when it was baked - only part of the way (the
exit, EXIT_FRACTIONS of the way to INTER), then joins DROP's route at
one of its later waypoints, skipping INTER and whatever came before:

    grab_NN -> exit -> DROP's waypoints k.. -> DROP

Every candidate is checked with DROP's gripper clearance from the
exit on (config.ROUTE_GRIPPER_CLEARANCE_M: laundry hangs from the
claw), and the fastest collision-free one (stopping at each via, the
fallback timing) is kept. The way back once the gripper let go is the
same search along drop_return's waypoints, at its clearance, replayed
backwards.

`laundry plan bake timing` then times them to pass through their
vias, and keeps a direct route only where it is faster than the way
through INTER; go_to uses it whenever it is there (transfers._pieces).
"""

import numpy as np

from .joint_path import densify, time_stop_at_each
from .. import config

# How far back along grab_NN's route (toward INTER) the exit is tried.
EXIT_FRACTIONS = (0.2, 0.35, 0.5, 0.65, 0.8)

# {direct route suffix: the INTER route whose waypoints it joins}.
# '>drop' carries laundry; '>drop_return' is the way back, empty.
JOINS = {
    'drop': 'drop',
    'drop_return': 'drop_return',
}


def name(grab, via_route):
    """Return the direct route's name: 'grab_NN>drop' or 'grab_NN>drop_return'."""
    return f'{grab}>{via_route}'


def split(route_name):
    """Return (grab, via_route) of a direct route's name, or None."""
    if '>' not in route_name:
        return None

    grab, via_route = route_name.split('>', 1)

    return (grab, via_route) if via_route in JOINS else None


def _timed_stop(waypoints):
    return float(time_stop_at_each(
        waypoints,
        config.TRANSFER_MAX_VELOCITY_RAD_S,
        max_acceleration=config.TRANSFER_MAX_ACCELERATION_RAD_S2,
        max_jerk=config.TRANSFER_MAX_JERK_RAD_S3,
    )[1][-1])


def candidates(grab_route, join_route):
    """
    Return the candidate paths [grab, exit, join_route[k:]...], fastest first.

    grab_route: INTER -> grab_NN as baked. join_route: INTER -> DROP
    (drop or drop_return) as baked. Each candidate is (seconds when
    stopping at each via, waypoints, index of the exit).
    """
    back = np.asarray(grab_route, dtype=np.float64)[::-1]
    join_route = np.asarray(join_route, dtype=np.float64)

    # Points back along grab_NN's route, by joint-space arc length.
    dense = densify(back)
    arc = np.concatenate(
        [[0.0], np.cumsum(np.abs(np.diff(dense, axis=0)).max(axis=1))]
    )
    found = []

    for fraction in EXIT_FRACTIONS:
        index = int(np.searchsorted(arc, fraction * arc[-1]))
        index = min(max(index, 1), len(dense) - 1)

        # The grab route's own waypoints up to the exit, then the exit.
        passed = [
            q for q, s in zip(back, _arc_at(back)) if 0.0 < s < arc[index]
        ]
        lead = [back[0]] + passed + [dense[index]]

        for k in range(1, len(join_route)):
            path = np.array(lead + list(join_route[k:]))
            found.append((_timed_stop(path), path, len(lead) - 1))

    found.sort(key=lambda item: item[0])

    return found


def _arc_at(waypoints):
    """Return the joint-space arc length at each waypoint."""
    return np.concatenate([
        [0.0],
        np.cumsum(np.abs(np.diff(waypoints, axis=0)).max(axis=1)),
    ])


def bake(arm, routes, log=print):
    """
    Find the direct routes for every sweep grab; return {name: entry}.

    entry: {'from', 'to', 'join', 'waypoints', 'exit', 'padding_m'}
    for transfers.save_direct. Checked against the planning scene as
    it is now; the padding is set here, one change per clearance
    (each change rebuilds move_group's padded meshes).
    """
    from . import scene
    from .transfers import (
        expected_gripper_clearance, route_arm_padding, route_pose,
    )

    grabs = [g for g in config.generated_grab_poses() if g in routes]
    entries = {}

    try:
        for join in JOINS.values():
            if join not in routes:
                log(f'  No {join} route baked: no direct routes along it.')
                continue

            clearance = expected_gripper_clearance(join)
            arm_padding = route_arm_padding(join)
            scene.set_padding(
                arm, arm_padding, {config.GRIPPER_LINK: clearance}
            )

            log(
                f'Direct routes grab_NN <-> {route_pose(join).upper()} '
                f'along {join} ({clearance * 100:g} cm gripper '
                'clearance)...'
            )

            for grab in grabs:
                via_inter = _timed_stop(
                    np.concatenate([routes[grab][::-1], routes[join][1:]])
                )

                for seconds, path, exit_index in candidates(
                    routes[grab], routes[join]
                ):
                    if seconds >= via_inter:
                        log(f'  {name(grab, join):<20} none faster and clear')
                        break

                    # Up to the exit the path lies on grab_NN's own
                    # route, checked (with its clearance) when it was
                    # baked; the replay re-checks all of it.
                    if arm.first_invalid_state(path[exit_index:]) is None:
                        entries[name(grab, join)] = {
                            'from': grab,
                            'to': route_pose(join),
                            'join': join,
                            'waypoints': path,
                            'exit': exit_index,
                            'padding_m': {
                                'arm_links': arm_padding,
                                'gripper': clearance,
                            },
                        }
                        log(
                            f'  {name(grab, join):<20} {seconds:.2f} s '
                            f'stopping at each (via INTER {via_inter:.2f} s)'
                        )
                        break
                else:
                    log(f'  {name(grab, join):<20} none faster and clear')
    finally:
        scene.set_padding(
            arm,
            config.OBSTACLE_PADDING_M,
            {config.GRIPPER_LINK: config.GRIPPER_PADDING_M},
        )

    log(f'{len(entries)} of {len(grabs) * len(JOINS)} direct route(s) found.')

    return entries
