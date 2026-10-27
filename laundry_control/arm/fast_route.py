#!/usr/bin/env python3

"""
Re-bake a transfer for the shortest replay time instead of least travel.

WHY
---
transfers.bake_transfer picks the collision-free route with the least
WEIGHTED joint travel (cable-twisting joints count more) and never
looks at time. For INTER <-> HOME that gave a route that swings the
base J1 49 deg out and back (97 deg of travel for a 1 deg change) and
J4 68 deg extra; turning the base swings the whole arm, so the claw
ran up against its speed cap (config.TRANSFER_MAX_TIP_SPEED_M_S) and
the route was slowed to 5.2 s (2026-10-26).

WHAT
----
faster() searches the same family of routes - back out along the tool
axis (now up to 25 cm), then one or two vias around the direct line -
in two stages:

  1. screening: every sampled route is scored by its stop-at-each
     replay time under the joint limits (joint_path.time_stop_at_each,
     cheap), and collision-checked (every 1 deg state, the route's own
     padding) only if it could make the 8 fastest;
  2. exact: those, plus the route as baked now (so the result is never
     slower), are timed as a replay would be - the pass-through timing
     under the joint limits and the claw cap, its rounded corners
     collision-checked, else stopping at the vias slowed for the claw
     (pass_through.time_routes) - and the fastest wins.

The winner is a plain route like any other: `laundry plan bake timing`
times it, go_to replays it and re-checks it live.

RESULT (2026-10-26, HOME)
-------------------------
5.13 s -> 3.10 s, passing through its via, claw 808 mm/s. The base
swing went (J1 travel 98 -> 34 deg), but the wrist moves more: J4 79 ->
97, J5 15 -> 60, J6 14 -> 138 deg - the time search does not weigh
cable twist the way transfers.TRAVEL_WEIGHTS does. Kept by the user's
choice, since HOME is a manual parking move. If the wrist wiring
suffers, re-search with a limit on wrist travel instead.
"""

import heapq

import numpy as np

from .joint_path import time_stop_at_each
from .. import config

# Routes re-baked for time by `laundry plan bake transfers`.
TIME_ROUTES = ('home',)

RETREAT_DISTANCES_M = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25)
SAMPLES_PER_CONFIG = 600
VIA_JITTER_RAD = np.deg2rad(30.0)
KEEP = 8
SEED = 0


def _stop_time(path):
    return float(time_stop_at_each(
        path,
        config.TRANSFER_MAX_VELOCITY_RAD_S,
        max_acceleration=config.TRANSFER_MAX_ACCELERATION_RAD_S2,
        max_jerk=config.TRANSFER_MAX_JERK_RAD_S3,
    )[1][-1])


def screen(arm, inter, target, retreat_of, rng=None, log=print):
    """
    Return up to KEEP collision-free routes, fastest stop-at-each first.

    retreat_of(distance) gives the backed-out state or None (transfers.
    _retreat). Collision checks run under the padding the caller set.
    """
    rng = rng or np.random.default_rng(SEED)
    best = []          # heap of (-seconds, index, path)
    count = 0

    def bar():
        return -best[0][0] if len(best) >= KEEP else np.inf

    for distance in RETREAT_DISTANCES_M:
        if distance > 0.0:
            retreat = retreat_of(distance)

            if (
                retreat is None
                or arm.first_invalid_state([inter, retreat]) is not None
            ):
                continue

            start = [inter, retreat]
        else:
            start = [inter]

        candidates = [start + [target]]

        for via_count in (1, 2):
            for _ in range(SAMPLES_PER_CONFIG):
                along = np.sort(rng.uniform(0.15, 0.85, via_count))
                candidates.append(start + [
                    start[-1] + u * (target - start[-1])
                    + rng.normal(0.0, VIA_JITTER_RAD, 7)
                    for u in along
                ] + [target])

        for path in candidates:
            seconds = _stop_time(path)

            if seconds >= bar():
                continue

            if arm.first_invalid_state(path) is not None:
                continue

            count += 1
            entry = (-seconds, count, [np.asarray(q) for q in path])

            if len(best) < KEEP:
                heapq.heappush(best, entry)
            else:
                heapq.heapreplace(best, entry)

        log(
            f'    backed out {distance * 100:2.0f} cm: best so far '
            + (f'{-max(best)[0]:.2f} s stopping at the vias' if best else 'none')
        )

    return [path for _neg, _i, path in sorted(best, key=lambda e: -e[0])]


def replay_seconds(name, path, timed, stop, arm):
    """Return how long go_to would take to replay path, and how it would."""
    from . import pass_through

    scale = stop.get(name, (1.0, 0.0))[0]
    stopping = _stop_time(path) / scale

    if timed.get(name) is not None and len(path) > 2:
        times, positions, velocities, _settings = timed[name]

        if arm.first_invalid_state(
            pass_through.dense_states(times, positions, velocities)
        ) is None and times[-1] < stopping:
            return float(times[-1]), 'passing through'

    return stopping, (
        'stopping at the vias' + (f', slowed x{scale:.3f}' if scale < 1 else '')
    )


def faster(arm, target_name, current, log=print):
    """
    Return the fastest route INTER -> target_name found, and its seconds.

    current: the route as baked now; it competes too, so the result is
    never slower. The padding is the route's own (transfers.
    route_arm_padding, expected_gripper_clearance), set here.
    """
    from . import pass_through, scene, transfers

    inter = np.array(config.get_named_pose('inter'))
    target = np.array(config.get_named_pose(transfers.route_pose(target_name)))

    scene.set_padding(
        arm, transfers.route_arm_padding(target_name),
        {config.GRIPPER_LINK: transfers.expected_gripper_clearance(target_name)},
    )

    try:
        log(f'  {target_name}: searching for the fastest route (screening)...')
        found = screen(
            arm, inter, target, lambda d: transfers._retreat(arm, d), log=log
        )

        candidates = {'current': [np.asarray(q) for q in current]}
        candidates.update({f'c{k}': path for k, path in enumerate(found)})

        log(
            f'  {target_name}: timing {len(candidates)} candidate(s) as a '
            'replay would (claw cap included)...'
        )
        timed, stop = pass_through.time_routes(
            candidates,
            config.TRANSFER_MAX_VELOCITY_RAD_S,
            config.TRANSFER_MAX_ACCELERATION_RAD_S2,
            config.TRANSFER_MAX_JERK_RAD_S3,
            tip_speed=config.TRANSFER_MAX_TIP_SPEED_M_S,
            corners=pass_through.gripper_corners(),
        )

        results = []

        for name, path in candidates.items():
            seconds, how = replay_seconds(name, path, timed, stop, arm)
            results.append((seconds, name, how, path))
            travel = transfers.per_joint_travel_deg(path)
            log(
                f'    {name:<8} {seconds:5.2f} s ({how}); {len(path) - 2} '
                f'via(s); J1 travel {travel[0]:.0f} deg, J4 {travel[3]:.0f}'
            )
    finally:
        scene.set_padding(
            arm, config.OBSTACLE_PADDING_M,
            {config.GRIPPER_LINK: config.GRIPPER_PADDING_M},
        )

    seconds, name, how, path = min(results, key=lambda r: r[0])
    log(f'  {target_name}: kept {name}, {seconds:.2f} s ({how}).')

    return path, seconds
