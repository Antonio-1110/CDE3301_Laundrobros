#!/usr/bin/env python3

"""
Fixed, minimum-twist moves between INTER and HOME / DROP.

WHY
---
These moves used to go through move_joints(): Pilz PTP if loaded,
else OMPL. OMPL samples a new route every run, and its routes can be
wild - measured on the fake controller, DROP -> INTER once turned J7
through 536 deg when the direct change is 12 deg, and INTER -> HOME
swung J4 through up to 208 deg instead of 10 - which is what pulls
on the wrist wiring.

WHAT
----
A straight joint-space line is the minimum-twist route: every joint
moves monotonically, by exactly its start-to-end difference. INTER's
gripper, though, points into the bucket mouth, so the straight lines
to HOME and DROP drag it through the rim (and HOME's then sweeps it
through the table). Each transfer is therefore:

    INTER -> back out along the tool axis (the retreat) -> 1-2
    intermediate poses -> target,

all straight joint-space segments, every 1 deg state collision-
checked, found once by `laundry plan bake transfers` and saved to
scan_plans/transfers.yaml:

  - the retreat is solved by IK, seeded at INTER;
  - intermediate poses are sampled around the direct line (fixed
    seed) and the collision-free path with the least WEIGHTED joint
    travel wins; J1 and J4-J7 weigh more (TRAVEL_WEIGHTS), since they
    twist the cables;
  - each intermediate pose is then pulled joint by joint back toward
    the direct line while it stays collision-free, so as many joints
    as possible move monotonically, and redundant vias are dropped.

Randomness exists only while baking; the saved path is replayed as
is. Going back to INTER replays the same states in reverse. The arm
stops briefly at each via (arm/joint_path.time_stop_at_each), so it
follows exactly the checked straight segments; each segment is a
jerk-limited S-curve that cruises at the top speed
(config.TRANSFER_MAX_ACCELERATION_RAD_S2 / _JERK_RAD_S3).

go_to() is the single way to reach a named pose: a baked transfer
when one applies, else a collision-checked straight joint move, and
only if that would collide, the planner (with a warning).
"""

import datetime
import os

from geometry_msgs.msg import Pose
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .joint_path import densify, time_path, time_stop_at_each
from .joint_ranges import task_limits
from .. import config

PLAN_VERSION = 1

# Targets reached from INTER through a baked transfer. INTER is the
# hub: a move between two of these (grab_NN -> DROP, say) replays
# one route back to INTER and the other out from it - through the
# bucket mouth, never across the rim.
TRANSFER_TARGETS = (
    'home',
    'drop',
    'drop_return',
    'bottom',
)

# Routes that end at another route's pose, baked with other padding:
# {route: the named pose it reaches}. drop_return is INTER <-> DROP
# at the empty gripper's clearance; go_to takes it, reversed, to leave
# DROP once the caller says the gripper let go (gripper_empty).
ROUTE_POSES = {
    'drop_return': 'drop',
}

# {pose: the route to leave it by when the gripper is empty}.
EMPTY_GRIPPER_RETURNS = {
    'drop': 'drop_return',
}


def route_pose(name):
    """Return the named pose the route `name` ends at."""
    return ROUTE_POSES.get(name, name)


def transfer_targets():
    """Return TRANSFER_TARGETS plus the generated grab_NN approach poses."""
    return TRANSFER_TARGETS + tuple(config.generated_grab_poses())


# Joint-travel weights for choosing a route: J1 and J4-J7 twist the
# cables running down the arm, so their travel costs more.
TRAVEL_WEIGHTS = np.array([2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0])

# How far the gripper backs out of the bucket along the tool axis
# before swinging away, tried in this order (metres).
# 0 means no retreat: vias straight from INTER, which is what targets
# INSIDE the bucket usually want.
RETREAT_DISTANCES_M = (0.0, 0.05, 0.10, 0.15)

# Via-pose sampling: poses placed along the direct line, then jittered.
SAMPLES_PER_CONFIG = 400
VIA_JITTER_RAD = np.deg2rad(30.0)
BAKE_SEED = 0

# A joint state counts as "at" a named pose within this (radians).
AT_POSE_TOLERANCE_RAD = np.deg2rad(2.0)


def default_plan_path():
    """Return <repo>/scan_plans/transfers.yaml."""
    return os.path.join(config.repo_root(), 'scan_plans', 'transfers.yaml')


def weighted_travel_deg(path, weights=TRAVEL_WEIGHTS):
    """Return the weighted sum of per-joint travel along a polyline."""
    steps = np.abs(np.diff(np.asarray(path, dtype=np.float64), axis=0))
    return float(np.degrees((steps * weights).sum()))


def per_joint_travel_deg(path):
    """Return each joint's total travel along a polyline, in degrees."""
    steps = np.abs(np.diff(np.asarray(path, dtype=np.float64), axis=0))
    return np.degrees(steps.sum(axis=0))


class BakeError(RuntimeError):
    """No collision-free transfer was found."""


def _retreat(arm, distance_m):
    """
    Return the joint state with the flange backed out of INTER, or None.

    INTER's flange pose comes from forward kinematics of the recorded
    joint angles, not from where the arm happens to be, so the same
    INTER always gives the same retreat - and the same baked route.
    (Reading the live flange made HOME's route differ between bakes:
    the arm stops within MoveIt's tolerance of INTER, not exactly on it.)
    """
    inter = config.get_named_pose('inter')
    fk = arm.compute_fk(inter)

    if fk is None:
        return None

    position, quaternion = fk

    tool_z = Rotation.from_quat(quaternion).as_matrix()[:, 2]

    pose = Pose()
    position = position - distance_m * tool_z
    pose.position.x, pose.position.y, pose.position.z = map(float, position)
    pose.orientation.x, pose.orientation.y = map(float, quaternion[:2])
    pose.orientation.z, pose.orientation.w = map(float, quaternion[2:])

    solution = arm.compute_ik(pose, inter)

    return None if solution is None else np.array(solution)


def _refine(arm, path, first_free=2):
    """
    Pull via poses toward monotone joint values, then drop spare vias.

    Only changes that keep the whole path collision-free AND lower
    the weighted travel are kept, so the result is never worse. Poses
    before index first_free (INTER, and the retreat if there is one)
    and the target are never moved.
    """
    def valid(candidate):
        return arm.first_invalid_state(candidate) is None

    path = [np.array(p, dtype=np.float64) for p in path]

    improved = True
    rounds = 0

    while improved and rounds < 20:
        improved = False
        rounds += 1

        for index in range(first_free, len(path) - 1):
            before, after = path[index - 1], path[index + 1]

            for joint in range(7):
                goal = np.clip(
                    path[index][joint],
                    min(before[joint], after[joint]),
                    max(before[joint], after[joint]),
                )

                for fraction in (1.0, 0.5, 0.25, 0.1):
                    candidate = [p.copy() for p in path]
                    candidate[index][joint] += fraction * (
                        goal - path[index][joint]
                    )

                    if (
                        weighted_travel_deg(candidate)
                        < weighted_travel_deg(path) - 0.05
                        and valid(candidate)
                    ):
                        path = candidate
                        improved = True
                        break

    index = first_free
    while index < len(path) - 1:
        candidate = path[:index] + path[index + 1:]

        if (
            valid(candidate)
            and weighted_travel_deg(candidate) <= weighted_travel_deg(path)
        ):
            path = candidate
        else:
            index += 1

    return path


def bake_transfer(arm, target_name, log=print):
    """
    Find the INTER -> target transfer; returns its joint waypoints.

    Does not move the arm: every check is on joint states. Everything
    is checked against the planning scene as it is now, so the caller
    sets the padding the route must keep.
    """
    inter = np.array(config.get_named_pose('inter'))
    target = np.array(config.get_named_pose(route_pose(target_name)))

    if not arm.state_is_valid(target):
        raise BakeError(
            f'{target_name.upper()} itself collides with the bucket/table '
            '(with the current padding); re-record it or check config.OBSTACLES.'
        )

    if arm.first_invalid_state([inter, target]) is None:
        log(f'  INTER -> {target_name}: the straight line is collision-free.')
        return [inter, target]

    rng = np.random.default_rng(BAKE_SEED)

    best = None

    for distance in RETREAT_DISTANCES_M:
        if distance > 0.0:
            retreat = _retreat(arm, distance)

            if (
                retreat is None
                or arm.first_invalid_state([inter, retreat]) is not None
            ):
                continue

            start = [inter, retreat]
        else:
            start = [inter]

        for via_count in (1, 2):
            for _ in range(SAMPLES_PER_CONFIG):
                along = np.sort(rng.uniform(0.15, 0.85, via_count))

                vias = [
                    start[-1]
                    + u * (target - start[-1])
                    + rng.normal(0.0, VIA_JITTER_RAD, 7)
                    for u in along
                ]

                path = start + vias + [target]
                cost = weighted_travel_deg(path)

                if best is not None and cost >= best[0]:
                    continue

                if arm.first_invalid_state(path) is None:
                    best = (cost, path, len(start))

    if best is None:
        raise BakeError(f'No collision-free route found from INTER to {target_name}.')

    path = _refine(arm, best[1], first_free=best[2])

    excess = per_joint_travel_deg(path) - np.degrees(np.abs(target - inter))

    log(
        f'  INTER -> {target_name}: {len(path) - 2} via(s); extra travel per '
        f'joint over the (colliding) direct line, deg: {np.round(excess, 0)}'
    )

    return path


def bake(arm, targets=None, log=print):
    """
    Bake every transfer that can be baked.

    Returns ({target: [joint waypoints]}, {target: gripper clearance
    used, metres}, {target: reason it failed}). Routes to
    Gripper padding is expected_gripper_clearance(target).
    Arm links get route_arm_padding(target).
    """
    from . import scene

    targets = transfer_targets() if targets is None else targets

    log('Baking transfers from INTER...')

    routes = {}
    clearances = {}
    failures = {}

    for name in targets:
        extra = expected_gripper_clearance(name)

        scene.set_padding(
            arm, route_arm_padding(name), {config.GRIPPER_LINK: extra}
        )

        try:
            routes[name] = bake_transfer(arm, name, log=log)
            clearances[name] = extra
        except BakeError as exc:
            failures[name] = str(exc)
            log(f'  INTER -> {name}: FAILED - {exc}')

    scene.set_padding(
        arm,
        config.OBSTACLE_PADDING_M,
        {config.GRIPPER_LINK: config.GRIPPER_PADDING_M},
    )

    return routes, clearances, failures


def routes_to_keep(path, replacing):
    """
    Return the raw entries of routes a partial bake should carry over.

    Only from a file baked against the current scene, and only routes
    to poses that still exist, are not being re-baked (`replacing`)
    and start at the current INTER.
    """
    from . import scene

    document = _read(path)

    if not document or document.get('scene') != scene.signature():
        return {}

    valid = set(transfer_targets()) - set(replacing)
    moved = set(from_other_inter({
        name: np.array(route['waypoints'])
        for name, route in document['routes'].items()
    }))

    return {
        name: route for name, route in document['routes'].items()
        if name in valid and name not in moved
    }


def save(
    path, routes, max_velocity_rad_s, baked_on='', clearances=None, keep=None,
):
    """
    Write baked transfers as YAML, stamped with the current scene.

    keep: raw route entries (see routes_to_keep) written unchanged
    ahead of the new ones.
    """
    from . import scene

    clearances = clearances or {}

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    document = {
        'version': PLAN_VERSION,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'baked_on': baked_on,
        'joint_names': list(config.JOINT_NAMES),
        'max_velocity_rad_s': float(max_velocity_rad_s),
        'from': 'inter',
        # The obstacle geometry the routes were checked against
        # (arm/scene.py); go_to() warns when config.OBSTACLES differs.
        'scene': scene.signature(),
        'obstacles': scene.describe(),
        'routes': dict(keep or {}, **{
            name: {
                'waypoints': [[round(float(v), 6) for v in q] for q in waypoints],
                'per_joint_travel_deg': [
                    round(float(v), 1) for v in per_joint_travel_deg(waypoints)
                ],
                'padding_m': {
                    'arm_links': float(route_arm_padding(name)),
                    'gripper': float(
                        clearances.get(name, expected_gripper_clearance(name))
                    ),
                },
            }
            for name, waypoints in routes.items()
        }),
    }

    _write(path, document)


def route_arm_padding(name):
    """Return the arm-link padding a route is baked and replayed with."""
    return float(
        config.ROUTE_ARM_PADDING_M.get(name, config.OBSTACLE_PADDING_M)
    )


def _read(path):
    return config.read_yaml(path or default_plan_path())


def load(path=None):
    """Return ({target: waypoints array}, max_velocity), or ({}, None) if absent."""
    document = _read(path)

    if document is None:
        return {}, None

    if document.get('version') != PLAN_VERSION:
        raise ValueError(f'{path!r} is an unknown transfers version; re-bake.')

    if list(document['joint_names']) != list(config.JOINT_NAMES):
        raise ValueError(f'{path!r} was baked for different joints.')

    routes = {
        name: np.array(route['waypoints'])
        for name, route in document['routes'].items()
    }

    return routes, float(document['max_velocity_rad_s'])


def load_timings(path=None):
    """
    Return {route: (times, positions, velocities)} to pass through the vias.

    Only timings made for the transfer limits config sets now, and for
    the route as it is now (same first and last pose); the others are
    left out, so those routes stop at their vias.
    """
    from . import pass_through

    document = _read(path)

    if document is None:
        return {}

    limits = pass_through.current_limits()
    timings = {}

    for name, route in document['routes'].items():
        timed = route.get('timed')

        if not timed or timed.get('limits') != limits:
            continue

        times, positions, velocities = pass_through.from_document(timed)
        waypoints = np.asarray(route['waypoints'])

        if (
            np.abs(positions[0] - waypoints[0]).max() < 1e-5
            and np.abs(positions[-1] - waypoints[-1]).max() < 1e-5
        ):
            timings[name] = (times, positions, velocities)

    for name, entry in (document.get('direct') or {}).items():
        timed = entry.get('timed')

        if not timed or timed.get('limits') != limits:
            continue

        times, positions, velocities = pass_through.from_document(timed)
        waypoints = np.asarray(entry['waypoints'])

        if (
            np.abs(positions[0] - waypoints[0]).max() < 1e-5
            and np.abs(positions[-1] - waypoints[-1]).max() < 1e-5
        ):
            timings[name] = (times, positions, velocities)

    # Chains (chain_pairs): only for the routes exactly as they are now.
    routes = {
        name: np.asarray(route['waypoints'])
        for name, route in document['routes'].items()
    }

    for name, chain in (document.get('chains') or {}).items():
        back, out = chain.get('back'), chain.get('out')
        timed = chain.get('timed')

        if (
            back not in routes or out not in routes or not timed
            or timed.get('limits') != limits
            or chain.get('digest')
            != route_digest(chain_waypoints(routes, back, out))
        ):
            continue

        timings[name] = pass_through.from_document(timed)

    return timings


def load_direct(path=None):
    """
    Return {direct route: waypoints} (arm/direct.py), or {}.

    Only routes that still start at their grab pose and end at their
    target pose as config has them now.
    """
    from . import direct

    document = _read(path)

    if document is None:
        return {}

    poses = config.named_poses()
    found = {}

    for name, entry in (document.get('direct') or {}).items():
        if direct.split(name) is None:
            continue

        waypoints = np.asarray(entry['waypoints'], dtype=np.float64)
        start, end = poses.get(entry['from']), poses.get(entry['to'])

        if (
            start is not None and end is not None
            and np.abs(waypoints[0] - start).max() < 1e-5
            and np.abs(waypoints[-1] - end).max() < 1e-5
        ):
            found[name] = waypoints

    return found


def save_direct(path, entries):
    """Replace the direct routes in transfers.yaml (arm/direct.bake)."""
    document = _read(path)
    document['direct'] = {
        name: {
            'from': entry['from'],
            'to': entry['to'],
            'join': entry['join'],
            'exit': int(entry['exit']),
            'padding_m': entry['padding_m'],
            'waypoints': [
                [round(float(v), 6) for v in q] for q in entry['waypoints']
            ],
        }
        for name, entry in entries.items()
    }
    _write(path, document)


def save_timings(path, timings, chains=None, direct=None):
    """
    Store pass-through timings with their routes in transfers.yaml.

    timings: {route: pass_through.to_document(...)}, or None for a
    route to drop its timing. The routes themselves are not changed.
    chains: {chain name (chain_pairs): {'back', 'out', 'timed'}}, to
    replace the stored chains; None keeps them.
    direct: {direct route: pass_through.to_document(...) or None (no
    timing)}, the direct routes to keep; the others are removed. None
    keeps them all as they are.
    """
    document = _read(path)

    if direct is not None:
        kept = {}

        for name, entry in (document.get('direct') or {}).items():
            if name not in direct:
                continue

            entry.pop('timed', None)

            if direct[name] is not None:
                entry['timed'] = direct[name]

            kept[name] = entry

        document['direct'] = kept

    for name, timed in timings.items():
        if timed is None:
            document['routes'][name].pop('timed', None)
        else:
            document['routes'][name]['timed'] = timed

    if chains is not None:
        routes = {
            name: np.asarray(route['waypoints'])
            for name, route in document['routes'].items()
        }
        document['chains'] = {
            name: {
                'back': chain['back'],
                'out': chain['out'],
                # The routes it was timed along: a re-baked route
                # leaves its chains unused (load_timings).
                'digest': route_digest(
                    chain_waypoints(routes, chain['back'], chain['out'])
                ),
                'timed': chain['timed'],
            }
            for name, chain in chains.items()
        }

    _write(path, document)


def _write(path, document):
    with open(path, 'w') as handle:
        handle.write(
            '# Baked transfers from INTER to the named poses '
            '(laundry_control/arm/transfers.py).\n'
            '# Generated by `laundry plan bake transfers` - re-bake after '
            'changing\n# a recorded pose, the padding or config.OBSTACLES.\n'
            '# `timed`: pass-through timing, `laundry plan bake timing`.\n'
        )
        yaml.safe_dump(document, handle, sort_keys=False, width=100)


def baked_scene(path=None):
    """Return the scene stamp a transfers file was baked against ('' if none)."""
    document = _read(path)

    return '' if document is None else str(document.get('scene', ''))


def expected_gripper_clearance(name):
    """Return the gripper padding a route to `name` should be baked with."""
    return float(
        config.ROUTE_GRIPPER_CLEARANCE_M.get(name, config.GRIPPER_PADDING_M)
    )


def padding_mismatches(path=None):
    """
    Return {route: explanation} for routes baked with other padding.

    Compares what each route recorded against what config asks for now
    (route_arm_padding, expected_gripper_clearance). Replays are
    re-checked under the CURRENT arm padding, so a route baked with
    less may now be refused; one baked with more, or with a different
    gripper clearance, still runs but is not what a fresh bake gives.
    """
    document = _read(path)

    if document is None:
        return {}

    mismatches = {}

    for name, route in document['routes'].items():
        recorded = route.get('padding_m', {})
        wanted = {
            'arm_links': route_arm_padding(name),
            'gripper': expected_gripper_clearance(name),
        }

        changed = [
            f'{part} {recorded.get(part, 0.0) * 100:g} -> {value * 100:g} cm'
            for part, value in wanted.items()
            if abs(recorded.get(part, 0.0) - value) > 1e-9
        ]

        if changed:
            mismatches[name] = (
                'baked with other padding than config now sets ('
                + ', '.join(changed) + '); re-bake: laundry plan bake transfers'
            )

    return mismatches


def _at(joints, pose):
    return bool(np.abs(np.asarray(joints) - np.asarray(pose)).max() <= AT_POSE_TOLERANCE_RAD)


def from_other_inter(routes):
    """
    Return the routes that do not start at the current INTER, sorted.

    Every route is baked from INTER, so its first waypoint IS the
    INTER it was baked from. Once INTER changes (derived from a moved
    bucket, or re-recorded), replaying such a route would first swing
    the arm from the new INTER to the old one.
    """
    inter = config.get_named_pose('inter')

    return sorted(
        name for name, route in routes.items()
        if len(route) and not _at(route[0], inter) and '>' not in name
    )


def _where(current, routes):
    """Return 'inter', the baked target the arm is at, or None."""
    poses = config.named_poses()

    if _at(current, poses['inter']):
        return 'inter'

    # A file baked before a pose was retired can still hold its route.
    for name in routes:
        if name in poses and _at(current, poses[name]):
            return name

    return None


def _leaving_by(here, routes, gripper_empty):
    """Return the route to leave the baked pose `here` by, back to INTER."""
    empty_route = EMPTY_GRIPPER_RETURNS.get(here)

    if gripper_empty and empty_route in routes:
        return empty_route

    return here


def route_for(current, target_name, routes, gripper_empty=False):
    """
    Return the baked waypoints from `current` to target_name, or None.

    INTER is the hub every route starts from:

      - at INTER, heading for a baked target: that route;
      - at a baked target, heading for INTER: the same route reversed;
      - at one baked target, heading for another: back to INTER along
        the first route, then out along the second (the arm stops at
        INTER in between).

    gripper_empty: the gripper holds nothing, so a pose with an
    empty-gripper return (EMPTY_GRIPPER_RETURNS) is left by that
    tighter, faster route instead of its own. Off by default: only
    say so when the gripper is known to have let go.

    The first waypoint is the pose itself; callers bridge the small
    gap from `current` with a straight move.
    """
    pieces = _pieces(current, target_name, routes, gripper_empty)

    if pieces is None:
        return None

    return np.concatenate([
        _piece_waypoints(routes, name, backward)[0 if k == 0 else 1:]
        for k, (name, backward) in enumerate(pieces)
    ])


def _pieces(current, target_name, routes, gripper_empty=False):
    """
    Return the baked routes a move consists of, as [(name, backward)], or None.

    See route_for: one route out from INTER, one back to it (backward),
    or back along one and out along another.
    """
    target_name = target_name.lower()

    here = _where(current, routes)

    if here is None or here == target_name:
        return None

    direct = _direct_piece(here, target_name, routes, gripper_empty)

    if direct is not None:
        return [direct]

    if here == 'inter':
        return [(target_name, False)] if target_name in routes else None

    back = (_leaving_by(here, routes, gripper_empty), True)

    if target_name == 'inter':
        return [back]

    if target_name in routes:
        return [back, (target_name, False)]

    return None


def _direct_piece(here, target_name, routes, gripper_empty):
    """
    Return (direct route, backward) between a grab and DROP, or None.

    grab_NN -> DROP carrying laundry: 'grab_NN>drop'; DROP -> grab_NN
    once the gripper let go: 'grab_NN>drop_return', backwards (see
    arm/direct.py). Present only where it was found faster.
    """
    from . import direct

    for route_name in routes:
        parts = direct.split(route_name)

        if parts is None:
            continue

        grab, join = parts

        if here == grab and target_name == route_pose(join) and join == 'drop':
            return route_name, False

        if (
            here == route_pose(join) and target_name == grab
            and join == EMPTY_GRIPPER_RETURNS.get(here) and gripper_empty
        ):
            return route_name, True

    return None


def _piece_waypoints(routes, name, backward):
    return routes[name][::-1] if backward else routes[name]


def go_to(
    arm,
    target_name,
    routes=None,
    max_velocity_rad_s=None,
    time_scale=1.0,
    arm_paddings=None,
    gripper_empty=False,
    timings=None,
    before=(),
    after=(),
):
    """
    Move to a named pose along the most repeatable route available.

    timings: {route: (times, positions, velocities)} to pass through the
    vias (pass_through.py); read from transfers.yaml when routes is.

    gripper_empty: the gripper is known to hold nothing (it just let go
    at DROP), so DROP is left by drop_return (see route_for).

    before / after: straight moves to make first / afterwards, in the
    same trajectory; see prepare_go_to.

    1. A baked transfer (see route_for), re-checked against the
       current planning scene, then replayed exactly - under the
       arm-link padding config sets for it now (route_arm_padding;
       the smaller one when two routes are chained through INTER).
       Routes baked with other padding are warned about; routes baked
       from another INTER (see from_other_inter) are not used.
    2. Otherwise a straight, collision-checked joint move.
    3. Only if that would collide: the planner (move_joints), with a
       warning, since its route is not repeatable.
    """
    move = prepare_go_to(
        arm, target_name, routes=routes,
        max_velocity_rad_s=max_velocity_rad_s, time_scale=time_scale,
        arm_paddings=arm_paddings, gripper_empty=gripper_empty,
        timings=timings, before=before, after=after,
    )

    return move is not None and move()


def prepare_go_to(
    arm,
    target_name,
    routes=None,
    max_velocity_rad_s=None,
    time_scale=1.0,
    arm_paddings=None,
    gripper_empty=False,
    timings=None,
    before=(),
    after=(),
):
    """
    Choose and collision-check go_to's move now; return it, to make later.

    Returns a function that makes the move (True if it succeeded), or
    None if no move is allowed (the reason is logged). The arm must
    stay where it is until the move is made: the checks take ~0.5 s,
    and this lets a caller make them while the gripper opens or
    closes (grasp/retrieve_grid.run) instead of with the arm standing
    still.

    before: joint poses to go to first, each in a straight joint move
    (as move_joints_linear: config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S,
    collision-checked); the route is then the one from the last of
    them. after: poses to go on to the same way from the target. With
    a baked route the whole move is ONE trajectory - it comes to rest
    at each join, but does not wait there for a new goal and its
    checks. Otherwise the moves are made one after another.
    """
    stamp = None
    mismatches = {}

    if routes is None:
        routes, _baked_velocity = load()
        stamp = baked_scene()
        mismatches = padding_mismatches()
        timings = load_timings()

    moved = from_other_inter(routes)

    if moved:
        arm.get_logger().warning(
            'Baked transfers to ' + ', '.join(n.upper() for n in moved)
            + ' start from another INTER than the current one (INTER was '
            're-derived or re-recorded); they are not used. Re-bake: '
            'laundry plan bake transfers',
            throttle_duration_sec=60.0,
        )
        routes = {
            name: route for name, route in routes.items() if name not in moved
        }

    if stamp is not None:
        # Direct grab <-> DROP routes do not start at INTER: added after
        # the check above.
        routes = dict(routes, **load_direct())

    max_velocity_rad_s = (
        max_velocity_rad_s or config.TRANSFER_MAX_VELOCITY_RAD_S
    )

    target = config.get_named_pose(target_name)
    current = arm.get_current_joints()

    if current is None:
        return None

    before = [list(q) for q in before]
    after = [list(q) for q in after]
    origin = before[-1] if before else current

    route = route_for(origin, target_name, routes, gripper_empty)

    if route is None:
        return _prepare_unbaked(
            arm, target_name, current, target, before, after,
            max_velocity_rad_s, time_scale,
            {'routes': routes, 'timings': timings,
             'arm_paddings': arm_paddings, 'gripper_empty': gripper_empty},
        )

    here = _where(origin, routes)
    leaving = (
        None if here in (None, 'inter')
        else _leaving_by(here, routes, gripper_empty)
    )
    used = [
        name for name in (leaving, target_name.lower())
        if name in routes
    ]

    for name in used:
        if name in mismatches:
            arm.get_logger().warning(
                f'Route to {name.upper()} {mismatches[name]}',
                throttle_duration_sec=60.0,
            )

    # Checked under the arm padding config asks for NOW (arm_paddings
    # overrides it, for tests), so raising OBSTACLE_PADDING_M is
    # enforced on routes baked before - they are refused if they no
    # longer keep it - rather than replayed at their old clearance.
    wanted = dict(
        {name: route_arm_padding(name) for name in used},
        **(arm_paddings or {}),
    )
    padding = min(
        [config.OBSTACLE_PADDING_M]
        + [wanted.get(name, config.OBSTACLE_PADDING_M) for name in used]
    )

    pieces = _pieces(origin, target_name, routes, gripper_empty)

    if '>' in pieces[0][0]:
        note = f', direct by {pieces[0][0]}, not via INTER'
    else:
        note = (
            f', leaving {here.upper()} by {leaving}'
            if leaving not in (None, here) else ''
        )

    if before or after:
        note += f', with {len(before) + len(after)} straight move(s)'

    # The padding matters only to the checks: set it for them alone.
    changed = padding != config.OBSTACLE_PADDING_M

    if changed:
        arm.set_arm_padding(padding)

    try:
        plan = _checked_plan(
            arm, target_name, current, origin, route, before, after,
            pieces, routes, timings or {}, max_velocity_rad_s, stamp,
        )
    finally:
        if changed:
            arm.set_arm_padding(config.OBSTACLE_PADDING_M)

    if plan is None:
        return None

    message, times, positions, velocities = plan

    def move():
        arm.get_logger().info(message + note + ').')
        return arm.execute_joint_path(
            positions, times, velocities, time_scale=time_scale
        )

    return move


def _prepare_unbaked(
    arm, target_name, current, target, before, after, max_velocity_rad_s,
    time_scale, go_to_options,
):
    """prepare_go_to without a baked route: straight, else the planner."""
    if before or after:
        # One move at a time, each checked as it is made.
        def move():
            return (
                all(
                    arm.move_joints_linear(q, time_scale=time_scale)
                    for q in before
                )
                and go_to(
                    arm, target_name, max_velocity_rad_s=max_velocity_rad_s,
                    time_scale=time_scale, **go_to_options,
                )
                and all(
                    arm.move_joints_linear(q, time_scale=time_scale)
                    for q in after
                )
            )

        return move

    # Widened to where the arm is, as move_joints_linear checks it: a
    # hand move may have left a joint outside the task ranges.
    if arm.first_invalid_state(
        [current, target], limits=task_limits(start=current)
    ) is None:
        return lambda: arm.move_joints_linear(
            target,
            max_velocity_rad_s=max_velocity_rad_s,
            time_scale=time_scale,
        )

    arm.get_logger().warning(
        f'No baked or straight collision-free route to {target_name.upper()} '
        'from here; falling back to the planner (route not repeatable, but '
        'kept inside the task joint ranges).'
    )

    return lambda: arm.move_joints(target)


def _checked_plan(
    arm, target_name, current, origin, route, before, after, pieces, routes,
    timings, max_velocity_rad_s, stamp,
):
    """
    Return (log message, times, positions, velocities), or None.

    The first of the route's timings (_route_plans) whose states are
    all collision-free in the live planning scene, with the straight
    moves before and after it. None if the straight moves, or every
    timing, collide.
    """
    if stamp is not None:
        from . import scene

        stale = scene.stale_plan_message(
            stamp, 'scan_plans/transfers.yaml',
            'laundry plan bake transfers',
        )

        if stale:
            arm.get_logger().warning(stale, throttle_duration_sec=60.0)

    straight_before = _straight_legs([current] + before)
    straight_after = _straight_legs([route[-1]] + after)

    for start, leg in straight_before + straight_after:
        limits = task_limits(start=start)
        bad = arm.first_invalid_state(leg[1], limits=limits)

        if bad is not None:
            arm.get_logger().error(
                f'Straight joint move on the way to {target_name.upper()} '
                f'refused at {100.0 * bad / (len(leg[1]) - 1):.0f}% of the '
                f'way ({arm._why_invalid(leg[1][bad], limits)}); not moving.'
            )
            return None

    vias = len(route) - 2

    # Re-check the route as it will be run - including the (< 2 deg)
    # bridge from where the arm actually is - against the planning
    # scene loaded NOW. The route was checked when it was baked, but
    # possibly on another machine or against an older bucket/table
    # pose; replaying it unchecked would trust geometry that may have
    # moved since. A pass-through timing's rounded corners leave the
    # straight lines, so its own curve is what is checked.
    for how, legs, check in _route_plans(
        origin, pieces, routes, timings, max_velocity_rad_s
    ):
        bad = arm.first_invalid_state(check)

        if bad is None:
            times, positions, velocities = _join(
                [leg for _start, leg in straight_before]
                + legs
                + [leg for _start, leg in straight_after],
                current,
            )
            return (
                f'Baked transfer to {target_name.upper()} ({vias} via(s), '
                + how,
                times, positions, velocities,
            )

        if how.startswith('passing'):
            arm.get_logger().warning(
                f'The rounded corners of the transfer to '
                f'{target_name.upper()} collide with the current planning '
                f'scene ({how}); trying the next timing. Re-time: laundry '
                'plan bake timing'
            )

    arm.get_logger().error(
        f'Baked transfer to {target_name.upper()} collides with the '
        f'current planning scene (at checked state {bad}); the '
        'obstacles, padding or poses changed since it was baked. Not '
        'moving. Re-bake: laundry plan bake transfers'
    )
    return None


# Bridges shorter than this (rad, largest joint) are absorbed into the
# first point instead of moved as a leg of their own.
_BRIDGE_TOLERANCE_RAD = 1e-3


def _straight_legs(poses):
    """
    Return [(start, (times, states, velocities))] between consecutive poses.

    Each a straight joint move timed as move_joints_linear times it;
    the states are its 1 deg checks. Moves of nothing are left out.
    """
    legs = []

    for start, end in zip(poses[:-1], poses[1:]):
        states = densify([start, end])

        if np.abs(states[-1] - states[0]).max() < 1e-4:
            continue

        times, velocities = time_path(
            states, config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S
        )
        legs.append((np.asarray(start), (times, states, velocities)))

    return legs


def chain_name(pieces):
    """
    Return the name of the timing that runs `pieces` in one go, or None.

    Back to INTER along one route and out along another, passing
    through INTER: '<back>..<out>' (see chain_pairs).
    """
    if (
        pieces is not None and len(pieces) == 2
        and pieces[0][1] and not pieces[1][1]
    ):
        return f'{pieces[0][0]}..{pieces[1][0]}'

    return None


def chain_waypoints(routes, back, out):
    """Return the waypoints back along `back` and out along `out`."""
    return np.concatenate([
        _piece_waypoints(routes, back, True),
        _piece_waypoints(routes, out, False)[1:],
    ])


def chain_pairs(routes):
    """
    Return {chain name: (back, out)} for the moves a sweep makes.

    Each grab_NN -> DROP (back along grab_NN, out along drop), and
    DROP -> grab_NN once the gripper let go (back along drop_return):
    each is timed in one piece, passing through INTER, so the arm
    does not stop there (`laundry plan bake timing`).
    """
    empty = EMPTY_GRIPPER_RETURNS.get('drop')
    pairs = {}

    for grab in config.generated_grab_poses():
        if grab not in routes:
            continue

        for back, out in ((grab, 'drop'), (empty, grab)):
            if back in routes and out in routes:
                pairs[chain_name([(back, True), (out, False)])] = (back, out)

    return pairs


def route_digest(waypoints):
    """Return a short digest of waypoints, to tell when a chain's routes changed."""
    import hashlib

    rounded = np.round(np.asarray(waypoints, dtype=np.float64), 5) + 0.0

    return hashlib.sha1(rounded.tobytes()).hexdigest()[:16]


def _route_plans(origin, pieces, routes, timings, max_velocity_rad_s):
    """
    Yield the ways to time a move, fastest first: (how, legs, check states).

    1. pass through it all, INTER included (a chain timing, chain_name);
    2. pass through each piece's vias, stopping at INTER between them
       (the pieces' own timings);
    3. stop at every via.

    Each leg is (times, positions, velocities), starting and ending at
    rest; the first starts at origin (bridged with a short straight
    move when the route starts more than _BRIDGE_TOLERANCE_RAD away).
    1 and 2 only when config.TRANSFER_PASS_THROUGH_VIAS.
    """
    from . import pass_through

    def stop_at_each(waypoints):
        dense, times, velocities = time_stop_at_each(
            waypoints,
            max_velocity_rad_s,
            max_acceleration=config.TRANSFER_MAX_ACCELERATION_RAD_S2,
            max_jerk=config.TRANSFER_MAX_JERK_RAD_S3,
        )
        return (times, dense, velocities), dense

    def timed(name, backward):
        times, positions, velocities = timings[name]

        if backward:
            times, positions, velocities = pass_through.reverse(
                times, positions, velocities
            )

        return (
            (times, positions, velocities),
            pass_through.dense_states(times, positions, velocities),
        )

    def with_bridge(parts):
        start = parts[0][0][1][0]
        if np.abs(np.asarray(origin) - start).max() > _BRIDGE_TOLERANCE_RAD:
            parts = [stop_at_each([origin, start])] + parts

        return (
            [leg for leg, _check in parts],
            np.concatenate([check for _leg, check in parts]),
        )

    if config.TRANSFER_PASS_THROUGH_VIAS and pieces:
        chain = chain_name(pieces)

        if chain in timings:
            yield ('passing through, INTER included',) + with_bridge(
                [timed(chain, False)]
            )

        if any(name in timings for name, _backward in pieces):
            yield (
                'passing through' + (
                    ', stopping at INTER' if len(pieces) > 1 else ''
                ),
            ) + with_bridge([
                timed(name, backward) if name in timings
                else stop_at_each(_piece_waypoints(routes, name, backward))
                for name, backward in pieces
            ])

    route = np.concatenate([
        _piece_waypoints(routes, name, backward)[0 if k == 0 else 1:]
        for k, (name, backward) in enumerate(pieces)
    ])
    leg, check = stop_at_each([origin] + list(route))

    yield 'stopping at each', [leg], check


def _join(legs, start):
    """
    Return one (times, positions, velocities) from legs run one after another.

    Every leg starts and ends at rest: each is joined end to start
    (its first point dropped), and the first point is set to `start`,
    where the arm is (within the bridge tolerance).
    """
    times, positions, velocities = [], [], []
    elapsed = 0.0

    for k, (t, q, qd) in enumerate(legs):
        first = 0 if k == 0 else 1
        times.append(elapsed + np.asarray(t)[first:])
        positions.append(np.asarray(q)[first:])
        velocities.append(np.asarray(qd)[first:])
        elapsed += float(t[-1])

    positions = np.concatenate(positions)
    positions[0] = start

    return np.concatenate(times), positions, np.concatenate(velocities)
