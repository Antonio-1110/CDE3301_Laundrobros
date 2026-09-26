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
follows exactly the checked straight segments.

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

from .joint_path import time_stop_at_each
from .. import config

PLAN_VERSION = 1

# Targets reached from INTER through a baked transfer. INTER is the
# hub: a move between two of these (RETRIEVE_n -> DROP, say) replays
# one route back to INTER and the other out from it - through the
# bucket mouth, never across the rim.
TRANSFER_TARGETS = (
    'home',
    'drop',
    'bottom',
    'retrieve_0',
    'retrieve_1',
    'retrieve_2',
    'retrieve_3',
)


def transfer_targets():
    """Return TRANSFER_TARGETS plus the generated grab_NN approach poses."""
    return TRANSFER_TARGETS + tuple(config.generated_grab_poses())


# Targets outside the bucket. Their routes are baked with extra gripper
# clearance (config.BAKE_GRIPPER_CLEARANCE_M) on top of the live
# padding, since nothing on the way there needs the gripper close to
# the bucket. Routes into it (BOTTOM, RETRIEVE_n) end with the gripper
# at the floor on purpose, so they are baked at the live padding.
OUTSIDE_TARGETS = ('home', 'drop')

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
    fk = arm.compute_fk(config.INTER)

    if fk is None:
        return None

    position, quaternion = fk

    tool_z = Rotation.from_quat(quaternion).as_matrix()[:, 2]

    pose = Pose()
    position = position - distance_m * tool_z
    pose.position.x, pose.position.y, pose.position.z = map(float, position)
    pose.orientation.x, pose.orientation.y = map(float, quaternion[:2])
    pose.orientation.z, pose.orientation.w = map(float, quaternion[2:])

    solution = arm.compute_ik(pose, config.INTER)

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
    inter = np.array(config.INTER)
    target = np.array(config.get_named_pose(target_name))

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
    OUTSIDE_TARGETS are found with the gripper padded by
    config.BAKE_GRIPPER_CLEARANCE_M; the rest at the live padding.
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
    to poses that still exist and are not being re-baked (`replacing`).
    """
    from . import scene

    document = _read(path)

    if not document or document.get('scene') != scene.signature():
        return {}

    valid = set(transfer_targets()) - set(replacing)

    return {
        name: route for name, route in document['routes'].items()
        if name in valid
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

    with open(path, 'w') as handle:
        handle.write(
            '# Baked transfers from INTER to the named poses '
            '(laundry_control/arm/transfers.py).\n'
            '# Generated by `laundry plan bake transfers` - re-bake after '
            'changing\n# a recorded pose, the padding or config.OBSTACLES.\n'
        )
        yaml.safe_dump(document, handle, sort_keys=False, width=100)


def route_arm_padding(name):
    """Return the arm-link padding a route is baked and replayed with."""
    return float(
        config.ROUTE_ARM_PADDING_M.get(name, config.OBSTACLE_PADDING_M)
    )


def _read(path):
    path = path or default_plan_path()

    if not os.path.isfile(path):
        return None

    with open(path) as handle:
        return yaml.safe_load(handle)


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


def baked_scene(path=None):
    """Return the scene stamp a transfers file was baked against ('' if none)."""
    document = _read(path)

    return '' if document is None else str(document.get('scene', ''))


def expected_gripper_clearance(name):
    """Return the gripper padding a route to `name` should be baked with."""
    return float(
        config.BAKE_GRIPPER_CLEARANCE_M if name in OUTSIDE_TARGETS
        else config.GRIPPER_PADDING_M
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


def _where(current, routes):
    """Return 'inter', the baked target the arm is at, or None."""
    if _at(current, config.INTER):
        return 'inter'

    for name in routes:
        if _at(current, config.get_named_pose(name)):
            return name

    return None


def route_for(current, target_name, routes):
    """
    Return the baked waypoints from `current` to target_name, or None.

    INTER is the hub every route starts from:

      - at INTER, heading for a baked target: that route;
      - at a baked target, heading for INTER: the same route reversed;
      - at one baked target, heading for another: back to INTER along
        the first route, then out along the second (the arm stops at
        INTER in between).

    The first waypoint is the pose itself; callers bridge the small
    gap from `current` with a straight move.
    """
    target_name = target_name.lower()

    here = _where(current, routes)

    if here is None or here == target_name:
        return None

    if here == 'inter':
        return routes.get(target_name)

    back = routes[here][::-1]

    if target_name == 'inter':
        return back

    if target_name in routes:
        return np.concatenate([back, routes[target_name][1:]])

    return None


def go_to(
    arm,
    target_name,
    routes=None,
    max_velocity_rad_s=None,
    time_scale=1.0,
    arm_paddings=None,
):
    """
    Move to a named pose along the most repeatable route available.

    1. A baked transfer (see route_for), re-checked against the
       current planning scene, then replayed exactly - under the
       arm-link padding config sets for it now (route_arm_padding;
       the smaller one when two routes are chained through INTER).
       Routes baked with other padding are warned about.
    2. Otherwise a straight, collision-checked joint move.
    3. Only if that would collide: the planner (move_joints), with a
       warning, since its route is not repeatable.
    """
    stamp = None
    mismatches = {}

    if routes is None:
        routes, baked_velocity = load()
        max_velocity_rad_s = max_velocity_rad_s or baked_velocity
        stamp = baked_scene()
        mismatches = padding_mismatches()

    max_velocity_rad_s = (
        max_velocity_rad_s or config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S
    )

    target = config.get_named_pose(target_name)
    current = arm.get_current_joints()

    if current is None:
        return False

    route = route_for(current, target_name, routes)

    if route is not None:
        used = [
            name for name in (_where(current, routes), target_name.lower())
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

        return _replay(
            arm, target_name, current, route, max_velocity_rad_s,
            time_scale, stamp, padding,
        )

    if arm.first_invalid_state([current, target]) is None:
        return arm.move_joints_linear(
            target,
            max_velocity_rad_s=max_velocity_rad_s,
            time_scale=time_scale,
        )

    arm.get_logger().warning(
        f'No baked or straight collision-free route to {target_name.upper()} '
        'from here; falling back to the planner (route not repeatable).'
    )

    return arm.move_joints(target)


def _replay(
    arm, target_name, current, route, max_velocity_rad_s, time_scale, stamp,
    padding,
):
    """Re-check and replay a baked route under its arm-link padding."""
    changed = padding != config.OBSTACLE_PADDING_M

    if changed:
        arm.set_arm_padding(padding)

    try:
        return _replay_checked(
            arm, target_name, current, route, max_velocity_rad_s,
            time_scale, stamp,
        )
    finally:
        if changed:
            arm.set_arm_padding(config.OBSTACLE_PADDING_M)


def _replay_checked(
    arm, target_name, current, route, max_velocity_rad_s, time_scale, stamp,
):
    """Re-check a baked route against the live scene, then replay it."""
    if stamp is not None:
        from . import scene

        stale = scene.stale_plan_message(
            stamp, 'scan_plans/transfers.yaml',
            'laundry plan bake transfers',
        )

        if stale:
            arm.get_logger().warning(stale, throttle_duration_sec=60.0)

    arm.get_logger().info(
        f'Baked transfer to {target_name.upper()} '
        f'({len(route) - 2} via(s), stopping at each).'
    )

    path = [current] + [list(q) for q in route]

    # Re-check the WHOLE route - including the (< 2 deg) bridge from
    # where the arm actually is - against the planning scene loaded
    # NOW. The route was checked when it was baked, but possibly on
    # another machine or against an older bucket/table pose;
    # replaying it unchecked would trust geometry that may have
    # moved since.
    bad = arm.first_invalid_state(path)

    if bad is not None:
        arm.get_logger().error(
            f'Baked transfer to {target_name.upper()} collides with the '
            f'current planning scene (at checked state {bad}); the '
            'obstacles, padding or poses changed since it was baked. Not '
            'moving. Re-bake: laundry plan bake transfers'
        )
        return False

    waypoints, times, velocities = time_stop_at_each(
        path, max_velocity_rad_s
    )

    return arm.execute_joint_path(
        waypoints, times, velocities, time_scale=time_scale
    )
