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

# Targets reached from INTER through a baked transfer.
TRANSFER_TARGETS = ('home', 'drop')

# Joint-travel weights for choosing a route: J1 and J4-J7 twist the
# cables running down the arm, so their travel costs more.
TRAVEL_WEIGHTS = np.array([2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0])

# How far the gripper backs out of the bucket along the tool axis
# before swinging away, tried in this order (metres).
RETREAT_DISTANCES_M = (0.05, 0.10, 0.15)

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
    """Return the joint state with the flange backed out of INTER, or None."""
    tf = arm.get_flange_transform()
    t = tf.transform.translation
    q = tf.transform.rotation

    tool_z = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2]

    pose = Pose()
    position = np.array([t.x, t.y, t.z]) - distance_m * tool_z
    pose.position.x, pose.position.y, pose.position.z = map(float, position)
    pose.orientation = q

    solution = arm.compute_ik(pose, config.INTER)

    return None if solution is None else np.array(solution)


def _refine(arm, path):
    """
    Pull via poses toward monotone joint values, then drop spare vias.

    Only changes that keep the whole path collision-free AND lower
    the weighted travel are kept, so the result is never worse.
    """
    def valid(candidate):
        return arm.first_invalid_state(candidate) is None

    path = [np.array(p, dtype=np.float64) for p in path]

    improved = True
    rounds = 0

    while improved and rounds < 20:
        improved = False
        rounds += 1

        # Never move the endpoints or the retreat (index 1).
        for index in range(2, len(path) - 1):
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

    index = 2
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

    MOVES THE ARM to INTER (to read the flange pose for the retreat).
    """
    inter = np.array(config.INTER)
    target = np.array(config.get_named_pose(target_name))

    if not arm.move_joints(config.INTER):
        raise BakeError('Could not reach INTER.')

    if arm.first_invalid_state([inter, target]) is None:
        log(f'  INTER -> {target_name}: the straight line is collision-free.')
        return [inter, target]

    rng = np.random.default_rng(BAKE_SEED)

    best = None

    for distance in RETREAT_DISTANCES_M:
        retreat = _retreat(arm, distance)

        if retreat is None or arm.first_invalid_state([inter, retreat]) is not None:
            continue

        for via_count in (1, 2):
            for _ in range(SAMPLES_PER_CONFIG):
                along = np.sort(rng.uniform(0.15, 0.85, via_count))

                vias = [
                    retreat
                    + u * (target - retreat)
                    + rng.normal(0.0, VIA_JITTER_RAD, 7)
                    for u in along
                ]

                path = [inter, retreat] + vias + [target]
                cost = weighted_travel_deg(path)

                if best is not None and cost >= best[0]:
                    continue

                if arm.first_invalid_state(path) is None:
                    best = (cost, path)

    if best is None:
        raise BakeError(f'No collision-free route found from INTER to {target_name}.')

    path = _refine(arm, best[1])

    excess = per_joint_travel_deg(path) - np.degrees(np.abs(target - inter))

    log(
        f'  INTER -> {target_name}: {len(path) - 2} via(s); extra travel per '
        f'joint over the (colliding) direct line, deg: {np.round(excess, 0)}'
    )

    return path


def bake(arm, targets=TRANSFER_TARGETS, log=print):
    """Bake every transfer; returns {target: [joint waypoints]}."""
    log('Baking transfers from INTER...')

    routes = {name: bake_transfer(arm, name, log=log) for name in targets}

    arm.move_joints(config.INTER)

    return routes


def save(path, routes, max_velocity_rad_s, baked_on=''):
    """Write baked transfers as YAML."""
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
        'routes': {
            name: {
                'waypoints': [[round(float(v), 6) for v in q] for q in waypoints],
                'per_joint_travel_deg': [
                    round(float(v), 1) for v in per_joint_travel_deg(waypoints)
                ],
            }
            for name, waypoints in routes.items()
        },
    }

    with open(path, 'w') as handle:
        handle.write(
            '# Baked INTER <-> HOME/DROP transfers (laundry_control/arm/'
            'transfers.py).\n'
            '# Generated by `laundry plan bake transfers` - re-bake after '
            'changing\n# INTER, HOME, DROP or the URDF.\n'
        )
        yaml.safe_dump(document, handle, sort_keys=False, width=100)


def load(path=None):
    """Return ({target: waypoints array}, max_velocity), or ({}, None) if absent."""
    path = path or default_plan_path()

    if not os.path.isfile(path):
        return {}, None

    with open(path) as handle:
        document = yaml.safe_load(handle)

    if document.get('version') != PLAN_VERSION:
        raise ValueError(f'{path!r} is an unknown transfers version; re-bake.')

    if list(document['joint_names']) != list(config.JOINT_NAMES):
        raise ValueError(f'{path!r} was baked for different joints.')

    routes = {
        name: np.array(route['waypoints'])
        for name, route in document['routes'].items()
    }

    return routes, float(document['max_velocity_rad_s'])


def _at(joints, pose):
    return bool(np.abs(np.asarray(joints) - np.asarray(pose)).max() <= AT_POSE_TOLERANCE_RAD)


def route_for(current, target_name, routes):
    """
    Return the baked waypoints from `current` to target_name, or None.

    Applies when the arm is at INTER heading for a baked target, or at
    a baked target heading for INTER (the same route, reversed). The
    first waypoint is the pose itself; callers bridge the small gap
    from `current` with a straight move.
    """
    target_name = target_name.lower()

    if target_name in routes and _at(current, config.INTER):
        return routes[target_name]

    if target_name == 'inter':
        for name, waypoints in routes.items():
            if _at(current, config.get_named_pose(name)):
                return waypoints[::-1]

    return None


def go_to(arm, target_name, routes=None, max_velocity_rad_s=None, time_scale=1.0):
    """
    Move to a named pose along the most repeatable route available.

    1. A baked transfer (INTER <-> HOME/DROP), replayed exactly.
    2. Otherwise a straight, collision-checked joint move.
    3. Only if that would collide: the planner (move_joints), with a
       warning, since its route is not repeatable.
    """
    if routes is None:
        routes, baked_velocity = load()
        max_velocity_rad_s = max_velocity_rad_s or baked_velocity

    max_velocity_rad_s = (
        max_velocity_rad_s or config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S
    )

    target = config.get_named_pose(target_name)
    current = arm.get_current_joints()

    if current is None:
        return False

    route = route_for(current, target_name, routes)

    if route is not None:
        arm.get_logger().info(
            f'Baked transfer to {target_name.upper()} '
            f'({len(route) - 2} via(s)).'
        )

        # Bridge the (< 2 deg) gap from where the arm actually is to the
        # route's own first pose - checked like everything else.
        if arm.first_invalid_state([current, route[0]]) is not None:
            arm.get_logger().error('Start of the baked transfer is blocked.')
            return False

        waypoints, times, velocities = time_stop_at_each(
            [current] + [list(q) for q in route], max_velocity_rad_s
        )

        return arm.execute_joint_path(
            waypoints, times, velocities, time_scale=time_scale
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
