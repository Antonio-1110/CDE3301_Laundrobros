#!/usr/bin/env python3

"""
Generated grab poses for the sensorless first pass (`laundry preplanned`).

WHY
---
The four grab poses once jogged by hand (RETRIEVE_0..3, now only
config.GRAB_IK_SEEDS) all sat on one line down the middle of the
floor, 11-39 cm from the closed end, two of them on nearly the same
spot, and one inside the modelled wall. Laundry lies across the whole
bottom of the bucket, so a sweep that is meant to save a scan should
cover it evenly.

WHAT
----
The sweep's grabs are placed by hand, relative to the bucket, in
scan_plans/grab_targets.yaml (grasp/grab_targets.py; edit them by
dragging in RViz with `laundry plan edit-grabs`). solve() takes each
one EXACTLY as placed - depth, floor angle, height, tilt and approach
distance - in the configured bucket (config.OBSTACLES, via
perception.bucket_model.seed_cone), and checks by IK that it is
collision-free with the gripper padded by RETRIEVE_GRID's
clearance_m, that its approach pose (approach_m back along the tool
axis) is too, and that the straight joint-space descent between the
two is clear. A grab that fails is reported with the reason and left
out; nothing moves it to a height or tilt that would work. IK is
seeded from INTER and config.GRAB_IK_SEEDS - the arm postures known
to reach this bucket's floor - and among seeds the one closest to
INTER in weighted joint travel wins (less cable twist). The claw
keeps the seeds' roll: local +X toward the mouth.

Everything is saved to scan_plans/retrieve.yaml, stamped with the
scene and the targets; each approach pose becomes the named pose
grab_NN (see config.named_poses), which `laundry plan bake` gives a
baked route from INTER like any other.

run() replays them: baked route to grab_NN, straight down, close,
straight up, DROP, open.

plan_floor_grab() does a search for a DETECTED item (the
sensor-guided grasp), over config.RETRIEVE_GRID's heights and tilts:
the grab is placed over the item, sunk into it,
and reached by a straight, collision-checked joint move from INTER
(or, when that line collides, from the nearest baked grab_NN) - so
the whole approach is as repeatable as the grid's.
"""

import contextlib
import datetime
import os

from geometry_msgs.msg import Pose
import numpy as np
import yaml

from . import grab_targets
from .. import config
from ..arm.geometry import look_at_quaternion
from ..perception.bucket_model import _axis_basis, seed_cone, to_cylindrical

PLAN_VERSION = 1

# An IK solution with any joint further than this from INTER is a
# flipped posture, not a grab: the IK solver restarts from random
# joint values when it cannot converge from the seed, and can land
# with J1 turned a full revolution or the wrist (J5, J7) flipped.
# Such a grab has no clean route from INTER. Measured (fake
# controller, 2026-09-28, three bakes of the grid): real grabs stay
# within 111 deg of INTER in every joint (approach poses within 96);
# flipped ones were 176-448 deg. Rejected, so the grab falls back to
# its next height/tilt instead.
MAX_TRAVEL_FROM_INTER_DEG = 150.0

# How far into a detected pile to sink the grab, as in grasp/plan.py:
# half the pile's sensed height above the floor, at most 5 cm.
FLOOR_GRAB_SINK_FRACTION = 0.5
FLOOR_GRAB_MAX_SINK_M = 0.05


def plan_path():
    """Return <repo>/scan_plans/retrieve.yaml."""
    return config.retrieve_plan_path()


def grab_name(index):
    """Return the named-pose name of the index-th grab (from 1)."""
    return f'grab_{index:02d}'


def grab_geometry(depth_m, floor_angle_deg, height_m, tilt_deg, cone=None):
    """
    Return (contact point, tool +Z, claw +X) for one grab, in link_base.

    The contact point is height_m in from the wall along the bucket's
    radius at floor_angle_deg (0 = the floor's lowest line, positive
    toward +x), depth_m from the closed end. Tool +Z points at the
    floor along that radius, tilted by tilt_deg toward the closed end.
    """
    cone = cone or seed_cone()
    axis = cone.axis_dir
    up, side = _axis_basis(axis)

    if side[0] < 0.0:
        side = -side

    phi = np.radians(floor_angle_deg)
    outward = -np.cos(phi) * up + np.sin(phi) * side

    contact = (
        cone.axis_point
        + depth_m * axis
        + (cone.radius_at(depth_m) - height_m) * outward
    )

    tilt = np.radians(tilt_deg)
    tool_z = np.cos(tilt) * outward - np.sin(tilt) * axis

    return contact, tool_z, axis


def _flange_pose(contact, tool_z, claw_x, back_off_m=0.0):
    flange = contact - (config.GRIPPER_OFFSET_Z + back_off_m) * tool_z

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, flange)
    pose.orientation = look_at_quaternion(tool_z, claw_x)

    return pose


def _seeds():
    return [config.get_named_pose('inter')] + [
        list(seed) for seed in config.GRAB_IK_SEEDS
    ]


def flipped(joints, inter):
    """Return True if any joint is past MAX_TRAVEL_FROM_INTER_DEG from INTER."""
    travel = np.degrees(np.abs(np.asarray(joints) - np.asarray(inter)))

    return bool(travel.max() > MAX_TRAVEL_FROM_INTER_DEG)


def solve_one(arm, depth_m, floor_angle_deg, grid=None, seeds=None):
    """
    Return the best grab at one spot, searching heights then tilts.

    Used for detected items (plan_floor_grab). Must run with the
    gripper padded by the grid's clearance (gripper_padding).
    Checks: grab and approach states, and the straight descent
    between them; solutions in a flipped posture (see
    MAX_TRAVEL_FROM_INTER_DEG) are skipped. seeds default to INTER
    and config.GRAB_IK_SEEDS. grid needs 'heights_m' (tried in
    order, each at every tilt) on top of RETRIEVE_GRID's keys.
    """
    grid = grid or config.RETRIEVE_GRID
    inter = np.array(config.get_named_pose('inter'))
    seeds = seeds or _seeds()

    candidates = [
        (height, tilt)
        for height in grid['heights_m']
        for tilt in grid['tilts_deg']
    ]

    for height, tilt in candidates:
        contact, tool_z, claw_x = grab_geometry(
            depth_m, floor_angle_deg, height, tilt
        )

        found, _ = _try_pose(
            arm, contact, tool_z, claw_x, grid['approach_m'], seeds, inter
        )

        if found is not None:
            return dict(
                found,
                depth_m=float(depth_m),
                floor_angle_deg=float(floor_angle_deg),
                height_m=float(height),
                tilt_deg=float(tilt),
            )

    return None


# Why _try_pose() found nothing, by how far the best seed got.
_FAILURES = (
    'no IK for the grab (out of reach, or the gripper collides there)',
    'the grab only solves in a flipped posture',
    'no IK for the approach pose (it collides, or is out of reach)',
    'the straight descent from the approach pose collides',
)


def _try_pose(arm, contact, tool_z, claw_x, approach_m, seeds, inter):
    """
    Solve one exact grab pose; return (dict, None) or (None, reason).

    The dict has 'contact', 'approach', 'grab' and 'travel_deg': the
    seed whose approach pose is nearest INTER in weighted joint travel.
    """
    from ..arm.transfers import weighted_travel_deg

    best = None
    stage = 0

    for seed in seeds:
        grab = arm.compute_ik(_flange_pose(contact, tool_z, claw_x), seed)

        if grab is None:
            continue

        if flipped(grab, inter):
            stage = max(stage, 1)
            continue

        approach = arm.compute_ik(
            _flange_pose(contact, tool_z, claw_x, approach_m), grab,
        )

        if approach is None or flipped(approach, inter):
            stage = max(stage, 2)
            continue

        if arm.first_invalid_state([approach, grab]) is not None:
            stage = max(stage, 3)
            continue

        cost = weighted_travel_deg([inter, approach])

        if best is None or cost < best['travel_deg']:
            best = {
                'contact': [round(float(v), 4) for v in contact],
                'approach': [round(float(v), 6) for v in approach],
                'grab': [round(float(v), 6) for v in grab],
                'travel_deg': round(cost, 1),
            }

    if best is None:
        return None, _FAILURES[stage]

    return best, None


def solve_target(arm, target, seeds=None):
    """
    Solve one grab of grab_targets.yaml exactly as placed.

    Returns (plan, None) or (None, reason). Must run with the gripper
    padded (gripper_padding). The plan is the target's fields plus
    'contact', 'approach', 'grab' and 'travel_deg'.
    """
    contact, tool_z, claw_x = grab_targets.geometry(target)

    found, reason = _try_pose(
        arm, contact, tool_z, claw_x, target['approach_m'],
        seeds or _seeds(), np.array(config.get_named_pose('inter')),
    )

    if found is None:
        return None, reason

    return dict(target, **found), None


@contextlib.contextmanager
def gripper_padding(arm, clearance_m=None):
    """Pad the gripper by clearance_m (default RETRIEVE_GRID's) for a block."""
    from ..arm import scene

    if clearance_m is None:
        clearance_m = config.RETRIEVE_GRID['clearance_m']

    scene.set_padding(
        arm,
        config.OBSTACLE_PADDING_M,
        {config.GRIPPER_LINK: config.GRIPPER_PADDING_M + clearance_m},
    )

    try:
        yield
    finally:
        scene.set_padding(
            arm,
            config.OBSTACLE_PADDING_M,
            {config.GRIPPER_LINK: config.GRIPPER_PADDING_M},
        )


def solve(arm, targets=None, log=print):
    """
    Solve every grab of grab_targets.yaml exactly as placed, in order.

    Returns (grabs, misses): grabs are solve_target() plans with a
    grab_NN 'name'; misses are (position in the list from 1, target,
    reason) for the grabs nothing reached.
    """
    targets = grab_targets.load() if targets is None else targets

    grabs = []
    misses = []

    with gripper_padding(arm):
        for index, target in enumerate(targets, start=1):
            plan, reason = solve_target(arm, target)

            where = f'#{index:<2} {grab_targets.describe(target)}'

            if plan is None:
                misses.append((index, target, reason))
                log(f'  {where}: NOT REACHABLE - {reason}')
                continue

            plan['name'] = grab_name(len(grabs) + 1)
            grabs.append(plan)

            log(f'  {plan["name"]}  {where}')

    return grabs, misses


def floor_position(point, cone=None):
    """
    Return (depth, floor angle deg, height above floor) of a point.

    In the configured bucket, with solve()'s conventions: depth from
    the closed end, angle from the floor's lowest line (positive toward
    +x), height inward from the wall.
    """
    cone = cone or seed_cone()

    s, theta, r = to_cylindrical(np.asarray(point, dtype=float)[None], cone)

    # theta is measured from "up" toward +x; the floor is at pi.
    angle = np.degrees(np.pi - theta[0])
    angle = (angle + 180.0) % 360.0 - 180.0

    return float(s[0]), float(angle), float(cone.radius_at(s[0]) - r[0])


def plan_floor_grab(arm, point, grabs, grid=None):
    """
    Plan a grid-style grab at a detected item's sensed top, or None.

    Returns the solve_one() dict plus 'via': the baked grab_NN (or
    INTER, as a last resort) it is reached from by a straight joint
    move. The grab sinks FLOOR_GRAB_SINK_FRACTION of the pile's sensed
    height into it (at most FLOOR_GRAB_MAX_SINK_M, never lower than
    the grid's lowest height); on a wall, "height" is the distance in
    from the wall. None if the item is beyond
    config.DETECTED_GRAB_MAX_ANGLE_DEG, out of the bucket, or nothing
    reachable was found.
    """
    from ..arm import scene

    grid = dict(grid or config.RETRIEVE_GRID)

    depth, angle, top_height = floor_position(point)

    if abs(angle) > config.DETECTED_GRAB_MAX_ANGLE_DEG or not (
        0.0 < depth < grid['max_depth_m']
    ):
        return None

    top_height = max(0.0, top_height)
    sink = min(FLOOR_GRAB_SINK_FRACTION * top_height, FLOOR_GRAB_MAX_SINK_M)
    lowest = max(grid['min_height_m'], top_height - sink)
    grid['heights_m'] = [lowest + 0.01 * k for k in range(4)]

    # Seed IK from the grabs nearest this spot: same elbow, same wrist.
    nearby = sorted(
        grabs,
        key=lambda g: np.hypot(
            g['depth_m'] - depth, np.radians(g['floor_angle_deg'] - angle) * 0.2
        ),
    )
    seeds = [g['grab'] for g in nearby[:3]] + _seeds()

    scene.set_padding(
        arm,
        config.OBSTACLE_PADDING_M,
        {config.GRIPPER_LINK: config.GRIPPER_PADDING_M + grid['clearance_m']},
    )

    try:
        plan = solve_one(arm, depth, angle, grid, seeds=seeds)
    finally:
        scene.set_padding(
            arm,
            config.OBSTACLE_PADDING_M,
            {config.GRIPPER_LINK: config.GRIPPER_PADDING_M},
        )

    if plan is None:
        return None

    # Straight from INTER when that line is clear, else from the nearest
    # (in joint space) grid grab with a clear line. Checked on the fake
    # controller (2026-10-06, 39 positions): INTER's line was clear for
    # 38, and skipping the stop at a grab_NN saved ~1.8 s each way.
    approach = np.array(plan['approach'])

    vias = [('inter', config.get_named_pose('inter'))] + [
        (grab['name'], grab['approach'])
        for grab in sorted(
            grabs,
            key=lambda g: np.abs(np.array(g['approach']) - approach).max(),
        )
    ]

    for name, joints in vias:
        if arm.first_invalid_state([joints, plan['approach']]) is None:
            plan['via'] = name
            return plan

    return None


def run_floor_grab(arm, gripper, plan, go_to, drop=True, log=print):
    """
    Execute a plan_floor_grab() plan; True if the item was taken (and dropped).

    via (INTER, or a baked grab_NN when INTER's line collides) ->
    approach -> down -> close -> up -> back to via -> DROP -> open ->
    INTER. Always lifts back out.
    """
    via = plan['via']

    log(
        f'Floor grab via {via}: {plan["depth_m"] * 100:.0f} cm deep, '
        f'{plan["floor_angle_deg"]:+.0f} deg, {plan["height_m"] * 100:.1f} cm '
        f'from the wall, tilt {plan["tilt_deg"]:.0f} deg'
    )

    if not gripper.open_blocking():
        log('Gripper did not confirm it opened; aborting before the approach.')
        return False

    if not (
        go_to(arm, via)
        and arm.move_joints_linear(plan['approach'])
        and arm.move_joints_linear(plan['grab'])
    ):
        log('Could not reach the grab; returning to INTER.')
        arm.move_joints_linear(plan['approach'])
        go_to(arm, 'inter')
        return False

    closed = gripper.close_blocking()

    # Back out the way it came, so the route to DROP starts at a
    # baked pose.
    if not (
        arm.move_joints_linear(plan['approach'])
        and arm.move_joints_linear(config.get_named_pose(via))
    ):
        log('Could not lift back out; stopping here.')
        return False

    if not closed:
        log('Gripper did not confirm it closed; returning to INTER.')
        go_to(arm, 'inter')
        return False

    if not drop:
        return go_to(arm, 'inter')

    if not go_to(arm, 'drop'):
        log('Failed to reach DROP.')
        return False

    released = gripper.open_blocking()

    if not released:
        log('Gripper did not confirm it opened at DROP; the item may still '
            'be held.')

    return go_to(arm, 'inter', gripper_empty=released) and released


def save(grabs, path=None, baked_on='', targets=None):
    """Write the grabs as YAML, stamped with the targets and the scene."""
    from ..arm import scene

    path = path or plan_path()
    targets = grab_targets.load() if targets is None else targets

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    document = {
        'version': PLAN_VERSION,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'baked_on': baked_on,
        'joint_names': list(config.JOINT_NAMES),
        'scene': scene.signature(),
        'obstacles': scene.describe(),
        'targets': [grab_targets.clean(target) for target in targets],
        'clearance_m': float(config.RETRIEVE_GRID['clearance_m']),
        'gripper_offset_z': float(config.GRIPPER_OFFSET_Z),
        'grabs': [
            {key: grab[key] for key in (
                'name', 'depth_m', 'floor_angle_deg', 'height_m', 'tilt_deg',
                'approach_m', 'contact', 'approach', 'grab',
            ) if key in grab}
            for grab in grabs
        ],
    }

    with open(path, 'w') as handle:
        handle.write(
            '# Generated grab poses for `laundry preplanned` '
            '(laundry_control/grasp/retrieve_grid.py).\n'
            '# Generated by `laundry plan bake retrieve` - re-bake after '
            'changing\n# scan_plans/grab_targets.yaml, config.OBSTACLES, '
            'the clearance or the gripper offset.\n'
        )
        yaml.safe_dump(document, handle, sort_keys=False, width=100)


def grid_mismatch(path=None, targets=None):
    """
    Return why retrieve.yaml no longer matches what it was baked for, or None.

    The grab targets (scan_plans/grab_targets.yaml), the gripper
    clearance and the gripper offset, against now.
    """
    path = path or plan_path()

    if not os.path.isfile(path):
        return None

    with open(path) as handle:
        document = yaml.safe_load(handle) or {}

    targets = grab_targets.load() if targets is None else targets

    def same(a, b):
        return np.allclose(np.asarray(a, float), np.asarray(b, float))

    def values(items):
        return [[t[f] for f in grab_targets.FIELDS] for t in items]

    baked = [grab_targets.clean(t) for t in document.get('targets', [])]
    changed = []

    if len(baked) != len(targets) or (
        baked and not same(values(baked), values(targets))
    ):
        changed.append('grab_targets.yaml')

    if not same(
        document.get('clearance_m', -1.0), config.RETRIEVE_GRID['clearance_m']
    ):
        changed.append('clearance_m')

    if not same(document.get('gripper_offset_z', 0.0), config.GRIPPER_OFFSET_Z):
        changed.append('GRIPPER_OFFSET_Z')

    if not changed:
        return None

    return (
        'scan_plans/retrieve.yaml was solved for other settings ('
        + ', '.join(changed) + ' changed); re-bake: laundry plan bake retrieve'
    )


def load(path=None):
    """Return (grabs in visiting order, scene stamp); ([], '') if absent."""
    path = path or plan_path()

    if not os.path.isfile(path):
        return [], ''

    with open(path) as handle:
        document = yaml.safe_load(handle) or {}

    if document.get('version') != PLAN_VERSION:
        raise ValueError(f'{path!r} is an unknown retrieve-plan version; re-bake.')

    if list(document['joint_names']) != list(config.JOINT_NAMES):
        raise ValueError(f'{path!r} was baked for different joints.')

    return list(document.get('grabs', [])), str(document.get('scene', ''))


def run(arm, gripper, grabs, go_to, time_scale=1.0, log=print):
    """
    Visit each grab: route in, down, close, up, DROP, open.

    Returns True if every motion succeeded. Stops at the first failure
    after getting the arm back to INTER where it can.
    """
    log('Opening gripper...')

    if not gripper.open_blocking():
        log('Gripper did not confirm it opened; aborting before any grab.')
        return False

    for grab in grabs:
        name = grab['name']

        log(
            f'{name}: {grab["depth_m"] * 100:.0f} cm deep, '
            f'{grab["floor_angle_deg"]:+.0f} deg, '
            f'{grab["height_m"] * 100:.0f} cm above the floor'
        )

        # The gripper confirmed it opened (before the first grab, else
        # at DROP below), so DROP is left by the empty-gripper route.
        if not go_to(arm, name, time_scale=time_scale, gripper_empty=True):
            log(f'Failed to reach {name}; aborting.')
            go_to(arm, 'inter', time_scale=time_scale)
            return False

        if not arm.move_joints_linear(grab['grab'], time_scale=time_scale):
            log(f'Descent at {name} would collide; aborting.')
            go_to(arm, 'inter', time_scale=time_scale)
            return False

        closed = gripper.close_blocking()

        # Always lift back out, even with nothing held.
        if not arm.move_joints_linear(grab['approach'], time_scale=time_scale):
            log(f'Could not lift back out at {name}; stopping here.')
            return False

        if not closed:
            log('Gripper did not confirm it closed; returning to INTER and '
                'aborting.')
            go_to(arm, 'inter', time_scale=time_scale)
            return False

        if not go_to(arm, 'drop', time_scale=time_scale):
            log('Failed to reach DROP; aborting.')
            return False

        if not gripper.open_blocking():
            log('Gripper did not confirm it opened at DROP; returning to '
                'INTER and aborting.')
            go_to(arm, 'inter', time_scale=time_scale)
            return False

    log('Returning to INTER...')

    return go_to(arm, 'inter', time_scale=time_scale, gripper_empty=True)
