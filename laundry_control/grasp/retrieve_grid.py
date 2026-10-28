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
bucket.seed_cone), and checks by IK that it is
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
sensor-guided grasp), over config.RETRIEVE_GRID's tilts: the grab is
placed over the item and dug as deep into it as is collision-free
(deepest_grab, config.DETECTED_GRAB_DIG), and reached by a straight,
collision-checked joint move from INTER (or, when that line collides,
from the nearest baked grab_NN) - so the whole approach is as
repeatable as the grid's.
"""

import contextlib
import datetime
import os

from geometry_msgs.msg import Pose

import numpy as np
import yaml

from . import carry, grab_targets
from .. import config
from ..arm.geometry import look_at_quaternion
from ..bucket import axis_basis, seed_cone, to_cylindrical

PLAN_VERSION = 1

# The approach pose of a detected item's grab starts at least this far
# above the pile's sensed top (along the tool axis), so the move to it
# never drags the claw sideways through the laundry: all the digging
# is the straight descent.
ABOVE_PILE_M = 0.03


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
    up, side = axis_basis(axis)

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


def down_grab_geometry(depth_m, lateral_m, height_m, tilt_deg, cone=None):
    """
    Return (contact point, tool +Z, claw +X) for a DOWN grab, in link_base.

    The contact point is lateral_m sideways of the drum's middle
    (positive toward +x, as grab_geometry's floor angles) and height_m
    above the floor's lowest line, depth_m from the closed end. Tool +Z
    points straight down, tilted by tilt_deg toward the closed end
    (grab_targets: down and radial grabs).
    """
    cone = cone or seed_cone()
    axis = cone.axis_dir
    up, side = axis_basis(axis)

    if side[0] < 0.0:
        side = -side

    contact = (
        cone.axis_point
        + depth_m * axis
        + (height_m - cone.radius_at(depth_m)) * up
        + lateral_m * side
    )

    tilt = np.radians(tilt_deg)
    tool_z = -np.cos(tilt) * up - np.sin(tilt) * axis

    return contact, tool_z, axis


def angle_about_axis(depth_m, lateral_m, height_m, cone=None):
    """Return where a down grab's contact point lies around the axis (deg)."""
    cone = cone or seed_cone()

    return float(np.degrees(
        np.arctan2(lateral_m, cone.radius_at(depth_m) - height_m)
    ))


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


def solve_one(arm, depth_m, floor_angle_deg, grid=None, seeds=None):
    """
    Return the best grab at one spot, searching heights then tilts.

    Used for detected items (plan_floor_grab). Must run with the
    gripper padded by the grid's clearance (gripper_padding).
    Checks: grab and approach states, and the straight descent
    between them. IK only answers inside the task joint ranges
    (arm/joint_ranges), so a flipped posture - J1 a full turn round,
    or the wrist flipped - is never offered. seeds default to INTER
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
    'no IK for the grab within the task joint ranges (out of reach, '
    'the gripper collides there, or it needs a flipped posture)',
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

        approach = arm.compute_ik(
            _flange_pose(contact, tool_z, claw_x, approach_m), grab,
        )

        if approach is None:
            stage = max(stage, 1)
            continue

        if arm.first_invalid_state([approach, grab]) is not None:
            stage = max(stage, 2)
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


def deepest_grab(arm, depth, angle, top_height, grid, seeds, dig=None):
    """
    Return the deepest collision-free grab into a pile, or None.

    top_height: the pile's sensed top, in from the bucket wall at
    (depth, angle) - the physical room there is to dig. For each tilt
    of grid['tilts_deg'], heights are tried from the deepest allowed,
    max(dig['wall_clearance_m'], top - dig['max_sink_m']), up to the
    sensed top, 2 x resolution_m apart; the first whose grab pose,
    approach pose and straight descent are all clear (_try_pose: IK
    inside the task ranges, MoveIt's collision checks with the
    caller's gripper padding) is then refined one resolution_m step
    down. The deepest over all tilts wins, the smaller tilt on a tie;
    a later tilt is only tried below the best so far.

    Scanned from the bottom up, not bisected: clear heights need not
    be one interval. Near the mouth a grab high in a tall pile puts
    the arm against the top of the bucket, while lower ones are clear
    (fake controller, 2026-10-22).

    The approach pose starts ABOVE_PILE_M above the sensed top, so the
    dig is the straight descent and the move to the approach never
    drags the claw through the pile; where that does not fit,
    grid['approach_m'] back along the tool (the grab then starts
    inside the pile, as before 2026-10-22).

    Returns solve_one()'s dict plus 'sink_m' (how far below the sensed
    top the claw closes) and 'approach_m'.
    """
    dig = dig or config.DETECTED_GRAB_DIG
    inter = np.array(config.get_named_pose('inter'))
    resolution = dig['resolution_m']
    coarse = 2.0 * resolution

    top = top_height
    lowest = min(top, max(dig['wall_clearance_m'], top - dig['max_sink_m']))

    def probe(height, tilt):
        contact, tool_z, claw_x = grab_geometry(depth, angle, height, tilt)
        # Along the tilted tool axis, the pile's top is further back.
        above = (top - height + ABOVE_PILE_M) / max(
            np.cos(np.radians(tilt)), 0.2
        )

        for approach in dict.fromkeys(
            [max(grid['approach_m'], above), grid['approach_m']]
        ):
            found, _ = _try_pose(
                arm, contact, tool_z, claw_x, approach, seeds, inter
            )

            if found is not None:
                return dict(
                    found, height_m=float(height), tilt_deg=float(tilt),
                    approach_m=float(approach),
                )

        return None

    best = None

    for tilt in grid['tilts_deg']:
        # Only worth trying below the best so far.
        ceiling = top if best is None else best['height_m'] - resolution
        height = lowest
        blocked = None
        clear = None

        while height <= ceiling + 1e-9:
            clear = probe(height, tilt)

            if clear is not None:
                break

            blocked = height
            height = min(height + coarse, ceiling) if height < ceiling else (
                ceiling + 1.0
            )

        if clear is None:
            continue

        # One step finer between the last blocked height and this one.
        if blocked is not None and clear['height_m'] - blocked > resolution:
            finer = probe(clear['height_m'] - resolution, tilt)

            if finer is not None:
                clear = finer

        best = clear

        if best['height_m'] <= lowest + 1e-9:
            break

    if best is None:
        return None

    return dict(
        best,
        depth_m=float(depth),
        floor_angle_deg=float(angle),
        sink_m=float(top - best['height_m']),
    )


def plan_floor_grab(arm, point, grabs, grid=None):
    """
    Plan a grid-style grab at a detected item's sensed top, or None.

    Returns the deepest_grab() dict plus 'via': the baked grab_NN (or
    INTER, as a last resort) it is reached from by a straight joint
    move. The grab digs as deep into the pile as is collision-free
    (deepest_grab, config.DETECTED_GRAB_DIG); on a wall, "height" is
    the distance in from the wall. None if the item is beyond
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
        plan = deepest_grab(arm, depth, angle, top_height, grid, seeds)
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


def run_floor_grab(arm, gripper, plan, go_to, drop=True, log=print,
                   prepare=None):
    """
    Execute a plan_floor_grab() plan; True if the item was taken (and dropped).

    via (INTER, or a baked grab_NN when INTER's line collides) ->
    approach -> down, one move, the claw closing as it arrives -> up
    -> back to via -> DROP, one move, carried with the claw turned on
    the way and opening as it arrives (grasp/carry.py) -> INTER. The
    move up and out is checked while the arm goes down, and the way
    back to INTER while it carries. Always lifts back out.

    prepare: transfers.prepare_go_to, or a stand-in for tests.
    """
    if prepare is None:
        from ..arm.transfers import prepare_go_to as prepare

    via = plan['via']
    via_pose = config.get_named_pose(via)

    log(
        f'Floor grab via {via}: {plan["depth_m"] * 100:.0f} cm deep, '
        f'{plan["floor_angle_deg"]:+.0f} deg, {plan["height_m"] * 100:.1f} cm '
        f'from the wall, tilt {plan["tilt_deg"]:.0f} deg, '
        f'{plan.get("sink_m", 0.0) * 100:.1f} cm into the pile'
    )

    if not gripper.open_blocking():
        log('Gripper did not confirm it opened; aborting before the approach.')
        return False

    down = prepare(arm, via, after=[plan['approach'], plan['grab']])
    close_on_arrival, closing = carry.on_arrival(gripper.start_close)
    planned = {}

    def check_lift():
        # While the arm goes down: the way back out the way it came (so
        # the route on starts at a baked pose), from the grab.
        out = [plan['approach'], via_pose]
        planned['lift'] = (
            prepare(arm, 'drop', before=out, turn_rad=carry.turn_rad(),
                    start=plan['grab'])
            if drop else prepare(arm, 'inter', before=out, start=plan['grab'])
        )

    if down is None or not down(
        before_end=close_on_arrival, while_moving=check_lift
    ):
        log('Could not reach the grab; returning to INTER.')
        arm.move_joints_linear(plan['approach'])
        go_to(arm, 'inter')
        return False

    lift = planned.get('lift')

    closed = carry.wait(closing, gripper.start_close)

    if not closed or lift is None:
        if not (
            arm.move_joints_linear(plan['approach'])
            and arm.move_joints_linear(via_pose)
        ):
            log('Could not lift back out; stopping here.')
            return False

        log('Gripper did not confirm it closed; returning to INTER.'
            if not closed else 'No collision-free way out and to DROP.')
        go_to(arm, 'inter')
        return False

    if not drop:
        return lift()

    open_on_arrival, opening = carry.on_arrival(gripper.start_open)
    back = {}

    def check_return():
        # While the arm carries: the way back to INTER once the claw let
        # go (it stood ~0.6 s at DROP for it on the rig, 2026-10-28).
        back['inter'] = prepare(
            arm, 'inter', gripper_empty=True,
            start=config.get_named_pose('drop'),
        )

    if not lift(before_end=open_on_arrival, while_moving=check_return):
        log('Failed to reach DROP.')
        return False

    released = carry.wait(opening, gripper.start_open)

    if not released:
        log('Gripper did not confirm it opened at DROP; the item may still '
            'be held.')
        return go_to(arm, 'inter') and released

    if back.get('inter') is not None:
        return back['inter']()

    return go_to(arm, 'inter', gripper_empty=True)


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
                'name', 'layer', 'depth_m', 'floor_angle_deg', 'lateral_m',
                'height_m', 'tilt_deg', 'approach_m', 'contact', 'approach',
                'grab',
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


def grid_mismatch(path=None, targets=None, generated=None):
    """
    Return why retrieve.yaml no longer matches what it was baked for, or None.

    The grab targets (scan_plans/grab_targets.yaml), the gripper
    clearance and the gripper offset, against now; and, when the
    targets were generated from config.PILE (pile_layers), whether
    PILE has changed since (`generated`: what they were generated
    from, default: read from the targets file).
    """
    from . import pile_layers

    path = path or plan_path()
    generated = grab_targets.generated_for() if targets is None else generated

    if not os.path.isfile(path):
        return None

    with open(path) as handle:
        document = yaml.safe_load(handle) or {}

    targets = grab_targets.load() if targets is None else targets

    def same(a, b):
        # NaN: no lateral_m (a radial grab) matches no lateral_m.
        return np.allclose(
            np.asarray(a, float), np.asarray(b, float), equal_nan=True
        )

    def values(items):
        return [
            [t[f] for f in grab_targets.FIELDS]
            + [t.get('lateral_m', float('nan')), t.get('layer', float('nan'))]
            for t in items
        ]

    baked = [grab_targets.clean(t) for t in document.get('targets', [])]
    changed = []

    if len(baked) != len(targets) or (
        baked and not same(values(baked), values(targets))
    ):
        changed.append('grab_targets.yaml')

    if pile_layers.pile_changed(generated):
        changed.append('config.PILE')

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


def run(arm, gripper, grabs, go_to, prepare, time_scale=1.0, log=print):
    """
    Visit each grab: route in and down, close, up and to DROP, open.

    go_to / prepare: transfers.go_to and transfers.prepare_go_to (or
    stand-ins, for tests). "Route in and down" and "up and to DROP"
    are one trajectory each (prepare_go_to's after / before), and the
    arm does not stand still for their checks: each is checked while
    the arm is still on the move before it - a grab's way up and to
    DROP while it goes down to the grab, the next grab's way in while
    it carries to DROP (prepare_go_to's while_moving). Checked after
    the claw closed instead, the checks kept the arm standing 2.7-6.1
    s at every grab on the rig (2026-10-28; 0.6-1.1 s on the fake
    controller).

    The claw is told to close or open as the arm arrives (grasp/
    carry.py, on_arrival), and the laundry is carried with the claw
    turned on the way (carry.turn_rad: J7 turns as the lift starts
    and is back at DROP, without stopping).

    Returns True if every motion succeeded. Stops at the first failure
    after getting the arm back to INTER where it can.
    """
    log('Opening gripper...')

    opening = gripper.start_open()

    # Checked while the gripper opens; used only once it confirms it
    # did, so DROP is left by the empty-gripper route.
    descend = prepare(
        arm, grabs[0]['name'], after=[grabs[0]['grab']],
        time_scale=time_scale, gripper_empty=True,
    ) if grabs else None

    for index, grab in enumerate(grabs):
        name = grab['name']

        where = (
            f'{grab["lateral_m"] * 100:+.0f} cm sideways'
            if 'lateral_m' in grab else f'{grab["floor_angle_deg"]:+.0f} deg'
        )
        log(
            f'{name}: {grab["depth_m"] * 100:.0f} cm deep, {where}, '
            f'{grab["height_m"] * 100:.0f} cm above the floor'
        )

        if not opening.wait():
            if index == 0:
                log('Gripper did not confirm it opened; aborting before any '
                    'grab.')
            else:
                log('Gripper did not confirm it opened at DROP; returning '
                    'to INTER and aborting.')
                go_to(arm, 'inter', time_scale=time_scale)

            return False

        if descend is None:
            log(f'No collision-free way to {name} and down; aborting.')
            go_to(arm, 'inter', time_scale=time_scale)
            return False

        close_on_arrival, closing = carry.on_arrival(gripper.start_close)
        planned = {}

        def check_lift(grab=grab):
            # While the arm goes down: this grab's way up and to DROP,
            # carried turned, from the grab (where the arm is not yet).
            planned['lift'] = prepare(
                arm, 'drop', before=[grab['approach']], time_scale=time_scale,
                turn_rad=carry.turn_rad(), start=grab['grab'],
            )

        if not descend(before_end=close_on_arrival, while_moving=check_lift):
            log(f'The move to {name} and down failed; lifting out and '
                'aborting.')

            if arm.move_joints_linear(grab['approach'], time_scale=time_scale):
                go_to(arm, 'inter', time_scale=time_scale)

            return False

        lift = planned.get('lift')
        following = grabs[index + 1] if index + 1 < len(grabs) else None

        closed = carry.wait(closing, gripper.start_close)

        if not closed or lift is None:
            # Always lift back out, even with nothing held.
            if not arm.move_joints_linear(
                grab['approach'], time_scale=time_scale
            ):
                log(f'Could not lift back out at {name}; stopping here.')
                return False

            if not closed:
                log('Gripper did not confirm it closed; returning to INTER '
                    'and aborting.')
                go_to(arm, 'inter', time_scale=time_scale)
            else:
                log('No collision-free route to DROP; aborting.')

            return False

        open_on_arrival, released = carry.on_arrival(gripper.start_open)
        nxt = {}

        def check_descent(following=following):
            # While the arm carries to DROP: the next grab's way in.
            if following is not None:
                nxt['descend'] = prepare(
                    arm, following['name'], after=[following['grab']],
                    time_scale=time_scale, gripper_empty=True,
                    start=config.get_named_pose('drop'),
                )

        if not lift(before_end=open_on_arrival, while_moving=check_descent):
            log('Failed to lift out and reach DROP; aborting.')
            return False

        descend = nxt.get('descend')

        opening = released[0] if released else gripper.start_open()

    if not opening.wait():
        log('Gripper did not confirm it opened at DROP; returning to INTER '
            'and aborting.')
        go_to(arm, 'inter', time_scale=time_scale)
        return False

    log('Returning to INTER...')

    return go_to(arm, 'inter', time_scale=time_scale, gripper_empty=True)
