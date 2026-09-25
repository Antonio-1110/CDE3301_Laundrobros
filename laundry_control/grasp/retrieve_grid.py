#!/usr/bin/env python3

"""
Generated grab poses for the sensorless first pass (`laundry preplanned`).

WHY
---
The hand-recorded RETRIEVE_0..3 all sit on one line down the middle
of the floor, 11-39 cm from the closed end, two of them on nearly the
same spot, and RETRIEVE_3 is inside the modelled wall. Laundry lies
across the whole bottom of the bucket, so a sweep that is meant to
save a scan should cover it evenly.

WHAT
----
solve() places a grab at every (depth, floor angle) of
config.RETRIEVE_GRID, in the configured bucket (config.OBSTACLES, via
perception.bucket_model.seed_cone), and for each finds by IK:

  - the LOWEST contact-point height above the floor, then
  - the most VERTICAL approach (tool axis tilted toward the closed end
    only as far as needed to reach in),

that is collision-free with the gripper padded by clearance_m, whose
approach pose (approach_m back along the tool axis) is too, and
whose straight joint-space descent between the two is clear. IK is
seeded from INTER and the recorded RETRIEVE poses - the elbow
configurations known to work in this bucket - and among seeds the one
closest to INTER in weighted joint travel wins (less cable twist).
The claw keeps the recorded poses' roll: local +X toward the mouth.

Everything is saved to scan_plans/retrieve.yaml, stamped with the
scene; each approach pose becomes the named pose grab_NN (see
config.named_poses), which `laundry plan bake` gives a baked route
from INTER like any other.

run() replays them: baked route to grab_NN, straight down, close,
straight up, DROP, open.
"""

import datetime
import os

from geometry_msgs.msg import Pose
import numpy as np
import yaml

from .. import config
from ..arm.geometry import look_at_quaternion
from ..perception.bucket_model import _axis_basis, seed_cone

PLAN_VERSION = 1


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
    return [
        config.INTER,
        config.RETRIEVE_0,
        config.RETRIEVE_1,
        config.RETRIEVE_2,
        config.RETRIEVE_3,
    ]


def solve_one(arm, depth_m, floor_angle_deg, grid=None):
    """
    Return the best grab at one grid point as a dict, or None.

    Must run with the gripper padded by the grid's clearance (see
    solve). Checks: grab and approach states, and the straight
    descent between them.
    """
    from ..arm.transfers import weighted_travel_deg

    grid = grid or config.RETRIEVE_GRID
    inter = np.array(config.INTER)

    for height in grid['heights_m']:
        for tilt in grid['tilts_deg']:
            contact, tool_z, claw_x = grab_geometry(
                depth_m, floor_angle_deg, height, tilt
            )

            best = None

            for seed in _seeds():
                grab = arm.compute_ik(_flange_pose(contact, tool_z, claw_x), seed)

                if grab is None:
                    continue

                approach = arm.compute_ik(
                    _flange_pose(contact, tool_z, claw_x, grid['approach_m']),
                    grab,
                )

                if approach is None:
                    continue

                if arm.first_invalid_state([approach, grab]) is not None:
                    continue

                cost = weighted_travel_deg([inter, approach])

                if best is None or cost < best['travel_deg']:
                    best = {
                        'depth_m': float(depth_m),
                        'floor_angle_deg': float(floor_angle_deg),
                        'height_m': float(height),
                        'tilt_deg': float(tilt),
                        'contact': [round(float(v), 4) for v in contact],
                        'approach': [round(float(v), 6) for v in approach],
                        'grab': [round(float(v), 6) for v in grab],
                        'travel_deg': round(cost, 1),
                    }

            if best is not None:
                return best

    return None


def solve(arm, grid=None, log=print):
    """
    Solve every grid point, in visiting order.

    Returns (grabs, misses): grabs are dicts with a grab_NN 'name';
    misses are the (depth, angle) points nothing reached.
    """
    from ..arm import scene

    grid = grid or config.RETRIEVE_GRID

    scene.set_padding(
        arm,
        config.OBSTACLE_PADDING_M,
        {'gripper_link': config.GRIPPER_PADDING_M + grid['clearance_m']},
    )

    grabs = []
    misses = []

    try:
        for depth in grid['depths_m']:
            for angle in grid['floor_angles_deg']:
                grab = solve_one(arm, depth, angle, grid)

                where = f'depth {depth * 100:4.0f} cm, angle {angle:+4.0f} deg'

                if grab is None:
                    misses.append((depth, angle))
                    log(f'  {where}: NOT REACHABLE')
                    continue

                grab['name'] = grab_name(len(grabs) + 1)
                grabs.append(grab)

                log(
                    f'  {grab["name"]}  {where}: {grab["height_m"] * 100:.0f} cm '
                    f'above the floor, tilt {grab["tilt_deg"]:.0f} deg'
                )

    finally:
        scene.set_padding(
            arm,
            config.OBSTACLE_PADDING_M,
            {'gripper_link': config.GRIPPER_PADDING_M},
        )

    return grabs, misses


def save(grabs, path=None, baked_on=''):
    """Write the grabs as YAML, stamped with the grid and the scene."""
    from ..arm import scene

    path = path or plan_path()

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
        'grid': dict(config.RETRIEVE_GRID),
        'gripper_offset_z': float(config.GRIPPER_OFFSET_Z),
        'grabs': [
            {key: grab[key] for key in (
                'name', 'depth_m', 'floor_angle_deg', 'height_m', 'tilt_deg',
                'contact', 'approach', 'grab',
            )}
            for grab in grabs
        ],
    }

    with open(path, 'w') as handle:
        handle.write(
            '# Generated grab poses for `laundry preplanned` '
            '(laundry_control/grasp/retrieve_grid.py).\n'
            '# Generated by `laundry plan bake retrieve` - re-bake after '
            'changing\n# config.RETRIEVE_GRID, config.OBSTACLES or the '
            'gripper offset.\n'
        )
        yaml.safe_dump(document, handle, sort_keys=False, width=100)


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

        if not go_to(arm, name, time_scale=time_scale):
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

    return go_to(arm, 'inter', time_scale=time_scale)
