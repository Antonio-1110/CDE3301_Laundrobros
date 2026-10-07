#!/usr/bin/env python3

"""
INTER and BOTTOM, derived from where the bucket is.

WHY
---
INTER (the scan's start, on the bucket axis at the mouth) and BOTTOM
(the tilted pose at the deepest stroke) were jogged by eye and saved
as joint angles, so they stayed where they were when the bucket
moved, and they are only as well aligned as an eye can judge. In the
modelled bucket, the recorded INTER's flange sits ~3 cm off the axis
with the tool 7.8 deg off it.

WHAT
----
config.BUCKET_POSES says where each pose is relative to the bucket
of config.OBSTACLES (see targets()). solve() finds their joint angles
by IK against a running MoveIt, seeded from the recorded angles, so
the arm keeps the elbow posture they were jogged in, checks them
against the obstacles, and reports how far each moved from the
recorded one. save() writes scan_plans/bucket_poses.yaml; from then
on config.named_poses() returns these in place of the recorded ones.

Nothing here moves the arm. Everything baked from INTER (transfers,
the grab grid, the end scan) must be re-baked after these change -
`laundry plan bake` does it all, in order.
"""

import datetime
import os

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .. import config

PLAN_VERSION = 1

POSE_NAMES = ('inter', 'bottom')

# A derived pose further than this from the recorded one (flange
# position, metres / tool axis, degrees) is called out: the bucket
# model and the arm then disagree by more than jogging error.
LARGE_SHIFT_M = 0.03
LARGE_TILT_DEG = 5.0

# An IK solution that turns any joint further than this from the
# recorded pose is a different posture (the solver can flip the
# wrist or elbow to get clear), not the same one moved a little: it
# would change every route. It is rejected, and the pose not derived.
MAX_JOINT_CHANGE_DEG = 30.0


def targets(cone=None, spec=None):
    """
    Return {name: (flange position, tool +Z, boresight +X)} in link_base.

    cone defaults to the configured bucket (perception.bucket_model.
    seed_cone), spec to config.BUCKET_POSES. Pure geometry: no ROS.
    """
    from ..perception.bucket_model import MESH_DEPTH_M, seed_cone
    from ..scan.endcap import precession_axes, reference_axes

    cone = cone or seed_cone()
    spec = spec or config.BUCKET_POSES

    # The axis points from the closed end to the mouth; the tool
    # points the other way, into the bucket.
    z0, up, side = reference_axes(-np.asarray(cone.axis_dir, dtype=float))

    inter = (
        np.asarray(cone.axis_point, dtype=float)
        + (MESH_DEPTH_M + spec['inter']['standoff_m']) * cone.axis_dir
    )
    bottom = inter + spec['bottom']['depth_m'] * z0

    result = {}

    for name, position, tilt in (
        ('inter', inter, 0.0),
        ('bottom', bottom, spec['bottom']['tilt_deg']),
    ):
        tool_z, boresight = precession_axes(tilt, 0.0, z0, up, side)
        result[name] = (position, tool_z, boresight)

    return result


def target_pose(position, tool_z, boresight):
    """Return the flange geometry_msgs Pose for a target."""
    from ..scan.endcap import precession_pose

    # The end scan's alpha = 0 pose about this target's own axes (its
    # "up" is opposite the boresight): exactly this frame.
    up = -np.asarray(boresight, dtype=float)

    return precession_pose(
        position, 0.0, 0.0, np.asarray(tool_z, dtype=float), up,
        np.cross(tool_z, up),
    )


def _flange_frame(arm, joints):
    """Return (position, tool +Z, boresight +X) of the flange by FK, or None."""
    fk = arm.compute_fk(joints)

    if fk is None:
        return None

    position, quaternion = fk
    matrix = Rotation.from_quat(quaternion).as_matrix()

    return np.asarray(position), matrix[:, 2], matrix[:, 0]


def shift(frame_a, frame_b):
    """Return (flange distance m, tool-axis angle deg) between two frames."""
    distance = float(np.linalg.norm(np.asarray(frame_a[0]) - frame_b[0]))
    cosine = float(np.clip(np.dot(frame_a[1], frame_b[1]), -1.0, 1.0))

    return distance, float(np.degrees(np.arccos(cosine)))


def solve(arm, log=print):
    """
    Solve INTER and BOTTOM in the configured bucket.

    Returns ({name: joints}, {name: reason it failed}). Each pose is
    checked collision-free under the live arm padding (config.
    OBSTACLE_PADDING_M), like every named pose: the moves that end
    there (e.g. the planner's, in the BOTTOM detour) keep it. Of the
    IK solutions from the seeds (the recorded pose, then the
    previously derived one), the one closest to the recorded pose in
    weighted joint travel wins.
    """
    from .transfers import weighted_travel_deg

    recorded = config.recorded_poses()
    previous = config.derived_bucket_poses()

    solved = {}
    failures = {}

    arm.set_arm_padding(config.OBSTACLE_PADDING_M)

    for name, (position, tool_z, boresight) in targets().items():
        pose = target_pose(position, tool_z, boresight)
        seeds = [recorded[name]] + (
            [previous[name]] if name in previous else []
        )

        candidates = []

        for seed in seeds:
            joints = arm.compute_ik(pose, seed)

            if (
                joints is not None
                and np.degrees(
                    np.abs(np.asarray(joints) - recorded[name])
                ).max() <= MAX_JOINT_CHANGE_DEG
                and arm.state_is_valid(joints)
            ):
                candidates.append(joints)

        if not candidates:
            failures[name] = (
                'no collision-free IK solution (arm links padded '
                f'{config.OBSTACLE_PADDING_M * 100:g} cm) within '
                f'{MAX_JOINT_CHANGE_DEG:g} deg per joint of the recorded pose'
            )
            log(f'  {name.upper()}: NOT REACHABLE - {failures[name]}')
            continue

        joints = min(
            candidates,
            key=lambda q: weighted_travel_deg([recorded[name], q]),
        )
        solved[name] = [float(v) for v in joints]

        joint_change = np.degrees(
            np.abs(np.asarray(joints) - recorded[name])
        )
        old = _flange_frame(arm, recorded[name])

        if old is None:
            moved = 'recorded pose: FK failed'
        else:
            distance, angle = shift(old, (position, tool_z))
            moved = (
                f'flange {distance * 100:.1f} cm and tool axis '
                f'{angle:.1f} deg from the recorded pose'
            )

            if distance > LARGE_SHIFT_M or angle > LARGE_TILT_DEG:
                moved += (
                    ' - LARGE: check config.OBSTACLES against the real '
                    'bucket before using it'
                )

        log(
            f'  {name.upper()}: {moved}; joints changed by up to '
            f'{joint_change.max():.1f} deg '
            f'(J{int(joint_change.argmax()) + 1})'
        )

    return solved, failures


def plan_path():
    """Return <repo>/scan_plans/bucket_poses.yaml."""
    return config.bucket_poses_path()


def save(poses, path=None, baked_on=''):
    """Write the derived poses, stamped with the scene and BUCKET_POSES."""
    from . import scene

    path = path or plan_path()

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    geometry = targets()

    document = {
        'version': PLAN_VERSION,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'baked_on': baked_on,
        'joint_names': list(config.JOINT_NAMES),
        'scene': scene.signature(),
        'obstacles': scene.describe(),
        'spec': {
            name: dict(values) for name, values in config.BUCKET_POSES.items()
        },
        'poses': {
            name: {
                'joints': [round(float(v), 6) for v in joints],
                'flange_xyz': [round(float(v), 4) for v in geometry[name][0]],
                'tool_z': [round(float(v), 4) for v in geometry[name][1]],
            }
            for name, joints in poses.items()
        },
    }

    with open(path, 'w') as handle:
        handle.write(
            '# INTER and BOTTOM derived from the bucket '
            '(laundry_control/arm/bucket_poses.py).\n'
            '# Generated by `laundry plan bake poses` - re-bake after '
            'changing\n# config.OBSTACLES or config.BUCKET_POSES, then '
            're-bake everything else.\n'
        )
        yaml.safe_dump(document, handle, sort_keys=False, width=100)


def _read(path=None):
    path = path or plan_path()

    if not os.path.isfile(path):
        return None

    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def baked_scene(path=None):
    """Return the scene stamp the poses were derived in ('' if none)."""
    document = _read(path)

    return '' if document is None else str(document.get('scene', ''))


def spec_mismatch(path=None):
    """Return why bucket_poses.yaml no longer matches BUCKET_POSES, or None."""
    document = _read(path)

    if document is None:
        return None

    baked = document.get('spec', {})
    changed = [
        f'{name}.{key}'
        for name, values in config.BUCKET_POSES.items()
        for key, value in values.items()
        if not np.isclose(baked.get(name, {}).get(key, np.nan), value)
    ]

    if not changed:
        return None

    return (
        'scan_plans/bucket_poses.yaml was derived with other '
        'config.BUCKET_POSES (' + ', '.join(changed) + ' changed); '
        're-bake: laundry plan bake'
    )


def status():
    """Return one line per pose: derived from the bucket, or recorded."""
    derived = config.derived_bucket_poses()

    return {
        name: (
            'derived from the bucket (scan_plans/bucket_poses.yaml)'
            if name in derived else
            'RECORDED by hand (not derived yet: laundry plan bake poses)'
        )
        for name in POSE_NAMES
    }
