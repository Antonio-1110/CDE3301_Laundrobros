#!/usr/bin/env python3

"""
Padded obstacles: the bucket and table as MoveIt WORLD objects.

WHY
---
The bucket and table used to be links in our xarm_ros2 fork's URDF,
which made them part of the ROBOT: MoveIt treated touching them as
self-collision, and it checks self-collision UNPADDED
(PlanningScene::checkCollision pads only robot-vs-world), so link
padding never kept the arm away from them. Moving the bucket also
meant editing the manufacturer's description package.

WHAT
----
apply() gives move_group:

  - the meshes (meshes/bucket.obj, meshes/table.obj) as world
    collision objects, at the poses in config.OBSTACLES - the one
    place they are defined, so moving the bucket is a config edit;
  - link padding on every moving arm link, so everything the arm does
    - planner routes, Cartesian strokes, our own straight-line and baked
    checks - keeps at least that clearance from them.

It runs on every `laundry` connect (skipped when move_group already
has exactly this scene), and `laundry scene apply` runs it alone -
laundry_bringup.launch.py does that at start-up, so RViz and hand
planning see the obstacles too.

If the URDF still has the old bucket/table links (an older
xarm_ros2 build), those are made collision-free so each obstacle is
checked once, padded.

The gripper gets its own (smaller) padding: it works inside the
bucket, close to the floor, on purpose. See config.OBSTACLE_PADDING_M
and GRIPPER_PADDING_M.
"""

import hashlib
import json
import os

from geometry_msgs.msg import Point, Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    CollisionObject,
    LinkPadding,
    PlanningScene,
    PlanningSceneComponents,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from scipy.spatial.transform import Rotation
from shape_msgs.msg import Mesh, MeshTriangle

from .. import config

# Links the xarm_ros2 fork used to carry the obstacles as. If a build
# still has them, they are disabled in favour of the world objects.
LEGACY_URDF_LINKS = ('laundry_bucket_link', 'table_link')

# World-object ids earlier versions of this module used; removed on
# apply so a long-running move_group never holds two copies.
LEGACY_OBJECT_IDS = ('bucket_obstacle', 'table_obstacle')

# A world object whose pose is within this of the configured one is
# left as is (metres, and quaternion components).
POSE_TOLERANCE = 1e-6

# Arm links that get padded. link_base is left out: it is bolted to
# the table, so any padding would put it in permanent contact.
PADDED_LINKS = (
    'link1',
    'link2',
    'link3',
    'link4',
    'link5',
    'link6',
    'link7',
    'gripper_link',
)


def mesh_dir():
    """Return the directory holding bucket.obj and table.obj."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = os.path.join(
            get_package_share_directory('laundry_control'), 'meshes'
        )

        if os.path.isdir(share):
            return share

    except Exception:
        pass

    return os.path.join(config.repo_root(), 'meshes')


def load_obj_mesh(path):
    """
    Read a Wavefront .obj file as a shape_msgs/Mesh (metres, as-is).

    Only vertices and faces matter; polygons are fan-triangulated and
    texture/normal indices ('v/vt/vn') are ignored.
    """
    mesh = Mesh()

    with open(path) as handle:
        for line in handle:
            parts = line.split()

            if not parts:
                continue

            if parts[0] == 'v':
                mesh.vertices.append(
                    Point(
                        x=float(parts[1]),
                        y=float(parts[2]),
                        z=float(parts[3]),
                    )
                )

            elif parts[0] == 'f':
                indices = [int(token.split('/')[0]) for token in parts[1:]]
                # OBJ indices are 1-based; negative ones count from the end.
                indices = [
                    i - 1 if i > 0 else len(mesh.vertices) + i
                    for i in indices
                ]

                for k in range(1, len(indices) - 1):
                    mesh.triangles.append(
                        MeshTriangle(
                            vertex_indices=[
                                indices[0], indices[k], indices[k + 1]
                            ]
                        )
                    )

    if not mesh.triangles:
        raise ValueError(f'{path!r} contains no faces.')

    return mesh


def obstacle_pose(spec):
    """Return a config.OBSTACLES entry's xyz/rpy as a Pose in link_base."""
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (
        float(v) for v in spec['xyz']
    )

    # URDF rpy: fixed axes X, Y, Z - scipy's lower-case (extrinsic) 'xyz'.
    x, y, z, w = Rotation.from_euler('xyz', spec['rpy']).as_quat()
    pose.orientation.x, pose.orientation.y = float(x), float(y)
    pose.orientation.z, pose.orientation.w = float(z), float(w)

    return pose


def _collision_objects(obstacles):
    objects = []

    for object_id, spec in obstacles.items():
        obj = CollisionObject()
        obj.id = object_id
        obj.header.frame_id = config.BASE_FRAME
        obj.pose = obstacle_pose(spec)
        obj.meshes.append(
            load_obj_mesh(os.path.join(mesh_dir(), spec['mesh']))
        )
        obj.mesh_poses.append(Pose())
        obj.mesh_poses[0].orientation.w = 1.0
        obj.operation = CollisionObject.ADD
        objects.append(obj)

    return objects


def _removals(present):
    return [
        CollisionObject(id=object_id, operation=CollisionObject.REMOVE)
        for object_id in LEGACY_OBJECT_IDS
        if object_id in present
    ]


def signature(obstacles=None):
    """
    Return a short fingerprint of the obstacle geometry.

    Covers every obstacle's pose, mesh file contents and allowed links.
    Baked plans store it, so replaying one against a scene that has
    since moved can say so (see transfers.go_to, endcap.run_plan).
    """
    obstacles = config.OBSTACLES if obstacles is None else obstacles

    digest = hashlib.sha1()

    for object_id in sorted(obstacles):
        spec = obstacles[object_id]
        digest.update(json.dumps(
            {
                'id': object_id,
                'xyz': [round(float(v), 6) for v in spec['xyz']],
                'rpy': [round(float(v), 6) for v in spec['rpy']],
                'allowed_links': sorted(spec.get('allowed_links', [])),
            },
            sort_keys=True,
        ).encode())

        try:
            with open(os.path.join(mesh_dir(), spec['mesh']), 'rb') as handle:
                digest.update(handle.read())
        except OSError:
            digest.update(spec['mesh'].encode())

    return digest.hexdigest()[:12]


def stale_plan_message(stamp, what, rebake):
    """
    Return a warning if a baked plan's scene stamp is not the current scene.

    None when they match. Replays still re-check every state against
    the live planning scene, so a stale plan that now collides is
    refused anyway; this says why a plan that still fits may no longer
    be the route a fresh bake would pick.
    """
    current = signature()

    if stamp == current:
        return None

    if not stamp:
        return (
            f'{what} predates scene stamps, so it may have been baked '
            f'against other obstacle poses than config.OBSTACLES '
            f'(scene {current}). Re-bake to be sure: {rebake}'
        )

    return (
        f'{what} was baked against scene {stamp}, but config.OBSTACLES is '
        f'now scene {current} (the bucket or table moved). It is still '
        f'collision-checked before it runs; re-bake: {rebake}'
    )


def describe(obstacles=None):
    """Return the obstacle poses as plain data, for a baked plan's header."""
    obstacles = config.OBSTACLES if obstacles is None else obstacles

    return {
        object_id: {
            'mesh': spec['mesh'],
            'xyz': [round(float(v), 6) for v in spec['xyz']],
            'rpy': [round(float(v), 6) for v in spec['rpy']],
        }
        for object_id, spec in obstacles.items()
    }


def _allow(acm, name_a, name_b):
    """Set a pair allowed in an AllowedCollisionMatrix message, adding names."""
    for name in (name_a, name_b):
        if name not in acm.entry_names:
            acm.entry_names.append(name)

            for entry in acm.entry_values:
                entry.enabled.append(False)

            acm.entry_values.append(
                AllowedCollisionEntry(enabled=[False] * len(acm.entry_names))
            )

    a = acm.entry_names.index(name_a)
    b = acm.entry_names.index(name_b)

    acm.entry_values[a].enabled[b] = True
    acm.entry_values[b].enabled[a] = True


def _configure_acm(acm, obstacles):
    """
    Return the ACM with each obstacle's allowed_links allowed to touch it.

    Legacy URDF copies of the obstacles, if this build still has them,
    are allowed to touch anything: the padded world objects replace
    them.
    """
    for object_id, spec in obstacles.items():
        for link in spec.get('allowed_links', []):
            _allow(acm, link, object_id)

    legacy = [name for name in LEGACY_URDF_LINKS if name in acm.entry_names]

    everything = list(acm.entry_names) + list(obstacles)

    for urdf_link in legacy:
        for other in everything:
            if other != urdf_link:
                _allow(acm, urdf_link, other)

        if urdf_link not in acm.default_entry_names:
            acm.default_entry_names.append(urdf_link)
            acm.default_entry_values.append(True)

    return acm


class SceneError(RuntimeError):
    """move_group refused or never answered a planning-scene change."""


def _get_scene(arm, components):
    client = arm._client(
        '_get_scene_client', GetPlanningScene, '/get_planning_scene'
    )

    request = GetPlanningScene.Request()
    request.components.components = components

    response = arm._call(client, request)

    if response is None:
        raise SceneError('No answer from /get_planning_scene.')

    return response.scene


def _apply(arm, scene):
    client = arm._client(
        '_apply_scene_client', ApplyPlanningScene, '/apply_planning_scene'
    )

    request = ApplyPlanningScene.Request()
    request.scene = scene
    request.scene.is_diff = True

    response = arm._call(client, request)

    if response is None or not response.success:
        raise SceneError('move_group did not apply the planning-scene change.')

    _wait_until_answering(arm)


def _wait_until_answering(arm, timeout_sec=60.0):
    """
    Block until move_group answers collision checks again.

    Changing a link's padding makes MoveIt rebuild that link's padded
    mesh; for the gripper's large STL that takes seconds, during which
    validity checks go unanswered.
    """
    import time

    from moveit_msgs.srv import GetStateValidity

    client = arm._client(
        '_validity_client', GetStateValidity, '/check_state_validity'
    )

    request = GetStateValidity.Request()
    request.group_name = arm.group_name

    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        if arm._spin_until_done(client.call_async(request), 5.0):
            return

    raise SceneError(
        f'move_group stopped answering for {timeout_sec:g}s after a '
        'planning-scene change.'
    )


def _paddings(padding_m, overrides=None):
    overrides = overrides or {}

    return {
        link: float(overrides.get(link, padding_m)) for link in PADDED_LINKS
    }


def _padding_messages(paddings):
    return [
        LinkPadding(link_name=link, padding=value)
        for link, value in paddings.items()
    ]


def _current_paddings(scene):
    return {entry.link_name: entry.padding for entry in scene.link_padding}


def _same_pose(a, b):
    values_a = (
        a.position.x, a.position.y, a.position.z,
        a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w,
    )
    values_b = (
        b.position.x, b.position.y, b.position.z,
        b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w,
    )

    same = all(abs(x - y) <= POSE_TOLERANCE for x, y in zip(values_a, values_b))

    # q and -q are the same rotation.
    flipped = all(
        abs(x - y) <= POSE_TOLERANCE for x, y in zip(values_a[:3], values_b[:3])
    ) and all(
        abs(x + y) <= POSE_TOLERANCE for x, y in zip(values_a[3:], values_b[3:])
    )

    return same or flipped


def _is_configured(scene, paddings, obstacles):
    """Return True if the scene has exactly these obstacles and paddings."""
    present = {obj.id: obj for obj in scene.world.collision_objects}

    if any(object_id in present for object_id in LEGACY_OBJECT_IDS):
        return False

    for object_id, spec in obstacles.items():
        obj = present.get(object_id)

        if obj is None or not _same_pose(obj.pose, obstacle_pose(spec)):
            return False

    current = _current_paddings(scene)

    return all(
        abs(current.get(link, -1.0) - value) < 1e-6
        for link, value in paddings.items()
    )


def apply(arm, padding_m, gripper_padding_m=0.0, log=None, obstacles=None):
    """
    Install the padded obstacles in move_group, if not already there.

    Obstacles default to config.OBSTACLES. Arm links get padding_m,
    gripper_link gripper_padding_m. A scene that already has these
    obstacles at these poses, with exactly these paddings, is left
    alone (changing the gripper's padding costs a seconds-long mesh
    rebuild). Returns True if anything changed. Raises SceneError if
    move_group refuses.
    """
    obstacles = config.OBSTACLES if obstacles is None else obstacles

    paddings = _paddings(padding_m, {'gripper_link': gripper_padding_m})

    current = _get_scene(
        arm,
        PlanningSceneComponents.WORLD_OBJECT_NAMES
        | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
        | PlanningSceneComponents.LINK_PADDING_AND_SCALING,
    )

    if _is_configured(current, paddings, obstacles):
        return False

    if log is not None:
        log(
            f'Adding the obstacles to MoveIt ({", ".join(obstacles)}; '
            f'scene {signature(obstacles)}): arm links padded '
            f'{padding_m * 100:g} cm, gripper {gripper_padding_m * 100:g} cm '
            '(takes a few seconds)...'
        )

    acm = _get_scene(
        arm, PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    ).allowed_collision_matrix

    present = {obj.id for obj in current.world.collision_objects}

    scene = PlanningScene()
    scene.world.collision_objects = (
        _removals(present) + _collision_objects(obstacles)
    )
    scene.allowed_collision_matrix = _configure_acm(acm, obstacles)
    scene.link_padding = _padding_messages(paddings)

    _apply(arm, scene)

    return True


def set_padding(arm, padding_m, overrides=None):
    """Change the arm links' padding (metres; per-link overrides)."""
    paddings = _paddings(padding_m, overrides)

    current = _get_scene(
        arm, PlanningSceneComponents.LINK_PADDING_AND_SCALING
    )

    if all(
        abs(_current_paddings(current).get(link, -1.0) - value) < 1e-6
        for link, value in paddings.items()
    ):
        return

    scene = PlanningScene()
    scene.link_padding = _padding_messages(paddings)

    _apply(arm, scene)
