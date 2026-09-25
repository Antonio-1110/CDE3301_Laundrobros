#!/usr/bin/env python3

"""
Padded obstacles: the bucket and table as MoveIt WORLD objects.

WHY
---
The bucket and table are links in our xarm_ros2 fork's URDF
(laundry_bucket_link, table_link), so MoveIt treats touching them as
SELF-collision - and MoveIt always checks self-collision UNPADDED
(PlanningScene::checkCollision pads only robot-vs-world). Link padding
therefore never kept the arm away from them: every check, the planner's
included, allowed the arm to pass within a hair of the bucket rim.

WHAT
----
apply() gives move_group padded copies:

  - the same meshes (meshes/bucket.obj, meshes/table.obj) added as
    world collision objects, anchored to the URDF links' own frames,
    so the URDF stays the one place their poses are defined;
  - the URDF copies made collision-free (allowed with everything in
    the allowed collision matrix), so each obstacle is checked once;
  - link padding on every moving arm link, so everything the arm does
    - planner routes, Cartesian strokes, our own straight-line and baked
    checks - keeps at least that clearance from the bucket and table.

Nothing is changed in the fork. If apply() never ran, move_group still
has the URDF's unpadded obstacles - less margin, but never none.

The gripper gets its own (smaller) padding: it works inside the
bucket, close to the floor, on purpose. See config.OBSTACLE_PADDING_M
and GRIPPER_PADDING_M.
"""

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
from shape_msgs.msg import Mesh, MeshTriangle

# World object id -> (URDF link it copies, mesh file under meshes/).
OBSTACLES = {
    'bucket_obstacle': ('laundry_bucket_link', 'bucket.obj'),
    'table_obstacle': ('table_link', 'table.obj'),
}

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

# Links that never move relative to the table, so their contact with
# it is irrelevant (and link_base rests on it).
TABLE_ALLOWED_LINKS = ('link_base',)


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

    from .. import config

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


def _collision_objects():
    objects = []

    for object_id, (frame, filename) in OBSTACLES.items():
        obj = CollisionObject()
        obj.id = object_id
        obj.header.frame_id = frame
        obj.pose.orientation.w = 1.0
        obj.meshes.append(load_obj_mesh(os.path.join(mesh_dir(), filename)))
        obj.mesh_poses.append(Pose())
        obj.mesh_poses[0].orientation.w = 1.0
        obj.operation = CollisionObject.ADD
        objects.append(obj)

    return objects


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


def _acm_with_urdf_copies_disabled(acm):
    """Return the ACM with the URDF bucket/table allowed to touch anything."""
    urdf_links = [frame for frame, _file in OBSTACLES.values()]

    everything = list(acm.entry_names) + list(OBSTACLES)

    for urdf_link in urdf_links:
        for other in everything:
            if other != urdf_link:
                _allow(acm, urdf_link, other)

        if urdf_link not in acm.default_entry_names:
            acm.default_entry_names.append(urdf_link)
            acm.default_entry_values.append(True)

    for link in TABLE_ALLOWED_LINKS:
        _allow(acm, link, 'table_obstacle')

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


def _is_configured(scene, paddings):
    names = {obj.id for obj in scene.world.collision_objects}

    if not set(OBSTACLES) <= names:
        return False

    current = _current_paddings(scene)

    return all(
        abs(current.get(link, -1.0) - value) < 1e-6
        for link, value in paddings.items()
    )


def apply(arm, padding_m, gripper_padding_m=0.0, log=None):
    """
    Install the padded bucket/table in move_group, if not already there.

    Arm links get padding_m, gripper_link gripper_padding_m. A scene
    that already has both obstacles and exactly these paddings is left
    alone (changing the gripper's padding costs a seconds-long mesh
    rebuild). Raises SceneError if move_group refuses.
    """
    paddings = _paddings(padding_m, {'gripper_link': gripper_padding_m})

    current = _get_scene(
        arm,
        PlanningSceneComponents.WORLD_OBJECT_NAMES
        | PlanningSceneComponents.LINK_PADDING_AND_SCALING,
    )

    if _is_configured(current, paddings):
        return False

    if log is not None:
        log(
            f'Padding the bucket/table in MoveIt: arm links '
            f'{padding_m * 100:g} cm, gripper {gripper_padding_m * 100:g} cm '
            '(one-off; takes a few seconds)...'
        )

    acm = _get_scene(
        arm, PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    ).allowed_collision_matrix

    scene = PlanningScene()
    scene.world.collision_objects = _collision_objects()
    scene.allowed_collision_matrix = _acm_with_urdf_copies_disabled(acm)
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
