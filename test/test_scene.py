"""The padded bucket/table planning-scene helpers (arm/scene.py)."""

import copy
import os

from laundry_control import config
from laundry_control.arm import scene
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    CollisionObject,
    LinkPadding,
    PlanningScene,
)
import numpy as np
from scipy.spatial.transform import Rotation

MESHES = os.path.join(os.path.dirname(__file__), '..', 'meshes')


def test_obj_meshes_load_as_triangles():
    bucket = scene.load_obj_mesh(os.path.join(MESHES, 'bucket.obj'))
    table = scene.load_obj_mesh(os.path.join(MESHES, 'table.obj'))

    assert len(bucket.vertices) == 717
    assert len(bucket.triangles) == 1430
    assert len(table.triangles) == 12

    for mesh in (bucket, table):
        top = len(mesh.vertices)
        assert all(
            0 <= i < top for t in mesh.triangles for i in t.vertex_indices
        )


def test_quads_are_triangulated(tmp_path):
    path = tmp_path / 'quad.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1 2/2 3/3 4/4\n')

    mesh = scene.load_obj_mesh(str(path))

    assert [list(t.vertex_indices) for t in mesh.triangles] == [
        [0, 1, 2], [0, 2, 3],
    ]


def _acm(names):
    return AllowedCollisionMatrix(
        entry_names=list(names),
        entry_values=[
            AllowedCollisionEntry(enabled=[False] * len(names)) for _ in names
        ],
    )


def _allowed(acm, a, b):
    return acm.entry_values[acm.entry_names.index(a)].enabled[
        acm.entry_names.index(b)
    ]


def test_allowed_links_and_legacy_urdf_copies():
    names = ['link_base', 'link3', 'gripper_link', 'laundry_bucket_link',
             'table_link']

    acm = scene._configure_acm(_acm(names), config.OBSTACLES)

    # An older xarm_ros2 build's URDF copies no longer collide...
    assert _allowed(acm, 'laundry_bucket_link', 'gripper_link')
    assert _allowed(acm, 'table_link', 'link3')
    assert _allowed(acm, 'laundry_bucket_link', 'bucket')
    # ...while the padded world objects still do.
    assert not _allowed(acm, 'bucket', 'gripper_link')
    assert not _allowed(acm, 'table', 'link3')
    # Except the base, which sits on the table.
    assert _allowed(acm, 'link_base', 'table')

    size = len(acm.entry_names)
    assert all(len(entry.enabled) == size for entry in acm.entry_values)


def test_without_legacy_links_only_allowed_links_change():
    names = ['link_base', 'link3', 'gripper_link']

    acm = scene._configure_acm(_acm(names), config.OBSTACLES)

    assert 'laundry_bucket_link' not in acm.entry_names
    assert _allowed(acm, 'link_base', 'table')
    assert 'bucket' not in acm.entry_names
    assert not acm.default_entry_names


def test_obstacle_pose_follows_urdf_rpy():
    spec = {'xyz': [0.1, -0.2, 0.3], 'rpy': [0.3, -0.4, 1.2]}

    pose = scene.obstacle_pose(spec)

    q = [pose.orientation.x, pose.orientation.y, pose.orientation.z,
         pose.orientation.w]
    # URDF: R = Rz(yaw) Ry(pitch) Rx(roll).
    expected = (
        Rotation.from_euler('z', 1.2)
        * Rotation.from_euler('y', -0.4)
        * Rotation.from_euler('x', 0.3)
    )
    assert np.allclose(Rotation.from_quat(q).as_matrix(), expected.as_matrix())
    assert (pose.position.x, pose.position.y, pose.position.z) == (0.1, -0.2, 0.3)


def _scene(objects, paddings):
    msg = PlanningScene()
    msg.world.collision_objects = [
        CollisionObject(id=i, pose=scene.obstacle_pose(spec))
        for i, spec in objects.items()
    ]
    msg.link_padding = [
        LinkPadding(link_name=k, padding=v) for k, v in paddings.items()
    ]
    return msg


def test_an_already_configured_scene_is_left_alone():
    wanted = scene._paddings(0.03, {'gripper_link': 0.0})
    obstacles = config.OBSTACLES

    assert scene._is_configured(_scene(obstacles, wanted), wanted, obstacles)
    # A missing obstacle or a different padding means apply again...
    only_bucket = {'bucket': obstacles['bucket']}
    assert not scene._is_configured(
        _scene(only_bucket, wanted), wanted, obstacles
    )
    other = dict(wanted, link3=0.02)
    assert not scene._is_configured(_scene(obstacles, other), wanted, obstacles)
    # ...and so does a moved bucket.
    moved = copy.deepcopy(obstacles)
    moved['bucket']['xyz'][0] += 0.01
    assert not scene._is_configured(_scene(obstacles, wanted), wanted, moved)
    # Leftover objects from the old ids are cleaned up too.
    legacy = dict(obstacles, bucket_obstacle=obstacles['bucket'])
    assert not scene._is_configured(_scene(legacy, wanted), wanted, obstacles)


def test_signature_tracks_the_geometry():
    moved = copy.deepcopy(config.OBSTACLES)

    assert scene.signature(moved) == scene.signature()

    moved['bucket']['rpy'][2] += 0.01
    assert scene.signature(moved) != scene.signature()


def test_stale_plan_message():
    assert scene.stale_plan_message(scene.signature(), 'x', 'bake') is None
    assert 'moved' in scene.stale_plan_message('0123456789ab', 'x', 'bake')
    assert 'predates' in scene.stale_plan_message('', 'x', 'bake')


def test_link_base_is_never_padded():
    assert 'link_base' not in scene._paddings(0.03)
    assert scene._paddings(0.03, {'gripper_link': 0.0})['gripper_link'] == 0.0
