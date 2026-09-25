"""The padded bucket/table planning-scene helpers (arm/scene.py)."""

import os

from laundry_control.arm import scene
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    CollisionObject,
    LinkPadding,
    PlanningScene,
)

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


def test_urdf_copies_are_ignored_and_the_world_copies_checked():
    names = ['link_base', 'link3', 'gripper_link', 'laundry_bucket_link',
             'table_link']

    acm = scene._acm_with_urdf_copies_disabled(_acm(names))

    # The URDF copies no longer collide with anything...
    assert _allowed(acm, 'laundry_bucket_link', 'gripper_link')
    assert _allowed(acm, 'table_link', 'link3')
    assert _allowed(acm, 'laundry_bucket_link', 'bucket_obstacle')
    # ...while the padded world copies still do.
    assert not _allowed(acm, 'bucket_obstacle', 'gripper_link')
    assert not _allowed(acm, 'table_obstacle', 'link3')
    # Except the base, which sits on the table.
    assert _allowed(acm, 'link_base', 'table_obstacle')

    size = len(acm.entry_names)
    assert all(len(entry.enabled) == size for entry in acm.entry_values)


def _scene(objects, paddings):
    msg = PlanningScene()
    msg.world.collision_objects = [CollisionObject(id=i) for i in objects]
    msg.link_padding = [
        LinkPadding(link_name=k, padding=v) for k, v in paddings.items()
    ]
    return msg


def test_an_already_padded_scene_is_left_alone():
    wanted = scene._paddings(0.03, {'gripper_link': 0.0})

    assert scene._is_configured(_scene(scene.OBSTACLES, wanted), wanted)
    # Missing obstacles, or a different padding, means apply again.
    assert not scene._is_configured(_scene(['bucket_obstacle'], wanted), wanted)
    other = dict(wanted, link3=0.02)
    assert not scene._is_configured(_scene(scene.OBSTACLES, other), wanted)


def test_link_base_is_never_padded():
    assert 'link_base' not in scene._paddings(0.03)
    assert scene._paddings(0.03, {'gripper_link': 0.0})['gripper_link'] == 0.0
