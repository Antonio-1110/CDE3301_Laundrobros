"""Tests for config.py and arm/geometry.py (pure, no ROS graph)."""

import math

from geometry_msgs.msg import Quaternion
from laundry_control import config
from laundry_control.arm.flange_check import describe_alignment
from laundry_control.arm.geometry import (
    angle_between_deg,
    look_at_quaternion,
    tool_z_from_quaternion,
)
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _quat(rotation):
    x, y, z, w = rotation.as_quat()
    return Quaternion(x=x, y=y, z=z, w=w)


def test_named_poses_cover_every_recorded_pose():
    poses = config.named_poses()

    recorded = {
        'home', 'inter', 'bottom', 'drop',
        'retrieve_0', 'retrieve_1', 'retrieve_2', 'retrieve_3',
    }
    # Plus the generated grab_NN poses, once scan_plans/retrieve.yaml
    # is baked.
    assert set(poses) == recorded | set(config.generated_grab_poses())
    assert all(
        name.startswith('grab_') for name in set(poses) - recorded
    )

    assert all(len(joints) == 7 for joints in poses.values())


def test_get_named_pose_is_case_insensitive_and_copies():
    pose = config.get_named_pose('  InTeR ')

    assert pose == config.INTER

    pose[0] = 99.0

    assert config.INTER[0] != 99.0


def test_get_named_pose_lists_alternatives_on_typo():
    with pytest.raises(KeyError, match='inter'):
        config.get_named_pose('intr')


def test_repo_root_is_the_source_checkout(monkeypatch):
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)

    root = config.repo_root()

    assert (root.rstrip('/').endswith('CDE3301_Laundrobros')
            or 'package.xml' in __import__('os').listdir(root))
    assert config.baseline_dir().startswith(root)


def test_repo_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))

    assert config.baseline_dir() == str(tmp_path / 'baseline_scans')
    assert config.scan_records_dir() == str(tmp_path / 'scan_records')


@pytest.mark.parametrize('seed', range(5))
def test_tool_z_matches_rotation_matrix(seed):
    rotation = Rotation.random(random_state=seed)

    expected = rotation.as_matrix()[:, 2]

    assert np.allclose(tool_z_from_quaternion(_quat(rotation)), expected)


def test_angle_between():
    assert angle_between_deg([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert angle_between_deg([1, 0, 0], [2, 0, 0]) == pytest.approx(0.0)


def test_look_at_points_local_z_along_direction():
    direction = np.array([0.3, -0.9, 0.1])

    q = look_at_quaternion(direction, [1.0, 0.0, 0.0])

    z_axis = tool_z_from_quaternion(q)

    assert np.allclose(z_axis, direction / np.linalg.norm(direction))


def test_inter_insertion_axis_against_the_bucket():
    # Tool +Z measured at INTER on the MoveIt fake controller.
    tool_z = (0.010, -1.000, -0.009)

    # The bucket axis this was measured against (the bucket-model seed
    # of 2026-09-24); the configured one is checked in
    # test_bucket_model.
    bucket_axis = (-0.0014, 0.996, 0.0891)

    misalignment, elevation, drift, _verdict = describe_alignment(
        tool_z, bucket_axis
    )

    assert abs(elevation) < 1.0
    assert misalignment == pytest.approx(4.6, abs=0.1)
    assert drift == pytest.approx(0.42 * math.sin(math.radians(misalignment)))


def test_describe_alignment_verdict_bands():
    axis = np.array([0.0, 1.0, 0.0])

    assert describe_alignment((0.0, -1.0, 0.0), axis)[3].startswith('ALIGNED')
    assert describe_alignment((0.0, -1.0, 0.05), axis)[3].startswith('CLOSE')
    assert describe_alignment((0.0, 0.0, -1.0), axis)[3].startswith('OFFSET')
