"""Tests for INTER and BOTTOM derived from the bucket (arm/bucket_poses.py)."""

import dataclasses

from laundry_control import config
from laundry_control.arm import bucket_poses, transfers
from laundry_control.perception.bucket_model import MESH_DEPTH_M, seed_cone
from laundry_control.scan import endcap
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml


def _angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def test_inter_is_on_the_axis_outside_the_mouth_looking_in():
    cone = seed_cone()
    position, tool_z, boresight = bucket_poses.targets()['inter']

    offset = position - cone.axis_point
    along = offset @ cone.axis_dir

    assert along == pytest.approx(
        MESH_DEPTH_M + config.BUCKET_POSES['inter']['standoff_m']
    )
    # On the axis, pointing straight in.
    assert np.linalg.norm(offset - along * cone.axis_dir) < 1e-9
    assert np.allclose(tool_z, -cone.axis_dir)
    # The sensor looks at the floor: perpendicular to the tool, and as
    # far down as it can be.
    assert boresight @ tool_z == pytest.approx(0.0, abs=1e-9)
    assert boresight[2] < -0.9


def test_bottom_is_the_scan_depth_in_tilted_up():
    cone = seed_cone()
    targets = bucket_poses.targets()
    inter = targets['inter'][0]
    position, tool_z, boresight = targets['bottom']

    spec = config.BUCKET_POSES['bottom']

    assert np.allclose(position, inter - spec['depth_m'] * cone.axis_dir)
    assert _angle_deg(tool_z, -cone.axis_dir) == pytest.approx(
        spec['tilt_deg']
    )
    # Tilted UP, so the beam leans down and toward the closed end.
    assert tool_z[2] > -cone.axis_dir[2]
    assert boresight[2] < 0.0
    assert boresight @ -cone.axis_dir > 0.0


def test_moving_the_bucket_moves_both_poses_with_it():
    cone = seed_cone()
    shift = np.array([0.05, -0.02, 0.01])
    moved = dataclasses.replace(cone, axis_point=cone.axis_point + shift)

    before = bucket_poses.targets(cone)
    after = bucket_poses.targets(moved)

    for name in bucket_poses.POSE_NAMES:
        assert np.allclose(after[name][0], before[name][0] + shift)
        assert np.allclose(after[name][1], before[name][1])


def test_turning_the_bucket_turns_both_poses_with_it():
    cone = seed_cone()
    turn = Rotation.from_euler('z', 20.0, degrees=True)
    turned = dataclasses.replace(cone, axis_dir=turn.apply(cone.axis_dir))

    after = bucket_poses.targets(turned)

    assert np.allclose(after['inter'][1], -turned.axis_dir)
    assert _angle_deg(after['bottom'][1], -turned.axis_dir) == pytest.approx(
        config.BUCKET_POSES['bottom']['tilt_deg']
    )


def test_target_pose_is_exactly_that_frame():
    for position, tool_z, boresight in bucket_poses.targets().values():
        pose = bucket_poses.target_pose(position, tool_z, boresight)
        q = pose.orientation
        matrix = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

        assert np.allclose(matrix[:, 2], tool_z)
        assert np.allclose(matrix[:, 0], boresight)
        assert np.allclose(
            [pose.position.x, pose.position.y, pose.position.z], position
        )


def _write_poses(path, poses, spec=None):
    from laundry_control.arm import scene

    document = {
        'scene': scene.signature(),
        'spec': spec or config.BUCKET_POSES,
        'poses': {name: {'joints': joints} for name, joints in poses.items()},
    }
    path.write_text(yaml.safe_dump(document))


def test_recorded_poses_are_used_until_derived(tmp_path, monkeypatch):
    path = tmp_path / 'bucket_poses.yaml'
    monkeypatch.setattr(config, 'bucket_poses_path', lambda: str(path))

    assert config.get_named_pose('inter') == config.INTER_RECORDED
    assert config.get_named_pose('bottom') == config.BOTTOM_RECORDED
    assert 'RECORDED' in bucket_poses.status()['inter']

    derived = [0.1 * k for k in range(7)]
    _write_poses(path, {'inter': derived})

    assert config.get_named_pose('inter') == derived
    # Only what was derived is replaced.
    assert config.get_named_pose('bottom') == config.BOTTOM_RECORDED
    assert 'derived' in bucket_poses.status()['inter']


def test_a_changed_spec_is_reported(tmp_path):
    path = tmp_path / 'bucket_poses.yaml'
    spec = {'inter': {'standoff_m': 0.10}, 'bottom': dict(
        config.BUCKET_POSES['bottom']
    )}
    _write_poses(path, {'inter': [0.0] * 7}, spec=spec)

    assert 'inter.standoff_m' in bucket_poses.spec_mismatch(str(path))

    _write_poses(path, {'inter': [0.0] * 7})

    assert bucket_poses.spec_mismatch(str(path)) is None


class _Logger:

    def warning(self, *_a, **_k):
        pass


class _IkArm:
    """IK lands a fixed offset from the seed; FK is the identity frame."""

    def __init__(self, reachable=True):
        self.reachable = reachable
        self.paddings = []

    def get_logger(self):
        return _Logger()

    def set_arm_padding(self, padding_m):
        self.paddings.append(padding_m)

    def compute_ik(self, pose, seed):
        if not self.reachable:
            return None
        return list(np.asarray(seed) + 0.01)

    def state_is_valid(self, joints):
        return True

    def compute_fk(self, joints):
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])


def test_solve_seeds_from_the_recorded_pose_under_the_live_padding(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        config, 'bucket_poses_path', lambda: str(tmp_path / 'none.yaml')
    )
    arm = _IkArm()

    poses, failures = bucket_poses.solve(arm, log=lambda *_: None)

    assert failures == {}
    assert np.allclose(poses['inter'], np.asarray(config.INTER_RECORDED) + 0.01)
    assert np.allclose(
        poses['bottom'], np.asarray(config.BOTTOM_RECORDED) + 0.01
    )
    assert arm.paddings == [config.OBSTACLE_PADDING_M]


def test_solve_reports_what_it_cannot_reach(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config, 'bucket_poses_path', lambda: str(tmp_path / 'none.yaml')
    )

    poses, failures = bucket_poses.solve(
        _IkArm(reachable=False), log=lambda *_: None
    )

    assert poses == {}
    assert set(failures) == {'inter', 'bottom'}


def test_solve_rejects_a_different_posture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config, 'bucket_poses_path', lambda: str(tmp_path / 'none.yaml')
    )

    class _FlippingArm(_IkArm):

        def compute_ik(self, pose, seed):
            flipped = list(seed)
            flipped[6] += np.radians(200.0)
            return flipped

    poses, failures = bucket_poses.solve(_FlippingArm(), log=lambda *_: None)

    assert poses == {}
    assert set(failures) == {'inter', 'bottom'}


def test_routes_baked_from_another_inter_are_not_used():
    inter = np.array(config.get_named_pose('inter'))
    drop = np.array(config.DROP)
    old_inter = inter + np.radians(5.0)

    routes = {
        'drop': np.array([inter, drop]),
        'home': np.array([old_inter, np.array(config.HOME)]),
    }

    assert transfers.from_other_inter(routes) == ['home']


class _FkArm:
    """FK puts the flange at `position`, tool +Z along -Y."""

    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def compute_fk(self, joints):
        return self.position, Rotation.from_euler('x', 90.0, degrees=True).as_quat()


def test_end_scan_from_another_inter_is_reported():
    tool_z = np.array([0.0, -1.0, 0.0])
    plan = endcap.EndcapPlan(
        depth_m=0.42,
        pivot=np.array([0.0, -0.42, 0.5]),
        tool_z=tool_z,
        waypoints=np.zeros((2, 7)),
        times=np.array([0.0, 1.0]),
        velocities=np.zeros((2, 7)),
        alpha_phi=np.zeros((2, 2)),
        rings=[],
        max_velocity_rad_s=0.7,
    )

    assert endcap.inter_mismatch(_FkArm([0.0, 0.0, 0.5]), plan) is None

    moved = endcap.inter_mismatch(_FkArm([0.03, 0.0, 0.5]), plan)

    assert moved is not None and '3.0 cm' in moved
