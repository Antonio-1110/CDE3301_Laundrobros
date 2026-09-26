"""Tests for planner-free joint paths and the baked precession end scan."""

import argparse
import math
import os

from laundry_control import config
from laundry_control.arm.joint_path import densify, time_path
from laundry_control.scan import coverage, endcap, pattern
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

Z0 = np.array([0.002, -1.0, -0.02])


# ------------------------------------------------------------ geometry


@pytest.mark.parametrize('alpha,phi', [(10, -90), (25, 0), (40, 45), (55, 70)])
def test_beam_is_perpendicular_to_tool_and_leans_to_the_end(alpha, phi):
    z0, up, side = endcap.reference_axes(Z0)

    tool_z, beam = endcap.precession_axes(alpha, phi, z0, up, side)

    assert tool_z @ beam == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(beam) == pytest.approx(1.0)
    # Leans toward the closed end (along the insertion axis) by alpha.
    assert beam @ z0 == pytest.approx(math.sin(math.radians(alpha)))


def test_zero_tilt_is_inter_orientation():
    z0, up, side = endcap.reference_axes(Z0)

    tool_z, beam = endcap.precession_axes(0.0, 0.0, z0, up, side)

    assert np.allclose(tool_z, z0)
    assert np.allclose(beam, -up)  # boresight straight down, as at INTER


def test_tilt_up_points_the_beam_down():
    z0, up, side = endcap.reference_axes(Z0)

    tool_z, beam = endcap.precession_axes(40.0, 0.0, z0, up, side)

    assert tool_z @ up > 0.0
    assert beam @ up < 0.0


def test_pose_orientation_matches_the_axes():
    z0, up, side = endcap.reference_axes(Z0)
    pose = endcap.precession_pose([0.1, -0.5, 0.4], 25.0, 30.0, z0, up, side)

    q = pose.orientation
    matrix = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    tool_z, beam = endcap.precession_axes(25.0, 30.0, z0, up, side)

    assert np.allclose(matrix[:, 2], tool_z)
    assert np.allclose(matrix[:, 0], beam)


# ------------------------------------------------------------ plan file


def _small_plan(depth=0.42):
    waypoints = densify(np.array([[0.0] * 7, [0.2] * 7, [0.0] * 7]))
    times, velocities = time_path(waypoints, 0.5)

    return endcap.EndcapPlan(
        depth_m=depth,
        pivot=np.array([0.15, -0.56, 0.46]),
        tool_z=Z0,
        waypoints=waypoints,
        times=times,
        velocities=velocities,
        alpha_phi=np.zeros((waypoints.shape[0], 2)),
        rings=[(10.0, -90.0, 90.0)],
        max_velocity_rad_s=0.5,
    )


def test_plan_round_trips(tmp_path):
    plan = _small_plan()
    path = str(tmp_path / 'plans' / 'endcap.yaml')

    plan.save(path)
    loaded = endcap.EndcapPlan.load(path)

    assert loaded.depth_m == pytest.approx(0.42)
    assert np.allclose(loaded.waypoints, plan.waypoints, atol=1e-6)
    assert np.allclose(loaded.times, plan.times, atol=1e-6)
    assert loaded.rings == [(10.0, -90.0, 90.0)]


def test_plan_rejects_other_versions(tmp_path):
    path = tmp_path / 'endcap.yaml'
    path.write_text('version: 99\n')

    with pytest.raises(ValueError, match='version'):
        endcap.EndcapPlan.load(str(path))


def test_committed_plan_is_sane():
    plan = endcap.EndcapPlan.load(endcap.default_plan_path())

    assert plan.depth_m == pytest.approx(pattern.DEFAULT_DEPTH_M)
    # Starts and ends at the pivot pose, so getting on and off is simple.
    assert np.allclose(plan.waypoints[0], plan.waypoints[-1], atol=1e-6)
    assert np.all(np.diff(plan.times) >= 0.0)
    assert np.abs(np.diff(plan.waypoints, axis=0)).max() <= math.radians(1.0) + 1e-6
    assert np.abs(plan.velocities).max() <= plan.max_velocity_rad_s * 1.05
    assert [ring[0] for ring in plan.rings] == [10.0, 25.0, 40.0, 55.0]


# ------------------------------------------------------------ runtime


class _Logger:

    def info(self, *_a, **_k):
        pass

    def error(self, *_a, **_k):
        pass

    def warning(self, *_a, **_k):
        pass


class _StubArm:

    def __init__(self, blocked=False):
        self.calls = []
        self.paddings = []
        self.blocked = blocked

    def get_logger(self):
        return _Logger()

    def first_invalid_state(self, waypoints):
        self.calls.append(('check', len(waypoints)))
        return 3 if self.blocked else None

    def set_arm_padding(self, padding_m):
        self.paddings.append(padding_m)

    def move_joints_linear(self, target, time_scale=1.0):
        self.calls.append(('linear', list(target), time_scale))
        return True

    def execute_joint_path(self, waypoints, times, velocities, time_scale=1.0):
        self.calls.append(('plan', len(waypoints), time_scale))
        return True


def test_run_plan_refuses_a_different_depth():
    with pytest.raises(ValueError, match='Re-bake'):
        endcap.run_plan(_StubArm(), _small_plan(depth=0.42), 0.40)


def test_run_plan_refuses_a_plan_that_collides_now():
    arm = _StubArm(blocked=True)

    assert not endcap.run_plan(arm, _small_plan(), 0.42)
    # Checked the whole plan, then never moved.
    assert arm.calls == [('check', len(_small_plan().waypoints))]


def test_run_plan_uses_the_plans_padding_then_restores_it():
    arm = _StubArm()
    plan = _small_plan()
    plan.padding_m = 0.01

    assert endcap.run_plan(arm, plan, 0.42)
    assert arm.paddings == [0.01, config.OBSTACLE_PADDING_M]

    blocked = _StubArm(blocked=True)
    assert not endcap.run_plan(blocked, plan, 0.42)
    # Restored even when the plan is refused.
    assert blocked.paddings == [0.01, config.OBSTACLE_PADDING_M]


def test_precession_end_scan_gets_on_replays_and_turns_around():
    arm = _StubArm()
    plan = _small_plan()
    pre = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -2.0]

    assert pattern._precession_end_scan(arm, plan, 0.42, pre, 150.0, 0.5)

    kinds = [call[0] for call in arm.calls]
    assert kinds == ['check', 'linear', 'plan', 'linear']
    assert arm.calls[1][1] == pytest.approx(plan.start)
    assert all(call[-1] == 0.5 for call in arm.calls[1:])

    exit_joints = arm.calls[3][1]
    assert exit_joints[:6] == pre[:6]
    assert exit_joints[6] == pytest.approx(-2.0 + math.radians(150.0))


def _scan_args(**overrides):
    parser = argparse.ArgumentParser()
    pattern.add_scan_arguments(parser)
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_scan_kwargs_default_to_the_committed_precession_plan():
    kwargs = pattern.scan_kwargs_from_args(_scan_args())

    assert kwargs['end_scan'] == 'precession'
    assert isinstance(kwargs['end_plan'], endcap.EndcapPlan)


def test_bottom_end_scan_needs_no_plan():
    kwargs = pattern.scan_kwargs_from_args(
        _scan_args(end_scan='bottom', end_plan='/does/not/exist.yaml')
    )

    assert kwargs['end_scan'] == 'bottom'
    assert kwargs['end_plan'] is None


def test_missing_plan_is_reported_before_the_arm_moves(tmp_path):
    with pytest.raises(FileNotFoundError, match='plan bake'):
        pattern.scan_kwargs_from_args(
            _scan_args(end_plan=str(tmp_path / 'missing.yaml'))
        )


def test_depth_mismatch_is_reported_before_the_arm_moves(tmp_path):
    path = str(tmp_path / 'endcap.yaml')
    _small_plan(depth=0.42).save(path)

    with pytest.raises(ValueError, match='Re-bake'):
        pattern.scan_kwargs_from_args(_scan_args(end_plan=path, depth=0.36, step=0.03))


def test_plan_beams_follow_the_plan_timing():
    plan = endcap.EndcapPlan.load(endcap.default_plan_path())

    origins, directions = coverage.plan_beams(plan)

    assert origins.shape[0] == pytest.approx(plan.duration_s * 20.0, abs=1)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_config_speed_is_within_every_joint_limit():
    assert config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S < config.JOINT7_MAX_VELOCITY_RAD_S
    assert os.path.isfile(endcap.default_plan_path())


def test_measured_coverage_uses_recorded_ray_origins(tmp_path):
    from builtin_interfaces.msg import Time
    from laundry_control.scan.cloud_io import save_xyz_csv

    # One beam: origin at the pivot, 0.1 m straight down.
    origin = np.array([0.15, -0.56, 0.46])
    end = origin + np.array([0.0, 0.0, -0.1])
    path = str(tmp_path / 'scan.csv')
    save_xyz_csv(path, [(*end, Time(), 0.1, *origin, 0.0)])

    rays = coverage.rays_from_csv(path)

    assert np.allclose(rays.origin[0], origin)
    assert rays.range_m[0] == pytest.approx(0.1)
