"""Generated grab poses for the sensorless sweep (grasp/retrieve_grid.py)."""

from laundry_control import config
from laundry_control.grasp import retrieve_grid
from laundry_control.perception.bucket_model import seed_cone, to_cylindrical
import numpy as np
import pytest


def test_grab_point_sits_the_given_height_above_the_floor():
    cone = seed_cone()

    contact, tool_z, _x = retrieve_grid.grab_geometry(0.30, 0.0, 0.03, 0.0)

    s, theta, r = to_cylindrical(contact[None], cone)
    assert s[0] == pytest.approx(0.30)
    assert np.degrees(theta[0]) == pytest.approx(180.0, abs=1e-6)
    assert cone.radius_at(0.30) - r[0] == pytest.approx(0.03)
    # Untilted, the tool points at the floor: down, square to the axis.
    assert tool_z @ cone.axis_dir == pytest.approx(0.0, abs=1e-9)
    assert tool_z[2] < -0.98


def test_positive_floor_angles_are_toward_plus_x():
    left, _z, _x = retrieve_grid.grab_geometry(0.30, -20.0, 0.03, 0.0)
    right, _z, _x = retrieve_grid.grab_geometry(0.30, 20.0, 0.03, 0.0)

    assert right[0] > left[0]
    assert right[2] == pytest.approx(left[2])


def test_tilt_leans_toward_the_closed_end():
    _c, tool_z, _x = retrieve_grid.grab_geometry(0.10, 0.0, 0.03, 30.0)

    assert np.degrees(np.arccos(-tool_z @ seed_cone().axis_dir)) == (
        pytest.approx(60.0)
    )


class _Arm:

    def __init__(self):
        self.calls = []

    def move_joints_linear(self, target, time_scale=1.0):
        self.calls.append(('linear', tuple(target)))
        return True


class _Gripper:

    def __init__(self, arm, fail_close=False):
        self.arm = arm
        self.fail_close = fail_close

    def open_blocking(self):
        self.arm.calls.append(('open',))
        return True

    def close_blocking(self):
        self.arm.calls.append(('close',))
        return not self.fail_close


def _grab(name, value):
    return {
        'name': name, 'depth_m': 0.3, 'floor_angle_deg': 0.0,
        'height_m': 0.02, 'approach': [value] * 7, 'grab': [value + 1] * 7,
    }


def _go_to(arm, name, **_kwargs):
    arm.calls.append(('go_to', name))
    return True


def test_each_grab_goes_down_closes_lifts_and_drops():
    arm = _Arm()

    assert retrieve_grid.run(
        arm, _Gripper(arm), [_grab('grab_01', 0.0), _grab('grab_02', 5.0)],
        _go_to,
    )

    assert arm.calls == [
        ('open',),
        ('go_to', 'grab_01'), ('linear', (1.0,) * 7), ('close',),
        ('linear', (0.0,) * 7), ('go_to', 'drop'), ('open',),
        ('go_to', 'grab_02'), ('linear', (6.0,) * 7), ('close',),
        ('linear', (5.0,) * 7), ('go_to', 'drop'), ('open',),
        ('go_to', 'inter'),
    ]


def test_a_failed_close_still_lifts_out_then_stops():
    arm = _Arm()

    assert not retrieve_grid.run(
        arm, _Gripper(arm, fail_close=True), [_grab('grab_01', 0.0)], _go_to
    )

    assert arm.calls[-3:] == [
        ('close',), ('linear', (0.0,) * 7), ('go_to', 'inter'),
    ]


def test_saved_grabs_become_named_poses(tmp_path, monkeypatch):
    path = tmp_path / 'scan_plans' / 'retrieve.yaml'
    monkeypatch.setattr(config, 'retrieve_plan_path', lambda: str(path))

    grab = dict(
        _grab('grab_01', 0.5), tilt_deg=0.0, contact=[0.1, -0.4, 0.3],
    )
    retrieve_grid.save([grab], path=str(path))

    assert config.get_named_pose('grab_01') == [0.5] * 7
    grabs, stamp = retrieve_grid.load(str(path))
    assert [g['name'] for g in grabs] == ['grab_01'] and stamp
