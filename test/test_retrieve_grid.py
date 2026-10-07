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

    def __init__(self, arm, fail_close=False, fail_open_at=None):
        self.arm = arm
        self.fail_close = fail_close
        # Which open (counting from 1) fails, if any.
        self.fail_open_at = fail_open_at
        self.opens = 0

    def open_blocking(self):
        self.arm.calls.append(('open',))
        self.opens += 1
        return self.opens != self.fail_open_at

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


def _go_to_recording_empty(arm, name, gripper_empty=False, **_kwargs):
    arm.calls.append(('go_to', name, gripper_empty))
    return True


def test_after_a_confirmed_open_drop_is_left_as_empty():
    arm = _Arm()

    assert retrieve_grid.run(
        arm, _Gripper(arm), [_grab('grab_01', 0.0), _grab('grab_02', 5.0)],
        _go_to_recording_empty,
    )

    leaving = [c for c in arm.calls if c[0] == 'go_to' and c[1] != 'drop']
    assert leaving == [
        ('go_to', 'grab_01', True), ('go_to', 'grab_02', True),
        ('go_to', 'inter', True),
    ]


def test_an_unconfirmed_open_at_drop_never_leaves_as_empty():
    arm = _Arm()

    # Open 1 is before the first grab; open 2 is at DROP.
    assert not retrieve_grid.run(
        arm, _Gripper(arm, fail_open_at=2), [_grab('grab_01', 0.0)],
        _go_to_recording_empty,
    )

    assert arm.calls[-1] == ('go_to', 'inter', False)


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


def test_a_changed_grid_is_reported(tmp_path, monkeypatch):
    path = str(tmp_path / 'retrieve.yaml')
    grab = dict(_grab('grab_01', 0.5), tilt_deg=0.0, contact=[0.1, -0.4, 0.3])
    retrieve_grid.save([grab], path=path)

    assert retrieve_grid.grid_mismatch(path) is None

    monkeypatch.setitem(config.RETRIEVE_GRID, 'clearance_m', 0.02)
    assert 'clearance_m' in retrieve_grid.grid_mismatch(path)


class _IkArm:
    """IK flips J1 a full turn at the first height, and is fine after."""

    def __init__(self):
        self.heights = []

    def compute_ik(self, pose, seed):
        # The approach pose is solved seeded from the grab.
        solution = np.array(config.get_named_pose('inter')) + 0.1

        if len(self.heights) == 1:
            solution[0] += np.radians(360.0)

        return list(solution)

    def first_invalid_state(self, waypoints):
        return None


def test_a_flipped_ik_solution_is_skipped_for_the_next_height(monkeypatch):
    arm = _IkArm()
    grid = dict(config.RETRIEVE_GRID, heights_m=[0.02, 0.03], tilts_deg=[0.0])

    real_geometry = retrieve_grid.grab_geometry

    def geometry(depth, angle, height, tilt, cone=None):
        arm.heights.append(height)
        return real_geometry(depth, angle, height, tilt, cone)

    monkeypatch.setattr(retrieve_grid, 'grab_geometry', geometry)

    grab = retrieve_grid.solve_one(arm, 0.30, 0.0, grid, seeds=[[0.0] * 7])

    assert grab['height_m'] == pytest.approx(0.03)
    assert not retrieve_grid.flipped(grab['grab'], config.get_named_pose('inter'))


class _ReachArm:
    """Reaches a grab only at the given (height, tilt) pairs."""

    def __init__(self, reachable):
        self.reachable = reachable
        self.current = None
        self.seen = []

    def compute_ik(self, pose, seed):
        if self.current not in self.reachable:
            return None

        return list(np.array(config.get_named_pose('inter')) + 0.1)

    def first_invalid_state(self, waypoints):
        return None


def _recording_geometry(monkeypatch, arm):
    real_geometry = retrieve_grid.grab_geometry

    def geometry(depth, angle, height, tilt, cone=None):
        arm.current = (round(height, 3), round(tilt, 1))
        arm.seen.append(arm.current)
        return real_geometry(depth, angle, height, tilt, cone)

    monkeypatch.setattr(retrieve_grid, 'grab_geometry', geometry)


def test_detected_grabs_search_heights_then_tilts(monkeypatch):
    arm = _ReachArm({(0.03, 30.0)})
    _recording_geometry(monkeypatch, arm)
    grid = dict(config.RETRIEVE_GRID, heights_m=[0.02, 0.03], tilts_deg=[0.0, 30.0])

    grab = retrieve_grid.solve_one(arm, 0.30, 0.0, grid, seeds=[[0.0] * 7])

    assert (grab['height_m'], grab['tilt_deg']) == (0.03, 30.0)
    assert arm.seen == [(0.02, 0.0), (0.02, 30.0), (0.03, 0.0), (0.03, 30.0)]


def test_detected_grabs_may_go_down_to_the_floor():
    # A flat item must not be grabbed above itself.
    assert config.RETRIEVE_GRID['min_height_m'] <= 0.02


def _target(**changes):
    target = {
        'depth_m': 0.30, 'floor_angle_deg': 0.0, 'height_m': 0.11,
        'tilt_deg': 15.0, 'approach_m': 0.05,
    }
    target.update(changes)
    return target


def test_the_sweep_solves_each_target_exactly_as_placed(monkeypatch):
    import contextlib

    arm = _ReachArm({(0.11, 15.0)})
    _recording_geometry(monkeypatch, arm)
    monkeypatch.setattr(
        retrieve_grid, 'gripper_padding',
        lambda *a, **k: contextlib.nullcontext(),
    )
    targets = [_target(), _target(height_m=0.13)]

    grabs, misses = retrieve_grid.solve(arm, targets, log=lambda *_: None)

    # No search: one try each, at exactly the placed height and tilt.
    assert arm.seen == [(0.11, 15.0), (0.13, 15.0)]
    assert [g['name'] for g in grabs] == ['grab_01']
    assert grabs[0]['approach_m'] == 0.05
    assert [(index, reason) for index, _, reason in misses] == [
        (2, retrieve_grid._FAILURES[0])
    ]


def test_an_unreachable_approach_is_named(monkeypatch):
    class _NoApproach(_ReachArm):
        def compute_ik(self, pose, seed):
            self.calls = getattr(self, 'calls', 0) + 1
            return super().compute_ik(pose, seed) if self.calls % 2 else None

    arm = _NoApproach({(0.11, 15.0)})
    _recording_geometry(monkeypatch, arm)

    plan, reason = retrieve_grid.solve_target(arm, _target(), seeds=[[0.0] * 7])

    assert plan is None and 'approach' in reason


def test_changed_targets_are_reported(tmp_path):
    path = str(tmp_path / 'retrieve.yaml')
    grab = dict(_grab('grab_01', 0.5), tilt_deg=0.0, contact=[0.1, -0.4, 0.3])
    retrieve_grid.save([grab], path=path, targets=[_target()])

    assert retrieve_grid.grid_mismatch(path, targets=[_target()]) is None
    assert 'grab_targets.yaml' in retrieve_grid.grid_mismatch(
        path, targets=[_target(height_m=0.12)]
    )
    assert 'grab_targets.yaml' in retrieve_grid.grid_mismatch(
        path, targets=[_target(), _target()]
    )
