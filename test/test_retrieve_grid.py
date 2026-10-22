"""Generated grab poses for the sensorless sweep (grasp/retrieve_grid.py)."""

from laundry_control import config
from laundry_control.bucket import seed_cone, to_cylindrical
from laundry_control.grasp import retrieve_grid
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


class _Pending:

    def __init__(self, arm, what, ok):
        self.arm, self.what, self.ok = arm, what, ok

    def wait(self):
        self.arm.calls.append((self.what + 'ed',))
        return self.ok


class _Gripper:
    """Records when each claw move starts and when it is waited for."""

    def __init__(self, arm, fail_close=False, fail_open_at=None):
        self.arm = arm
        self.fail_close = fail_close
        # Which open (counting from 1) fails, if any.
        self.fail_open_at = fail_open_at
        self.opens = 0

    def start_open(self):
        self.arm.calls.append(('open',))
        self.opens += 1
        return _Pending(self.arm, 'open', self.opens != self.fail_open_at)

    def start_close(self):
        self.arm.calls.append(('close',))
        return _Pending(self.arm, 'clos', not self.fail_close)


def _grab(name, value):
    return {
        'name': name, 'depth_m': 0.3, 'floor_angle_deg': 0.0,
        'height_m': 0.02, 'approach': [value] * 7, 'grab': [value + 1] * 7,
    }


def _go_to(arm, name, gripper_empty=False, **_kwargs):
    arm.calls.append(('go_to', name, gripper_empty))
    return True


def _prepare(refuse=()):
    """Return a prepare_go_to stand-in that records the check, then the move."""
    def prepare(arm, name, before=(), after=(), gripper_empty=False,
                **_kwargs):
        straight = tuple(tuple(q) for q in list(before) + list(after))
        arm.calls.append(('check', name, straight, gripper_empty))

        if name in refuse:
            return None

        def move():
            arm.calls.append(('move', name))
            return True

        return move

    return prepare


def test_each_grab_goes_down_closes_lifts_and_drops():
    arm = _Arm()

    assert retrieve_grid.run(
        arm, _Gripper(arm), [_grab('grab_01', 0.0), _grab('grab_02', 5.0)],
        _go_to, _prepare(),
    )

    assert arm.calls == [
        ('open',),
        ('check', 'grab_01', ((1.0,) * 7,), True), ('opened',),
        ('move', 'grab_01'),
        ('close',), ('check', 'drop', ((0.0,) * 7,), False), ('closed',),
        ('move', 'drop'),
        ('open',),
        ('check', 'grab_02', ((6.0,) * 7,), True), ('opened',),
        ('move', 'grab_02'),
        ('close',), ('check', 'drop', ((5.0,) * 7,), False), ('closed',),
        ('move', 'drop'),
        ('open',), ('opened',),
        ('go_to', 'inter', True),
    ]


def test_an_unconfirmed_open_at_drop_never_leaves_as_empty():
    arm = _Arm()

    # Open 1 is before the first grab; open 2 is at DROP.
    assert not retrieve_grid.run(
        arm, _Gripper(arm, fail_open_at=2), [_grab('grab_01', 0.0)],
        _go_to, _prepare(),
    )

    assert arm.calls[-2:] == [('opened',), ('go_to', 'inter', False)]


def test_an_unconfirmed_first_open_does_not_move():
    arm = _Arm()

    assert not retrieve_grid.run(
        arm, _Gripper(arm, fail_open_at=1), [_grab('grab_01', 0.0)],
        _go_to, _prepare(),
    )

    assert not [c for c in arm.calls if c[0] in ('move', 'go_to', 'linear')]


def test_a_failed_close_still_lifts_out_then_stops():
    arm = _Arm()

    assert not retrieve_grid.run(
        arm, _Gripper(arm, fail_close=True), [_grab('grab_01', 0.0)],
        _go_to, _prepare(),
    )

    assert arm.calls[-3:] == [
        ('closed',), ('linear', (0.0,) * 7), ('go_to', 'inter', False),
    ]


def test_a_refused_route_to_drop_still_lifts_out_then_stops():
    arm = _Arm()

    assert not retrieve_grid.run(
        arm, _Gripper(arm), [_grab('grab_01', 0.0)], _go_to,
        _prepare(refuse=('drop',)),
    )

    assert arm.calls[-2:] == [('closed',), ('linear', (0.0,) * 7)]


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


class _ReachArm:
    """Reaches a grab only at the given (height, tilt) pairs."""

    def __init__(self, reachable):
        self.reachable = reachable
        self.current = None
        self.seen = []

    def compute_ik(self, pose, seed, limits=None):
        if self.current not in self.reachable:
            return None

        return list(np.array(config.get_named_pose('inter')) + 0.1)

    def first_invalid_state(self, waypoints, limits=None):
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


class _DigArm:
    """Clear at a tilt only from a given height up: {tilt: lowest clear}."""

    def __init__(self, clear_from):
        self.clear_from = clear_from
        self.current = None
        self.seen = []

    def compute_ik(self, pose, seed, limits=None):
        height, tilt = self.current

        if height < self.clear_from.get(tilt, np.inf) - 1e-9:
            return None

        return list(np.array(config.get_named_pose('inter')) + 0.1)

    def first_invalid_state(self, waypoints, limits=None):
        return None


DIG = {'max_sink_m': 0.15, 'wall_clearance_m': 0.02, 'resolution_m': 0.01}


def _dig(monkeypatch, clear_from, top, tilts=(0.0, 30.0)):
    arm = _DigArm(clear_from)
    _recording_geometry(monkeypatch, arm)
    grid = dict(config.RETRIEVE_GRID, tilts_deg=list(tilts))
    grab = retrieve_grid.deepest_grab(
        arm, 0.30, 0.0, top, grid, [[0.0] * 7], dig=DIG
    )
    return arm, grab


def test_with_nothing_in_the_way_the_grab_digs_down_to_the_wall(monkeypatch):
    _arm, grab = _dig(monkeypatch, {0.0: 0.0}, top=0.10)

    assert grab['height_m'] == pytest.approx(DIG['wall_clearance_m'])
    assert grab['sink_m'] == pytest.approx(0.08)


def test_a_tall_pile_is_dug_at_most_max_sink(monkeypatch):
    _arm, grab = _dig(monkeypatch, {0.0: 0.0}, top=0.25)

    assert grab['sink_m'] == pytest.approx(DIG['max_sink_m'])
    assert grab['height_m'] == pytest.approx(0.10)


def test_the_grab_stops_where_it_would_collide(monkeypatch):
    # Below 6.3 cm the claw would hit the bucket at tilt 0.
    arm, grab = _dig(monkeypatch, {0.0: 0.063}, top=0.12, tilts=(0.0,))

    assert grab['height_m'] >= 0.063
    assert grab['height_m'] - 0.063 <= DIG['resolution_m']
    # Bisection, not a probe per centimetre.
    assert len(arm.seen) <= 6


def test_a_tilt_that_digs_deeper_wins(monkeypatch):
    _arm, grab = _dig(monkeypatch, {0.0: 0.08, 30.0: 0.04}, top=0.12)

    assert grab['tilt_deg'] == 30.0
    assert grab['height_m'] - 0.04 <= DIG['resolution_m']


def test_a_tilt_that_cannot_beat_the_best_is_not_searched(monkeypatch):
    arm, grab = _dig(monkeypatch, {0.0: 0.02, 30.0: 0.0}, top=0.10)

    # Tilt 0 already reached the wall clearance: tilt 30 is not tried.
    assert grab['tilt_deg'] == 0.0
    assert all(tilt == 0.0 for _height, tilt in arm.seen)


def test_a_blocked_top_does_not_stop_the_search_lower_down(monkeypatch):
    # Near the mouth, high in a tall pile the arm meets the top of the
    # bucket: clear only between 5 and 10 cm from the wall.
    class Band(_DigArm):
        def compute_ik(self, pose, seed, limits=None):
            height, _tilt = self.current
            if not 0.05 - 1e-9 <= height <= 0.10 + 1e-9:
                return None
            return list(np.array(config.get_named_pose('inter')) + 0.1)

    arm = Band({})
    _recording_geometry(monkeypatch, arm)
    grid = dict(config.RETRIEVE_GRID, tilts_deg=[0.0])
    grab = retrieve_grid.deepest_grab(
        arm, 0.15, 0.0, 0.20, grid, [[0.0] * 7], dig=DIG
    )

    assert grab is not None
    assert grab['height_m'] - 0.05 <= DIG['resolution_m'] + 1e-9


def test_a_long_approach_that_collides_falls_back_to_the_short_one(
    monkeypatch,
):
    real = retrieve_grid._try_pose

    def try_pose(arm, contact, tool_z, claw_x, approach_m, seeds, inter):
        if approach_m > config.RETRIEVE_GRID['approach_m'] + 1e-9:
            return None, 'the approach pose collides'
        return real(arm, contact, tool_z, claw_x, approach_m, seeds, inter)

    monkeypatch.setattr(retrieve_grid, '_try_pose', try_pose)
    _arm, grab = _dig(monkeypatch, {0.0: 0.0}, top=0.10, tilts=(0.0,))

    assert grab['height_m'] == pytest.approx(DIG['wall_clearance_m'])
    assert grab['approach_m'] == pytest.approx(
        config.RETRIEVE_GRID['approach_m']
    )


def test_nothing_clear_even_at_the_top_gives_none(monkeypatch):
    _arm, grab = _dig(monkeypatch, {}, top=0.10)

    assert grab is None


def test_the_approach_starts_above_the_pile(monkeypatch):
    seen = []
    real = retrieve_grid._try_pose

    def try_pose(arm, contact, tool_z, claw_x, approach_m, seeds, inter):
        seen.append(approach_m)
        return real(arm, contact, tool_z, claw_x, approach_m, seeds, inter)

    monkeypatch.setattr(retrieve_grid, '_try_pose', try_pose)
    _arm, grab = _dig(monkeypatch, {0.0: 0.0}, top=0.20, tilts=(0.0,))

    # Dug 15 cm in: the approach is at least that plus 3 cm back.
    assert grab['sink_m'] == pytest.approx(0.15)
    assert max(seen) >= 0.15 + retrieve_grid.ABOVE_PILE_M - 1e-9


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
        def compute_ik(self, pose, seed, limits=None):
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
