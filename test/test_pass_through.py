"""Passing through the vias of baked transfers (arm/pass_through.py)."""

import math

from laundry_control import config
from laundry_control.arm import pass_through, transfers
from laundry_control.arm.joint_path import time_stop_at_each
import numpy as np
import pytest
import yaml

INTER = np.array(config.get_named_pose('inter'))
DROP = np.array(config.DROP)
VIA = INTER + 0.1
ROUTES = {'drop': np.array([INTER, VIA, DROP])}


def _timed(waypoints, duration=2.0, n=11):
    """Return a stand-in timing: straight through the waypoints, at rest at both ends."""
    waypoints = np.asarray(waypoints)
    s = np.linspace(0.0, 1.0, n)
    arc = np.linspace(0.0, 1.0, len(waypoints))
    positions = np.array([
        [np.interp(x, arc, waypoints[:, j]) for j in range(7)] for x in s
    ])
    times = s * duration
    velocities = np.gradient(positions, times, axis=0)
    velocities[0] = velocities[-1] = 0.0
    return times, positions, velocities


def test_a_short_final_interval_is_folded_into_the_one_before():
    times = np.array([0.0, 0.2, 0.4, 0.403])
    q = np.arange(4.0)[:, None] * np.ones(7)

    t, p, v = pass_through.merge_short_tail(times, q, q)

    assert list(t) == [0.0, 0.2, 0.403]
    assert np.allclose(p[-1], q[-1])


def test_a_normal_final_interval_is_kept():
    times = np.array([0.0, 0.2, 0.4, 0.55])
    q = np.zeros((4, 7))

    assert len(pass_through.merge_short_tail(times, q, q)[0]) == 4


def test_curve_peaks_match_the_cubic_the_controller_draws():
    # Rest to rest over h: the cubic peaks at 1.5 d/h, 6 d/h^2, 12 d/h^3.
    d, h = 0.5, 2.0
    times = np.array([0.0, h])
    positions = np.array([np.zeros(7), np.r_[d, np.zeros(6)]])

    v, a, j = pass_through.curve_peaks(times, positions, np.zeros((2, 7)))

    assert v == pytest.approx(1.5 * d / h, rel=1e-3)
    assert a == pytest.approx(6 * d / h ** 2)
    assert j == pytest.approx(12 * d / h ** 3)


def test_limits_are_checked_on_all_three_peaks():
    assert pass_through.within_limits((0.7, 1.9, 16.0), 0.785, 2.0, 17.0)
    assert not pass_through.within_limits((0.8, 1.9, 16.0), 0.785, 2.0, 17.0)
    assert not pass_through.within_limits((0.7, 2.1, 16.0), 0.785, 2.0, 17.0)
    assert not pass_through.within_limits((0.7, 1.9, 18.0), 0.785, 2.0, 17.0)


def test_reversing_runs_the_same_motion_backwards():
    t, q, qd = _timed([INTER, VIA, DROP])

    rt, rq, rqd = pass_through.reverse(t, q, qd)

    assert rt[0] == 0.0 and rt[-1] == pytest.approx(t[-1])
    assert np.allclose(rq[0], DROP) and np.allclose(rq[-1], INTER)
    assert np.allclose(rqd, -qd[::-1])
    assert all(np.allclose(a, b) for a, b in zip(pass_through.reverse(rt, rq, rqd), (t, q, qd)))


def test_dense_states_are_close_enough_to_collision_check():
    t, q, qd = _timed([INTER, VIA, DROP])

    states = pass_through.dense_states(t, q, qd)

    assert np.allclose(states[0], INTER) and np.allclose(states[-1], DROP)
    steps = np.abs(np.diff(states, axis=0)).max(axis=1)
    assert steps.max() <= math.radians(1.0) + 0.02


def test_a_timing_round_trips_through_the_document():
    t, q, qd = _timed([INTER, VIA, DROP])

    document = pass_through.to_document(
        t, q, qd, pass_through.current_limits(), {'path_tolerance_rad': 0.05}
    )
    back = pass_through.from_document(yaml.safe_load(yaml.safe_dump(document)))

    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(back, (t, q, qd)))


def _write_plan(tmp_path, timed):
    path = tmp_path / 'transfers.yaml'
    transfers.save(str(path), ROUTES, 0.785)
    transfers.save_timings(str(path), {'drop': timed})
    return str(path)


def test_timings_for_the_current_limits_are_loaded(tmp_path):
    t, q, qd = _timed([INTER, VIA, DROP])
    path = _write_plan(tmp_path, pass_through.to_document(
        t, q, qd, pass_through.current_limits(), {}
    ))

    assert set(transfers.load_timings(path)) == {'drop'}
    # The routes themselves are unchanged by storing a timing.
    assert np.allclose(transfers.load(path)[0]['drop'], ROUTES['drop'])


def test_timings_for_other_limits_are_ignored(tmp_path):
    t, q, qd = _timed([INTER, VIA, DROP])
    limits = dict(pass_through.current_limits(), velocity_rad_s=1.0)
    path = _write_plan(tmp_path, pass_through.to_document(t, q, qd, limits, {}))

    assert transfers.load_timings(path) == {}


def test_a_timing_for_another_route_is_ignored(tmp_path):
    t, q, qd = _timed([INTER, VIA, DROP + 0.2])
    path = _write_plan(tmp_path, pass_through.to_document(
        t, q, qd, pass_through.current_limits(), {}
    ))

    assert transfers.load_timings(path) == {}


class _Logger:

    def __init__(self):
        self.lines = []

    def info(self, text, **_k):
        self.lines.append(text)

    warning = error = info


class _Arm:
    """first_invalid_state answers from `verdicts`, in call order."""

    def __init__(self, at, verdicts=()):
        self.at = list(at)
        self.verdicts = list(verdicts)
        self.logger = _Logger()
        self.executed = []

    def get_logger(self):
        return self.logger

    def get_current_joints(self):
        return list(self.at)

    def set_arm_padding(self, padding_m):
        pass

    def first_invalid_state(self, waypoints, **_k):
        return self.verdicts.pop(0) if self.verdicts else None

    def execute_joint_path(self, waypoints, times, velocities, time_scale=1.0):
        self.executed.append((np.array(waypoints), np.array(times)))
        return True


def _go(arm, target, timings):
    return transfers.go_to(
        arm, target, routes=ROUTES, max_velocity_rad_s=0.785, timings=timings
    )


def test_a_timed_route_passes_through_its_vias():
    timing = _timed([INTER, VIA, DROP])
    arm = _Arm(INTER)

    assert _go(arm, 'drop', {'drop': timing})

    (positions, times), = arm.executed
    assert times[-1] == pytest.approx(timing[0][-1])
    assert np.allclose(positions[-1], DROP)
    assert any('passing through' in line for line in arm.logger.lines)


def test_the_way_back_runs_the_timing_backwards():
    arm = _Arm(DROP)

    assert _go(arm, 'inter', {'drop': _timed([INTER, VIA, DROP])})

    (positions, _times), = arm.executed
    assert np.allclose(positions[0], DROP) and np.allclose(positions[-1], INTER)


def test_colliding_corners_fall_back_to_stopping_at_the_vias():
    # 1st check (the straight route) clear; 2nd (the rounded curve) hits.
    arm = _Arm(INTER, verdicts=[None, 5])

    assert _go(arm, 'drop', {'drop': _timed([INTER, VIA, DROP])})

    (_positions, times), = arm.executed
    stop = time_stop_at_each(
        [INTER, INTER, VIA, DROP], 0.785,
        max_acceleration=config.TRANSFER_MAX_ACCELERATION_RAD_S2,
        max_jerk=config.TRANSFER_MAX_JERK_RAD_S3,
    )[1][-1]
    assert times[-1] == pytest.approx(stop)
    assert any('stopping at each' in line for line in arm.logger.lines)


def test_without_a_timing_the_route_stops_at_its_vias():
    arm = _Arm(INTER)

    assert _go(arm, 'drop', {})

    assert any('stopping at each' in line for line in arm.logger.lines)


def test_pass_through_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, 'TRANSFER_PASS_THROUGH_VIAS', False)
    arm = _Arm(INTER)

    assert _go(arm, 'drop', {'drop': _timed([INTER, VIA, DROP])})

    assert any('stopping at each' in line for line in arm.logger.lines)


def test_a_small_gap_to_the_route_is_bridged_first():
    start = INTER + math.radians(1.0)
    arm = _Arm(start)

    assert _go(arm, 'drop', {'drop': _timed([INTER, VIA, DROP])})

    (positions, _times), = arm.executed
    assert np.allclose(positions[0], start)
    assert np.allclose(positions[-1], DROP)


def test_a_timing_starts_and_ends_exactly_at_rest():
    # The controller rejects a last point with any velocity at all.
    velocities = np.full((3, 7), 1e-3)

    rested = pass_through.at_rest_at_both_ends(velocities)

    assert np.all(rested[0] == 0.0) and np.all(rested[-1] == 0.0)
    assert np.all(rested[1] == 1e-3)


def test_a_stored_timing_is_loaded_at_rest_at_both_ends():
    t, q, qd = _timed([INTER, VIA, DROP])
    qd[-1] = 1e-3
    document = pass_through.to_document(t, q, qd, pass_through.current_limits(), {})

    _t, _q, loaded = pass_through.from_document(document)

    assert np.all(loaded[-1] == 0.0)
