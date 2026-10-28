"""The claw on arrival and the turned carry (grasp/carry.py), and `laundry grasp`."""

from types import SimpleNamespace

from laundry_control import config
from laundry_control.arm.joint_path import (
    add_j7_turn, resample_by_step, time_stop_at_each,
)
from laundry_control.grasp import carry, execute
import numpy as np
import pytest


class _Pending:

    def __init__(self, calls, what):
        self.calls, self.what = calls, what

    def wait(self):
        self.calls.append(self.what + 'ed')
        return True


def test_the_claw_starts_on_arrival_and_is_waited_for_once():
    calls = []

    before_end, started = carry.on_arrival(
        lambda: calls.append('close') or _Pending(calls, 'clos')
    )

    assert before_end[0] == config.CLAW_COMMAND_LEAD_SEC
    before_end[1]()
    assert carry.wait(started, lambda: pytest.fail('started twice'))
    assert calls == ['close', 'closed']


def test_a_claw_not_started_on_arrival_is_started_when_waited_for():
    calls = []

    _before_end, started = carry.on_arrival(lambda: pytest.fail('early'))

    assert carry.wait(
        started, lambda: calls.append('close') or _Pending(calls, 'clos')
    )
    assert calls == ['close', 'closed']


def _carry_path(travel_rad):
    """Return a timed carry moving J1-J6 travel_rad (J7 little): ~1 s/rad."""
    waypoints = np.array([[0.0] * 7, [travel_rad] * 6 + [0.1]])
    return time_stop_at_each(
        waypoints, np.radians(60), max_acceleration=7.0, max_jerk=59.0,
    )


def _turn(times, positions, velocities):
    return add_j7_turn(
        times, positions, velocities, carry.turn_rad(),
        config.CARRY_TURN_MAX_VELOCITY_RAD_S,
        config.CARRY_TURN_MAX_ACCELERATION_RAD_S2,
        config.TRANSFER_MAX_JERK_RAD_S3,
    )


def test_the_turn_rides_on_the_carry_within_j7s_limits():
    # ~3.1 s, like the shortest baked carries.
    positions, times, velocities = _carry_path(3.0)

    grid, turned, rates, ramp = _turn(times, positions, velocities)

    # Same start, end and duration; turned the full 90 deg midway.
    assert np.allclose(turned[0], positions[0])
    assert np.allclose(turned[-1], positions[-1])
    assert grid[-1] == pytest.approx(times[-1])
    middle = np.argmin(np.abs(grid - grid[-1] / 2))
    plain = np.interp(grid[middle], times, positions[:, 6])
    assert turned[middle, 6] - plain == pytest.approx(carry.turn_rad())
    # Within J7's limits for the turn, the carry's own J7 motion added.
    assert np.abs(rates[:, 6]).max() <= config.CARRY_TURN_MAX_VELOCITY_RAD_S
    assert 2 * ramp <= grid[-1]


def test_a_carry_too_short_to_turn_and_back_is_not_turned():
    # ~1 s: too short to turn 90 deg and back.
    positions, times, velocities = _carry_path(0.75)

    assert _turn(times, positions, velocities) is None


def test_off_carries_as_before(monkeypatch):
    monkeypatch.setattr(config, 'CARRY_J7_TURN_DEG', 0.0)
    assert carry.turn_rad() == 0.0


class _Arm:

    def __init__(self):
        self.calls = []

    def move_to_pose(self, *_args, **_kwargs):
        self.calls.append('pose')
        return True


class _Gripper:

    def __init__(self, arm):
        self.arm = arm

    def open_blocking(self):
        self.arm.calls.append('open')
        return True

    def close_blocking(self):
        self.arm.calls.append('close')
        return True

    def start_open(self):
        self.arm.calls.append('open')
        return _Pending(self.arm.calls, 'open')


@pytest.fixture
def grasp_moves(monkeypatch):
    arm = _Arm()

    def go_to(arm_, name, **_kwargs):
        arm.calls.append(f'go_to {name}')
        return True

    def prepare_go_to(arm_, name, before=(), turn_rad=0.0, **_kwargs):
        arm.calls.append(f'check {name}' + (' turned' if turn_rad else ''))

        def move(before_end=None):
            arm.calls.append(f'move {name}')
            if before_end is not None:
                before_end[1]()
            return True

        return move

    monkeypatch.setattr(execute, 'go_to', go_to)
    monkeypatch.setattr(execute, 'prepare_go_to', prepare_go_to)

    return arm, SimpleNamespace(tcp_position=(0.1, -0.4, 0.3), orientation=None)


def test_laundry_grasp_carries_out_and_to_drop_turned_in_one_move(grasp_moves):
    arm, grasp = grasp_moves

    assert execute.execute_grasp(arm, _Gripper(arm), grasp, drop=True)

    assert arm.calls == [
        'open', 'pose', 'close',
        'check drop turned', 'move drop',
        # Opening as it arrives, then back.
        'open', 'opened',
        'go_to inter',
    ]


def test_laundry_grasp_unturned_is_as_before(grasp_moves, monkeypatch):
    monkeypatch.setattr(config, 'CARRY_J7_TURN_DEG', 0.0)
    arm, grasp = grasp_moves

    assert execute.execute_grasp(arm, _Gripper(arm), grasp, drop=True)
    assert arm.calls == [
        'open', 'pose', 'close', 'go_to inter', 'go_to drop', 'open',
        'go_to inter',
    ]


def test_checks_are_spaced_evenly_at_most_one_degree_apart():
    # Uneven samples: dense where slow, sparse where fast.
    t = np.linspace(0.0, 1.0, 400) ** 3
    path = np.outer(t, np.radians([30, 10, 0, 0, 0, 0, 90]))

    states = resample_by_step(path)

    steps = np.abs(np.diff(states, axis=0)).max(axis=1)
    assert steps.max() <= np.radians(1.0) + 1e-9
    assert len(states) == 91  # 90 deg of J7 travel, plus the start
    assert np.allclose(states[[0, -1]], path[[0, -1]])
