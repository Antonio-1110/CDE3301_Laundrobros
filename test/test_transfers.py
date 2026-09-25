"""Tests for the baked transfers from INTER (arm/transfers.py)."""

import math

from laundry_control import config
from laundry_control.arm import transfers
from laundry_control.arm.joint_path import time_stop_at_each
import numpy as np
import pytest

INTER = np.array(config.INTER)
DROP = np.array(config.DROP)
HOME = np.array(config.HOME)
BOTTOM = np.array(config.BOTTOM)


def test_time_stop_at_each_rests_at_every_waypoint():
    waypoints = np.array([[0.0] * 7, [0.3] + [0.0] * 6, [0.3, 0.3] + [0.0] * 5])

    dense, times, velocities = time_stop_at_each(waypoints, 0.5)

    assert np.all(np.diff(times) > 0.0)
    for waypoint in waypoints:
        index = int(np.argmin(np.abs(dense - waypoint).max(axis=1)))
        assert np.allclose(dense[index], waypoint)
        assert np.allclose(velocities[index], 0.0)


def test_weighted_travel_counts_cable_joints_more():
    j2_only = [np.zeros(7), np.array([0, 1.0, 0, 0, 0, 0, 0])]
    j7_only = [np.zeros(7), np.array([0, 0, 0, 0, 0, 0, 1.0])]

    assert transfers.weighted_travel_deg(j7_only) == pytest.approx(
        3.0 * transfers.weighted_travel_deg(j2_only)
    )


ROUTES = {'drop': np.array([INTER, INTER + 0.1, DROP])}


def test_route_applies_from_inter_and_reverses_back_to_inter():
    forward = transfers.route_for(INTER + math.radians(1.0), 'drop', ROUTES)
    assert np.allclose(forward[0], INTER) and np.allclose(forward[-1], DROP)

    back = transfers.route_for(DROP, 'inter', ROUTES)
    assert np.allclose(back[0], DROP) and np.allclose(back[-1], INTER)


def test_route_between_two_targets_goes_through_inter():
    routes = {
        'drop': np.array([INTER, INTER + 0.1, DROP]),
        'home': np.array([INTER, INTER - 0.1, HOME]),
    }

    path = transfers.route_for(HOME, 'drop', routes)

    # Back along HOME's route to INTER, then out along DROP's.
    assert np.allclose(path[0], HOME)
    assert np.allclose(path[2], INTER)
    assert np.allclose(path[-1], DROP)
    assert len(path) == 5


def test_route_to_where_the_arm_already_is_is_none():
    assert transfers.route_for(DROP, 'drop', ROUTES) is None


def test_route_does_not_apply_elsewhere():
    assert transfers.route_for(HOME, 'drop', ROUTES) is None
    assert transfers.route_for(INTER + math.radians(5.0), 'drop', ROUTES) is None
    assert transfers.route_for(INTER, 'home', ROUTES) is None


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / 'transfers.yaml')
    transfers.save(path, {'drop': list(ROUTES['drop'])}, 0.7)

    routes, speed = transfers.load(path)

    assert speed == pytest.approx(0.7)
    assert np.allclose(routes['drop'], ROUTES['drop'], atol=1e-6)


def test_saved_routes_record_scene_and_paddings(tmp_path):
    from laundry_control.arm import scene

    path = str(tmp_path / 'transfers.yaml')
    transfers.save(
        path, {'drop': list(ROUTES['drop']), 'bottom': [INTER, BOTTOM]}, 0.7
    )

    assert transfers.baked_scene(path) == scene.signature()
    assert transfers.baked_arm_paddings(path) == {
        'drop': config.OBSTACLE_PADDING_M,
        'bottom': config.ROUTE_ARM_PADDING_M['bottom'],
    }


def test_missing_file_means_no_routes(tmp_path):
    assert transfers.load(str(tmp_path / 'none.yaml')) == ({}, None)


class _Logger:

    def info(self, *_a, **_k):
        pass

    def warning(self, *_a, **_k):
        pass

    def error(self, *_a, **_k):
        pass


class _StubArm:

    def __init__(self, at, blocked=False):
        self.at = list(at)
        self.blocked = blocked
        self.calls = []

    def get_logger(self):
        return _Logger()

    def get_current_joints(self):
        return list(self.at)

    def set_arm_padding(self, padding_m):
        self.calls.append(('padding', padding_m))

    def first_invalid_state(self, waypoints):
        return 3 if self.blocked else None

    def execute_joint_path(self, waypoints, times, velocities, time_scale=1.0):
        self.calls.append(('baked', np.array(waypoints)))
        return True

    def move_joints_linear(self, target, **_kwargs):
        self.calls.append(('linear', target))
        return True

    def move_joints(self, target):
        self.calls.append(('planner', target))
        return True


def test_go_to_replays_the_baked_route():
    arm = _StubArm(INTER)

    assert transfers.go_to(arm, 'drop', routes=ROUTES, max_velocity_rad_s=0.7)

    kind, waypoints = arm.calls[0]
    assert kind == 'baked'
    # Passes through the via exactly and ends at DROP.
    assert np.any(np.all(np.isclose(waypoints, INTER + 0.1), axis=1))
    assert np.allclose(waypoints[-1], DROP)


def test_go_to_refuses_a_baked_route_that_collides_now():
    arm = _StubArm(INTER, blocked=True)

    assert not transfers.go_to(
        arm, 'drop', routes=ROUTES, max_velocity_rad_s=0.7
    )
    # No replay, and no planner fallback either.
    assert arm.calls == []


def test_go_to_replays_under_the_routes_padding_then_restores_it():
    routes = dict(ROUTES, bottom=np.array([INTER, BOTTOM]))
    arm = _StubArm(DROP)

    assert transfers.go_to(
        arm, 'bottom', routes=routes, max_velocity_rad_s=0.7,
        arm_paddings={'drop': 0.03, 'bottom': 0.02},
    )

    kinds = [call[0] for call in arm.calls]
    assert kinds == ['padding', 'baked', 'padding']
    assert arm.calls[0][1] == 0.02
    assert arm.calls[2][1] == config.OBSTACLE_PADDING_M


def test_go_to_uses_a_straight_move_without_a_route():
    arm = _StubArm(HOME)

    assert transfers.go_to(arm, 'bottom', routes=ROUTES, max_velocity_rad_s=0.7)
    assert arm.calls[0][0] == 'linear'


def test_go_to_falls_back_to_the_planner_only_when_blocked():
    arm = _StubArm(HOME, blocked=True)

    assert transfers.go_to(arm, 'bottom', routes=ROUTES, max_velocity_rad_s=0.7)
    assert arm.calls[0][0] == 'planner'


def test_committed_transfers_are_sane():
    routes, speed = transfers.load()

    assert {'home', 'drop'} <= set(routes) <= set(transfers.TRANSFER_TARGETS)
    assert speed == pytest.approx(config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S)

    for name, route in routes.items():
        target = np.array(config.get_named_pose(name))

        assert np.allclose(route[0], INTER, atol=1e-5)
        assert np.allclose(route[-1], target, atol=1e-5)

        # The wrist-cable joint moves by (near enough) the direct amount.
        j7_travel = transfers.per_joint_travel_deg(route)[6]
        assert j7_travel <= abs(math.degrees(target[6] - INTER[6])) + 5.0
