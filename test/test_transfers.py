"""Tests for the baked transfers from INTER (arm/transfers.py)."""

import math

from laundry_control import config
from laundry_control.arm import transfers
from laundry_control.arm.joint_path import time_stop_at_each
import numpy as np
import pytest

INTER = np.array(config.get_named_pose('inter'))
DROP = np.array(config.DROP)
HOME = np.array(config.HOME)
BOTTOM = np.array(config.get_named_pose('bottom'))


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


# The 10 cm DROP route goes via INTER + 0.1, the empty-gripper one via
# EMPTY_VIA - off that line, so no replayed segment passes INTER + 0.1.
EMPTY_VIA = INTER + np.array([0.2, -0.2, 0.2, -0.2, 0.2, -0.2, 0.2])
EMPTY_ROUTES = dict(
    ROUTES, drop_return=np.array([INTER, EMPTY_VIA, DROP])
)


def test_drop_is_left_by_the_empty_gripper_route_only_when_told_so():
    loaded = transfers.route_for(DROP, 'inter', EMPTY_ROUTES)
    assert np.allclose(loaded[1], INTER + 0.1)

    empty = transfers.route_for(DROP, 'inter', EMPTY_ROUTES, gripper_empty=True)
    assert np.allclose(empty[0], DROP) and np.allclose(empty[-1], INTER)
    assert np.allclose(empty[1], EMPTY_VIA)


def test_the_way_to_drop_never_takes_the_empty_gripper_route():
    out = transfers.route_for(INTER, 'drop', EMPTY_ROUTES, gripper_empty=True)

    assert np.allclose(out[1], INTER + 0.1)


def test_empty_gripper_without_its_route_falls_back_to_drops_own():
    back = transfers.route_for(DROP, 'inter', ROUTES, gripper_empty=True)

    assert np.allclose(back[1], INTER + 0.1)


def test_empty_gripper_route_chains_on_to_the_next_target():
    routes = dict(EMPTY_ROUTES, home=np.array([INTER, INTER - 0.1, HOME]))

    path = transfers.route_for(DROP, 'home', routes, gripper_empty=True)

    assert np.allclose(path[1], EMPTY_VIA)
    assert np.allclose(path[2], INTER)
    assert np.allclose(path[-1], HOME)


def test_the_empty_gripper_route_is_baked_to_drop():
    assert transfers.route_pose('drop_return') == 'drop'
    assert transfers.route_pose('home') == 'home'
    assert 'drop_return' in transfers.transfer_targets()
    assert transfers.expected_gripper_clearance('drop_return') < (
        transfers.expected_gripper_clearance('drop')
    )


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
    assert transfers.padding_mismatches(path) == {}


def test_padding_changes_are_reported(tmp_path, monkeypatch):
    path = str(tmp_path / 'transfers.yaml')
    transfers.save(path, {'drop': list(ROUTES['drop'])}, 0.7)

    monkeypatch.setattr(config, 'OBSTACLE_PADDING_M', 0.05)

    report = transfers.padding_mismatches(path)
    assert set(report) == {'drop'}
    assert 'arm_links 3 -> 5 cm' in report['drop']


def test_replay_is_checked_under_the_current_padding(monkeypatch):
    monkeypatch.setattr(config, 'OBSTACLE_PADDING_M', 0.05)
    arm = _StubArm(INTER)

    # Routes supplied directly carry no file; the padding comes from
    # config, so the raised value is what the route is checked under.
    assert transfers.go_to(arm, 'drop', routes=ROUTES, max_velocity_rad_s=0.7)
    assert arm.calls[0][0] == 'baked'  # 5 cm is config now: no switch needed

    monkeypatch.setattr(config, 'ROUTE_ARM_PADDING_M', {'drop': 0.04})
    arm = _StubArm(INTER)
    assert transfers.go_to(arm, 'drop', routes=ROUTES, max_velocity_rad_s=0.7)
    assert [c for c in arm.calls if c[0] == 'padding'] == [
        ('padding', 0.04), ('padding', 0.05)
    ]


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


def test_go_to_leaves_drop_by_the_empty_gripper_route_when_told_so():
    arm = _StubArm(DROP)

    assert transfers.go_to(
        arm, 'inter', routes=EMPTY_ROUTES, max_velocity_rad_s=0.7,
        gripper_empty=True,
    )

    kind, waypoints = arm.calls[0]
    assert kind == 'baked'
    assert np.any(np.all(np.isclose(waypoints, EMPTY_VIA), axis=1))
    assert not np.any(np.all(np.isclose(waypoints, INTER + 0.1), axis=1))


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

    assert {'home', 'drop'} <= set(routes) <= set(transfers.transfer_targets())
    assert speed == pytest.approx(config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S)

    for name, route in routes.items():
        target = np.array(config.get_named_pose(transfers.route_pose(name)))

        assert np.allclose(route[0], INTER, atol=1e-5)
        assert np.allclose(route[-1], target, atol=1e-5)

        # The wrist-cable joint moves by (near enough) the direct amount.
        j7_travel = transfers.per_joint_travel_deg(route)[6]
        assert j7_travel <= abs(math.degrees(target[6] - INTER[6])) + 5.0


def test_a_route_to_a_retired_pose_is_ignored():
    # A transfers.yaml baked before RETRIEVE_0..3 were retired.
    routes = dict(ROUTES, retrieve_0=np.array([INTER, HOME]))

    assert transfers.route_for(HOME, 'inter', routes) is None
    assert transfers.route_for(DROP, 'inter', routes) is not None
