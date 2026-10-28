"""Direct routes between a sweep grab and DROP (arm/direct.py)."""

from laundry_control import config
from laundry_control.arm import direct, transfers
import numpy as np

INTER = np.array(config.get_named_pose('inter'))
DROP = np.array(config.DROP)
GRAB = np.array(config.get_named_pose('grab_01'))
VIA = INTER + 0.1
EMPTY_VIA = INTER - 0.1
ROUTES = {
    'drop': np.array([INTER, VIA, DROP]),
    'drop_return': np.array([INTER, EMPTY_VIA, DROP]),
    'grab_01': np.array([INTER, GRAB]),
}


def test_names_round_trip():
    assert direct.split(direct.name('grab_01', 'drop')) == ('grab_01', 'drop')
    assert direct.split('grab_01..drop') is None
    assert direct.split('grab_01>home') is None


def test_candidates_leave_along_the_grab_route_and_join_later():
    found = direct.candidates(ROUTES['grab_01'], ROUTES['drop'])

    times = [seconds for seconds, _path, _exit in found]
    assert times == sorted(times)

    for _seconds, path, exit_index in found:
        assert np.allclose(path[0], GRAB) and np.allclose(path[-1], DROP)
        # The exit lies on the straight line from the grab to INTER.
        exit_pose = path[exit_index]
        u = (exit_pose - GRAB) / (INTER - GRAB)
        assert np.allclose(u, u[0], atol=0.02) and 0.0 < u[0] < 1.0
        # INTER itself is skipped.
        assert not np.any(np.all(np.isclose(path, INTER), axis=1))


def _with_direct():
    return dict(
        ROUTES,
        **{
            'grab_01>drop': np.array([GRAB, 0.5 * (GRAB + INTER), VIA, DROP]),
            'grab_01>drop_return': np.array(
                [GRAB, 0.5 * (GRAB + INTER), EMPTY_VIA, DROP]
            ),
        },
    )


def test_a_direct_route_is_taken_from_the_grab_to_drop():
    pieces = transfers._pieces(GRAB, 'drop', _with_direct())

    assert pieces == [('grab_01>drop', False)]


def test_the_empty_way_back_runs_its_direct_route_backwards():
    routes = _with_direct()

    assert transfers._pieces(DROP, 'grab_01', routes, gripper_empty=True) == [
        ('grab_01>drop_return', True)
    ]
    # Not known to be empty: back through INTER as before.
    assert transfers._pieces(DROP, 'grab_01', routes) == [
        ('drop', True), ('grab_01', False)
    ]

    path = transfers.route_for(DROP, 'grab_01', routes, gripper_empty=True)
    assert np.allclose(path[0], DROP) and np.allclose(path[-1], GRAB)
    assert np.any(np.all(np.isclose(path, EMPTY_VIA), axis=1))


def test_without_a_direct_route_the_move_goes_through_inter():
    assert transfers._pieces(GRAB, 'drop', ROUTES) == [
        ('grab_01', True), ('drop', False)
    ]


def test_direct_routes_are_not_mistaken_for_routes_from_another_inter():
    assert transfers.from_other_inter(_with_direct()) == []


def test_stored_direct_routes_load_while_their_poses_hold(tmp_path):
    path = str(tmp_path / 'transfers.yaml')
    transfers.save(path, ROUTES, 0.785)
    transfers.save_direct(path, {
        'grab_01>drop': {
            'from': 'grab_01', 'to': 'drop', 'join': 'drop', 'exit': 1,
            'padding_m': {'arm_links': 0.03, 'gripper': 0.05},
            'waypoints': _with_direct()['grab_01>drop'],
        },
        'grab_01>drop_return': {
            'from': 'grab_01', 'to': 'drop', 'join': 'drop_return',
            'exit': 1, 'padding_m': {'arm_links': 0.03, 'gripper': 0.01},
            'waypoints': _with_direct()['grab_01>drop_return'] + 0.01,
        },
    })

    # The second no longer starts at grab_01: left out.
    assert set(transfers.load_direct(path)) == {'grab_01>drop'}

    # Keeping only one, without a timing, removes the other.
    transfers.save_timings(path, {}, direct={'grab_01>drop_return': None})
    assert set(transfers._read(path)['direct']) == {'grab_01>drop_return'}


class _FkArm:
    """compute_fk for a claw `metres_per_rad` from the base on J1 alone."""

    def __init__(self, metres_per_rad):
        self.radius = metres_per_rad

    def compute_fk(self, joints):
        angle = joints[0]
        return (
            np.array([self.radius * np.cos(angle),
                      self.radius * np.sin(angle), 0.0]),
            np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]),
        )


def test_a_straight_leg_is_slowed_to_keep_the_claw_under_its_cap():
    from laundry_control.arm.joint_path import densify

    states = densify([np.zeros(7), np.r_[1.5, np.zeros(6)]])

    near = transfers._time_straight(_FkArm(0.05), states)
    far = transfers._time_straight(_FkArm(2.0), states)
    unknown = transfers._time_straight(object(), states)

    assert near[2] < config.TRANSFER_MAX_TIP_SPEED_M_S
    assert far[2] == config.TRANSFER_MAX_TIP_SPEED_M_S
    assert far[0][-1] > near[0][-1]
    # Without forward kinematics: timed as before (min-jerk, 45 deg/s).
    assert unknown[2] is None
    assert unknown[0][-1] > near[0][-1]
