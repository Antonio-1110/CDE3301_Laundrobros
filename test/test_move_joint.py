"""`laundry move joint1 .. joint7`: one joint alone, straight and checked."""

import math

from laundry_control import config
from laundry_control.arm.joint_path import limits_no_closer_than
from laundry_control.cli import arm as cli_arm
from laundry_control.cli import build_parser
import pytest

CURRENT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


class _Arm:

    def __init__(self, current=CURRENT, collides=False):
        self.current = list(current)
        self.collides = collides
        self.calls = []

    def get_current_joints(self):
        return list(self.current)

    def move_joints_linear(self, target, time_scale=1.0, **kwargs):
        self.calls.append(('linear', list(target), time_scale))
        self.kwargs = kwargs
        return not self.collides

    def move_joints(self, *_args, **_kwargs):
        self.calls.append(('planner',))
        return True


@pytest.fixture
def run(monkeypatch):
    """Run `laundry move ...` against a stub arm; return (status, arm)."""

    def _run(argv, arm=None):
        arm = arm or _Arm()

        class _Session:

            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return arm

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(cli_arm, 'RosSession', _Session)
        args = build_parser().parse_args(['move'] + argv)
        return args.func(args), arm

    return _run


def test_relative_turn_moves_only_that_joint(run):
    status, arm = run(['joint3', '-10'])

    assert status == 0
    (kind, target, time_scale), = arm.calls
    assert kind == 'linear'
    assert target[2] == pytest.approx(0.3 - math.radians(10))
    assert [q for i, q in enumerate(target) if i != 2] == [
        q for i, q in enumerate(CURRENT) if i != 2
    ]
    assert time_scale == cli_arm.SINGLE_JOINT_DEFAULT_SPEED


def test_to_sets_the_angle_and_speed_scales_the_move(run):
    status, arm = run(['joint7', '90', '--to', '--speed', '0.3'])

    assert status == 0
    (_kind, target, time_scale), = arm.calls
    assert target[6] == pytest.approx(math.radians(90))
    assert time_scale == 0.3


@pytest.mark.parametrize('joint', range(1, 8))
def test_every_joint_has_a_command(run, joint):
    status, arm = run([f'joint{joint}', '1'])

    assert status == 0
    assert arm.calls[0][1][joint - 1] == pytest.approx(
        CURRENT[joint - 1] + math.radians(1)
    )


def test_never_uses_the_planner(run):
    _status, arm = run(['joint6', '5'])

    assert ('planner',) not in arm.calls


def test_out_of_range_absolute_is_refused_before_ros(run, monkeypatch):
    monkeypatch.setattr(
        cli_arm, 'RosSession',
        lambda *a, **k: pytest.fail('connected to ROS for a bad target'),
    )
    args = build_parser().parse_args(['move', 'joint2', '130', '--to'])

    # J2's upper limit is 120 deg.
    assert args.func(args) == 2


def test_out_of_range_relative_is_refused_without_moving(run):
    # J4 sits at 0.4 rad (~23 deg); its lower limit is -11 deg.
    status, arm = run(['joint4', '-40'])

    assert status == 1
    assert arm.calls == []


def test_a_colliding_move_reports_failure(run):
    status, _arm = run(['joint1', '20'], arm=_Arm(collides=True))

    assert status == 1


def test_bad_speed_is_refused(run):
    status, arm = run(['joint1', '5', '--speed', '1.5'])

    assert status == 2
    assert arm.calls == []


def test_the_range_keeps_the_limit_margin():
    low, high = cli_arm._limits_deg(1)

    margin = math.degrees(config.JOINT_LIMIT_MARGIN_RAD)
    assert low == pytest.approx(math.degrees(config.JOINT_LOWER_LIMITS_RAD[1]) + margin)
    assert high == pytest.approx(math.degrees(config.JOINT_UPPER_LIMITS_RAD[1]) - margin)


def test_single_joint_moves_may_leave_the_limit_margin(run):
    _status, arm = run(['joint1', '5'])

    assert arm.kwargs == {'no_closer_to_limits': True}


# J2's hard limit is 120 deg; with the 2 deg margin, 118 is the most.
AT_J2_MARGIN = [1.84, math.radians(118.1), -2.52, 0.32, 0.16, 0.95, -5.2]


def _bounds(start):
    lower, upper = limits_no_closer_than(
        start, config.JOINT_LOWER_LIMITS_RAD, config.JOINT_UPPER_LIMITS_RAD,
        config.JOINT_LIMIT_MARGIN_RAD,
    )
    return lower, upper


def test_a_joint_inside_its_margin_keeps_its_angle_as_the_bound():
    lower, upper = _bounds(AT_J2_MARGIN)

    assert upper[1] == pytest.approx(math.radians(118.1))
    # Every other joint keeps the full margin.
    assert upper[0] == pytest.approx(
        config.JOINT_UPPER_LIMITS_RAD[0] - config.JOINT_LIMIT_MARGIN_RAD
    )
    assert lower[1] == pytest.approx(
        config.JOINT_LOWER_LIMITS_RAD[1] + config.JOINT_LIMIT_MARGIN_RAD
    )


def test_the_hard_limit_still_holds():
    start = list(AT_J2_MARGIN)
    start[1] = math.radians(125.0)  # Beyond the hard limit already.

    _lower, upper = _bounds(start)

    assert upper[1] == pytest.approx(config.JOINT_UPPER_LIMITS_RAD[1])


def test_other_joints_may_move_while_one_sits_in_its_margin(run):
    status, arm = run(['joint1', '10'], arm=_Arm(current=AT_J2_MARGIN))

    assert status == 0
    assert arm.calls[0][1][1] == pytest.approx(math.radians(118.1))


def test_a_joint_in_its_margin_may_move_out_but_not_further_in(run):
    out_status, _ = run(['joint2', '-0.05'], arm=_Arm(current=AT_J2_MARGIN))
    in_status, arm = run(['joint2', '0.5'], arm=_Arm(current=AT_J2_MARGIN))

    assert out_status == 0
    assert in_status == 1 and arm.calls == []
