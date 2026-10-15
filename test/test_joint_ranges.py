"""Task joint ranges (arm/joint_ranges.py) and their use in IK requests."""

import glob
import os
from types import SimpleNamespace

from geometry_msgs.msg import Pose
from laundry_control import config
from laundry_control.arm import joint_ranges
from laundry_control.arm.kinematics import MoveItQueries
import numpy as np
import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_task_limits_sit_inside_the_hard_limits_less_the_margin():
    lower, upper = joint_ranges.task_limits()
    margin = config.JOINT_LIMIT_MARGIN_RAD

    assert np.all(lower >= np.asarray(config.JOINT_LOWER_LIMITS_RAD) + margin)
    assert np.all(upper <= np.asarray(config.JOINT_UPPER_LIMITS_RAD) - margin)
    assert np.all(lower < upper)


@pytest.mark.parametrize('joint', [0, 2, 4, 6])
def test_a_full_turn_of_a_continuous_joint_is_outside(joint):
    # J1, J3, J5 and J7 may turn +/-360 deg: a posture one full turn
    # round must not be offered.
    inter = np.array(config.get_named_pose('inter'))
    limits = joint_ranges.task_limits()

    for turn in (-2.0 * np.pi, 2.0 * np.pi):
        flipped = inter.copy()
        flipped[joint] += turn
        assert joint_ranges.outside(flipped, limits)


def _baked_postures():
    """Every named pose and every posture in the baked plans."""
    postures = [config.get_named_pose(n) for n in config.named_poses()]
    plans = os.path.join(REPO, 'scan_plans')

    transfers = os.path.join(plans, 'transfers.yaml')
    if os.path.exists(transfers):
        for route in yaml.safe_load(open(transfers))['routes'].values():
            postures += route['waypoints']
            postures += route.get('timed', {}).get('positions', [])

    endcap = os.path.join(plans, 'endcap.yaml')
    if os.path.exists(endcap):
        postures += [p['q'] for p in yaml.safe_load(open(endcap))['points']]

    for path in glob.glob(os.path.join(plans, 'retrieve.yaml')):
        for grab in yaml.safe_load(open(path)).get('grabs', []):
            postures += [grab['approach'], grab['grab']]

    return np.array(postures, dtype=float)


def test_every_baked_motion_and_named_pose_is_inside():
    limits = joint_ranges.task_limits()

    beyond = {
        i: joint_ranges.outside(q, limits)
        for i, q in enumerate(_baked_postures())
        if joint_ranges.outside(q, limits)
    }

    assert beyond == {}


def test_narrowing_around_a_posture():
    inter = np.array(config.get_named_pose('inter'))
    lower, upper = joint_ranges.task_limits(
        around=inter, within_rad=np.radians(30.0)
    )

    assert np.all(lower >= inter - np.radians(30.0) - 1e-12)
    assert np.all(upper <= inter + np.radians(30.0) + 1e-12)
    assert np.all(lower <= inter) and np.all(inter <= upper)


def test_a_start_outside_widens_only_that_side_to_the_start():
    base_lower, base_upper = joint_ranges.task_limits()
    start = np.array(config.get_named_pose('inter'))
    start[6] = np.radians(-298.0)  # J7 left there by a hand move

    lower, upper = joint_ranges.task_limits(start=start)

    assert lower[6] == pytest.approx(np.radians(-298.0))
    assert upper[6] == base_upper[6]
    assert np.allclose(lower[:6], base_lower[:6])
    assert np.allclose(upper, base_upper)


def test_constraints_message_spans_the_limits():
    lower, upper = joint_ranges.task_limits()
    message = joint_ranges.constraints((lower, upper))

    assert [c.joint_name for c in message.joint_constraints] == (
        config.JOINT_NAMES
    )
    for c, lo, hi in zip(message.joint_constraints, lower, upper):
        assert c.position - c.tolerance_below == pytest.approx(lo)
        assert c.position + c.tolerance_above == pytest.approx(hi)


class _Logger:

    def __init__(self):
        self.errors = []

    def warning(self, message, **_k):
        self.errors.append(message)


class _IkQueries(MoveItQueries):
    """MoveItQueries with the service call replaced by a canned answer."""

    JOINT_NAMES = config.JOINT_NAMES
    group_name = 'xarm7'
    flange_link = 'link7'
    base_frame = 'link_base'

    def __init__(self, answer):
        self.answer = answer
        self.requests = []
        self.logger = _Logger()

    def get_logger(self):
        return self.logger

    def _client(self, *_a):
        return None

    def _call(self, _client, request):
        self.requests.append(request)
        return SimpleNamespace(
            error_code=SimpleNamespace(val=1),
            solution=SimpleNamespace(joint_state=SimpleNamespace(
                name=list(config.JOINT_NAMES),
                position=list(self.answer),
            )),
        )


def test_the_ranges_go_into_moveits_joint_limits():
    # The shape of xarm_moveit_config's joint_limits.yaml.
    planning = {'joint_limits': {
        name: {'has_velocity_limits': True, 'max_velocity': 2.14}
        for name in config.JOINT_NAMES
    }}

    joint_ranges.add_to_moveit_limits(planning)

    lower, upper = joint_ranges.task_limits()
    for name, lo, hi in zip(config.JOINT_NAMES, lower, upper):
        assert planning['joint_limits'][name] == {
            'has_velocity_limits': True, 'max_velocity': 2.14,
            'has_position_limits': True,
            'min_position': pytest.approx(lo), 'max_position': pytest.approx(hi),
        }


def test_compute_ik_leaves_the_task_ranges_to_moveits_model():
    inter = config.get_named_pose('inter')
    arm = _IkQueries(inter)

    assert arm.compute_ik(Pose(), inter) == pytest.approx(inter)
    assert not arm.requests[0].ik_request.constraints.joint_constraints


def test_compute_ik_sends_narrower_limits_as_constraints():
    inter = config.get_named_pose('inter')
    arm = _IkQueries(inter)
    narrow = joint_ranges.task_limits(around=inter, within_rad=0.5)

    assert arm.compute_ik(Pose(), inter, limits=narrow) == pytest.approx(inter)

    sent = arm.requests[0].ik_request.constraints.joint_constraints
    assert [c.position - c.tolerance_below for c in sent] == (
        pytest.approx(list(narrow[0]))
    )
    assert [c.position + c.tolerance_above for c in sent] == (
        pytest.approx(list(narrow[1]))
    )


def test_compute_ik_refuses_an_answer_outside_the_ranges():
    flipped = list(config.get_named_pose('inter'))
    flipped[0] += 2.0 * np.pi
    arm = _IkQueries(flipped)

    assert arm.compute_ik(Pose(), config.get_named_pose('inter')) is None
    assert arm.logger.errors
