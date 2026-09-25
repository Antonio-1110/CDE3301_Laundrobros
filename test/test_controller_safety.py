"""XArm7Controller safety behaviour, exercised on stubs (no MoveIt)."""

import math

from builtin_interfaces.msg import Duration
from laundry_control import config
from laundry_control.arm.controller import XArm7Controller
from moveit_msgs.msg import RobotTrajectory
import numpy as np
import pytest
from trajectory_msgs.msg import JointTrajectoryPoint


class _Logger:

    def info(self, *_a, **_k):
        pass

    debug = warning = error = info


class _Self:
    """Just enough of an XArm7Controller for the unbound methods."""

    JOINT_NAMES = list(config.JOINT_NAMES)
    _duration_to_seconds = staticmethod(XArm7Controller._duration_to_seconds)

    def get_logger(self):
        return _Logger()


# ---------------------------------------------------------------
# Fix 2: fall back to OMPL only when the first attempt failed to PLAN
# ---------------------------------------------------------------

def _move_joints_with(outcomes):
    arm = _Self()
    arm.attempts = []

    def once(joints, **kwargs):
        arm.attempts.append(kwargs['pipeline_id'])
        return outcomes[len(arm.attempts) - 1]

    arm._move_joints_once = once

    ok = XArm7Controller.move_joints(arm, [0.0] * 7)

    return ok, arm.attempts


def test_planning_failure_falls_back_to_ompl():
    ok, attempts = _move_joints_with(
        [(False, '-2 (PLANNING_FAILED)', True), (True, None, False)]
    )

    assert ok
    assert attempts == [config.PILZ_PIPELINE_ID, config.OMPL_PIPELINE_ID]


def test_unplanned_failures_are_retryable_whatever_the_code():
    from moveit_msgs.action import MoveGroup
    from trajectory_msgs.msg import JointTrajectoryPoint

    from laundry_control.arm.controller import failure_is_retryable

    unloaded_pipeline = MoveGroup.Result()
    unloaded_pipeline.error_code.val = 0
    assert failure_is_retryable(unloaded_pipeline)

    stopped = MoveGroup.Result()
    stopped.error_code.val = -4  # CONTROL_FAILED, after execution began
    stopped.planned_trajectory.joint_trajectory.points = [
        JointTrajectoryPoint()
    ]
    assert not failure_is_retryable(stopped)

    stopped.error_code.val = -2  # PLANNING_FAILED
    assert failure_is_retryable(stopped)


def test_execution_failure_is_never_retried():
    ok, attempts = _move_joints_with([(False, '-5 (CONTROL_FAILED)', False)])

    assert not ok
    assert attempts == [config.PILZ_PIPELINE_ID]


# ---------------------------------------------------------------
# Fix 1: Ctrl+C stops the arm at the trajectory controller
# ---------------------------------------------------------------

class _Future:

    def __init__(self, result=None, done=True):
        self._result = result
        self._done = done

    def done(self):
        return self._done

    def result(self):
        return self._result


class _GoalHandle:

    accepted = True

    def __init__(self):
        self.cancelled = False
        self.result_future = _Future(done=False)

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancelled = True
        # MoveIt finishes the goal once the controller has stopped.
        self.result_future._done = True
        return _Future()


class _Client:

    def __init__(self, handle):
        self.handle = handle

    def send_goal_async(self, goal):
        return _Future(self.handle)


class _CancelClient:

    def __init__(self):
        self.calls = 0

    def service_is_ready(self):
        return True

    def call_async(self, request):
        self.calls += 1
        response = type('Response', (), {'goals_canceling': ['goal']})()
        return _Future(response)


def test_ctrl_c_while_moving_stops_the_controller_and_the_goal():
    arm = _Self()
    handle = _GoalHandle()
    arm._controller_cancel_client = _CancelClient()
    interrupted = []

    def spin_until_done(future, timeout_sec=None):
        if not future.done() and not interrupted:
            # Ctrl+C lands while waiting for the motion to finish.
            interrupted.append(True)
            raise KeyboardInterrupt
        return future.done()

    arm._spin_until_done = spin_until_done
    arm._cancel_controller_goals = (
        lambda: XArm7Controller._cancel_controller_goals(arm)
    )
    arm._stop_after_interrupt = (
        lambda *args: XArm7Controller._stop_after_interrupt(arm, *args)
    )

    with pytest.raises(KeyboardInterrupt):
        XArm7Controller._run_goal(arm, _Client(handle), object(), 'test move')

    assert arm._controller_cancel_client.calls >= 1
    assert handle.cancelled


# ---------------------------------------------------------------
# Fix 3: the J7 twist keeps positions, velocities and timing consistent
# ---------------------------------------------------------------

def _stroke(duration=2.0, count=7):
    """Return a straight 7-joint stroke, at rest at both ends."""
    times = np.linspace(0.0, duration, count)
    s = 0.5 - 0.5 * np.cos(np.pi * times / duration)
    ds = 0.5 * np.pi / duration * np.sin(np.pi * times / duration)
    dds = 0.5 * (np.pi / duration) ** 2 * np.cos(np.pi * times / duration)
    delta = np.array([0.02, -0.03, 0.04, 0.01, -0.02, 0.03, 0.001])

    trajectory = RobotTrajectory()
    trajectory.joint_trajectory.joint_names = list(config.JOINT_NAMES)

    for t, si, dsi, ddsi in zip(times, s, ds, dds):
        point = JointTrajectoryPoint()
        point.positions = list(np.array(config.INTER) + si * delta)
        point.velocities = list(dsi * delta)
        point.accelerations = list(ddsi * delta)
        nanoseconds = int(round(t * 1e9))
        point.time_from_start = Duration(
            sec=nanoseconds // 1000000000, nanosec=nanoseconds % 1000000000
        )
        trajectory.joint_trajectory.points.append(point)

    return trajectory


def _arrays(trajectory):
    points = trajectory.joint_trajectory.points
    times = np.array([XArm7Controller._duration_to_seconds(p.time_from_start)
                      for p in points])
    q = np.array([p.positions for p in points])
    v = np.array([p.velocities for p in points])
    return times, q, v


def test_twist_is_added_in_full_and_starts_and_ends_at_rest():
    trajectory = _stroke()
    _t0, q0, _v0 = _arrays(trajectory)

    assert XArm7Controller._add_joint7_twist(_Self(), trajectory, 30.0)

    times, q, v = _arrays(trajectory)

    assert q[-1, 6] - q0[-1, 6] == pytest.approx(math.radians(30.0))
    assert np.allclose(q[:, :6], q0[:, :6])
    assert v[0, 6] == pytest.approx(0.0, abs=1e-9)
    assert v[-1, 6] == pytest.approx(0.0, abs=1e-9)
    # Between the ends J7 actually moves - velocities are not left at
    # MoveIt's near-zero J7 values, the bug on the real arm.
    assert np.all(v[1:-1, 6] > 0.0)
    assert times[-1] == pytest.approx(2.0)


def test_twist_velocities_match_positions():
    trajectory = _stroke(count=41)

    assert XArm7Controller._add_joint7_twist(_Self(), trajectory, 30.0)

    times, q, v = _arrays(trajectory)

    # The trajectory controller interpolates with these velocities, so
    # they must agree with the positions' own rate of change.
    differenced = np.gradient(q[:, 6], times)
    assert np.allclose(v[1:-1, 6], differenced[1:-1], atol=0.02)


def test_twist_too_fast_for_j7_slows_the_whole_stroke():
    trajectory = _stroke(duration=0.5)

    assert XArm7Controller._add_joint7_twist(_Self(), trajectory, 150.0)

    times, q, v = _arrays(trajectory)

    assert times[-1] > 0.5
    assert np.abs(v[:, 6]).max() <= config.JOINT7_MAX_VELOCITY_RAD_S + 1e-9
    # Same path, only slower.
    assert q[-1, 6] - config.INTER[6] == pytest.approx(
        math.radians(150.0) + 0.001
    )
