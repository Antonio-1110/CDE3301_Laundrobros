"""Tests for arm/joint_path.py: planner-free joint-space paths."""

import math

from laundry_control.arm.joint_path import (
    densify,
    min_jerk,
    split_at_reversals,
    time_path,
    within_joint_limits,
)
import numpy as np
import pytest


def test_within_joint_limits_keeps_the_margin():
    lower, upper = [-1.0] * 7, [1.0] * 7
    margin = math.radians(2.0)

    assert within_joint_limits([0.0] * 7, lower, upper, margin)
    assert within_joint_limits([1.0] * 7, lower, upper)
    assert not within_joint_limits([1.0] + [0.0] * 6, lower, upper, margin)
    assert not within_joint_limits([0.0] * 6 + [-1.01], lower, upper)


def test_drop_via_that_tripped_c23_is_outside_the_limits():
    from laundry_control import config

    # INTER -> DROP via baked 2026-09-29: J2 at 125.3 deg, limit 120.
    via = np.radians([17.5, 125.3, -193.4, 25.0, -1.0, 55.5, -167.4])

    assert not within_joint_limits(
        via,
        config.JOINT_LOWER_LIMITS_RAD,
        config.JOINT_UPPER_LIMITS_RAD,
        config.JOINT_LIMIT_MARGIN_RAD,
    )


def test_min_jerk_starts_and_ends_at_rest():
    s, ds = min_jerk([0.0, 0.5, 1.0])
    assert s == pytest.approx([0.0, 0.5, 1.0])
    assert ds[0] == pytest.approx(0.0) and ds[2] == pytest.approx(0.0)
    assert ds[1] == pytest.approx(1.875)


def test_densify_keeps_waypoints_and_limits_steps():
    waypoints = np.array([[0.0] * 7, [0.1] + [0.0] * 6, [0.1, 0.2] + [0.0] * 5])
    dense = densify(waypoints, step_rad=math.radians(1.0))

    assert np.abs(np.diff(dense, axis=0)).max() <= math.radians(1.0) + 1e-12
    for waypoint in waypoints:
        assert np.any(np.all(np.isclose(dense, waypoint), axis=1))


def test_split_at_reversals_finds_out_and_back():
    out = [[x, 0, 0, 0, 0, 0, 0] for x in np.linspace(0.0, 1.0, 5)]
    back = out[::-1][1:]

    assert split_at_reversals(np.array(out + back)) == [(0, 4), (4, 8)]


def test_time_path_respects_speed_and_stops_at_reversals():
    out = [[x, 0.5 * x, 0, 0, 0, 0, 0] for x in np.linspace(0.0, 1.0, 50)]
    path = densify(np.array(out + out[::-1][1:]))

    vmax = 0.5
    times, velocities = time_path(path, vmax)

    assert np.all(np.diff(times) >= 0.0)
    assert np.abs(velocities).max() <= vmax * 1.02
    assert np.abs(velocities).max() >= vmax * 0.9

    turn = int(np.argmax(path[:, 0]))
    assert np.allclose(velocities[[0, turn, -1]], 0.0)
