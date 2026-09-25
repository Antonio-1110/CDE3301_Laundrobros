"""Tests for arm/joint_path.py: planner-free joint-space paths."""

import math

from laundry_control.arm.joint_path import (
    densify,
    min_jerk,
    split_at_reversals,
    time_path,
)
import numpy as np
import pytest


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
