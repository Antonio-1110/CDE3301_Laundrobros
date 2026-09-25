#!/usr/bin/env python3

"""
Planner-free joint-space paths: densify, time, and turn into trajectories.

Everything here is deterministic numpy - the same input always gives
the same trajectory - which is the point: a motion built this way
(and collision-checked state by state, see XArm7Controller.
move_joints_linear) repeats exactly every run, unlike a sampling
planner such as OMPL, and without depending on Pilz being loaded.

Timing uses a minimum-jerk profile per SEGMENT,

    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,   tau = t / T,

which starts and ends at zero velocity AND zero acceleration, so
every segment boundary is a smooth stop. A path is split into
segments wherever it reverses direction (the end of an out-and-back
spoke, say), since passing through such a point at speed would mean
an instantaneous velocity flip. T is chosen so the fastest joint
peaks exactly at max_velocity: min-jerk peaks at 1.875 L / T for a
segment of length L.
"""

import numpy as np

# 1.875 = peak of ds/dtau for the minimum-jerk profile.
MIN_JERK_PEAK = 1.875

# Dense interpolation step for collision checking: the largest joint
# change allowed between two checked states.
DEFAULT_CHECK_STEP_RAD = np.deg2rad(1.0)


def min_jerk(tau):
    """Return (s, ds/dtau) of the minimum-jerk profile at tau in [0, 1]."""
    tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4
    return s, ds


def densify(waypoints, step_rad=DEFAULT_CHECK_STEP_RAD):
    """
    Return waypoints with straight joint-space steps of at most step_rad.

    Every original waypoint is kept, so a densified path passes
    through exactly the states it was built from.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)

    out = [waypoints[0]]

    for start, end in zip(waypoints[:-1], waypoints[1:]):
        count = max(1, int(np.ceil(np.abs(end - start).max() / step_rad)))

        for t in np.linspace(0.0, 1.0, count + 1)[1:]:
            out.append(start + (end - start) * t)

    return np.array(out)


def split_at_reversals(waypoints):
    """
    Return index ranges [(i0, i1), ...] of monotonic stretches of a path.

    A new segment starts wherever consecutive steps point in opposing
    directions in joint space (negative dot product), and at
    zero-length steps.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)

    steps = np.diff(waypoints, axis=0)

    bounds = [0]

    for index in range(1, steps.shape[0]):
        if float(steps[index - 1] @ steps[index]) <= 0.0:
            bounds.append(index)

    bounds.append(waypoints.shape[0] - 1)

    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def time_path(waypoints, max_velocity_rad_s):
    """
    Return (times, velocities) for waypoints, min-jerk per segment.

    times[0] is 0; velocities are joint velocities at each waypoint,
    zero at every segment boundary. Along a segment the path is
    parameterised by cumulative max-joint distance, so the fastest
    joint peaks at max_velocity_rad_s.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)

    n = waypoints.shape[0]

    times = np.zeros(n)
    velocities = np.zeros_like(waypoints)

    elapsed = 0.0

    for a, b in split_at_reversals(waypoints):
        segment = waypoints[a:b + 1]

        arc = np.concatenate(
            [[0.0], np.cumsum(np.abs(np.diff(segment, axis=0)).max(axis=1))]
        )
        length = float(arc[-1])

        if length <= 0.0:
            times[a + 1:b + 1] = elapsed
            continue

        duration = MIN_JERK_PEAK * length / max_velocity_rad_s

        # Invert s(tau) on a fine grid: s is monotonic on [0, 1].
        grid = np.linspace(0.0, 1.0, 2001)
        s_grid, _ = min_jerk(grid)
        tau = np.interp(arc / length, s_grid, grid)

        _s, ds = min_jerk(tau)

        # dq/darc by finite differences along the segment.
        dq = np.gradient(segment, arc, axis=0)

        times[a:b + 1] = elapsed + tau * duration
        velocities[a:b + 1] = dq * (ds * length / duration)[:, None]

        velocities[a] = 0.0
        velocities[b] = 0.0

        elapsed += duration

    return times, velocities
