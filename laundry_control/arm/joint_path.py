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


def within_joint_limits(joints, lower, upper, margin_rad=0.0):
    """Return True if every joint is at least margin_rad inside [lower, upper]."""
    joints = np.asarray(joints, dtype=np.float64)

    return bool(
        np.all(joints >= np.asarray(lower) + margin_rad)
        and np.all(joints <= np.asarray(upper) - margin_rad)
    )


def limits_no_closer_than(start, lower, upper, margin_rad):
    """
    Return (lower, upper) bounds that keep margin_rad, or what start has.

    A joint at least margin_rad inside its limits keeps that margin. One
    already inside it (but within the hard limits) gets its start angle
    as the bound on that side: it may stay there or move away from the
    limit, never closer. Without this, an arm left inside the margin
    could not move at all - every state of every move, including its
    first, would fail the check.
    """
    start = np.asarray(start, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)

    return (
        np.maximum(low, np.minimum(low + margin_rad, start)),
        np.minimum(high, np.maximum(high - margin_rad, start)),
    )


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


def _s_curve_phases(length, v_max, a_max, j_max):
    """
    Return [(duration, jerk)] for a rest-to-rest jerk-limited move of `length`.

    The classic double-S: jerk up, hold the acceleration, jerk down to
    cruise speed, cruise, then the mirror image. When the move is too
    short to reach v_max, the peak speed is lowered (bisection) and
    there is no cruise; when a_max is not reached on the way to that
    speed, there is no constant-acceleration phase either.
    """

    def accel_phase(v_peak):
        if v_peak * j_max <= a_max ** 2:
            t_jerk = np.sqrt(v_peak / j_max)
            t_hold = 0.0
        else:
            t_jerk = a_max / j_max
            t_hold = v_peak / a_max - t_jerk

        # The phase is point-symmetric in velocity: average v_peak / 2.
        return t_jerk, t_hold, v_peak * (2.0 * t_jerk + t_hold) / 2.0

    t_jerk, t_hold, distance = accel_phase(v_max)
    v_peak = v_max

    if 2.0 * distance > length:
        low, high = 0.0, v_max

        for _ in range(60):
            v_peak = (low + high) / 2.0

            if 2.0 * accel_phase(v_peak)[2] > length:
                high = v_peak
            else:
                low = v_peak

        t_jerk, t_hold, distance = accel_phase(v_peak)

    t_cruise = max(0.0, (length - 2.0 * distance) / v_peak)

    accelerate = [(t_jerk, j_max), (t_hold, 0.0), (t_jerk, -j_max)]
    decelerate = [(t_jerk, -j_max), (t_hold, 0.0), (t_jerk, j_max)]

    return accelerate + [(t_cruise, 0.0)] + decelerate


def s_curve(length, v_max, a_max, j_max, sample_dt=5e-4):
    """
    Return (times, s, ds/dt) of a rest-to-rest jerk-limited move.

    Sampled every sample_dt (exactly, per constant-jerk phase); s runs
    from 0 to `length`. Speed, acceleration and jerk never exceed
    v_max, a_max and j_max.
    """
    times, positions, speeds = [0.0], [0.0], [0.0]
    t = pos = vel = acc = 0.0

    for duration, jerk in _s_curve_phases(length, v_max, a_max, j_max):
        if duration <= 0.0:
            continue

        steps = max(1, int(np.ceil(duration / sample_dt)))
        dt = duration / steps

        for _ in range(steps):
            pos += vel * dt + acc * dt ** 2 / 2.0 + jerk * dt ** 3 / 6.0
            vel += acc * dt + jerk * dt ** 2 / 2.0
            acc += jerk * dt
            t += dt
            times.append(t)
            positions.append(pos)
            speeds.append(vel)

    positions = np.array(positions)
    # Undo the bisection's last ~1e-15 of mismatch.
    scale = length / positions[-1] if positions[-1] > 0.0 else 1.0

    return np.array(times), positions * scale, np.array(speeds) * scale


def _time_s_curve(segment, max_velocity_rad_s, max_acceleration, max_jerk):
    """Like time_path() for one straight segment, but with an s_curve()."""
    arc = np.concatenate(
        [[0.0], np.cumsum(np.abs(np.diff(segment, axis=0)).max(axis=1))]
    )
    length = float(arc[-1])

    curve_t, curve_s, curve_v = s_curve(
        length, max_velocity_rad_s, max_acceleration, max_jerk
    )

    times = np.interp(arc, curve_s, curve_t)
    speed = np.interp(times, curve_t, curve_v)

    # Straight segment: every joint moves in proportion to the arc.
    direction = (segment[-1] - segment[0]) / length
    velocities = speed[:, None] * direction[None, :]
    velocities[0] = 0.0
    velocities[-1] = 0.0

    return times, velocities


def time_stop_at_each(
    waypoints,
    max_velocity_rad_s,
    step_rad=DEFAULT_CHECK_STEP_RAD,
    max_acceleration=None,
    max_jerk=None,
):
    """
    Densify and time a polyline that comes to rest at every waypoint.

    Returns (dense_waypoints, times, velocities). Each straight
    segment gets its own profile from rest to rest, so the arm follows
    exactly the straight joint-space lines that were collision-checked
    - no corner-cutting at the vias - at the cost of a brief stop at
    each one.

    The profile is minimum-jerk unless max_acceleration and max_jerk
    are given, then a jerk-limited S-curve (s_curve) that cruises at
    max_velocity_rad_s. Min-jerk averages only 1/1.875 (53%) of its
    peak speed, so for the same top speed the S-curve is faster on any
    segment long enough to cruise. The stops cost nothing extra with
    min-jerk - its duration is proportional to the distance - but
    each costs a speed-up and a slow-down with the S-curve.
    """
    cruise = max_acceleration is not None and max_jerk is not None
    waypoints = np.asarray(waypoints, dtype=np.float64)

    dense = [waypoints[:1]]
    times = [np.zeros(1)]
    velocities = [np.zeros((1, waypoints.shape[1]))]
    elapsed = 0.0

    for start, end in zip(waypoints[:-1], waypoints[1:]):
        if np.abs(end - start).max() < 1e-9:
            continue

        segment = densify([start, end], step_rad)

        if cruise:
            seg_times, seg_velocities = _time_s_curve(
                segment, max_velocity_rad_s, max_acceleration, max_jerk
            )
        else:
            seg_times, seg_velocities = time_path(segment, max_velocity_rad_s)

        dense.append(segment[1:])
        times.append(elapsed + seg_times[1:])
        velocities.append(seg_velocities[1:])

        elapsed += float(seg_times[-1])

    return (
        np.concatenate(dense),
        np.concatenate(times),
        np.concatenate(velocities),
    )
