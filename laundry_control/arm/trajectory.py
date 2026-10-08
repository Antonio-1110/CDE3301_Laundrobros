#!/usr/bin/env python3

"""
Edits to planned joint trajectories (trajectory_msgs), outside the node.

The scan's helical strokes are a MoveIt tool-Z stroke with a J7 twist
added on top (add_joint7_twist); XArm7Controller.move_tool_z_with_twist
plans the stroke, applies this, and executes it.
"""

import math

from builtin_interfaces.msg import Duration
import numpy as np

from .. import config


def duration_to_seconds(duration):
    """Return a builtin_interfaces Duration in seconds."""
    return float(duration.sec) + float(duration.nanosec) * 1e-9


def seconds_to_duration(seconds):
    """Return seconds as a builtin_interfaces Duration (nanosecond rounding)."""
    nanoseconds = int(round(seconds * 1e9))

    return Duration(
        sec=nanoseconds // 1000000000,
        nanosec=nanoseconds % 1000000000,
    )


def add_joint7_twist(trajectory, twist_deg, logger):
    """
    Add a J7 rotation that tracks the stroke's own progress.

    The final J7 trajectory is

        q7(t) = q7_moveit(t) + twist * p(t)

    where p(t) in [0, 1] is the stroke's progress along its path:
    its cumulative joint-space arc length, normalised. Because
    the stroke is a straight tool-Z line, J7 therefore turns a
    fixed angle per centimetre of insertion - a true helix, which
    is what perception/coverage.py models.

    Velocities and accelerations get the matching derivatives
    (twist * dp/dt, and its time derivative), so positions,
    velocities and accelerations stay consistent. That matters on
    the real arm: its trajectory controller interpolates BETWEEN
    points using the velocities, and the xArm driver streams the
    result to the joints at 150 Hz (servo mode). Editing positions
    alone - what this used to do - left J7's velocity near zero
    at every waypoint, so J7 would stop and start at each 5 mm
    point with peaks well above its average speed. MoveIt's
    profile already starts and ends at rest, so dp/dt does too.

    MoveIt never checks the added twist against J7's limits, so
    this does: if the twisted stroke would exceed
    config.JOINT7_MAX_VELOCITY_RAD_S or
    JOINT7_MAX_ACCELERATION_RAD_S2, the whole stroke is slowed
    uniformly (same path, longer duration) until it fits.
    """
    traj = trajectory.joint_trajectory

    if not traj.points:

        logger.error(
            'Cannot twist an empty trajectory.'
        )

        return False

    joint_names = list(
        traj.joint_names
    )

    if 'joint7' not in joint_names:

        logger.error(
            'joint7 not present in trajectory.'
        )

        return False

    j7_index = joint_names.index(
        'joint7'
    )

    times = np.array([
        duration_to_seconds(point.time_from_start)
        for point in traj.points
    ])

    if times[-1] <= 0.0 or np.any(np.diff(times) <= 0.0):

        logger.error(
            'Trajectory timing is invalid; cannot add a twist.'
        )

        return False

    positions = np.array([list(p.positions) for p in traj.points])

    dof = positions.shape[1]

    if all(len(p.velocities) == dof for p in traj.points):
        velocities = np.array([list(p.velocities) for p in traj.points])
    else:
        velocities = np.gradient(positions, times, axis=0)

    has_accelerations = all(
        len(p.accelerations) == dof for p in traj.points
    )

    # Progress along the stroke: normalised joint-space arc
    # length. ds/dt is the joint-space speed, taken from MoveIt's
    # own (consistent) velocities rather than differenced.
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))]
    )
    length = float(arc[-1])

    if length <= 1e-9:

        logger.error(
            'Stroke does not move; cannot add a twist.'
        )

        return False

    twist_rad = math.radians(
        twist_deg
    )

    progress = arc / length
    speed = np.linalg.norm(velocities, axis=1)

    q7_added = twist_rad * progress
    v7_added = twist_rad * speed / length
    a7_added = np.gradient(v7_added, times)

    v7_total = velocities[:, j7_index] + v7_added

    if has_accelerations:
        a7_total = (
            np.array([p.accelerations[j7_index] for p in traj.points])
            + a7_added
        )
    else:
        a7_total = a7_added

    # Slowing a trajectory by k divides velocities by k and
    # accelerations by k^2.
    peak_velocity = float(np.abs(v7_total).max())
    peak_acceleration = float(np.abs(a7_total).max())

    slowdown = max(
        1.0,
        peak_velocity / config.JOINT7_MAX_VELOCITY_RAD_S,
        math.sqrt(
            peak_acceleration / config.JOINT7_MAX_ACCELERATION_RAD_S2
        ),
    )

    # Diagnostic baseline
    logger.info(
        'J7 synchronized twist:'
    )

    logger.info(
        f'  MoveIt baseline: '
        f'{math.degrees(positions[0, j7_index]):+.2f} -> '
        f'{math.degrees(positions[-1, j7_index]):+.2f} deg'
    )

    logger.info(
        f'  Added twist: '
        f'{twist_deg:+.2f} deg'
    )

    logger.info(
        f'  Final target: '
        f'{math.degrees(positions[-1, j7_index] + twist_rad):+.2f} deg'
    )

    logger.info(
        f'  J7 peak: {math.degrees(peak_velocity / slowdown):.0f} deg/s, '
        f'{peak_acceleration / slowdown ** 2:.1f} rad/s^2 '
        f'over {times[-1] * slowdown:.2f} s'
    )

    if slowdown > 1.0:
        logger.warning(
            f'J7 twist would peak at {math.degrees(peak_velocity):.0f} '
            f'deg/s / {peak_acceleration:.1f} rad/s^2, over the J7 '
            f'limits; stroke slowed {slowdown:.2f}x to fit '
            '(lower --velocity or --sweep to avoid this).',
            throttle_duration_sec=30.0,
        )

    for index, point in enumerate(traj.points):

        point_positions = list(point.positions)
        point_positions[j7_index] += float(q7_added[index])
        point.positions = point_positions

        point_velocities = list(velocities[index])
        point_velocities[j7_index] = float(v7_total[index])
        point.velocities = [v / slowdown for v in point_velocities]

        if has_accelerations:
            point_accelerations = list(point.accelerations)
            point_accelerations[j7_index] = float(a7_total[index])
            point.accelerations = [
                a / slowdown ** 2 for a in point_accelerations
            ]

        point.time_from_start = seconds_to_duration(times[index] * slowdown)

    return True
