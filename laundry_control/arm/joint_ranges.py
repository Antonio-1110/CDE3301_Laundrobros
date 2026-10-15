#!/usr/bin/env python3

"""
The joint ranges automatic motion keeps to (config.JOINT_TASK_RANGES_DEG).

task_limits() gives the (lower, upper) bounds as arrays in radians,
optionally narrowed around a reference posture or widened to include
where the arm starts. constraints() turns them into the MoveIt
message the IK service and the planners take, so the solvers search
inside them instead of their answers being filtered afterwards.
"""

import math

import numpy as np

from .. import config


def task_limits(start=None, around=None, within_rad=None):
    """
    Return (lower, upper) in radians for automatic motion.

    The task ranges, inside the hard limits less JOINT_LIMIT_MARGIN_RAD.
    around/within_rad narrow each joint to within_rad of `around` (the
    same posture, moved a little). start widens a joint that starts
    outside to its start angle on that side: it may stay there or come
    back in, never go further out - so a move from where a hand move
    left the arm is still possible.
    """
    task = np.radians(np.asarray(config.JOINT_TASK_RANGES_DEG, dtype=float))
    margin = config.JOINT_LIMIT_MARGIN_RAD

    lower = np.maximum(
        task[:, 0], np.asarray(config.JOINT_LOWER_LIMITS_RAD) + margin
    )
    upper = np.minimum(
        task[:, 1], np.asarray(config.JOINT_UPPER_LIMITS_RAD) - margin
    )

    if around is not None:
        around = np.asarray(around, dtype=float)
        lower = np.maximum(lower, around - within_rad)
        upper = np.minimum(upper, around + within_rad)

    if start is not None:
        start = np.asarray(start, dtype=float)
        lower = np.minimum(lower, start)
        upper = np.maximum(upper, start)

    return lower, upper


def outside(joints, limits):
    """Return 'J7 at -290.0 deg, allowed -270.0 .. +30.0' for each joint outside."""
    return [
        f'J{i + 1} at {math.degrees(q):+.1f} deg, allowed '
        f'{math.degrees(lo):+.1f} .. {math.degrees(hi):+.1f}'
        for i, (q, lo, hi) in enumerate(zip(joints, *limits))
        if not lo - 1e-9 <= q <= hi + 1e-9
    ]


def add_to_moveit_limits(planning):
    """
    Put the task ranges into MoveIt's robot model, in place.

    planning is the robot_description_planning dict the manufacturer's
    MoveItConfigsBuilder makes from xarm_moveit_config's
    joint_limits.yaml. Each joint gets has_position_limits with
    task_limits() as its min/max_position, which MoveIt applies over
    the URDF's +/-360 deg: KDL then clips its iterations and random
    restarts to them, and OMPL samples only inside them, so a flipped
    posture is never found in the first place. Used by
    laundry_bringup.launch.py for move_group and RViz.
    """
    lower, upper = task_limits()
    limits = planning.setdefault('joint_limits', {})

    for name, lo, hi in zip(config.JOINT_NAMES, lower, upper):
        limits.setdefault(name, {}).update(
            has_position_limits=True,
            min_position=float(lo),
            max_position=float(hi),
        )

    return planning


def constraints(limits, joint_names=None):
    """Return a moveit_msgs Constraints keeping each joint inside limits."""
    from moveit_msgs.msg import Constraints, JointConstraint

    message = Constraints(name='task_joint_ranges')

    for name, lo, hi in zip(
        joint_names or config.JOINT_NAMES, *limits
    ):
        message.joint_constraints.append(JointConstraint(
            joint_name=name,
            position=float(lo + hi) / 2.0,
            tolerance_above=float(hi - lo) / 2.0,
            tolerance_below=float(hi - lo) / 2.0,
            weight=1.0,
        ))

    return message
