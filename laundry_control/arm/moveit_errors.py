#!/usr/bin/env python3

"""
What a MoveIt failure means, and whether it is safe to try again.

Pure functions over MoveIt result messages, so the controller's
retry rule (fall back to OMPL only when nothing can have moved) is
unit-testable without MoveIt.
"""

# moveit_msgs/MoveItErrorCodes, for logging a failure as something
# readable instead of a bare integer. "why did that move fail" is
# otherwise a trip to the message definition every time.
MOVEIT_ERROR_CODES = {
    1: 'SUCCESS',
    -1: 'FAILURE',
    -2: 'PLANNING_FAILED',
    -3: 'INVALID_MOTION_PLAN',
    -4: 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE',
    -5: 'CONTROL_FAILED',
    -6: 'UNABLE_TO_AQUIRE_SENSOR_DATA',
    -7: 'TIMED_OUT',
    -8: 'PREEMPTED',
    -10: 'START_STATE_IN_COLLISION',
    -11: 'START_STATE_VIOLATES_PATH_CONSTRAINTS',
    -12: 'GOAL_IN_COLLISION',
    -13: 'GOAL_VIOLATES_PATH_CONSTRAINTS',
    -14: 'GOAL_CONSTRAINTS_VIOLATED',
    -15: 'INVALID_GROUP_NAME',
    -16: 'INVALID_GOAL_CONSTRAINTS',
    -17: 'INVALID_ROBOT_STATE',
    -18: 'INVALID_LINK_NAME',
    -19: 'INVALID_OBJECT_NAME',
    -21: 'FRAME_TRANSFORM_FAILURE',
    -22: 'COLLISION_CHECKING_UNAVAILABLE',
    -23: 'ROBOT_STATE_STALE',
    -24: 'SENSOR_INFO_STALE',
    -25: 'COMMUNICATION_FAILURE',
    -31: 'NO_IK_SOLUTION',
}


def describe_moveit_error(code):
    """Return an error code with its name, e.g. '-2 (PLANNING_FAILED)'."""
    return f"{code} ({MOVEIT_ERROR_CODES.get(code, 'UNKNOWN')})"


# Error codes MoveIt returns before anything has moved: the request
# could not be planned. Only these justify retrying on the fallback
# pipeline. Anything else (CONTROL_FAILED, PREEMPTED, TIMED_OUT, a
# plan invalidated mid-motion, ...) can mean the arm was stopped
# part-way - by its own collision detection, the e-stop or a cancel -
# and must never be answered by commanding the motion again.
PLANNING_STAGE_ERROR_CODES = frozenset({
    -2,   # PLANNING_FAILED
    -3,   # INVALID_MOTION_PLAN
    -11,  # START_STATE_VIOLATES_PATH_CONSTRAINTS
    -12,  # GOAL_IN_COLLISION
    -13,  # GOAL_VIOLATES_PATH_CONSTRAINTS
    -14,  # GOAL_CONSTRAINTS_VIOLATED
    -15,  # INVALID_GROUP_NAME
    -16,  # INVALID_GOAL_CONSTRAINTS
    -17,  # INVALID_ROBOT_STATE (Pilz: non-zero start velocity)
    -31,  # NO_IK_SOLUTION
})


def usable_attempts(attempts, loaded, logger=None):
    """
    Drop (pipeline, planner) attempts whose pipeline move_group lacks.

    loaded None (unknown) keeps every attempt, as before. If nothing
    would be left, the attempts are kept as they are, so MoveIt's own
    error explains the failure.
    """
    if loaded is None:
        return attempts

    usable = [attempt for attempt in attempts if attempt[0] in loaded]

    if not usable:
        return attempts

    if len(usable) < len(attempts) and logger is not None:
        skipped = ', '.join(p for p, _ in attempts if p not in loaded)
        logger.info(
            f'Skipping {skipped} (not loaded in move_group: {loaded}).',
            once=True,
        )

    return usable


def failure_is_retryable(result):
    """
    Return True if a failed MoveGroup result cannot have moved the arm.

    That is a planning-stage error code, or no planned trajectory at
    all: nothing was planned, so nothing was executed. The second case
    covers codes MoveIt uses outside the planning-stage list - an
    unloaded pipeline (Pilz on the stock xArm launches) comes back as
    0, before anything moves, and must still fall back to OMPL.
    """
    return (
        result.error_code.val in PLANNING_STAGE_ERROR_CODES
        or not result.planned_trajectory.joint_trajectory.points
    )
