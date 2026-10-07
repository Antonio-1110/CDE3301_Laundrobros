#!/usr/bin/env python3

"""
How the scan's strokes divide the insertion into equal steps.

The scan moves straight in from INTER by entry_depth (the sensor is
then inside the bucket), then swings J7 through one stroke per step
down to depth. The strokes are spread EVENLY over that distance, as
few as keep them at most max_step apart:

    count = ceil((depth - entry_depth) / max_step)
    step  = (depth - entry_depth) / count

so any entry depth works, and max_step sets the scan's density: the
largest axial distance between consecutive sweeps.

Pure arithmetic, no ROS: scan.pattern (the arm) and perception.coverage
(the simulator) share it, so they can never disagree on the path.
"""

import math

# The largest allowed stroke spacing; `--step` above this is rejected,
# not clamped. Spacing over the 3 cm default leaves gaps the beam never
# sees, so a coarse scan is only for finding the bigger items fast: in
# simulation (8 baselines, 2026-10-01), 12 cm still found every towel
# and ~93% of shirts but only about half the socks and flat cloths.
# Saying the bucket is EMPTY needs the default spacing.
MAX_STEP_M = 0.12

# Slack for floating-point division: 0.36 / 0.03 must be 12, not 13.
_EPSILON = 1e-9


def stroke_plan(depth, entry_depth, max_step):
    """
    Return (stroke count, step) for scanning from entry_depth to depth.

    Raises ValueError, saying why, for a spacing over MAX_STEP_M, an
    entry depth outside [0, depth) or a non-positive depth or step.
    """
    if depth <= 0.0:
        raise ValueError(f'depth must be greater than zero (got {depth}).')

    if max_step <= 0.0:
        raise ValueError(f'step must be greater than zero (got {max_step}).')

    if max_step > MAX_STEP_M + _EPSILON:
        raise ValueError(
            f'step {max_step * 100:g} cm is over the {MAX_STEP_M * 100:g} cm '
            'maximum stroke spacing.'
        )

    if not 0.0 <= entry_depth < depth:
        raise ValueError(
            f'entry depth must be at least 0 and less than the depth '
            f'({depth:.3f} m); got {entry_depth:.3f} m.'
        )

    span = depth - entry_depth
    count = max(1, math.ceil(span / max_step - _EPSILON))

    return count, span / count
