#!/usr/bin/env python3

"""Where the grabs go: the grid the detected-item search uses, and its limits."""

# =============================================================
# GRABS
#
# The sensorless sweep (`laundry preplanned`, the start of `laundry
# clear`) visits the grabs placed by hand in scan_plans/grab_targets.
# yaml, in order (grasp/grab_targets.py): each is a depth, floor
# angle, height, tilt and approach distance relative to the bucket.
# Edit them by dragging in RViz (`laundry plan edit-grabs`), then
# `laundry plan bake retrieve` solves each EXACTLY as placed and
# stores it as two joint states in scan_plans/retrieve.yaml:
#
#   approach (named pose grab_NN): the gripper approach_m back along
#       its own axis - reached from INTER on a baked route;
#   grab: straight down onto the laundry from there, with a straight,
#       collision-checked joint move, then straight back up.
#
# The sweep runs on a drum expected ~2/3 full (the pile's top ~25 cm
# above the floor), so its grabs sit well up in the pile; the old
# grid's grabs, 1.5-4 cm off the floor, drove the claw ~20 cm through
# it. A vertical grab needs the 15 cm gripper plus its approach
# between the contact point and the drum's ceiling.
#
# RETRIEVE_GRID is the search for DETECTED items (grasp/
# retrieve_grid.plan_floor_grab), which sinks into the sensed item:
#
#   max_depth_m:   deepest item it will try, from the CLOSED end of
#                  the bucket (it is ~0.53 m deep)
#   min_height_m:  never place the contact point lower than this
#                  above the floor
#   tilts_deg:     tool axis from straight at the wall toward the
#                  closed end, tried smallest first
#   approach_m:    the approach pose's distance back along the tool
#   clearance_m:   extra gripper padding every grab - swept or
#                  detected - is solved with, on top of
#                  GRIPPER_PADDING_M: margin for the bucket model and
#                  the arm's accuracy
#
# Side grabs are the ones most sensitive to where the bucket really
# is (`laundry scene fit`): settle OBSTACLES first, then bake.
# =============================================================

RETRIEVE_GRID = {
    'max_depth_m': 0.55,
    'min_height_m': 0.01,
    'tilts_deg': [0.0, 15.0, 30.0, 45.0, 60.0],
    'approach_m': 0.08,
    # 0.5 cm (was 1 cm): the rig confirmed grabs by ruler (2026-09-29).
    'clearance_m': 0.005,
}

# DETECTED items are grabbed the same way (IK over the item, straight
# descent, reached from the nearest grab_NN - grasp/retrieve_grid.
# plan_floor_grab) anywhere up to this angle from the floor's lowest
# line on either side: 90 = the whole lower half of the drum, the
# project's scope. Measured on the fake controller (2026-09-26), grabs
# solve everywhere from 0 to +/-90 deg at depths 3-48 cm. Items beyond
# it fall back to the Cartesian reach (grasp/plan.py).
DETECTED_GRAB_MAX_ANGLE_DEG = 90.0
