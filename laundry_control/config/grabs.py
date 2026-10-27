#!/usr/bin/env python3

"""Where the grabs go: the pile-aware sweep layers and the detected-item grid."""

import os

from .robot import JOINT7_MAX_ACCELERATION_RAD_S2, JOINT7_MAX_VELOCITY_RAD_S

# =============================================================
# GRABS
#
# The sensorless sweep (`laundry preplanned`, the start of `laundry
# clear`) visits the grabs in scan_plans/grab_targets.yaml, in order
# (grasp/grab_targets.py): each is a depth, floor angle, height, tilt
# and approach distance relative to the bucket. `laundry plan bake
# retrieve` solves each EXACTLY as placed and stores it as two joint
# states in scan_plans/retrieve.yaml:
#
#   approach (named pose grab_NN): the gripper approach_m back along
#       its own axis - reached from INTER on a baked route;
#   grab: straight down onto the laundry from there, with a straight,
#       collision-checked joint move, then straight back up.
#
# The sweep runs on a drum that is already full, so what it needs
# depends on how full: PILE below. The bake GENERATES the grab list
# from it (grasp/pile_layers.py) whenever PILE changed since the list
# was generated:
#
#   top_height_m:  where the pile's top is, above the floor's lowest
#                  line. 2/3 of a ~37 cm drum = 0.25. Override per
#                  run with LAUNDRY_PILE_TOP_M (bake AND run: a plan
#                  baked for another level is reported as stale).
#   sink_m:        how far below the pile's surface the first layer's
#                  contact points go
#   layer_step_m:  how much lower each further layer goes than the one
#                  before ACTUALLY reached (the gripper cannot reach
#                  the top of a full drum, so layers step down from
#                  what is reachable, not from the pile's top)
#   min_height_m:  the lowest sweep grab; laundry below it is the
#                  scan-then-grasp stage's, so a pile this low gets no
#                  sweep grabs at all
#   depths_m, lateral_m: the spots - depths from the CLOSED end, each
#                  at these sideways offsets, the claw pointing down -
#                  in visiting order (the mouth first: the claw passes
#                  through the laundry there to get deeper)
#   tilts_deg:     tried smallest first; each spot takes the highest
#                  height that some tilt reaches. The 15 cm gripper
#                  plus its approach meet the drum's ceiling, so
#                  higher costs tilt (vertical to ~5-8 cm, 14 cm up
#                  needs 45-60 deg: fake controller, 2026-10-06)
#   approach_m:    the approach pose's distance back along the tool
#
# A layer is one pass over every spot; the sweep runs layer 1 (the
# pile's top) over all spots, then layer 2 a step lower, and so on,
# since taking laundry lowers the pile. A spot's layers stop at
# min_height_m.
#
# Edit them by dragging in RViz instead (`laundry plan edit-grabs`)
# and the bake keeps your list until PILE changes or you pass
# --regenerate.
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

PILE = {
    'top_height_m': float(os.environ.get('LAUNDRY_PILE_TOP_M', 0.25)),
    'sink_m': 0.06,
    'layer_step_m': 0.06,
    'min_height_m': 0.10,
    # Rows 7 cm apart from 32 cm, the mouth row's (40 until 2026-10-27:
    # the user moved it 8 cm in and asked for the rest re-spaced).
    'depths_m': (0.32, 0.25, 0.17, 0.10),
    # Sideways of the drum's middle, the widest each spot is wanted at;
    # one that does not fit tries 1 cm nearer the middle at a time. The
    # claw points straight down (grab_targets: down grabs). Until
    # 2026-10-27 the spots were floor angles 0/-20/+20 deg, which high
    # in the drum lie within ~1 cm of each other.
    'lateral_m': (0.0, -0.12, 0.12),
    'tilts_deg': (0.0, 15.0, 30.0, 45.0, 60.0),
    'approach_m': 0.05,
}

# Fill levels for the sweep, picked per run (`laundry preplanned
# --fill 0.75`, `clear --fill`, or LAUNDRY_FILL=0.75 for any command,
# bakes included): each is its own baked plan set in
# scan_plans/fill_<percent>/ - its grab_targets.yaml, retrieve.yaml
# and transfers.yaml - generated from PILE with the pile's top at that
# fraction of the drum's height (DRUM_HEIGHT_M), the rest of PILE as
# is. ./rebake_fills.sh bakes them all. With no fill chosen, the plans
# in scan_plans/ itself are used (PILE's own top_height_m).
#
# A level whose pile is too low for PILE's min_height_m (0.25, 0.3:
# their first layer aims at 4.6 and 6.8 cm) still gets one layer: its
# grabs may go LOW_FILL_DEPTH_M below the aim, but never under
# LOW_FILL_MIN_HEIGHT_M (config.set_fill). Fuller levels keep
# min_height_m.
FILL_LEVELS = (0.25, 0.3, 0.5, 0.6, 0.75, 0.9)
LOW_FILL_DEPTH_M = 0.03
LOW_FILL_MIN_HEIGHT_M = 0.03

# The drum's height at mid-depth: twice the fitted bucket's radius
# there (bucket.seed_cone; test_fill checks it against the model).
DRUM_HEIGHT_M = 0.425

RETRIEVE_GRID = {
    'max_depth_m': 0.55,
    'min_height_m': 0.01,
    'tilts_deg': [0.0, 15.0, 30.0, 45.0, 60.0],
    'approach_m': 0.08,
    # 0.5 cm (was 1 cm): the rig confirmed grabs by ruler (2026-09-29).
    'clearance_m': 0.005,
}

# How deep the grab of a DETECTED item digs into it. The sensed top is
# only the pile's surface; the claw closes on what is below it, down
# toward the bucket wall underneath (the fitted bucket model says how
# far that is). The grab goes to the DEEPEST height whose grab pose,
# approach pose and straight descent are all collision-free (IK +
# MoveIt's checks, the gripper padded by RETRIEVE_GRID's clearance_m),
# searched to resolution_m - so how close it gets to the bucket is set
# by the collision check, not by a fixed fraction of the pile.
#
#   max_sink_m:        never more than this below the sensed top
#   wall_clearance_m:  the contact point stays at least this far in
#                      from the bucket wall (the model is only as good
#                      as `laundry scene fit`; the claw mesh itself is
#                      kept clear by the padding)
#   resolution_m:      how finely the deepest clear height is found
#
# Until 2026-10-22 the sink was half the pile's height, at most 5 cm:
# on a 20 cm pile the claw closed 15 cm above the floor.
DETECTED_GRAB_DIG = {
    'max_sink_m': 0.15,
    'wall_clearance_m': 0.02,
    'resolution_m': 0.01,
}

# DETECTED items are grabbed the same way (IK over the item, straight
# descent, reached from the nearest grab_NN - grasp/retrieve_grid.
# plan_floor_grab) anywhere up to this angle from the floor's lowest
# line on either side: 90 = the whole lower half of the drum, the
# project's scope. Measured on the fake controller (2026-09-26), grabs
# solve everywhere from 0 to +/-90 deg at depths 3-48 cm. Items beyond
# it fall back to the Cartesian reach (grasp/plan.py).
DETECTED_GRAB_MAX_ANGLE_DEG = 90.0

# Carry turned: once the claw has closed, every grasp (the sweep,
# detected items, `laundry grasp`) turns J7 by this much while it lifts
# the laundry out and carries it to DROP, and back by the time it gets
# there, so it opens unturned (grasp/carry.py). The turn rides on the
# baked carry - it is not a move of its own, nothing stops for it
# (arm/joint_path.add_j7_turn). The jaws close along link7 Y, and every
# grab sets link7 X along the bucket axis (at INTER: X down), so
# unturned they close sideways and the laundry hangs in a pinch;
# turned, the lower jaw is under it. +90, not -90: -90 takes the
# deepest grabs past J7's -270 deg task limit. Fake controller,
# 2026-10-26: turning was collision-free for all 24 baked grabs, and
# every baked carry (3.2-4.9 s) fits the turn and back (1.3-1.7 s
# each way). 0 carries unturned, as before.
CARRY_J7_TURN_DEG = 90.0

# J7's limits for the turn, with the carry's own J7 motion added: 90%
# of J7's own.
CARRY_TURN_MAX_VELOCITY_RAD_S = 0.9 * JOINT7_MAX_VELOCITY_RAD_S
CARRY_TURN_MAX_ACCELERATION_RAD_S2 = 0.9 * JOINT7_MAX_ACCELERATION_RAD_S2
