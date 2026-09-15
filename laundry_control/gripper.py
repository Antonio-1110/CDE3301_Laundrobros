#!/usr/bin/env python3

"""
gripper.py

Gripper contact-point mounting offset, following the same
convention as tof_sensor.py's TOF_SENSOR_OFFSET_*: expressed in
link7's own local frame, measured at the INTER reference
orientation (where check_flange.py confirms link7's local +Z
points straight down).

Unlike the ToF sensor's offset (which has a nonzero local X
component - its outward-pointing boresight), the gripper's
contact point sits ENTIRELY along local +Z: it hangs straight
down from the TCP, on the wrist's own rotation axis. This has an
important consequence: rotating J7 sweeps the ToF sensor through
a horizontal circle (since its offset arm has X extent to sweep),
but rotating J7 does NOT move the gripper's contact point at all,
since a point on the rotation axis doesn't move when you rotate
about that axis. Reaching an arbitrary bucket-interior point with
the gripper therefore requires genuine XY translation of the TCP,
not just J7 rotation the way scanning does.

GRIPPER_OFFSET_Z is a user-reported measurement, not yet verified
against the physical hardware - confirm it (e.g. by comparing a
computed grasp point against the claw's real position in RViz or
in person) before trusting it for an unattended grasp.
"""

GRIPPER_OFFSET_X = 0.0
GRIPPER_OFFSET_Y = 0.0
GRIPPER_OFFSET_Z = 0.15

# The claw is roughly 6cm wide, so it doesn't need pinpoint lateral
# accuracy - being within this tolerance of the intended (x, y) is
# still graspable. This is a documented error budget, not a second
# offset vector - nothing currently applies it as a coordinate
# correction.
GRIPPER_LATERAL_TOLERANCE_M = 0.03
