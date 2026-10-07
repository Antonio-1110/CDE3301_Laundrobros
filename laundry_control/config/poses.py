#!/usr/bin/env python3

"""
Recorded joint poses, and where INTER and BOTTOM sit relative to the bucket.

Looked up by name through config.get_named_pose() - never read
INTER or BOTTOM from the lists here directly (see INTER_RECORDED).
"""

# =============================================================
# RECORDED JOINT POSES
#
# Seven absolute joint angles in RADIANS, J1..J7, recorded on the
# real arm. `laundry move <name>` accepts any upper-case list here
# by its lower-case name (see named_poses()).
# =============================================================

HOME = [
    1.57,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

# INTER, the reference pose everything else is measured from: the
# scan starts and ends here, and the ToF/gripper offsets (config/hardware.py)
# were measured here. These are the angles jogged by eye; once
# `laundry plan bake poses` has run, INTER is instead derived from
# the bucket (BUCKET_POSES below, scan_plans/bucket_poses.yaml), and
# these only seed its IK - they pick the elbow posture. Always look
# INTER up with get_named_pose('inter'), never from this list.
#
# Forward kinematics of these angles (same URDF as the real arm):
#
#   link7 local +Z = (0.00, -1.00, -0.00)  horizontal, along -Y:
#                                           into the bucket, 7.8 deg
#                                           off the modelled axis
#   link7 local +X = (-0.01, 0.00, -1.00)  straight DOWN - this is
#                                           the ToF boresight, so at
#                                           INTER's J7 the sensor
#                                           looks at the bucket floor
#
# Older comments said "+Z points straight down at INTER"; it is the
# sensor's +X that does. The scan's +/-75 deg J7 sweep is therefore
# centred on the floor, which is why the ceiling is never seen.
INTER_RECORDED = [
    1.5105054378509521,
    1.5552058219909668,
    -2.4133317470550537,
    0.17237895727157593,
    0.28598350286483765,
    0.12894250452518463,
    -2.70444223365,
]

# BOTTOM: tilted-up pose used at maximum insertion depth so the
# wrist-mounted sensor can see the closed end past the end effector
# (see scan.pattern's BOTTOM detour). Jogged by eye; derived from the
# bucket like INTER once baked, and then only an IK seed. FK: local
# +Z tilts 42 deg up from horizontal (50.6 deg from the modelled
# bucket axis), and the boresight (+X) points down and toward the
# closed end, ~48 deg below horizontal.
BOTTOM_RECORDED = [
    -0.6021117568016052,
    1.1247814893722534,
    -1.3280805349349976,
    1.6322089433670044,
    0.7299384474754333,
    -1.0157668590545654,
    -2.638805389404297,
]

# Where retrieved laundry is released.
DROP = [
    0.10072359442710876,
    1.4992231130599976,
    -3.0929934978485107,
    0.29612118005752563,
    -0.11858407407999039,
    0.7183085680007935,
    -2.913362979888916,
]

# Starting points (IK seeds) for the grab-grid solver, grasp/
# retrieve_grid.py - never moved to. They are the four grab poses
# once jogged by hand (the old RETRIEVE_0..3), kept because they show
# the solver which arm postures reach the bucket floor: from each, it
# converges on a grab in that same posture. The generated grab grid
# (config/grabs.py) replaced them as poses.
GRAB_IK_SEEDS = [
    [
        -0.16877898573875427,
        0.3103211522102356,
        -1.2240749597549438,
        1.2551662921905518,
        0.23523668944835663,
        0.6844640374183655,
        -3.019961357116699,
    ],
    [
        1.0142279863357544,
        0.7750195264816284,
        -2.239295482635498,
        0.6214537024497986,
        0.6356657147407532,
        0.6073369383811951,
        -3.199042797088623,
    ],
    [
        0.635716438293457,
        0.1730310469865799,
        -1.9444830417633057,
        1.1769441366195679,
        0.14949719607830048,
        1.2463109493255615,
        -2.922783851623535,
    ],
    [
        1.0214934349060059,
        0.7060800194740295,
        -2.34552264213562,
        0.7978871464729309,
        0.47409510612487793,
        1.36742103099823,
        -3.12626576423645,
    ],
]

# The poses above that `laundry move <name>` knows, under their
# lower-case names without _RECORDED (config.recorded_poses).
RECORDED_POSE_NAMES = (
    'HOME',
    'INTER_RECORDED',
    'BOTTOM_RECORDED',
    'DROP',
)

# =============================================================
# INTER AND BOTTOM, FROM THE BUCKET
#
# Where INTER and BOTTOM sit relative to the bucket of OBSTACLES, so
# that moving the bucket moves them too. `laundry plan bake poses`
# (the first step of `laundry plan bake`) solves their joint angles
# by IK in the configured bucket - seeded from INTER_RECORDED /
# BOTTOM_RECORDED, so the arm keeps the elbow posture those were
# jogged in - and saves them to scan_plans/bucket_poses.yaml. Until
# that file exists, the recorded angles are used as they are.
#
#   inter:
#     The flange on the bucket axis, standoff_m outside the mouth
#     plane; tool +Z straight down the axis into the bucket, the ToF
#     boresight (+X) at the floor.
#     standoff_m:  0.054 is where the recorded INTER's flange is.
#
#   bottom:
#     The flange depth_m in from INTER along the axis - the deepest
#     stroke of the scan (scan.pattern.DEFAULT_DEPTH_M) - with the
#     tool tilted up by tilt_deg from the axis, the boresight
#     leaning down toward the closed end (the end scan's
#     alpha = tilt_deg, phi = 0 pose, scan/endcap.py).
#     tilt_deg:    the recorded BOTTOM is 50.6 deg from the modelled
#                  axis, but 2.3 cm above it; on the axis, 50 deg puts
#                  link3 against the rim with 3 cm padding (fake
#                  controller, 2026-09-28). 45 deg is the reachable
#                  tilt closest to the recorded posture (at most 6.4
#                  deg per joint); 30-40 deg also solve.
#
# Derived poses are only as good as OBSTACLES: settle the bucket pose
# (`laundry scene fit`, a tape measure) before baking them.
# =============================================================

BUCKET_POSES = {
    'inter': {
        'standoff_m': 0.054,
    },
    'bottom': {
        'depth_m': 0.42,
        'tilt_deg': 45.0,
    },
}
