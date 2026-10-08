#!/usr/bin/env python3

"""The MoveIt planning scene: obstacle poses and the padding kept around them."""

# =============================================================
# OBSTACLES (the MoveIt planning scene)
#
# The bucket and table are MoveIt WORLD collision objects, added by
# arm/scene.py whenever `laundry` connects - not URDF links. Their
# poses live here and nowhere else.
#
# Each pose is the mesh origin in link_base: xyz in metres, rpy in
# radians with the URDF convention (fixed-axis roll about X, then
# pitch about Y, then yaw about Z) - the same numbers a URDF <origin>
# would take.
#
# AFTER MOVING THE BUCKET OR TABLE:
#
#   1. Edit its xyz/rpy below. `laundry scene check` shows the new
#      scene against every named pose and baked route without
#      moving the arm (run it with the fake controller up, too).
#   2. `laundry plan bake` (fake controller) and commit scan_plans/.
#      It derives INTER and BOTTOM from the new bucket pose
#      (BUCKET_POSES), then bakes everything else from that INTER.
#      Baked plans record the scene they were baked against;
#      replaying one baked against a different scene warns, one that
#      now collides is refused, and transfers baked from another
#      INTER are not used.
#   3. Re-record HOME or DROP only if the move put the bucket in
#      their way.
#   4. Collect new baselines (`laundry baseline collect --archive`):
#      the scan runs from the new INTER.
#
# The mesh file is under meshes/. allowed_links lists robot links
# allowed to touch the object (link_base rests on the table).
# =============================================================

OBSTACLES = {
    'bucket': {
        'mesh': 'bucket.obj',
        'xyz': [0.14, -0.72, 0.42],
        # Roll 1.68 (was 1.71): the axis rises 6.3 deg toward the mouth,
        # as `laundry scene fit` measures from the baselines (2026-09-28;
        # check-flange at the derived INTER agreed, 1.9 deg off at 1.71).
        # The fit's sideways shift (+3.1 cm x) is not applied: unverified.
        'rpy': [1.68, 3.14, -3.14],
        'allowed_links': [],
    },
    'table': {
        'mesh': 'table.obj',
        'xyz': [0.19, -0.43, -0.02],
        'rpy': [3.14, 3.14, 1.57],
        'allowed_links': ['link_base'],
    },
}

# Clearance MoveIt keeps between the obstacles and every moving arm
# link (link1-link7) - see arm/scene.py. Applied whenever `laundry`
# connects to MoveIt, so it governs everything: the planner, Cartesian
# strokes, and our own straight-line and baked-path checks.
#
# Measured on the fake controller (2026-09-25): at 3 cm the committed
# HOME/DROP transfers and every recorded pose still pass; the end
# scan's last ring brought the elbow (link3) within 2 cm of the rim,
# so it was re-baked with this padding.
OBSTACLE_PADDING_M = 0.03

# Arm-link padding the baked end scan is solved and replayed with, in
# place of OBSTACLE_PADDING_M, for its ~30 s at the bottom of the
# bucket (scan/endcap.py). The precession tilts the tool up to 55 deg,
# which swings the elbow (link3) toward the bucket rim; with 3 cm the
# outer rings are infeasible and simulated coverage of the closed
# end's lower half drops from 86% to 57%, no better than the old
# BOTTOM detour. Changing arm-link padding is quick (~0.3 s).
ENDCAP_PADDING_M = 0.01

# The gripper's own live padding. It is the part that works INSIDE the
# bucket: the RETRIEVE poses put it within 1-2 cm of the floor and
# walls by design, and the scan and grasp need it there, so it gets
# none. Changing it is also slow - MoveIt rebuilds the padded gripper
# mesh (3-6 s) - so it stays fixed at runtime.
GRIPPER_PADDING_M = 0.0

# Gripper padding each baked route (arm/transfers.py) is solved with,
# in place of GRIPPER_PADDING_M. The routes that LEAVE the bucket
# (INTER -> HOME/DROP) get extra, so the gripper clears the bucket
# mouth by at least this much on its way out, although the live check
# does not require it. Routes into the bucket (BOTTOM, grab_NN) end
# with the gripper at the floor on purpose, so they are left out.
# Re-bake after changing a value (`laundry plan bake transfers`).
ROUTE_GRIPPER_CLEARANCE_M = {
    'home': 0.01,
    # Laundry hanging from the gripper swings out on the way to DROP;
    # keep it clear of the rim so it does not snag. 10 cm until
    # 2026-10-07; at 5 cm the route is shorter (issue #11).
    'drop': 0.05,
    # The way back from DROP once the gripper has let go (go_to's
    # gripper_empty): nothing hangs from it, so 1 cm, like HOME. On
    # the fake controller (2026-10-01) that cut the return from 7.2 s
    # to 5.1 s, ~2 s off every grab (issue #11).
    'drop_return': 0.01,
}

# Baked routes (arm/transfers.py) that are solved and replayed with a
# smaller arm-link padding than OBSTACLE_PADDING_M, like the end scan.
# BOTTOM tilts the tool 42 deg up at the bottom of the bucket, which
# brings the elbow (link3) to the rim: measured on the fake controller
# (2026-09-26), no INTER -> BOTTOM route exists at 3 cm, while at 2 cm
# it is almost the straight line (1 via, ~1 deg extra travel).
ROUTE_ARM_PADDING_M = {
    'bottom': 0.02,
}
