#!/usr/bin/env python3

"""
retrieve.py

End-to-end: detect laundry (laundry_detect.detect_laundry), pick a
target cluster, compute its grasp target (grasp_plan.
compute_grasp_target), move to INTER (the reference orientation
all the offset math in gripper.py/grasp_plan.py assumes), then
move_to_pose() the flange so the gripper's contact point lands on
the target, open/close the gripper via gripper_node.py (over
open_gripper/close_gripper services - see gripper_client.py), and
return to INTER.

gripper_node.py must be running separately (it owns the actual
GPIO hardware access):

    ros2 run laundry_control gripper_node

Its open_angle_deg/close_angle_deg parameters default to the
calibrated values (45/100 deg) - override via --ros-args if that
ever changes:

    ros2 run laundry_control gripper_node --ros-args \\
        -p open_angle_deg:=20.0 -p close_angle_deg:=160.0

This module uses package-relative imports (like laundry_detect.py/
laundry_detect_cli.py), so - unlike move.py/move_cli.py/
scan_move.py, which use plain imports and run as bare scripts -
it must be run as a module or via the registered console script,
not as a bare `python3 retrieve.py`:

    ros2 run laundry_control retrieve path/to/candidate_scan.csv
    python3 -m laundry_control.retrieve path/to/candidate_scan.csv \\
        --baseline path/to/other_baseline.csv
"""

import argparse

import rclpy

from .arm_position import INTER
from .grasp_plan import compute_grasp_target
from .gripper_client import GripperClient
from .bucket_model import build_baseline_surface, fit_report, occupancy_summary
from .laundry_detect import detect_laundry, load_baseline_scans
from .move import XArm7Controller

# Absolute, not relative: `ros2 run` executes from the colcon
# build/install tree, not the source tree, and baseline_scans/
# only exists in the latter - a relative default would silently
# fail to resolve unless run from inside CDE3301_Laundrobros/
# itself. Matches real_arm_scan.launch.py's own `records_dir`
# default for the same reason.
# A DIRECTORY, not a single file: the detector builds its model
# from all the empty scans it finds there, and the per-cell noise
# map it derives the threshold from needs ~8-10 of them. A single
# CSV path still works if one is passed explicitly.
DEFAULT_BASELINE_PATH = (
    "/home/cde3301a/ros2_ws/src/CDE3301_Laundrobros/"
    "baseline_scans"
)


def rank_clusters(clusters):
    """
    Order detected clusters best-target-first: largest by
    integrated intrusion VOLUME, ties broken by point count.

    Volume rather than point count, because the scan's point
    density is strongly non-uniform - the helical path samples some
    parts of the bucket several times more densely than others - so
    cluster.size partly measures where an item happened to sit
    rather than how much fabric is there. volume_m3 is integrated
    per grid cell and is density-independent, so it compares two
    items fairly wherever they landed. It is also the quantity a
    gripper actually cares about.

    Callers should walk this list rather than committing to its
    first entry: being the biggest cluster does not make a target
    reachable, and an unreachable one is no reason to abandon a
    scan that found other candidates.
    """

    return sorted(
        clusters,
        key=lambda cluster: (cluster.volume_m3, cluster.size),
        reverse=True,
    )


def select_target_cluster(clusters):
    """
    The single best-ranked cluster, or None if there are none.
    """

    ranked = rank_clusters(clusters)

    return ranked[0] if ranked else None


def plan_first_reachable(clusters, surface, arm, compute_grasp_target):
    """
    Walk clusters best-first and return the first
    (cluster, GraspTarget) whose grasp target the arm can actually
    reach, or (None, None) if none of them can be.

    compute_grasp_target is injected rather than imported here so
    this stays unit-testable with a stub.
    """

    for cluster in rank_clusters(clusters):

        cx, cy, cz = cluster.centroid

        print(
            f"Trying cluster: size={cluster.size} "
            f"centroid=({cx:.3f}, {cy:.3f}, {cz:.3f}) "
            f"mean_dev={cluster.mean_deviation_m:.3f}m"
        )

        grasp = compute_grasp_target(cluster, surface, arm)

        if grasp is not None:
            return cluster, grasp

        print("  unreachable at every sink depth; trying the next cluster.")

    return None, None


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Detect laundry in a scan and move the arm's gripper "
            "contact point to the best target."
        )
    )

    parser.add_argument(
        "candidate_csv",
        type=str,
        help="Path to the scan CSV to search for laundry in.",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE_PATH,
        help=(
            "Path to the empty-bucket baseline scan CSV "
            f"(default: {DEFAULT_BASELINE_PATH})."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Move to INTER and compute/print the grasp target as "
            "usual, but stop there - never call move_to_pose(), "
            "operate the gripper, or return to INTER. Still moves "
            "the physical arm to INTER (needed for the reachability "
            "check to probe from the correct starting state), but "
            "the final approach/grasp motion is skipped."
        ),
    )

    return parser


def main():

    args = build_parser().parse_args()

    # The bucket model is built ONCE and used for both halves of
    # the job: finding the laundry, and working out how much room
    # is underneath it to sink the gripper into. Fitting it twice
    # would waste the expensive step and, worse, risk the grasp
    # being planned against a different surface than the detection
    # was judged against.
    baseline_scans = load_baseline_scans(args.baseline)
    surface = build_baseline_surface(baseline_scans)

    print(
        f"Bucket model from {len(baseline_scans)} baseline scan(s) "
        f"({sum(scan.shape[0] for scan in baseline_scans)} pts)"
    )
    print(fit_report(surface.cone))
    print(occupancy_summary(surface))
    print()

    clusters = detect_laundry(
        baseline_csv=args.baseline,
        candidate_csv=args.candidate_csv,
        surface=surface,
    )

    if not clusters:
        print("No laundry detected; nothing to retrieve.")
        return

    print(f"{len(clusters)} cluster(s) detected:")

    for rank, cluster in enumerate(rank_clusters(clusters), start=1):
        cx, cy, cz = cluster.centroid
        print(
            f"  #{rank}  vol={cluster.volume_m3 * 1e6:.1f}cm3  "
            f"size={cluster.size}  "
            f"centroid=({cx:.3f}, {cy:.3f}, {cz:.3f})"
            f"{'' if cluster.confident else '  [low confidence]'}"
        )

    rclpy.init()

    arm = XArm7Controller()
    gripper = GripperClient(arm)

    try:

        print("Moving to INTER (reference orientation for grasp math)...")

        if not arm.move_joints(INTER):

            print("Failed to reach INTER; aborting.")

            return

        target_cluster, grasp = plan_first_reachable(
            clusters,
            surface,
            arm,
            compute_grasp_target,
        )

        if grasp is None:

            print(
                f"None of the {len(clusters)} detected cluster(s) "
                "yielded a reachable grasp target, even at sink=0; "
                "aborting."
            )

            return

        print(
            f"Grasp target: TCP={grasp.tcp_position}, "
            f"orientation={grasp.orientation}, "
            f"contact point={grasp.grasp_point}, "
            f"sink={grasp.sink_amount_m * 100.0:.1f}cm "
            f"(fraction={grasp.sink_fraction_used}, "
            f"gap={grasp.gap_m * 100.0:.1f}cm, "
            f"reachability={grasp.reachability_fraction:.3f})"
        )

        if args.dry_run:

            print("--dry-run: stopping before move_to_pose().")

            return

        print("Opening gripper...")

        gripper.open_blocking()

        x, y, z = grasp.tcp_position

        if not arm.move_to_pose(x, y, z, orientation=grasp.orientation):

            print("Failed to reach grasp target; aborting.")

            return

        print("Closing gripper...")

        gripper.close_blocking()

        print("Returning to INTER...")

        arm.move_joints(INTER)

    finally:

        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
