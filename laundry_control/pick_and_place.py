#!/usr/bin/env python3

"""
pick_and_place.py

Full pipeline, in one continuous arm connection:

    scan -> detect + compute grasp -> approach + close -> return to
    INTER -> open (drop)

This is a bare script (like scan_move.py/move.py, and NOT
registered as a console_script), since it needs scan_move.py's
scan() function, and scan_move.py's OWN top-level imports
(arm_position/move/scan_record) are bare/non-relative themselves -
they only resolve when run this way (sys.path[0] = this script's
own directory). The newer modules this script also needs
(laundry_detect/grasp_plan/gripper_client/retrieve) use
package-relative imports internally, so they're reached instead via
their full laundry_control.* path, which resolves independently
through the sourced workspace's install tree - already a hard
requirement anyway, since even scan_move.py's own rclpy/moveit_msgs
imports need the workspace sourced.

CAUTION: because this script is never installed, its OWN bare
imports always run against the live SOURCE tree, but the
laundry_control.* imports resolve through whatever the sourced
workspace's BUILD/INSTALL tree currently has. Rebuild (colcon build
--packages-select laundry_control) after editing laundry_detect.py/
grasp_plan.py/gripper_client.py/retrieve.py, or use
--symlink-install, to keep both halves in sync.

Which CSV does detect_laundry() read back? scan_recorder_node
normally auto-generates a timestamped filename, which this script
would have no way to know in advance - so before scanning, this
script explicitly SETS scan_recorder_node's csv_path parameter
(via its /scan_recorder_node/set_parameters service) to a freshly
computed timestamped path, guaranteeing it knows exactly which file
the scan just produced.

gripper_node.py must be running separately (see retrieve.py).

Usage:
    python3 pick_and_place.py
    python3 pick_and_place.py --baseline path/to/other_baseline.csv
"""

import argparse
import os
from datetime import datetime

import rclpy

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

import arm_position
from move import XArm7Controller
from scan_move import scan
from scan_record import ScanRecorderClient

from laundry_control.grasp_plan import compute_grasp_target
from laundry_control.gripper_client import GripperClient
from laundry_control.laundry_detect import detect_laundry, load_points_xyz
from laundry_control.retrieve import select_target_cluster

DEFAULT_BASELINE_PATH = (
    "/home/cde3301a/ros2_ws/src/CDE3301_Laundrobros/"
    "baseline_scans/baseline.csv"
)

DEFAULT_SCAN_RECORDS_DIR = (
    "/home/cde3301a/ros2_ws/src/CDE3301_Laundrobros/scan_records"
)

SCAN_RECORDER_NODE_NAME = "scan_recorder_node"


def set_remote_string_param(node, remote_node_name, param_name, value, timeout_sec=5.0):
    """
    Set a string parameter on a DIFFERENT, already-running node
    (scan_recorder_node), via its own /<name>/set_parameters
    service - so this script can pin down exactly which CSV path
    the next scan will save to, rather than guessing at whatever
    auto-generated timestamped filename scan_recorder_node would
    otherwise have picked on its own.
    """

    client = node.create_client(
        SetParameters, f"/{remote_node_name}/set_parameters"
    )

    if not client.wait_for_service(timeout_sec=timeout_sec):

        node.get_logger().error(
            f"'/{remote_node_name}/set_parameters' not available "
            f"after {timeout_sec:.1f}s (is {remote_node_name} running?)."
        )

        return False

    request = SetParameters.Request()

    request.parameters = [
        Parameter(
            name=param_name,
            value=ParameterValue(
                type=ParameterType.PARAMETER_STRING,
                string_value=value,
            ),
        )
    ]

    future = client.call_async(request)

    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)

    if not future.done():
        node.get_logger().error(f"Setting '{param_name}' timed out.")
        return False

    response = future.result()

    if not all(result.successful for result in response.results):

        node.get_logger().error(
            f"Failed to set '{param_name}': "
            f"{[r.reason for r in response.results if not r.successful]}"
        )

        return False

    return True


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Scan the bucket, retrieve the best-detected laundry "
            "item, and drop it back at INTER."
        )
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

    return parser


def main():

    args = build_parser().parse_args()

    rclpy.init()

    arm = XArm7Controller()
    recorder = ScanRecorderClient(arm)
    gripper = GripperClient(arm)

    try:

        candidate_csv = os.path.join(
            DEFAULT_SCAN_RECORDS_DIR,
            f"pick_and_place_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        print(
            f"Pinning scan_recorder_node's csv_path to {candidate_csv} "
            "so we know exactly what this scan produces..."
        )

        if not set_remote_string_param(
            arm, SCAN_RECORDER_NODE_NAME, "csv_path", candidate_csv
        ):

            print("Failed to pin down the scan's CSV path; aborting.")

            return

        print("========== SCAN ==========")

        recorder.clear_blocking()

        if not scan(arm=arm, recorder=recorder):

            print("Scan failed; aborting.")

            return

        print("========== RETRIEVE ==========")

        clusters = detect_laundry(
            baseline_csv=args.baseline,
            candidate_csv=candidate_csv,
        )

        if not clusters:

            print("No laundry detected; nothing to retrieve.")

            return

        target_cluster = select_target_cluster(clusters)

        print(
            f"Targeting cluster: size={target_cluster.size} "
            f"centroid={target_cluster.centroid}"
        )

        baseline_xyz = load_points_xyz(args.baseline)

        # scan() already returns to INTER at its own end, but this
        # is repeated explicitly (matches retrieve.py) so the
        # reachability probing below always starts from a known,
        # correct reference orientation regardless.
        print("Moving to INTER (reference orientation for grasp math)...")

        if not arm.move_joints(arm_position.INTER):

            print("Failed to reach INTER; aborting.")

            return

        grasp = compute_grasp_target(target_cluster, baseline_xyz, arm)

        if grasp is None:

            print(
                "No reachable grasp target found, even at "
                "sink=0; aborting."
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

        print("Opening gripper...")

        gripper.open_blocking()

        x, y, z = grasp.tcp_position

        if not arm.move_to_pose(x, y, z, orientation=grasp.orientation):

            print("Failed to reach grasp target; aborting.")

            return

        print("Closing gripper...")

        gripper.close_blocking()

        print("Returning to INTER...")

        arm.move_joints(arm_position.INTER)

        print("========== DROP ==========")

        print("Opening gripper to drop the item at INTER...")

        gripper.open_blocking()

    finally:

        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
