#!/usr/bin/env python3

"""
pick_and_place.py

Full pipeline, in one continuous arm connection:

    open gripper -> scan -> detect + compute grasp -> approach +
    close (grasp) -> retract to INTER -> DROP -> open (release)
    -> INTER

The gripper is open for the whole run except while carrying the
item, from the grasp until the release at DROP. Scanning open (not
just idling open) is deliberate: the end effector occludes the
bucket, so the scan has to be taken in the same gripper state the
baseline was recorded in. The release happens at
arm_position.DROP rather than at INTER, with INTER used as the
known-clear waypoint between the two.

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
from laundry_control.bucket_model import build_baseline_surface
from laundry_control.laundry_detect import detect_laundry, load_baseline_scans
from laundry_control.retrieve import plan_first_reachable

# A DIRECTORY of empty-bucket scans - see retrieve.py.
DEFAULT_BASELINE_PATH = (
    "/home/cde3301a/ros2_ws/src/CDE3301_Laundrobros/"
    "baseline_scans"
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
            "item, release it at DROP, and return to INTER."
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

        # The gripper stays OPEN for everything except the actual
        # carry, and this makes that explicit rather than inheriting
        # whatever state the last run left it in. It matters for
        # detection, not just tidiness: the sensor is wrist-mounted
        # and the end effector already occludes part of the bucket
        # (see scan_move.py's BOTTOM detour), so a closed gripper
        # occludes differently from an open one - scanning in a
        # different gripper state than the baseline was recorded in
        # shifts returns in the affected region by more than
        # DEFAULT_THRESHOLD_M and shows up as laundry that isn't
        # there.
        print("Opening gripper so the scan matches the baseline's geometry...")

        if not gripper.open_blocking():

            print(
                "WARNING: could not confirm the gripper opened. If it "
                "is closed, this scan's end-effector occlusion differs "
                "from the baseline's and may produce phantom "
                "detections."
            )

        recorder.clear_blocking()

        if not scan(arm=arm, recorder=recorder):

            print("Scan failed; aborting.")

            return

        print("========== RETRIEVE ==========")

        # One model for both detection and grasp depth - see
        # retrieve.main() for why they must not be fitted twice.
        baseline_scans = load_baseline_scans(args.baseline)
        surface = build_baseline_surface(baseline_scans)

        clusters = detect_laundry(
            baseline_csv=args.baseline,
            candidate_csv=candidate_csv,
            surface=surface,
        )

        if not clusters:

            print("No laundry detected; nothing to retrieve.")

            return

        print(f"{len(clusters)} cluster(s) detected.")


        # scan() already returns to INTER at its own end, but this
        # is repeated explicitly (matches retrieve.py) so the
        # reachability probing below always starts from a known,
        # correct reference orientation regardless.
        print("Moving to INTER (reference orientation for grasp math)...")

        if not arm.move_joints(arm_position.INTER):

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

        print("Opening gripper...")

        gripper.open_blocking()

        x, y, z = grasp.tcp_position

        if not arm.move_to_pose(x, y, z, orientation=grasp.orientation):

            print("Failed to reach grasp target; aborting.")

            return

        print("Closing gripper...")

        gripper.close_blocking()

        print("========== DROP ==========")

        # Retract to INTER before traversing to DROP. The grasp pose
        # is deep inside the bucket at an arbitrary computed
        # position, so going straight to DROP would sweep the arm
        # (and whatever it is now holding) sideways through the
        # bucket wall. INTER is the withdrawn pose the scan itself
        # starts and ends at, so it is a known-clear waypoint out.
        print("Retracting to INTER...")

        if not arm.move_joints(arm_position.INTER):

            print("Failed to retract to INTER; aborting.")

            return

        print("Moving to DROP...")

        if not arm.move_joints(arm_position.DROP):

            print("Failed to reach DROP; aborting.")

            return

        print("Opening gripper to release the item...")

        gripper.open_blocking()

        print("Returning to INTER...")

        arm.move_joints(arm_position.INTER)

    finally:

        arm.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
