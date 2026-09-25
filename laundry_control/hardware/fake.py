#!/usr/bin/env python3

"""
Stand-ins for the gripper and the scan recorder, for `--fake-hardware`.

With these, the whole pipeline (scan motion -> detect -> grasp ->
drop) runs on a machine with no servo, no ToF sensor and no arm -
against the MoveIt fake controller (xarm7_moveit_fake.launch.py),
with a previously saved scan CSV standing in for the scan's output.

The arm itself is never faked: every motion still goes through
MoveIt, so planning failures, unreachable grasp targets and
collisions with the bucket/table geometry all still show up. Only
the two devices that need GPIO/I2C are replaced.

Each fake has exactly the interface of the real client it replaces
(hardware.gripper_client.GripperClient,
scan.recorder_client.ScanRecorderClient), so the stage code needs no
`if fake:` branches of its own.
"""

import os
import shutil


class FakeGripper:
    """Logs open/close requests and always reports success."""

    def __init__(self, node):
        self.node = node
        self.state = 'unknown'

    def open_blocking(self, timeout_sec=5.0):
        """Pretend to open the gripper."""
        self.state = 'open'
        self.node.get_logger().info('[fake gripper] open')
        return True

    def close_blocking(self, timeout_sec=5.0):
        """Pretend to close the gripper."""
        self.state = 'closed'
        self.node.get_logger().info('[fake gripper] close')
        return True


class FakeRecorder:
    """
    Stand in for scan_recorder_node by "recording" a saved scan.

    The scan motion runs for real (on the fake controller), but no
    ToF readings exist, so on the final save the source CSV is copied
    to wherever the real recorder would have written - after which
    the rest of the pipeline cannot tell the difference.
    """

    def __init__(self, node, source_csv, csv_path=None):
        if not os.path.isfile(source_csv):
            raise FileNotFoundError(
                f'--scan-from file not found: {source_csv!r}'
            )

        self.node = node
        self.source_csv = source_csv
        self.csv_path = csv_path

    def set_csv_path(self, csv_path, timeout_sec=5.0):
        """Record where the final save should land."""
        self.csv_path = csv_path
        return True

    def clear_blocking(self, timeout_sec=5.0):
        """Clear nothing: there is no accumulated state."""
        return True

    def clear_async(self):
        """Clear nothing: there is no accumulated state."""

    def save_async(self):
        """Skip mid-scan checkpoints; only the final save matters."""

    def save_blocking(self, timeout_sec=5.0):
        """Copy the source scan to csv_path, as the real save would."""
        if self.csv_path is None:
            self.node.get_logger().info(
                f'[fake recorder] scan is {self.source_csv}'
            )
            return True

        directory = os.path.dirname(self.csv_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        if os.path.abspath(self.csv_path) != os.path.abspath(self.source_csv):
            shutil.copyfile(self.source_csv, self.csv_path)

        self.node.get_logger().info(
            f'[fake recorder] {self.source_csv} -> {self.csv_path}'
        )

        return True
