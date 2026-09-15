#!/usr/bin/env python3

"""
gripper_client.py

Client for the separate gripper_node process (see gripper_node.py).
All servo/GPIO access happens over there, on its own executor - the
same split scan_record.py uses for scan_recorder_node.

Only blocking calls are provided: unlike a mid-scan checkpoint
save (safe to fire-and-forget), retrieve.py always needs to know
the gripper actually finished opening/closing before its next
motion step, so there is no async flavour here.
"""

import rclpy
from std_srvs.srv import Trigger


class GripperClient:

    def __init__(
        self,
        node,
        open_service="open_gripper",
        close_service="close_gripper",
    ):
        self.node = node

        self._open_client = node.create_client(Trigger, open_service)
        self._close_client = node.create_client(Trigger, close_service)

    def open_blocking(self, timeout_sec=5.0):
        """Open the gripper and wait for confirmation."""
        return self._call_blocking(self._open_client, "open_gripper", timeout_sec)

    def close_blocking(self, timeout_sec=5.0):
        """Close the gripper and wait for confirmation."""
        return self._call_blocking(self._close_client, "close_gripper", timeout_sec)

    def _call_blocking(self, client, name, timeout_sec):

        if not client.wait_for_service(timeout_sec=timeout_sec):

            self.node.get_logger().warning(
                f"'{name}' service not available after "
                f"{timeout_sec:.1f}s (is gripper_node running?)."
            )

            return False

        future = client.call_async(Trigger.Request())

        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=timeout_sec,
        )

        if not future.done():
            self.node.get_logger().warning(f"'{name}' timed out waiting for a response.")
            return False

        try:
            response = future.result()
        except Exception as exc:
            self.node.get_logger().warning(f"'{name}' call failed: {exc}")
            return False

        if not response.success:
            self.node.get_logger().warning(
                f"'{name}' reported failure: {response.message}"
            )
            return False

        self.node.get_logger().info(f"'{name}' succeeded: {response.message}")
        return True
