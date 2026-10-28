#!/usr/bin/env python3

"""
Client for the separate gripper_node process (see gripper_node.py).

All servo/GPIO access happens over there, on its own executor - the
same split scan/recorder_client.py uses for scan_recorder_node.

The grasp stage always needs to know the gripper actually finished
opening/closing before its next motion step. open_blocking() and
close_blocking() wait for that; start_open() and start_close() send
the request and return a Pending whose wait() does, so the arm can
check its next motion while the claw moves (grasp/retrieve_grid.run).

hardware.fake.FakeGripper has the same interface, for running the
pipeline with no servo attached.
"""

import rclpy
from std_srvs.srv import Trigger

OPEN_SERVICE = 'open_gripper'
CLOSE_SERVICE = 'close_gripper'


class Pending:
    """A gripper move under way; wait() returns True once it is confirmed."""

    def __init__(self, finish):
        self._finish = finish
        self._result = None

    def wait(self):
        """Wait for the gripper's answer (once); True if it confirmed the move."""
        if self._result is None:
            self._result = bool(self._finish())

        return self._result


class GripperClient:

    def __init__(
        self,
        node,
        open_service=OPEN_SERVICE,
        close_service=CLOSE_SERVICE,
    ):
        self.node = node

        self._open_client = node.create_client(Trigger, open_service)
        self._close_client = node.create_client(Trigger, close_service)

    def open_blocking(self, timeout_sec=5.0):
        """Open the gripper and wait for confirmation."""
        return self.start_open(timeout_sec).wait()

    def close_blocking(self, timeout_sec=5.0):
        """Close the gripper and wait for confirmation."""
        return self.start_close(timeout_sec).wait()

    def start_open(self, timeout_sec=5.0):
        """Ask the gripper to open; wait() on the result for confirmation."""
        return self._start(self._open_client, OPEN_SERVICE, timeout_sec)

    def start_close(self, timeout_sec=5.0):
        """Ask the gripper to close; wait() on the result for confirmation."""
        return self._start(self._close_client, CLOSE_SERVICE, timeout_sec)

    def _start(self, client, name, timeout_sec):
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.node.get_logger().warning(
                f"'{name}' service not available after "
                f'{timeout_sec:.1f}s (is gripper_node running?).'
            )

            return Pending(lambda: False)

        future = client.call_async(Trigger.Request())

        return Pending(lambda: self._finish(future, name, timeout_sec))

    def _finish(self, future, name, timeout_sec):
        rclpy.spin_until_future_complete(
            self.node,
            future,
            timeout_sec=timeout_sec,
        )

        if not future.done():
            self.node.get_logger().warning(
                f"'{name}' timed out waiting for a response."
            )
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
