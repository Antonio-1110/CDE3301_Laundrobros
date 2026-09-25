#!/usr/bin/env python3

"""
Client for the separate scan_recorder_node process (see recorder_node.py).

All ToF capture, TF lookups, point-cloud
building and CSV I/O happen over there, on their own executor.

Two flavours of call are provided:

    *_async()    - fire-and-forget, returns immediately. Use ONLY
                   while the arm may be moving (e.g. mid-scan
                   checkpoint saves), so this can never block the
                   arm-control node's spin loop (arm/controller.py).

    *_blocking() - waits for scan_recorder_node's response (bounded
                   by timeout_sec). Only safe to use when the arm is
                   stationary - e.g. clearing old points before a
                   scan starts, or the final save before shutdown,
                   where we actually need to know it succeeded and
                   there's no risk to arm control from waiting.

set_csv_path() pins exactly which file the next save writes, so a
caller (`laundry scan --save`, `laundry run`, baseline collection)
knows what the scan produced instead of guessing at the recorder's
auto-generated timestamped name.

hardware.fake.FakeRecorder has the same interface, for running with
no ToF sensor.
"""

from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
import rclpy
from std_srvs.srv import Trigger

SCAN_RECORDER_NODE_NAME = 'scan_recorder_node'
SAVE_SERVICE = 'save_scan'
CLEAR_SERVICE = 'clear_scan'


def set_remote_string_param(
    node, remote_node_name, param_name, value, timeout_sec=5.0
):
    """
    Set a string parameter on a DIFFERENT, already-running node.

    Goes through that node's own /<name>/set_parameters service.
    Returns True on success; logs why and returns False otherwise.
    """
    client = node.create_client(
        SetParameters, f'/{remote_node_name}/set_parameters'
    )

    if not client.wait_for_service(timeout_sec=timeout_sec):
        node.get_logger().error(
            f"'/{remote_node_name}/set_parameters' not available "
            f'after {timeout_sec:.1f}s (is {remote_node_name} running?).'
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
            f'{[r.reason for r in response.results if not r.successful]}'
        )

        return False

    return True


class ScanRecorderClient:

    def __init__(
        self,
        arm,
        save_service=SAVE_SERVICE,
        clear_service=CLEAR_SERVICE,
        recorder_node_name=SCAN_RECORDER_NODE_NAME,
    ):
        self.arm = arm
        self.recorder_node_name = recorder_node_name

        self._save_client = arm.create_client(Trigger, save_service)
        self._clear_client = arm.create_client(Trigger, clear_service)

        # Keep references so in-flight futures aren't garbage
        # collected before their done-callback fires.
        self._pending_futures = []

    def set_csv_path(self, csv_path, timeout_sec=5.0):
        """
        Pin the recorder's output file ("" restores auto-naming).

        Callers that pin a path must restore "" afterwards (in a
        finally): left pinned, the NEXT ordinary scan silently
        overwrites this one.
        """
        return set_remote_string_param(
            self.arm,
            self.recorder_node_name,
            'csv_path',
            csv_path,
            timeout_sec=timeout_sec,
        )

    def clear_async(self):
        """Reset scan_recorder_node's accumulated points (fire-and-forget)."""
        self._call_async(self._clear_client, 'clear_scan')

    def save_async(self):
        """Ask scan_recorder_node to checkpoint to CSV (fire-and-forget)."""
        self._call_async(self._save_client, 'save_scan')

    def clear_blocking(self, timeout_sec=5.0):
        """
        Reset the recorder's points and wait for confirmation.

        Reset scan_recorder_node's accumulated points and wait for
        confirmation. Only call this before the arm starts moving.
        """
        return self._call_blocking(self._clear_client, 'clear_scan', timeout_sec)

    def save_blocking(self, timeout_sec=5.0):
        """
        Save the recorder's CSV and wait for confirmation.

        Ask scan_recorder_node to save to CSV and wait for
        confirmation. Only call this once the arm has stopped
        moving (e.g. the final save before shutdown).
        """
        return self._call_blocking(self._save_client, 'save_scan', timeout_sec)

    def _call_async(self, client, name):

        if not client.service_is_ready():

            self.arm.get_logger().warning(
                f"'{name}' service not available "
                '(is scan_recorder_node running?); skipping.',
                throttle_duration_sec=5.0,
            )

            return

        future = client.call_async(Trigger.Request())

        self._pending_futures.append(future)

        future.add_done_callback(
            lambda f: self._on_done(f, name)
        )

    def _on_done(self, future, name):

        if future in self._pending_futures:
            self._pending_futures.remove(future)

        try:
            response = future.result()
        except Exception as exc:
            self.arm.get_logger().warning(f"'{name}' call failed: {exc}")
            return

        if not response.success:
            self.arm.get_logger().warning(
                f"'{name}' reported failure: {response.message}"
            )

    def _call_blocking(self, client, name, timeout_sec):

        if not client.wait_for_service(timeout_sec=timeout_sec):

            self.arm.get_logger().warning(
                f"'{name}' service not available after "
                f'{timeout_sec:.1f}s (is scan_recorder_node running?).'
            )

            return False

        future = client.call_async(Trigger.Request())

        rclpy.spin_until_future_complete(
            self.arm,
            future,
            timeout_sec=timeout_sec,
        )

        if not future.done():
            self.arm.get_logger().warning(f"'{name}' timed out waiting for a response.")
            return False

        try:
            response = future.result()
        except Exception as exc:
            self.arm.get_logger().warning(f"'{name}' call failed: {exc}")
            return False

        if not response.success:
            self.arm.get_logger().warning(
                f"'{name}' reported failure: {response.message}"
            )
            return False

        self.arm.get_logger().info(f"'{name}' succeeded: {response.message}")
        return True
