"""
The gripper servo on the ESP32, driven over MQTT (esp32_protocol.py).

MqttServoDriver has ServoDriver's interface (servo.py): move() sends
the command and blocks until the ESP32 acknowledges the finished move,
so gripper_node's open/close services mean the same thing with either
backend. A missing or failed ack raises, which gripper_node turns into
a failed service call - as it does for a GPIO error.

The ESP32 makes the pulses with its LEDC hardware PWM, so they are
exact however busy the Pi is (compare servo.py, SOFTWARE PWM). That
should also make hold_closed worth trying again - untested.
"""

import itertools
import threading
import time

from . import esp32_protocol
from .servo import SETTLE_SEC, validate_angle

# Seconds past settle_sec to wait for the ack: the WiFi round trip,
# plus slack for a reconnect in progress.
ACK_MARGIN_SEC = 2.0


class GripperLinkError(RuntimeError):
    """The ESP32 did not confirm a move."""


class MqttServoDriver:
    """
    ServoDriver's interface, with the servo on the ESP32.

    link: a started mqtt_link.MqttLink (or a stand-in, for tests).
    """

    def __init__(self, link, topics, settle_sec=SETTLE_SEC,
                 ack_margin_sec=ACK_MARGIN_SEC):
        self._link = link
        self._topics = topics
        self._settle_sec = settle_sec
        self._ack_margin_sec = ack_margin_sec

        # Ids start from the clock, so a late ack meant for a node that
        # has since restarted can never match this one's command.
        self._ids = itertools.count(int(time.time() * 1000) % 2**31)
        self._lock = threading.Lock()
        self._waiting_id = None
        self._ack = None
        self._acked = threading.Event()
        self.esp32_status = None

        link.subscribe(topics.gripper_ack, self._on_ack)
        link.subscribe(topics.status, self._on_status, qos=1)

    def wait_connected(self, timeout_sec):
        """Wait for the broker connection; False if it did not come."""
        return self._link.wait_connected(timeout_sec)

    def _on_status(self, payload):
        self.esp32_status = payload.decode(errors='replace')

    def _on_ack(self, payload):
        ack = esp32_protocol.decode_gripper_ack(payload)

        if ack is None:
            return

        with self._lock:
            if ack[0] != self._waiting_id:
                return  # A late ack for a move that already timed out.

            self._ack = ack
            self._acked.set()

    def move(self, angle, hold=False, settle_sec=None):
        """Move to `angle` degrees; return once the ESP32 says it is there."""
        validate_angle(angle)
        settle_sec = self._settle_sec if settle_sec is None else settle_sec

        if self.esp32_status == esp32_protocol.OFFLINE:
            raise GripperLinkError('the ESP32 is offline (its MQTT last will)')

        if not self._link.connected:
            raise GripperLinkError('not connected to the MQTT broker')

        with self._lock:
            cmd_id = next(self._ids)
            self._waiting_id = cmd_id
            self._ack = None
            self._acked.clear()

        payload = esp32_protocol.encode_gripper_cmd(
            cmd_id, angle, hold, settle_sec
        )

        try:
            if not self._link.publish(self._topics.gripper_cmd, payload, qos=1):
                raise GripperLinkError('could not publish the gripper command')

            timeout = settle_sec + self._ack_margin_sec

            if not self._acked.wait(timeout):
                raise GripperLinkError(
                    f'no ack from the ESP32 within {timeout:.1f} s'
                )

            _id, ok, message = self._ack

            if not ok:
                raise GripperLinkError(f'the ESP32 refused the move: {message}')

        finally:
            with self._lock:
                self._waiting_id = None

    def release(self):
        """
        Do nothing: the ESP32 releases the servo by itself.

        It stops the pulses after a move unless asked to hold. Losing
        the broker changes nothing, so a held claw stays held.
        """

    def close(self):
        """Stop the MQTT connection."""
        self._link.stop()
