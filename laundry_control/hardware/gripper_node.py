#!/usr/bin/env python3

r"""
Standalone node that owns the gripper's servo hardware.

The servo is on this Pi's GPIO (backend:=gpio, servo.py) or on the
ESP32, commanded over MQTT (backend:=mqtt, mqtt_servo.py). The
services behave the same either way - they return once the move has
finished. Default: config.GRIPPER_BACKEND.

Callers like the grasp stage never need direct GPIO access - the
same split already used for the ToF sensor (tof_sensor.py) and the
scan recorder (scan/recorder_node.py).

Services (std_srvs/Trigger):

    open_gripper  - move the servo to the `open_angle_deg` param,
                    then stop driving it.
    close_gripper - move the servo to the `close_angle_deg` param,
                    then stop driving it too - unless the
                    `hold_closed` param is true, which keeps driving
                    it until the next open (see servo.py, HOLDING).
                    The servo is released when the node shuts down.

Which raw servo angle actually opens/closes the physical claw
depends on how it's linked to the servo horn - the defaults
(config.GRIPPER_OPEN_ANGLE_DEG / GRIPPER_CLOSE_ANGLE_DEG) are
calibrated for the current hardware; override via --ros-args if
that ever changes, rather than editing code:

    ros2 run laundry_control gripper_node --ros-args \\
        -p open_angle_deg:=20.0 -p close_angle_deg:=160.0

HARDWARE ONLY. Usage:
    ros2 run laundry_control gripper_node
    ros2 run laundry_control gripper_node --ros-args -p backend:=mqtt
"""

import os

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from .gripper_client import CLOSE_SERVICE, OPEN_SERVICE
from .servo import SETTLE_SEC
from ..config import (
    GRIPPER_BACKEND,
    GRIPPER_BACKENDS,
    GRIPPER_CLOSE_ANGLE_DEG,
    GRIPPER_OPEN_ANGLE_DEG,
    MQTT_HOST,
    MQTT_PORT,
    MQTT_TOPIC_PREFIX,
)


def enter_realtime(priority):
    """
    Put the calling thread on real-time (SCHED_FIFO) scheduling.

    Threads it starts afterwards inherit it - lgpio's pulse thread
    included. Returns None on success, else why not (typically: this
    user may not use real-time priority).
    """
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except (OSError, AttributeError) as exc:
        return str(exc)

    return None


def make_mqtt_servo(host, port, prefix, client_id, log_info=print,
                    log_warn=print):
    """Return a started mqtt_servo.MqttServoDriver for the ESP32's servo."""
    from . import esp32_protocol
    from .mqtt_link import MqttLink
    from .mqtt_servo import MqttServoDriver

    link = MqttLink(host, port, client_id, log_info=log_info,
                    log_warn=log_warn)
    driver = MqttServoDriver(link, esp32_protocol.topics(prefix))
    link.start()
    return driver


def leave_realtime():
    """Put the calling thread back on normal scheduling."""
    try:
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
    except OSError:
        pass


class GripperNode(Node):

    def __init__(self):
        super().__init__('gripper_node')

        self.declare_parameter('backend', GRIPPER_BACKEND)
        self.declare_parameter('mqtt_host', MQTT_HOST)
        self.declare_parameter('mqtt_port', MQTT_PORT)
        self.declare_parameter('mqtt_prefix', MQTT_TOPIC_PREFIX)
        self.declare_parameter('open_angle_deg', GRIPPER_OPEN_ANGLE_DEG)
        self.declare_parameter('close_angle_deg', GRIPPER_CLOSE_ANGLE_DEG)
        # Off: on the rig, holding with the Pi's software-timed PWM
        # made the closed claw shake and grip worse (2026-09-29) than
        # stopping the pulses, which it holds without.
        self.declare_parameter('hold_closed', False)
        # Seconds each move keeps sending pulses before they stop (see
        # servo.SETTLE_SEC); longer if the claw does not finish moving.
        self.declare_parameter('settle_sec', SETTLE_SEC)
        # Real-time (SCHED_FIFO) priority for lgpio's pulse thread, so
        # the busy Pi does not delay its pulses (servo.py, REAL-TIME
        # PRIORITY); 0 turns it off. Above ros2_control's 50: it only
        # runs while the claw moves, when the arm is standing still.
        self.declare_parameter('rt_priority', 60)

        self._open_srv = self.create_service(
            Trigger, OPEN_SERVICE, self._open_callback
        )

        self._close_srv = self.create_service(
            Trigger, CLOSE_SERVICE, self._close_callback
        )

        # One driver for the node's lifetime, so a closed claw keeps
        # its PWM between service calls. The GPIO one is created
        # lazily (see _move_to); the MQTT one now, so it is connected
        # before the first call.
        self._driver = None
        self.backend = self.get_parameter('backend').value

        if self.backend not in GRIPPER_BACKENDS:
            raise ValueError(
                f'backend must be one of {GRIPPER_BACKENDS}, not '
                f'{self.backend!r}'
            )

        if self.backend == 'mqtt':
            self._driver = make_mqtt_servo(
                self.get_parameter('mqtt_host').value,
                int(self.get_parameter('mqtt_port').value),
                self.get_parameter('mqtt_prefix').value,
                client_id='laundry_gripper_node',
                log_info=self.get_logger().info,
                log_warn=self.get_logger().warning,
            )

        self.get_logger().info(f'gripper_node ready ({self.backend} servo).')

    def _open_callback(self, request, response):
        return self._move_to(
            self.get_parameter('open_angle_deg').value, response, hold=False
        )

    def _close_callback(self, request, response):
        return self._move_to(
            self.get_parameter('close_angle_deg').value,
            response,
            hold=bool(self.get_parameter('hold_closed').value),
        )

    def _move_to(self, angle, response, hold):
        realtime = False

        try:
            # Inside the try: the first servo call is what actually
            # initialises the GPIO pin factory (see servo.py), so a
            # hardware/permission problem there must fail this one
            # service call, not crash the whole node.
            if self._driver is None:
                from .servo import ServoDriver

                self._driver = ServoDriver()

                # The first move starts lgpio's pulse thread, which
                # inherits this thread's scheduling: see enter_realtime.
                realtime = self._enter_realtime()

            self._driver.move(
                angle,
                hold=hold,
                settle_sec=float(self.get_parameter('settle_sec').value),
            )

        except Exception as exc:
            if realtime:
                leave_realtime()

            response.success = False
            response.message = f'Failed to set servo to {angle} deg: {exc}'
            self.get_logger().error(response.message)
            return response

        if realtime:
            leave_realtime()

        response.success = True
        response.message = (
            f'Servo set to {angle} deg' + (', holding.' if hold else '.')
        )
        return response

    def _enter_realtime(self):
        """Switch to real-time priority for the pulse thread; True if it did."""
        priority = int(self.get_parameter('rt_priority').value)

        if priority <= 0:
            return False

        reason = enter_realtime(priority)

        if reason is None:
            self.get_logger().info(
                f'Servo pulse thread at real-time priority {priority}.'
            )
            return True

        self.get_logger().warning(
            f'Servo pulses at normal priority ({reason}): the claw may '
            'jitter under load. Allow real-time priority for this user '
            '(see servo.py, REAL-TIME PRIORITY) and restart.'
        )
        return False

    def release(self):
        """Stop driving the servo and free the pin (on shutdown)."""
        if self._driver is not None:
            self._driver.close()


def main(args=None):
    rclpy.init(args=args)

    node = GripperNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
