#!/usr/bin/env python3

"""
Drive the gripper's hobby servo from Raspberry Pi GPIO.

Uses the lgpio pin factory instead of gpiozero's default
(RPi.GPIO/software) factory: the default factory times PWM pulses
from a Python thread, which is subject to OS scheduling jitter -
especially on a Pi also running ROS 2/MoveIt under load - and that
jitter shows up directly as inconsistent servo angles (this is
exactly what gpiozero's own PWMSoftwareFallback warning is about).

NOTE: this used to use the pigpio pin factory instead, but pigpio
cannot work at all on a Raspberry Pi 5 - it memory-maps the legacy
BCM283x GPIO registers directly via /dev/mem, while the Pi 5 moved
GPIO handling to a separate RP1 chip with a different register
layout entirely. pigpio's author archived the project before ever
adding Pi 5 support (which is why it's no longer in Raspberry Pi
OS/Ubuntu's apt repos - there's no version that would work here).
lgpio is the actively-maintained replacement, talks to the kernel's
gpiochip character-device interface (which DOES support the Pi 5),
and ships as python3-lgpio - no extra daemon/install needed.

The gpiozero/lgpio imports happen on the first set_servo_angle()
call, not at import time, so this module (and the CLI that uses it)
can be imported on a machine with no GPIO at all. Importing gpiozero
with the lgpio factory is what claims the GPIO chip, so a missing
library or a permissions problem surfaces as an exception from that
first call - which gripper_node turns into a failed service response
instead of a crashed node.

HOLDING
-------
A hobby servo only holds its position while it receives pulses: with
PWM stopped it goes limp, and a claw closed on laundry springs open
under the load on the way to DROP. ServoDriver therefore keeps
driving the servo after a move with hold=True (gripper_node does this
for close), and stops only after hold=False moves (open - holding an
unloaded servo just makes it jitter and warm up). The driver stays
alive in gripper_node, so the hold lasts until the next command.

HARDWARE ONLY. From the terminal: `laundry gripper <ANGLE>` (a
one-shot process, so it never holds).
"""

import time

from ..config import (
    SERVO_MAX_ANGLE_DEG,
    SERVO_MAX_PULSE_WIDTH_S,
    SERVO_MIN_ANGLE_DEG,
    SERVO_MIN_PULSE_WIDTH_S,
    SERVO_PIN,
)

# Give the servo enough time to reach the target before PWM stops.
SETTLE_SEC = 0.7

_pin_factory_ready = False


def _ensure_pin_factory():
    """Select the lgpio pin factory once, on first use."""
    global _pin_factory_ready

    if _pin_factory_ready:
        return

    from gpiozero import Device
    from gpiozero.pins.lgpio import LGPIOFactory

    Device.pin_factory = LGPIOFactory()

    _pin_factory_ready = True


def validate_angle(angle):
    """Raise ValueError unless the angle is within the servo's range."""
    if not SERVO_MIN_ANGLE_DEG <= angle <= SERVO_MAX_ANGLE_DEG:
        raise ValueError(
            f'Angle must be between {SERVO_MIN_ANGLE_DEG:g} and '
            f'{SERVO_MAX_ANGLE_DEG:g} degrees.'
        )


def _make_servo():
    _ensure_pin_factory()

    from gpiozero import AngularServo

    return AngularServo(
        SERVO_PIN,
        min_angle=SERVO_MIN_ANGLE_DEG,
        max_angle=SERVO_MAX_ANGLE_DEG,
        min_pulse_width=SERVO_MIN_PULSE_WIDTH_S,
        max_pulse_width=SERVO_MAX_PULSE_WIDTH_S,
    )


class ServoDriver:
    """
    A long-lived servo that can keep holding its position.

    make_servo is injectable so this can be tested without GPIO.
    """

    def __init__(self, make_servo=_make_servo, settle_sec=SETTLE_SEC):
        self._make_servo = make_servo
        self._settle_sec = settle_sec
        self._servo = None

    @property
    def holding(self):
        """Return True while PWM is being sent."""
        return self._servo is not None and self._servo.value is not None

    def move(self, angle, hold=False):
        """
        Move to `angle` degrees and wait for it to get there.

        hold=True keeps driving it afterwards (until the next move or
        release()); hold=False stops the pulses.
        """
        validate_angle(angle)

        if self._servo is None:
            self._servo = self._make_servo()

        try:
            self._servo.angle = angle
            time.sleep(self._settle_sec)

        except BaseException:
            self._servo.detach()
            raise

        if not hold:
            self._servo.detach()

    def release(self):
        """Stop the pulses (the servo goes limp)."""
        if self._servo is not None:
            self._servo.detach()

    def close(self):
        """Stop the pulses and free the GPIO pin."""
        if self._servo is not None:
            self._servo.detach()
            self._servo.close()
            self._servo = None


def set_servo_angle(angle):
    """Move the servo to `angle` degrees, then stop driving it."""
    driver = ServoDriver()

    try:
        print(f'Moving servo to {angle:.1f} degrees')
        driver.move(angle, hold=False)

    finally:
        driver.close()
