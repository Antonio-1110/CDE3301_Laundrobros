#!/usr/bin/env python3

"""
Drive the gripper's hobby servo from Raspberry Pi GPIO.

HARDWARE PWM (preferred)
------------------------
When the Pi 5's RP1 PWM channel on GPIO 18 has been set up (see
setup/servo-pwm/README.md: a config.txt overlay plus a boot service
that exports the channel to this user), the pulses come from the PWM
hardware - exact whatever the CPU load. HardwarePwmServo drives it
through the kernel's sysfs PWM interface. Otherwise the lgpio
software PWM below is used, which on a loaded Pi makes the servo
jitter and miss moves (2026-09-29).

SOFTWARE PWM (fallback)
-----------------------
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
ServoDriver can keep driving the servo after a move (hold=True) until
the next move or release(), or stop the pulses (hold=False). Some
hobby servos go limp without pulses; the rig's does not - it held the
claw closed with the pulses stopped. Holding, on the other hand, made
it shake and grip worse (2026-09-29): lgpio's PWM is software timed
too (tx_pwm, from a C thread), so on the loaded Pi pulses come out
slightly off and the servo keeps chasing them. gripper_node therefore
stops the pulses after close as well, by default; its `hold_closed`
parameter turns holding back on (worth it with hardware-timed
pulses).

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

# How long a move keeps sending pulses (50 a second) before they stop,
# so the servo has time to reach the target - against a load too.
# 1.5 s, up from 0.7: at 0.7 the claw did not reliably finish its
# moves on the rig (2026-09-29). gripper_node's settle_sec overrides.
SETTLE_SEC = 1.5

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


# The RP1 PWM block whose channel 2 is GPIO 18 on the Pi 5, as its
# device path ends; and the servo's pulse period (50 Hz).
RP1_PWM_DEVICE = '1f00098000.pwm'
SERVO_PWM_CHANNEL = 2
SERVO_PERIOD_NS = 20_000_000
PWM_CLASS_DIR = '/sys/class/pwm'


def hardware_pwm_channel(pwm_class_dir=PWM_CLASS_DIR):
    """
    Return the sysfs directory of the servo's hardware PWM channel, or None.

    Only a channel that is exported and writable counts (the boot
    service in setup/servo-pwm does both), so without that setup the
    servo falls back to software PWM instead of failing.
    """
    import glob
    import os

    for chip in sorted(glob.glob(os.path.join(pwm_class_dir, 'pwmchip*'))):
        device = os.path.realpath(os.path.join(chip, 'device'))

        if not device.endswith(RP1_PWM_DEVICE):
            continue

        channel = os.path.join(chip, f'pwm{SERVO_PWM_CHANNEL}')

        if all(
            os.access(os.path.join(channel, name), os.W_OK)
            for name in ('period', 'duty_cycle', 'enable')
        ):
            return channel

    return None


class HardwarePwmServo:
    """
    The gpiozero AngularServo surface ServoDriver uses, on hardware PWM.

    angle = ... sets the pulse width and turns the output on; value is
    None while it is off (no pulses), as with gpiozero; detach() turns
    it off.
    """

    def __init__(self, channel_dir):
        self._dir = channel_dir
        self.value = None
        self._angle = None

        self._write('enable', 0)
        self._write('period', SERVO_PERIOD_NS)

    def _write(self, name, value):
        import os

        with open(os.path.join(self._dir, name), 'w') as handle:
            handle.write(str(int(value)))

    @property
    def angle(self):
        """Return the last commanded position in degrees, or None."""
        return self._angle

    @angle.setter
    def angle(self, angle):
        span = SERVO_MAX_ANGLE_DEG - SERVO_MIN_ANGLE_DEG
        fraction = (angle - SERVO_MIN_ANGLE_DEG) / span
        pulse_s = SERVO_MIN_PULSE_WIDTH_S + fraction * (
            SERVO_MAX_PULSE_WIDTH_S - SERVO_MIN_PULSE_WIDTH_S
        )

        self._write('duty_cycle', round(pulse_s * 1e9))
        self._write('enable', 1)

        self._angle = angle
        self.value = 2.0 * fraction - 1.0

    def detach(self):
        """Stop the pulses."""
        self._write('enable', 0)
        self.value = None

    def close(self):
        """Stop the pulses (the channel stays exported for next time)."""
        self.detach()


def _make_servo():
    channel = hardware_pwm_channel()

    if channel is not None:
        return HardwarePwmServo(channel)

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

    def move(self, angle, hold=False, settle_sec=None):
        """
        Move to `angle` degrees and wait for it to get there.

        Pulses go out for settle_sec (default: the driver's). hold=True
        keeps driving it afterwards (until the next move or release());
        hold=False stops the pulses then.
        """
        validate_angle(angle)

        if self._servo is None:
            self._servo = self._make_servo()

        try:
            self._servo.angle = angle
            time.sleep(self._settle_sec if settle_sec is None else settle_sec)

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
