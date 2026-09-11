#!/usr/bin/env python3

"""
servo_control.py

Control a hobby servo from Raspberry Pi GPIO.

Usage:
    python3 servo_control.py 90
    python3 servo_control.py 45
    python3 servo_control.py 180
"""

import argparse
import time
from gpiozero import AngularServo


SERVO_PIN = 18          # BCM GPIO number

MIN_ANGLE = 0
MAX_ANGLE = 180

# Typical hobby servo pulse widths, in seconds
MIN_PULSE_WIDTH = 0.5 / 1000
MAX_PULSE_WIDTH = 2.5 / 1000


def set_servo_angle(angle):
    """Move servo to specified angle."""

    if not MIN_ANGLE <= angle <= MAX_ANGLE:
        raise ValueError(
            f"Angle must be between {MIN_ANGLE} and {MAX_ANGLE} degrees."
        )

    servo = AngularServo(
        SERVO_PIN,
        min_angle=MIN_ANGLE,
        max_angle=MAX_ANGLE,
        min_pulse_width=MIN_PULSE_WIDTH,
        max_pulse_width=MAX_PULSE_WIDTH,
    )

    try:
        print(f"Moving servo to {angle:.1f} degrees")
        servo.angle = angle

        # Give servo enough time to reach the target.
        time.sleep(0.7)

    finally:
        # Stop sending PWM after reaching target.
        servo.detach()
        servo.close()


def main():
    parser = argparse.ArgumentParser(
        description="Move Raspberry Pi servo to specified angle."
    )

    parser.add_argument(
        "angle",
        type=float,
        help="Target angle in degrees (0-180)"
    )

    args = parser.parse_args()

    set_servo_angle(args.angle)


if __name__ == "__main__":
    main()
