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
import RPi.GPIO as GPIO


SERVO_PIN = 18          # BCM GPIO number
PWM_FREQUENCY = 50      # Standard hobby servo frequency: 50 Hz

MIN_ANGLE = 0
MAX_ANGLE = 180

# Typical hobby servo pulse widths
MIN_PULSE_MS = 0.5
MAX_PULSE_MS = 2.5
PERIOD_MS = 20.0        # 50 Hz = 20 ms period


def angle_to_duty_cycle(angle):
    """Convert servo angle to PWM duty cycle."""

    if not MIN_ANGLE <= angle <= MAX_ANGLE:
        raise ValueError(
            f"Angle must be between {MIN_ANGLE} and {MAX_ANGLE} degrees."
        )

    pulse_ms = MIN_PULSE_MS + (
        (angle - MIN_ANGLE)
        / (MAX_ANGLE - MIN_ANGLE)
        * (MAX_PULSE_MS - MIN_PULSE_MS)
    )

    return pulse_ms / PERIOD_MS * 100


def set_servo_angle(angle):
    """Move servo to specified angle."""

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SERVO_PIN, GPIO.OUT)

    pwm = GPIO.PWM(SERVO_PIN, PWM_FREQUENCY)
    pwm.start(0)

    try:
        duty_cycle = angle_to_duty_cycle(angle)

        print(f"Moving servo to {angle:.1f} degrees")
        pwm.ChangeDutyCycle(duty_cycle)

        # Give servo enough time to reach the target.
        time.sleep(0.7)

        # Stop sending PWM after reaching target.
        pwm.ChangeDutyCycle(0)

    finally:
        pwm.stop()
        GPIO.cleanup()


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