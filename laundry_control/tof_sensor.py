"""
tof_sensor.py

Simple VL53L0X interface for Raspberry Pi.

Wiring:
    VL53L0X VCC -> RPi 3.3V
    VL53L0X GND -> RPi GND
    VL53L0X SDA -> RPi GPIO 2  (Pin 3)
    VL53L0X SCL -> RPi GPIO 3  (Pin 5)

Library:
    pip install adafruit-circuitpython-vl53l0x
"""

import time

import board
import busio
import adafruit_vl53l0x


class ToFSensor:
    def __init__(self, offset_cm=-1.0):
        """
        Initialize VL53L0X sensor.

        Args:
            offset_cm:
                Calibration offset added to the raw measurement.
                Default -1.0 cm matches the Arduino code.
        """

        self.offset_cm = offset_cm

        print("Initializing VL53L0X...")

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl53l0x.VL53L0X(self.i2c)

        print("VL53L0X ready.")

    def get_distance_mm(self):
        """Return current distance in millimeters."""
        return self.sensor.range

    def get_distance_cm(self):
        """Return current calibrated distance in centimeters."""
        distance_mm = self.sensor.range

        distance_cm = distance_mm / 10.0
        distance_cm += self.offset_cm

        return distance_cm


def main():
    tof = ToFSensor(offset_cm=-1.0)

    try:
        while True:
            distance = tof.get_distance_cm()

            print(f"Distance: {distance:.1f} cm")

            time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nStopping VL53L0X.")


if __name__ == "__main__":
    main()