"""
tof_sensor.py

ROS2 node that publishes VL53L0X time-of-flight distance readings.

Wiring:
    VL53L0X VCC -> RPi 3.3V
    VL53L0X GND -> RPi GND
    VL53L0X SDA -> RPi GPIO 2  (Pin 3)
    VL53L0X SCL -> RPi GPIO 3  (Pin 5)

Library:
    pip install adafruit-circuitpython-vl53l0x

Usage:
    ros2 run laundry_control tof_sensor
"""

import board
import busio
import adafruit_vl53l0x

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range

# VL53L0X datasheet: ~25 degree field of view, usable range ~30mm-2000mm.
FIELD_OF_VIEW_RAD = 0.436
MIN_RANGE_M = 0.03
MAX_RANGE_M = 2.0

PUBLISH_PERIOD_SEC = 0.05


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

        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_vl53l0x.VL53L0X(self.i2c)

    def get_distance_mm(self):
        """Return current distance in millimeters."""
        return self.sensor.range

    def get_distance_cm(self):
        """Return current calibrated distance in centimeters."""
        distance_mm = self.sensor.range

        distance_cm = distance_mm / 10.0
        distance_cm += self.offset_cm

        return distance_cm


class ToFSensorNode(Node):
    def __init__(self):
        super().__init__('tof_sensor')

        self.declare_parameter('offset_cm', -1.0)
        self.declare_parameter('frame_id', 'tof_sensor_link')

        offset_cm = self.get_parameter('offset_cm').value
        self.frame_id = self.get_parameter('frame_id').value

        self.get_logger().info('Initializing VL53L0X...')
        self.tof = ToFSensor(offset_cm=offset_cm)
        self.get_logger().info('VL53L0X ready.')

        self.publisher = self.create_publisher(Range, 'tof_sensor/range', 10)
        self.timer = self.create_timer(PUBLISH_PERIOD_SEC, self.publish_reading)

    def publish_reading(self):
        distance_cm = self.tof.get_distance_cm()

        msg = Range()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = FIELD_OF_VIEW_RAD
        msg.min_range = MIN_RANGE_M
        msg.max_range = MAX_RANGE_M
        msg.range = distance_cm / 100.0

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = ToFSensorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
