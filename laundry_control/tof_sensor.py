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

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Range

# VL53L0X datasheet: ~25 degree field of view, usable range ~30mm-2000mm.
FIELD_OF_VIEW_RAD = 0.436
MIN_RANGE_M = 0.03
MAX_RANGE_M = 0.26

PUBLISH_PERIOD_SEC = 0.05

# =============================================================
# TOF SENSOR MOUNTING OFFSET
#
# Translation from the TCP frame (link7) to the ToF sensor's own
# frame. Axis-aligned with link7 (no rotation offset) - the
# sensor's +X (its boresight, per REP 117 / sensor_msgs/Range)
# points along link7's +X.
#
# Measured at the INTER pose, where check_flange.py confirms
# link7's local +Z points straight down:
#
#   - "2.8 cm below the TCP origin"  -> local +Z (down at INTER)
#   - "7.75 cm in front of the TCP"  -> local +X (the sensor's own
#     boresight axis - it's mounted looking forward/outward, offset
#     slightly along the same direction it looks)
#
# Because the offset is expressed in link7's own frame, it is
# joint7-invariant: it stays correct as J7 rotates during a scan.
#
# These constants have no hardware dependencies (unlike the rest
# of this file, which needs board/busio/adafruit_vl53l0x), so
# other modules (e.g. scan_recorder_node.py) can import just
# these without pulling in Pi-only libraries - the hardware
# imports below are deferred into ToFSensor.__init__ for exactly
# that reason.
# =============================================================

TOF_SENSOR_FRAME = "tof_sensor_link"

TOF_SENSOR_OFFSET_X = 0.0775
TOF_SENSOR_OFFSET_Y = 0.0
TOF_SENSOR_OFFSET_Z = 0.028


class ToFSensor:
    def __init__(self, offset_cm=-10.0):
        """
        Initialize VL53L0X sensor.

        Args:
            offset_cm:
                Calibration offset added to the raw measurement.
                Default -10.0 cm matches the Arduino code.
        """

        import board
        import busio
        import adafruit_vl53l0x

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

        self.declare_parameter('offset_cm', -10.0)
        self.declare_parameter('frame_id', TOF_SENSOR_FRAME)
        self.declare_parameter('flange_link', 'link7')

        offset_cm = self.get_parameter('offset_cm').value
        self.frame_id = self.get_parameter('frame_id').value
        self.flange_link = self.get_parameter('flange_link').value

        self.get_logger().info('Initializing VL53L0X...')
        self.tof = ToFSensor(offset_cm=offset_cm)
        self.get_logger().info('VL53L0X ready.')

        self.publisher = self.create_publisher(Range, 'tof_sensor/range', 10)
        self.timer = self.create_timer(PUBLISH_PERIOD_SEC, self.publish_reading)

        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._broadcast_mounting_tf()

    def _broadcast_mounting_tf(self):
        """Broadcast the fixed offset: flange_link -> this sensor's frame."""

        transform = TransformStamped()

        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.flange_link
        transform.child_frame_id = self.frame_id

        transform.transform.translation.x = TOF_SENSOR_OFFSET_X
        transform.transform.translation.y = TOF_SENSOR_OFFSET_Y
        transform.transform.translation.z = TOF_SENSOR_OFFSET_Z

        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0

        self.static_tf_broadcaster.sendTransform(transform)

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
