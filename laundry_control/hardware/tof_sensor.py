"""
ROS 2 node that publishes VL53L0X time-of-flight distance readings.

Also broadcasts the static flange_link -> tof_sensor_link mounting
transform (config.TOF_SENSOR_OFFSET_*), so scan_recorder_node can
place every reading through TF.

Wiring:
    VL53L0X VCC -> RPi 3.3V
    VL53L0X GND -> RPi GND
    VL53L0X SDA -> RPi GPIO 2  (Pin 3)
    VL53L0X SCL -> RPi GPIO 3  (Pin 5)

Library:
    pip install adafruit-circuitpython-vl53l0x

Usage:
    ros2 run laundry_control tof_sensor
    ros2 run laundry_control tof_sensor --ros-args -p offset_cm:=-8.5

HARDWARE ONLY: needs the sensor on the Pi's I2C bus.
"""

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
import tf2_ros

from ..config import (
    TOF_DEFAULT_OFFSET_CM,
    TOF_FIELD_OF_VIEW_RAD,
    TOF_MAX_RANGE_M,
    TOF_MIN_RANGE_M,
    TOF_PUBLISH_PERIOD_SEC,
    TOF_RANGE_TOPIC,
    TOF_SENSOR_FRAME,
    TOF_SENSOR_OFFSET_X,
    TOF_SENSOR_OFFSET_Y,
    TOF_SENSOR_OFFSET_Z,
)

# The mounting offset and range limits live in config.py (they are
# measured numbers, and scan_recorder_node needs the frame name
# without pulling in Pi-only libraries). The hardware imports
# (board/busio/adafruit_vl53l0x) are deferred into ToFSensor.__init__
# for the same reason: importing this module must not require them.


class ToFSensor:
    def __init__(self, offset_cm=TOF_DEFAULT_OFFSET_CM):
        """
        Initialize VL53L0X sensor.

        Constructor argument:
            offset_cm:
                Calibration offset added to the raw measurement.
                Default (config.TOF_DEFAULT_OFFSET_CM, -10 cm)
                matches the original Arduino code.
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

        self.declare_parameter('offset_cm', TOF_DEFAULT_OFFSET_CM)
        self.declare_parameter('frame_id', TOF_SENSOR_FRAME)
        self.declare_parameter('flange_link', 'link7')

        offset_cm = self.get_parameter('offset_cm').value
        self.frame_id = self.get_parameter('frame_id').value
        self.flange_link = self.get_parameter('flange_link').value

        self.get_logger().info('Initializing VL53L0X...')
        self.tof = ToFSensor(offset_cm=offset_cm)
        self.get_logger().info('VL53L0X ready.')

        self.publisher = self.create_publisher(Range, TOF_RANGE_TOPIC, 10)
        self.timer = self.create_timer(
            TOF_PUBLISH_PERIOD_SEC, self.publish_reading
        )

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
        msg.field_of_view = TOF_FIELD_OF_VIEW_RAD
        msg.min_range = TOF_MIN_RANGE_M
        msg.max_range = TOF_MAX_RANGE_M
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


if __name__ == '__main__':
    main()
