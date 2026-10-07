"""
ROS 2 node that publishes VL53L0X time-of-flight distance readings.

The sensor is on this Pi's I2C bus (source:=i2c), or on the ESP32,
which publishes its readings over MQTT (source:=mqtt; see
esp32_protocol.py). Either way the node publishes the same Range
messages, stamped in this Pi's clock, so scan_recorder_node and
everything after it cannot tell the difference. Default:
config.TOF_SOURCE.

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
    ros2 run laundry_control tof_sensor --ros-args -p source:=mqtt

TIMESTAMPS
----------
A reading takes the sensor's whole timing budget (~33 ms, measured
~36 ms): `sensor.range` starts a measurement and blocks until it is
done. Each reading is stamped at the MIDDLE of that window, not when
it returned - scan_recorder_node looks up the arm's pose at the
stamp, and during a scan J7 turns ~2 deg in 36 ms (~7 mm at the
bucket wall).

Over MQTT, the ESP32 stamps the middle of the window in its own
clock, and esp32_protocol.ClockSync maps that to this Pi's: WiFi
latency then does not move the stamp at all.

I2C ERRORS
----------
A bus glitch or a loose wire raises OSError from the read. That used
to escape the timer callback and kill the node mid-scan. Now the
reading is skipped with a (throttled) warning, and after
REINIT_AFTER_FAILURES failures in a row the sensor is re-initialised.

HARDWARE ONLY: needs the sensor on the Pi's I2C bus, or the ESP32
and the MQTT broker.
"""

from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Range
import tf2_ros

from . import esp32_protocol
from ..config import (
    ESP32_CLOCK_SYNC_BURST_PERIOD_SEC,
    ESP32_CLOCK_SYNC_PERIOD_SEC,
    ESP32_CLOCK_SYNC_WARN_MS,
    ESP32_CLOCK_SYNC_WINDOW,
    ESP32_READING_TIMEOUT_SEC,
    MQTT_HOST,
    MQTT_PORT,
    MQTT_TOPIC_PREFIX,
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
    TOF_SOURCE,
    TOF_SOURCES,
)

# The mounting offset and range limits live in config/hardware.py (they are
# measured numbers, and scan_recorder_node needs the frame name
# without pulling in Pi-only libraries). The hardware imports
# (board/busio/adafruit_vl53l0x) are deferred into ToFSensor.__init__
# for the same reason: importing this module must not require them.

# Consecutive failed reads (at 20 Hz: half a second) before the sensor
# is re-initialised.
REINIT_AFTER_FAILURES = 10

# I2C/driver failures a read can raise: OSError for bus errors
# (EIO/EREMOTEIO), RuntimeError for the adafruit driver's timeouts.
READ_ERRORS = (OSError, RuntimeError)


def midpoint_ns(before_ns, after_ns):
    """Return the middle of a measurement window, in nanoseconds."""
    return (before_ns + after_ns) // 2


class GuardedReader:
    """
    Read a sensor without letting I2C errors escape.

    read() returns (distance_cm, stamp_ns) or None. make_sensor builds
    (and rebuilds) the sensor; now_ns is the clock the stamps use.
    """

    def __init__(
        self,
        make_sensor,
        now_ns,
        warn=print,
        info=print,
        reinit_after=REINIT_AFTER_FAILURES,
    ):
        self._make_sensor = make_sensor
        self._now_ns = now_ns
        self._warn = warn
        self._info = info
        self._reinit_after = reinit_after

        self.sensor = make_sensor()
        self.failures = 0

    def read(self):
        before = self._now_ns()

        try:
            if self.sensor is None:
                raise OSError('sensor not initialised')

            distance_cm = self.sensor.get_distance_cm()

        except READ_ERRORS as exc:
            self._failed(exc)
            return None

        after = self._now_ns()

        if self.failures:
            self._info(f'ToF reads recovered after {self.failures} failure(s).')
            self.failures = 0

        return distance_cm, midpoint_ns(before, after)

    def _failed(self, exc):
        self.failures += 1

        self._warn(f'ToF read failed ({self.failures} in a row): {exc}')

        if self.failures % self._reinit_after == 0:
            try:
                self.sensor = self._make_sensor()
                self._info('ToF sensor re-initialised.')
            except READ_ERRORS as reinit_exc:
                self.sensor = None
                self._warn(f'ToF re-initialisation failed: {reinit_exc}')


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

        self.declare_parameter('source', TOF_SOURCE)
        self.declare_parameter('offset_cm', TOF_DEFAULT_OFFSET_CM)
        self.declare_parameter('frame_id', TOF_SENSOR_FRAME)
        self.declare_parameter('flange_link', 'link7')
        self.declare_parameter('mqtt_host', MQTT_HOST)
        self.declare_parameter('mqtt_port', MQTT_PORT)
        self.declare_parameter('mqtt_prefix', MQTT_TOPIC_PREFIX)

        source = self.get_parameter('source').value
        self.offset_cm = self.get_parameter('offset_cm').value
        self.frame_id = self.get_parameter('frame_id').value
        self.flange_link = self.get_parameter('flange_link').value

        if source not in TOF_SOURCES:
            raise ValueError(
                f'source must be one of {TOF_SOURCES}, not {source!r}'
            )

        self.publisher = self.create_publisher(Range, TOF_RANGE_TOPIC, 10)

        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._broadcast_mounting_tf()

        self.esp32 = None

        if source == 'i2c':
            self._start_i2c()
        else:
            self.esp32 = Esp32RangeSource(self)

    def _start_i2c(self):
        offset_cm = self.offset_cm

        self.get_logger().info('Initializing VL53L0X...')
        self.reader = GuardedReader(
            make_sensor=lambda: ToFSensor(offset_cm=offset_cm),
            now_ns=lambda: self.get_clock().now().nanoseconds,
            warn=lambda text: self.get_logger().warning(
                text, throttle_duration_sec=1.0
            ),
            info=self.get_logger().info,
        )
        self.get_logger().info('VL53L0X ready.')

        self.timer = self.create_timer(
            TOF_PUBLISH_PERIOD_SEC, self.publish_reading
        )

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
        reading = self.reader.read()

        if reading is None:
            return

        self.publish_range(*reading)

    def publish_range(self, distance_cm, stamp_ns):
        """Publish one calibrated reading, captured at stamp_ns (ROS clock)."""
        msg = Range()
        msg.header.stamp = Time(
            nanoseconds=stamp_ns, clock_type=self.get_clock().clock_type
        ).to_msg()
        msg.header.frame_id = self.frame_id
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = TOF_FIELD_OF_VIEW_RAD
        msg.min_range = TOF_MIN_RANGE_M
        msg.max_range = TOF_MAX_RANGE_M
        msg.range = distance_cm / 100.0

        self.publisher.publish(msg)


class Esp32RangeSource:
    """
    Readings from the ESP32 over MQTT, for ToFSensorNode (source:=mqtt).

    Pings the ESP32 to keep the clock sync current, stamps each
    reading through it, and publishes it from paho's thread (rclpy's
    publish is thread-safe). Readings are held back until the clock
    is synced, and the sync is redone whenever the ESP32 reboots.
    """

    def __init__(self, node):
        from .mqtt_link import MqttLink

        self._node = node
        self._log = node.get_logger()
        self._now_ns = lambda: node.get_clock().now().nanoseconds
        self._topics = esp32_protocol.topics(
            node.get_parameter('mqtt_prefix').value
        )
        self.sync = esp32_protocol.ClockSync(window=ESP32_CLOCK_SYNC_WINDOW)
        self.gaps = esp32_protocol.SequenceGaps()
        self._started_ns = self._now_ns()
        self._last_reading_ns = None
        self._reported_sync = False

        self.link = MqttLink(
            node.get_parameter('mqtt_host').value,
            int(node.get_parameter('mqtt_port').value),
            client_id='laundry_tof_sensor',
            log_info=self._log.info,
            log_warn=self._log.warning,
        )
        self.link.subscribe(self._topics.range, self._on_range)
        self.link.subscribe(self._topics.pong, self._on_pong)
        self.link.subscribe(self._topics.status, self._on_status, qos=1)
        self.link.start()

        self._last_ping_ns = 0
        node.create_timer(ESP32_CLOCK_SYNC_BURST_PERIOD_SEC, self._ping)
        node.create_timer(ESP32_READING_TIMEOUT_SEC, self._check_alive)

        self._log.info(
            f'Waiting for ToF readings from the ESP32 on {self._topics.range}.'
        )

    def _ping(self):
        """Ping every period, or every burst period while filling."""
        now_ns = self._now_ns()
        period = (
            ESP32_CLOCK_SYNC_BURST_PERIOD_SEC
            if self.sync.filling
            else ESP32_CLOCK_SYNC_PERIOD_SEC
        )

        if not self.link.connected or now_ns - self._last_ping_ns < (
            period * 1e9 * 0.9
        ):
            return

        self._last_ping_ns = now_ns
        # Take the time last, so it is as close to the send as can be.
        payload = self.sync.ping(self._now_ns())
        self.link.publish(self._topics.ping, payload)

    def _on_pong(self, payload):
        now_ns = self._now_ns()
        boot = self.sync.boot
        rtt_ns = self.sync.pong(payload, now_ns)

        if rtt_ns is None:
            return

        if self.sync.boot != boot:
            self._reported_sync = False

        if not self._reported_sync and not self.sync.filling:
            self._reported_sync = True
            self._log.info(
                f'Clock synced with ESP32 boot {self.sync.boot}: '
                f'+/- {self.sync.uncertainty_ns * 1e-6:.1f} ms.'
            )

    def _on_status(self, payload):
        status = payload.decode(errors='replace')

        if status == esp32_protocol.OFFLINE:
            self._log.warning('The ESP32 went offline (MQTT last will).')
        else:
            self._log.info(f'ESP32 status: {status}.')

    def _on_range(self, payload):
        now_ns = self._now_ns()
        reading = esp32_protocol.decode_range(payload)

        if reading is None:
            self._log.warning(
                f'Malformed ToF payload: {payload[:80]!r}',
                throttle_duration_sec=5.0,
            )
            return

        self._last_reading_ns = now_ns

        lost = self.gaps.update(reading)

        if lost:
            self._log.warning(
                f'{lost} ToF reading(s) lost in transit '
                f'({self.gaps.lost} so far).',
                throttle_duration_sec=2.0,
            )

        stamp_ns, reason = self.sync.stamp(reading, now_ns)

        if stamp_ns is None:
            self._log.warning(
                f'ToF reading dropped: {reason}.', throttle_duration_sec=2.0
            )
            return

        uncertainty_ms = self.sync.uncertainty_ns * 1e-6

        if uncertainty_ms > ESP32_CLOCK_SYNC_WARN_MS:
            self._log.warning(
                f'Clock sync with the ESP32 only +/- {uncertainty_ms:.1f} ms '
                '(slow WiFi round trips; is its power save off?): scan '
                'points may be misplaced while the arm moves.',
                throttle_duration_sec=5.0,
            )

        self._node.publish_range(
            reading.mm / 10.0 + self._node.offset_cm, stamp_ns
        )

    def _check_alive(self):
        last_ns = self._last_reading_ns or self._started_ns
        silent_sec = (self._now_ns() - last_ns) * 1e-9

        if silent_sec > ESP32_READING_TIMEOUT_SEC:
            self._log.warning(
                f'No ToF reading from the ESP32 for {silent_sec:.1f} s '
                f'(MQTT {"up" if self.link.connected else "DOWN"}; is the '
                'ESP32 powered and on WiFi?).',
                throttle_duration_sec=5.0,
            )

    def stop(self):
        self.link.stop()


def main(args=None):
    rclpy.init(args=args)

    node = ToFSensorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.esp32 is not None:
            node.esp32.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
