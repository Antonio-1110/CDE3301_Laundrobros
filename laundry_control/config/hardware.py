#!/usr/bin/env python3

"""The ToF sensor, the gripper servo, and the optional ESP32 link to both."""

# =============================================================
# TOF SENSOR (VL53L0X)
# =============================================================

TOF_RANGE_TOPIC = 'tof_sensor/range'
TOF_SENSOR_FRAME = 'tof_sensor_link'

# VL53L0X datasheet: ~25 degree field of view, usable range
# ~30mm-2000mm. MAX_RANGE_M is deliberately tightened below the
# sensor's own usable range: readings beyond the bucket's own depth
# aren't useful (they're outside the bucket entirely) and are
# dropped at the source, since scan_recorder_node just filters
# against whatever min_range/max_range the sensor node reports.
TOF_FIELD_OF_VIEW_RAD = 0.436
TOF_MIN_RANGE_M = 0.03
TOF_MAX_RANGE_M = 0.25

TOF_PUBLISH_PERIOD_SEC = 0.05

# Calibration offset added to raw VL53L0X readings, in cm. The node
# default; laundry_bringup.launch.py passes the rig's calibrated value
# (-8.5) explicitly.
TOF_DEFAULT_OFFSET_CM = -10.0

# Translation from the TCP frame (link7) to the ToF sensor's own
# frame. Axis-aligned with link7 (no rotation offset) - the sensor's
# +X (its boresight, per REP 117 / sensor_msgs/Range) points along
# link7's +X.
#
# Measured by ruler at the INTER pose, where link7's local +Z points
# horizontally INTO the bucket (see INTER_RECORDED, config/poses.py):
#
#   - 7.75 cm along local +X - the sensor's own boresight axis. It
#     looks radially outward from the insertion axis at the bucket
#     wall (straight down at INTER's J7), offset slightly along the
#     same direction it looks.
#   - 2.8 cm along local +Z - forward, further into the bucket.
#
# Because the offset is expressed in link7's own frame, it is
# joint7-invariant: it stays correct as J7 rotates during a scan,
# which is what sweeps the beam around the bucket.
TOF_SENSOR_OFFSET_X = 0.0775
TOF_SENSOR_OFFSET_Y = 0.0
TOF_SENSOR_OFFSET_Z = 0.028

# =============================================================
# GRIPPER
# =============================================================

# Gripper contact-point mounting offset, in link7's own local frame
# (same convention as the ToF offset above).
#
# Unlike the ToF sensor's offset (which has a nonzero local X
# component - its outward-pointing boresight), the gripper's contact
# point sits ENTIRELY along local +Z: it projects straight out from
# the TCP along the wrist's own rotation axis (at INTER, forward into
# the bucket). So rotating J7 sweeps the ToF sensor around a circle,
# but does NOT move the gripper's contact point at all - reaching an
# arbitrary bucket-interior point with the gripper therefore needs
# genuine translation/tilt of the TCP, not just J7 rotation the way
# scanning does.
#
# GRIPPER_OFFSET_Z is a user-reported measurement, not yet verified
# against the physical hardware - confirm it (HARDWARE_TESTS.md)
# before trusting it for an unattended grasp.
GRIPPER_OFFSET_X = 0.0
GRIPPER_OFFSET_Y = 0.0
GRIPPER_OFFSET_Z = 0.15

# The claw is roughly 6cm wide, so it doesn't need pinpoint lateral
# accuracy - being within this tolerance of the intended point is
# still graspable. A documented error budget, not an offset: nothing
# applies it as a coordinate correction.
GRIPPER_LATERAL_TOLERANCE_M = 0.03

# Raw servo angles that open/close the physical claw. Which angle
# does what depends on how the claw is linked to the servo horn;
# override on the node with --ros-args rather than editing code if
# the linkage changes.
GRIPPER_OPEN_ANGLE_DEG = 55.0
GRIPPER_CLOSE_ANGLE_DEG = 100.0

# BCM GPIO pin and pulse range of the hobby servo.
SERVO_PIN = 18
SERVO_MIN_ANGLE_DEG = 0.0
SERVO_MAX_ANGLE_DEG = 180.0
SERVO_MIN_PULSE_WIDTH_S = 0.5 / 1000
SERVO_MAX_PULSE_WIDTH_S = 2.5 / 1000

# =============================================================
# HARDWARE LINK: this Pi, or the ESP32 over MQTT
#
# Where the ToF readings come from and who drives the gripper servo.
# These are the defaults; laundry_bringup.launch.py overrides them
# per run (tof_source:=mqtt gripper_backend:=mqtt mqtt_host:=...),
# and `laundry gripper ANGLE` follows GRIPPER_BACKEND.
#
#   TOF_SOURCE       'i2c'  - the VL53L0X on this Pi's I2C bus
#                    'mqtt' - the ESP32 publishes the readings
#   GRIPPER_BACKEND  'gpio' - the servo on this Pi's GPIO 18
#                    'mqtt' - the ESP32 drives the servo
#
# The protocol, and how the ESP32's timestamps are synced to this
# Pi's clock: hardware/esp32_protocol.py. The ESP32's firmware:
# esp32/tof_gripper_bridge. The broker (mosquitto, on this Pi):
# setup/mosquitto/README.md.
# =============================================================

TOF_SOURCES = ('i2c', 'mqtt')
GRIPPER_BACKENDS = ('gpio', 'mqtt')

TOF_SOURCE = 'i2c'
GRIPPER_BACKEND = 'gpio'

# The broker runs on this Pi, so the nodes reach it on localhost;
# the ESP32 is given the Pi's LAN address in its own config.
MQTT_HOST = 'localhost'
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = 'laundrobros'

# Clock sync pings (esp32_protocol.ClockSync): every period, and the
# best of the last WINDOW used - 10 s of pings at 2 Hz. After each
# ESP32 boot they go every BURST period until the window is full.
ESP32_CLOCK_SYNC_PERIOD_SEC = 0.5
ESP32_CLOCK_SYNC_BURST_PERIOD_SEC = 0.05
ESP32_CLOCK_SYNC_WINDOW = 20

# Warn while the sync's error bound (half the best round trip) is
# above this: about 0.2 mm of scan error at the bucket wall per ms.
# Over a good WiFi link with power save off it is 1-3 ms.
ESP32_CLOCK_SYNC_WARN_MS = 5.0

# No reading from the ESP32 for this long warns (it is offline, or
# its sensor stopped), like an I2C failure does on this Pi.
ESP32_READING_TIMEOUT_SEC = 1.0
