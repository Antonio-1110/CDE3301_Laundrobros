#!/usr/bin/env python3

"""
Standalone node that records ToF scan points, publishes them, and saves CSVs.

Points are published as a PointCloud2 for RViz and saved to CSV on
request.

This runs as its OWN node/process, separate from xarm7_controller
(arm/controller.py). All it needs is /tf, /tf_static (already published
independently by robot_state_publisher, part of the MoveIt stack)
and the ToF Range topic - no MoveIt access required, since this is
pure kinematics via TF. (/joint_states only fills the J7 column.)

Each reading is placed with the transform at its own capture time,
so it waits briefly for /tf to catch up (TF_WAIT_SEC).

Because it is a separate process with its own executor, nothing
that happens here (TF lookups, point-cloud building, CSV I/O) can
ever stall the arm-control node's spin loop, regardless of scan
length or transform-lookup timing.

Frame chain (see config.py):

    base_frame ("link_base")
        -> flange_link ("link7", == TCP frame)
            -> TOF_SENSOR_FRAME ("tof_sensor_link")

Per REP 117 / sensor_msgs/Range, a reading is a distance along the
sensor frame's +X axis, so each reading is represented as the point
(range, 0, 0) in TOF_SENSOR_FRAME before being transformed.

Services (std_srvs/Trigger):

    save_scan  - write accumulated points to the `csv_path` param.
    clear_scan - reset accumulated points (e.g. before a new scan).

Parameter `recording` (default false): readings are only recorded
while it is true, which a scan sets once the sensor is inside the
bucket (scan.pattern) and clears afterwards - so the arm's other
motions never pile points into the cloud. For a manual recording:
`-p recording:=true`.

The cloud (for RViz) is republished only when it changes; clear_scan
publishes an empty one, so RViz drops the old scan too.

If `csv_path` is left empty (the default), points are saved under
`records_dir` (default: config.scan_records_dir(), the source tree's
scan_records/), in a file named with the scan's start time (kept
stable across repeated save_scan calls within the same scan).

Usage:
    ros2 run laundry_control scan_recorder_node
    ros2 run laundry_control scan_recorder_node --ros-args -p csv_path:=/tmp/scan.csv
"""

from collections import deque
from datetime import datetime
import os

from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState, PointCloud2, Range
from std_srvs.srv import Trigger
import tf2_geometry_msgs
import tf2_ros

from .cloud_io import build_cloud, POINT_CLOUD_QOS, save_xyz_csv
from .. import config
from ..config import TOF_SENSOR_FRAME

# The joint the ToF sensor sweeps with during a scan (see
# scan.pattern's helical strokes). Recorded per point so
# calibrate_extrinsics.py can look for a J7-periodic signature in
# the empty-bucket residual, which is what distinguishes a
# mis-measured sensor mounting from a mis-placed bucket - the cone
# fit alone cannot tell those apart (see bucket_model.fit_report).
SWEEP_JOINT_NAME = 'joint7'

# How long a reading may wait for /tf to reach its capture time
# before it is placed with the latest transform instead. /tf trails
# the ToF: robot_state_publisher computes it from /joint_states,
# which the stock xArm launch's joint_state_publisher republishes at
# only 10 Hz, so the newest transform is up to ~100 ms old when a
# reading arrives. Looking it up at once (as this node used to) fell
# back to "where the arm is now" for 99.5% of a real scan's readings
# (2026-09-28) - up to 100 ms of J7 sweep, i.e. several degrees of
# beam direction. Waiting until /tf has caught up lets the reading
# use the transform interpolated at its own capture time.
TF_WAIT_SEC = 0.3

# How often queued readings are retried, since /tf arriving does not
# itself trigger anything here.
DRAIN_RATE_HZ = 25.0

# Sweep-joint history kept for interpolating J7 at a reading's time.
SWEEP_HISTORY_SEC = 2.0


class ScanRecorderNode(Node):

    def __init__(self):
        super().__init__('scan_recorder_node')

        self.declare_parameter('range_topic', config.TOF_RANGE_TOPIC)
        self.declare_parameter('point_cloud_topic', 'scan_record/points')
        self.declare_parameter('base_frame', config.BASE_FRAME)
        self.declare_parameter('flange_link', config.FLANGE_LINK)
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('csv_path', '')
        self.declare_parameter('records_dir', '')
        self.declare_parameter('joint_state_topic', config.JOINT_STATE_TOPIC)
        # Readings are ignored unless this is true: a scan turns it on
        # only while the sensor is inside the bucket (scan.pattern,
        # entry_depth), and pipeline.run_scan turns it off after.
        # Left on between scans, every motion of the arm piled points
        # into the RViz cloud until the next scan cleared it.
        self.declare_parameter('recording', False)

        # Size of the cloud last published; -1 forces a publish (an
        # empty cloud after a clear, so RViz drops the old one).
        self._published_count = -1

        self.base_frame = self.get_parameter('base_frame').value
        self.flange_link = self.get_parameter('flange_link').value

        # Lazily-created, cached path used when csv_path is left at
        # its default (""), so repeated save_scan calls in one run
        # keep overwriting the same timestamped file.
        self._session_csv_path = None

        self.points_tcp_frame = []
        self.points_base_frame = []

        # Per-point ray metadata, index-aligned with
        # points_base_frame: (raw_range, ox, oy, oz, j7). The
        # endpoint alone loses where the beam started, which is what
        # range-space residuals and extrinsic calibration both need.
        self.point_rays = []

        # Recent sweep-joint angles from /joint_states, as (time,
        # angle), interpolated at each reading's capture time. Empty
        # until the first message arrives, so a scan recorded
        # without joint states is visibly missing the column (NaN)
        # rather than quietly full of zeros.
        self._sweep_history = deque()

        # Readings waiting for /tf to reach their capture time (see
        # TF_WAIT_SEC), oldest first.
        self._pending = deque()

        # TF lookup outcomes for the current scan. The fallback path
        # below substitutes "wherever the arm is NOW" for "where the
        # arm was when this reading was captured", which during a
        # moving scan misplaces the point by however far the arm
        # travelled in between - so a scan that silently took the
        # fallback for most of its readings is materially less
        # accurate than one that didn't, with no other visible
        # symptom. These counters (and the warning on first
        # fallback) are what make that visible.
        self._tf_exact_count = 0
        self._tf_fallback_count = 0
        self._tf_dropped_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._point_cloud_pub = self.create_publisher(
            PointCloud2,
            self.get_parameter('point_cloud_topic').value,
            POINT_CLOUD_QOS,
        )

        self._range_sub = self.create_subscription(
            Range,
            self.get_parameter('range_topic').value,
            self._range_callback,
            qos_profile_sensor_data,
        )

        self._joint_state_sub = self.create_subscription(
            JointState,
            self.get_parameter('joint_state_topic').value,
            self._joint_state_callback,
            10,
        )

        self._drain_timer = self.create_timer(
            1.0 / DRAIN_RATE_HZ, self._drain
        )

        publish_rate_hz = self.get_parameter('publish_rate_hz').value

        self._publish_timer = self.create_timer(
            1.0 / publish_rate_hz,
            self._publish_point_cloud,
        )

        self._save_srv = self.create_service(
            Trigger, 'save_scan', self._save_scan_callback
        )

        self._clear_srv = self.create_service(
            Trigger, 'clear_scan', self._clear_scan_callback
        )

        self.get_logger().info('scan_recorder_node ready.')

    def _joint_state_callback(self, msg):
        """
        Remember the sweep joint's recent angles.

        Only ever used for offline calibration diagnostics (the J7
        column), never for placing a point, which goes through TF.
        """
        if SWEEP_JOINT_NAME not in msg.name:
            return

        index = msg.name.index(SWEEP_JOINT_NAME)

        if index >= len(msg.position):
            return

        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9

        self._sweep_history.append((stamp, float(msg.position[index])))

        while stamp - self._sweep_history[0][0] > SWEEP_HISTORY_SEC:
            self._sweep_history.popleft()

    def _sweep_angle_at(self, stamp):
        """Return the sweep joint's angle at `stamp` (s), interpolated; NaN if unknown."""
        if not self._sweep_history:
            return float('nan')

        times, angles = zip(*self._sweep_history)

        return float(np.interp(stamp, times, angles))

    def _range_callback(self, msg):

        if not self.get_parameter('recording').value:
            return

        if not (msg.min_range <= msg.range <= msg.max_range):
            return

        self._pending.append(msg)
        self._drain()

    def _lookup(self, sensor_frame, when):
        """Return (flange, base) transforms of sensor_frame at `when`; raises if unavailable."""
        # Zero-timeout lookups only check what is already buffered,
        # so they never block this node's executor.
        return (
            self.tf_buffer.lookup_transform(
                self.flange_link, sensor_frame, when
            ),
            self.tf_buffer.lookup_transform(
                self.base_frame, sensor_frame, when
            ),
        )

    def _drain(self, force=False):
        """
        Place every queued reading whose capture time /tf has reached.

        Readings go in capture order. The oldest one waits (and with
        it the rest) until the transform at its own capture time is
        available, or until it is TF_WAIT_SEC old: then it is placed
        with the latest transform, and counted as a fallback. force
        (on save) places everything queued now, exact where possible.
        """
        now_ns = self.get_clock().now().nanoseconds

        while self._pending:
            msg = self._pending[0]
            sensor_frame = msg.header.frame_id or TOF_SENSOR_FRAME
            stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
            waiting = not force and (now_ns - stamp_ns) * 1e-9 < TF_WAIT_SEC

            try:
                transforms = self._lookup(sensor_frame, msg.header.stamp)
            except Exception as exact_exc:
                if waiting:
                    return

                self._pending.popleft()
                self._record_fallback(msg, sensor_frame, exact_exc)
                continue

            # The J7 column is interpolated from /joint_states, which
            # can arrive just after the /tf computed from it: wait for
            # a sample past the reading too, or it gets the last one
            # before it (up to 100 ms of sweep stale).
            if (
                waiting
                and self._sweep_history
                and self._sweep_history[-1][0] < stamp_ns * 1e-9
            ):
                return

            self._pending.popleft()
            self._tf_exact_count += 1
            self._record(msg, sensor_frame, *transforms)

    def _record_fallback(self, msg, sensor_frame, exact_exc):
        """Place a reading with the latest transform (the arm's pose NOW)."""
        try:
            transforms = self._lookup(sensor_frame, Time())

        except Exception as exc:

            self._tf_dropped_count += 1

            self.get_logger().warning(
                f'Could not transform ToF reading: {exc}',
                throttle_duration_sec=2.0,
            )

            return

        self._tf_fallback_count += 1

        # Warn on the very first fallback regardless of the throttle,
        # so a scan that silently degrades from the first reading
        # onward (e.g. /tf not arriving at all) is obvious
        # immediately rather than only in the end-of-scan tally.
        if self._tf_fallback_count == 1:

            self.get_logger().warning(
                'ToF reading could not be transformed at its own '
                f'capture time within {TF_WAIT_SEC:g} s ({exact_exc}); '
                'falling back to the latest available transform. Points '
                'captured this way are placed where the arm is NOW, not '
                'where it was when the reading was taken - expect reduced '
                'spatial accuracy while the arm is moving.'
            )

        else:

            self.get_logger().warning(
                'Still using latest-transform fallback '
                f'({self._tf_fallback_count} readings so far).',
                throttle_duration_sec=5.0,
            )

        self._record(msg, sensor_frame, *transforms)

    def _record(self, msg, sensor_frame, tcp_transform, base_transform):
        """Store one reading as a point (flange and base frames) plus its ray."""
        sensor_point = PointStamped()
        sensor_point.header.frame_id = sensor_frame
        sensor_point.point.x = float(msg.range)
        sensor_point.point.y = 0.0
        sensor_point.point.z = 0.0

        tcp_point = tf2_geometry_msgs.do_transform_point(
            sensor_point, tcp_transform
        )

        base_point = tf2_geometry_msgs.do_transform_point(
            sensor_point, base_transform
        )

        # Keep the sensor's own capture time for bookkeeping,
        # regardless of which transform lookup (exact stamp or
        # latest-available fallback) placed the point.
        tcp_point.header.stamp = msg.header.stamp
        base_point.header.stamp = msg.header.stamp

        # The beam's ORIGIN in base_frame: the same transform
        # applied to the sensor frame's own origin. Together with
        # the endpoint this reconstructs the full ray, which is
        # what range-space residuals and extrinsic calibration
        # need and what the endpoint alone cannot give back.
        origin_point = PointStamped()
        origin_point.header.frame_id = sensor_frame
        origin_point.point.x = 0.0
        origin_point.point.y = 0.0
        origin_point.point.z = 0.0

        base_origin = tf2_geometry_msgs.do_transform_point(
            origin_point, base_transform
        )

        self.points_tcp_frame.append(tcp_point)
        self.points_base_frame.append(base_point)

        self.point_rays.append(
            (
                float(msg.range),
                base_origin.point.x,
                base_origin.point.y,
                base_origin.point.z,
                self._sweep_angle_at(
                    Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
                ),
            )
        )

    def _publish_point_cloud(self):
        """Republish the cloud for RViz, only if it changed since last time."""
        if len(self.points_base_frame) == self._published_count:
            return

        xyz = [
            (p.point.x, p.point.y, p.point.z)
            for p in self.points_base_frame
        ]

        if self.points_base_frame:
            frame_id = self.points_base_frame[-1].header.frame_id
            stamp = self.points_base_frame[-1].header.stamp
        else:
            frame_id = self.base_frame
            stamp = self.get_clock().now().to_msg()

        cloud = build_cloud(frame_id=frame_id, stamp=stamp, xyz_points=xyz)

        self._point_cloud_pub.publish(cloud)
        self._published_count = len(self.points_base_frame)

    def _save_scan_callback(self, request, response):

        # Readings still waiting for /tf are part of this scan.
        self._drain(force=True)

        if not self.points_base_frame:
            response.success = False
            response.message = 'No points recorded yet.'
            return response

        csv_path = self._resolve_csv_path()

        points = [
            (p.point.x, p.point.y, p.point.z, p.header.stamp) + ray
            for p, ray in zip(self.points_base_frame, self.point_rays)
        ]

        try:
            save_xyz_csv(csv_path, points)
        except OSError as exc:
            response.success = False
            response.message = f'Failed to save to {csv_path}: {exc}'
            return response

        response.success = True
        response.message = (
            f'Saved {len(points)} points to {csv_path} '
            f'({self._tf_summary()})'
        )
        return response

    def _tf_summary(self):
        """
        Summarise TF lookup accuracy for the current scan in one line.

        One-line TF-accuracy tally for the current scan, reported on
        every save so it lands in the scan log (recorder_client.py logs
        the save response) rather than needing to be dug for.
        """
        total = (
            self._tf_exact_count
            + self._tf_fallback_count
            + self._tf_dropped_count
        )

        if total == 0:
            return 'no TF lookups yet'

        fallback_pct = 100.0 * self._tf_fallback_count / total

        summary = (
            f'TF: {self._tf_exact_count} exact, '
            f'{self._tf_fallback_count} fallback ({fallback_pct:.1f}%), '
            f'{self._tf_dropped_count} dropped'
        )

        if self._tf_fallback_count:
            summary += ' - fallback points are less accurate, see warnings'

        return summary

    def _resolve_csv_path(self):
        """
        Return the CSV path the next save_scan should write.

        Return the configured csv_path, or lazily create one under
        scan_records/ (package root) named after this session's
        start time, kept stable across repeated save_scan calls.
        """
        configured = self.get_parameter('csv_path').value

        if configured:
            return configured

        if self._session_csv_path is None:

            records_dir = (
                self.get_parameter('records_dir').value
                or config.scan_records_dir()
            )
            os.makedirs(records_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            self._session_csv_path = os.path.join(
                records_dir, f'scan_{timestamp}.csv'
            )

        return self._session_csv_path

    def _clear_scan_callback(self, request, response):

        count = len(self.points_base_frame)

        self.points_tcp_frame.clear()
        self.points_base_frame.clear()
        self.point_rays.clear()
        self._pending.clear()

        # The next publish sends an empty cloud, clearing RViz too.
        self._published_count = -1

        # clear_scan marks a scan boundary, so the TF tally starts
        # over with it - otherwise the next scan's accuracy report
        # would carry the previous scan's failures.
        self._tf_exact_count = 0
        self._tf_fallback_count = 0
        self._tf_dropped_count = 0

        # clear_scan marks the boundary between one scan and the
        # next (`laundry scan` calls it before every run). Drop the
        # cached auto-generated path so the *next* save_scan gets a
        # fresh timestamped filename instead of silently overwriting
        # the previous scan's CSV for the lifetime of this node.
        self._session_csv_path = None

        response.success = True
        response.message = f'Cleared {count} points.'
        return response


def main(args=None):
    rclpy.init(args=args)

    node = ScanRecorderNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
