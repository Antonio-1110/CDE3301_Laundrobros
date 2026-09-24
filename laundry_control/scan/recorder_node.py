#!/usr/bin/env python3

"""
Standalone node that records ToF scan points, publishes them, and saves CSVs.

Points are published as a PointCloud2 for RViz and saved to CSV on
request.

This runs as its OWN node/process, separate from xarm7_controller
(arm/controller.py). All it needs is /tf, /tf_static (already published
independently by robot_state_publisher, part of the MoveIt stack)
and the ToF Range topic - no joint-state or MoveIt access required,
since this is pure kinematics via TF.

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

If `csv_path` is left empty (the default), points are saved under
`records_dir` (default: config.scan_records_dir(), the source tree's
scan_records/), in a file named with the scan's start time (kept
stable across repeated save_scan calls within the same scan).

Usage:
    ros2 run laundry_control scan_recorder_node
    ros2 run laundry_control scan_recorder_node --ros-args -p csv_path:=/tmp/scan.csv
"""

import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

import tf2_ros
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Range, PointCloud2, JointState
from std_srvs.srv import Trigger

from .. import config
from ..config import TOF_SENSOR_FRAME
from .cloud_io import build_cloud, POINT_CLOUD_QOS, save_xyz_csv

# The joint the ToF sensor sweeps with during a scan (see
# scan.pattern's helical strokes). Recorded per point so
# calibrate_extrinsics.py can look for a J7-periodic signature in
# the empty-bucket residual, which is what distinguishes a
# mis-measured sensor mounting from a mis-placed bucket - the cone
# fit alone cannot tell those apart (see bucket_model.fit_report).
SWEEP_JOINT_NAME = "joint7"


class ScanRecorderNode(Node):

    def __init__(self):
        super().__init__("scan_recorder_node")

        self.declare_parameter("range_topic", config.TOF_RANGE_TOPIC)
        self.declare_parameter("point_cloud_topic", "scan_record/points")
        self.declare_parameter("base_frame", config.BASE_FRAME)
        self.declare_parameter("flange_link", config.FLANGE_LINK)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("csv_path", "")
        self.declare_parameter("records_dir", "")
        self.declare_parameter("joint_state_topic", config.JOINT_STATE_TOPIC)

        self.base_frame = self.get_parameter("base_frame").value
        self.flange_link = self.get_parameter("flange_link").value

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

        # Latest sweep-joint angle, cached from /joint_states. NaN
        # until the first message arrives, so a scan recorded
        # without joint states is visibly missing the column rather
        # than quietly full of zeros.
        self._latest_sweep_angle = float("nan")

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
            self.get_parameter("point_cloud_topic").value,
            POINT_CLOUD_QOS,
        )

        self._range_sub = self.create_subscription(
            Range,
            self.get_parameter("range_topic").value,
            self._range_callback,
            qos_profile_sensor_data,
        )

        self._joint_state_sub = self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._joint_state_callback,
            10,
        )

        publish_rate_hz = self.get_parameter("publish_rate_hz").value

        self._publish_timer = self.create_timer(
            1.0 / publish_rate_hz,
            self._publish_point_cloud,
        )

        self._save_srv = self.create_service(
            Trigger, "save_scan", self._save_scan_callback
        )

        self._clear_srv = self.create_service(
            Trigger, "clear_scan", self._clear_scan_callback
        )

        self.get_logger().info("scan_recorder_node ready.")

    def _joint_state_callback(self, msg):
        """
        Cache the sweep joint's angle.

        Latest-value caching is good enough here: this is only ever
        used for offline calibration diagnostics, never for placing
        a point (which goes through TF at the reading's own
        timestamp). A few milliseconds of staleness shifts the
        fitted phase of a J7 sinusoid slightly and changes nothing
        about the scan geometry.
        """

        if SWEEP_JOINT_NAME not in msg.name:
            return

        index = msg.name.index(SWEEP_JOINT_NAME)

        if index < len(msg.position):
            self._latest_sweep_angle = float(msg.position[index])

    def _range_callback(self, msg):

        if not (msg.min_range <= msg.range <= msg.max_range):
            return

        sensor_frame = msg.header.frame_id or TOF_SENSOR_FRAME

        sensor_point = PointStamped()
        sensor_point.header.frame_id = sensor_frame
        sensor_point.point.x = float(msg.range)
        sensor_point.point.y = 0.0
        sensor_point.point.z = 0.0

        # Prefer the transform AT the reading's own capture time, for
        # spatial accuracy - fall back to the latest available one
        # if that's not resolvable (e.g. clock skew between this PC
        # and the ToF sensor's Raspberry Pi makes msg.header.stamp
        # look like it's in the future relative to /tf, or the
        # buffer just hasn't received that far yet). Zero-timeout
        # lookups only check what's already buffered, so neither
        # attempt can block this node's executor.
        try:
            tcp_transform = self.tf_buffer.lookup_transform(
                self.flange_link,
                sensor_frame,
                msg.header.stamp,
            )

            base_transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                sensor_frame,
                msg.header.stamp,
            )

            self._tf_exact_count += 1

        except Exception as exact_exc:

            try:
                tcp_transform = self.tf_buffer.lookup_transform(
                    self.flange_link,
                    sensor_frame,
                    Time(),
                )

                base_transform = self.tf_buffer.lookup_transform(
                    self.base_frame,
                    sensor_frame,
                    Time(),
                )

            except Exception as exc:

                self._tf_dropped_count += 1

                self.get_logger().warning(
                    f"Could not transform ToF reading: {exc}",
                    throttle_duration_sec=2.0,
                )

                return

            self._tf_fallback_count += 1

            # Warn on the very first fallback regardless of the
            # throttle, so a scan that silently degrades from the
            # first reading onward (e.g. a sensor whose clock never
            # got synchronised to this machine's) is obvious
            # immediately rather than only in the end-of-scan tally.
            if self._tf_fallback_count == 1:

                self.get_logger().warning(
                    "ToF reading could not be transformed at its own "
                    f"capture time ({exact_exc}); falling back to the "
                    "latest available transform. Points captured this "
                    "way are placed where the arm is NOW, not where it "
                    "was when the reading was taken - expect reduced "
                    "spatial accuracy while the arm is moving."
                )

            else:

                self.get_logger().warning(
                    "Still using latest-transform fallback "
                    f"({self._tf_fallback_count} readings so far).",
                    throttle_duration_sec=5.0,
                )

        tcp_point = tf2_geometry_msgs.do_transform_point(
            sensor_point, tcp_transform
        )

        base_point = tf2_geometry_msgs.do_transform_point(
            sensor_point, base_transform
        )

        # Keep the sensor's own capture time for bookkeeping,
        # regardless of which transform lookup (exact stamp or
        # latest-available fallback) actually succeeded above.
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
                self._latest_sweep_angle,
            )
        )

    def _publish_point_cloud(self):

        if not self.points_base_frame:
            return

        xyz = [
            (p.point.x, p.point.y, p.point.z)
            for p in self.points_base_frame
        ]

        cloud = build_cloud(
            frame_id=self.points_base_frame[-1].header.frame_id,
            stamp=self.points_base_frame[-1].header.stamp,
            xyz_points=xyz,
        )

        self._point_cloud_pub.publish(cloud)

    def _save_scan_callback(self, request, response):

        if not self.points_base_frame:
            response.success = False
            response.message = "No points recorded yet."
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
            response.message = f"Failed to save to {csv_path}: {exc}"
            return response

        response.success = True
        response.message = (
            f"Saved {len(points)} points to {csv_path} "
            f"({self._tf_summary()})"
        )
        return response

    def _tf_summary(self):
        """
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
            return "no TF lookups yet"

        fallback_pct = 100.0 * self._tf_fallback_count / total

        summary = (
            f"TF: {self._tf_exact_count} exact, "
            f"{self._tf_fallback_count} fallback ({fallback_pct:.1f}%), "
            f"{self._tf_dropped_count} dropped"
        )

        if self._tf_fallback_count:
            summary += " - fallback points are less accurate, see warnings"

        return summary

    def _resolve_csv_path(self):
        """
        Return the configured csv_path, or lazily create one under
        scan_records/ (package root) named after this session's
        start time, kept stable across repeated save_scan calls.
        """

        configured = self.get_parameter("csv_path").value

        if configured:
            return configured

        if self._session_csv_path is None:

            records_dir = (
                self.get_parameter("records_dir").value
                or config.scan_records_dir()
            )
            os.makedirs(records_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            self._session_csv_path = os.path.join(
                records_dir, f"scan_{timestamp}.csv"
            )

        return self._session_csv_path

    def _clear_scan_callback(self, request, response):

        count = len(self.points_base_frame)

        self.points_tcp_frame.clear()
        self.points_base_frame.clear()
        self.point_rays.clear()

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
        response.message = f"Cleared {count} points."
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


if __name__ == "__main__":
    main()
