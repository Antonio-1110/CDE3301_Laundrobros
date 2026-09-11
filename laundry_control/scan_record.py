#!/usr/bin/env python3

"""
scan_record.py

Records ToF sensor readings during a bucket/drum scan, transformed
into the TCP frame (and the robot base frame), so the accumulated
points can be inspected afterward or visualized live in RViz.

Frame chain (see move.py):

    base_frame ("link_base")
        -> flange_link ("link7", == TCP frame)
            -> TOF_SENSOR_FRAME ("tof_sensor_link")

The flange_link -> TOF_SENSOR_FRAME offset is a static, axis-aligned
translation (TOF_SENSOR_OFFSET_* in move.py) - a placeholder until
the real mounting offset is measured. Because it is expressed
relative to link7, it automatically rotates with joint 7.

Per REP 117 / sensor_msgs/Range, a reading is a distance along the
sensor frame's +X axis, so each reading is represented as the point
(range, 0, 0) in TOF_SENSOR_FRAME before being transformed.
"""

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Range, PointCloud2
import tf2_geometry_msgs

from move import TOF_SENSOR_FRAME
from scan_cloud_util import build_cloud, save_xyz_csv, POINT_CLOUD_QOS


class TofScanRecorder:
    """
    Subscribes to ToF range readings on an existing XArm7Controller
    node and accumulates each reading as a 3D point.

    Points are stored in two frames:

        - TCP frame (link7)   -> point_stamped_tcp
        - base frame          -> point_stamped_base (for building a
          map of the drum interior, since the TCP moves during the
          scan but the base does not)
    """

    def __init__(
        self,
        arm,
        range_topic="tof_sensor/range",
        point_cloud_topic="scan_record/points",
    ):
        self.arm = arm

        self.points_tcp_frame = []
        self.points_base_frame = []

        self._point_cloud_pub = arm.create_publisher(
            PointCloud2,
            point_cloud_topic,
            POINT_CLOUD_QOS,
        )

        self._range_sub = arm.create_subscription(
            Range,
            range_topic,
            self._range_callback,
            qos_profile_sensor_data,
        )

    def _range_callback(self, msg):

        if not (msg.min_range <= msg.range <= msg.max_range):
            return

        sensor_frame = msg.header.frame_id or TOF_SENSOR_FRAME

        sensor_point = PointStamped()
        sensor_point.header.frame_id = sensor_frame
        sensor_point.point.x = float(msg.range)
        sensor_point.point.y = 0.0
        sensor_point.point.z = 0.0

        # Look up the LATEST available transform (Time() == "latest")
        # instead of the one at msg.header.stamp. Waiting for TF to
        # catch up to an exact past/future stamp would block this
        # callback - and since this callback runs on the same
        # single-threaded executor that also processes incoming TF
        # updates and the arm's own trajectory execution, blocking
        # here starves both, causing TF lag AND incorrect motion.
        try:
            tcp_transform = self.arm.tf_buffer.lookup_transform(
                self.arm.flange_link,
                sensor_frame,
                Time(),
            )

            base_transform = self.arm.tf_buffer.lookup_transform(
                self.arm.base_frame,
                sensor_frame,
                Time(),
            )

        except Exception as exc:

            self.arm.get_logger().warning(
                f"Could not transform ToF reading: {exc}",
                throttle_duration_sec=2.0,
            )

            return

        tcp_point = tf2_geometry_msgs.do_transform_point(
            sensor_point, tcp_transform
        )

        base_point = tf2_geometry_msgs.do_transform_point(
            sensor_point, base_transform
        )

        # Keep the sensor's own capture time for bookkeeping, even
        # though the transform used was the latest available one.
        tcp_point.header.stamp = msg.header.stamp
        base_point.header.stamp = msg.header.stamp

        self.points_tcp_frame.append(tcp_point)
        self.points_base_frame.append(base_point)

        self._publish_point_cloud()

    def _publish_point_cloud(self):

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

    def clear(self):
        self.points_tcp_frame.clear()
        self.points_base_frame.clear()

    def save_csv(self, path):
        """
        Save accumulated points (base frame) to a CSV file that
        scan_replay.py can later republish for RViz, or that can be
        loaded offline with pandas/matplotlib/Open3D.
        """

        xyz = [
            (p.point.x, p.point.y, p.point.z)
            for p in self.points_base_frame
        ]

        save_xyz_csv(path, xyz)
