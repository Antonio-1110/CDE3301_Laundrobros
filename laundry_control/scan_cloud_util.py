#!/usr/bin/env python3

"""
scan_cloud_util.py

Shared helpers for building/publishing/loading ToF scan point
clouds, used by both the live recorder (scan_record.py) and the
offline replay tool (scan_replay.py) so RViz sees identical
PointCloud2 data whether the scan is happening live or being
replayed from a saved CSV file.
"""

import csv

from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# TRANSIENT_LOCAL durability means a late-joining subscriber (e.g.
# RViz opened after the scan/replay already started publishing)
# still receives the last published cloud immediately, instead of
# only future messages.
POINT_CLOUD_QOS = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


def build_cloud(frame_id, stamp, xyz_points):
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    return point_cloud2.create_cloud_xyz32(
        header=header,
        points=xyz_points,
    )


def save_xyz_csv(path, points):
    # points: iterable of (x, y, z, stamp), stamp being a
    # builtin_interfaces/Time (or anything with .sec/.nanosec) so
    # scan_replay.py can reconstruct the real capture cadence.
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "stamp_sec", "stamp_nanosec"])

        for x, y, z, stamp in points:
            writer.writerow([x, y, z, stamp.sec, stamp.nanosec])


def load_xyz_csv(path):
    # Returns (x, y, z, stamp) tuples. Missing/older-format stamp
    # columns default to zero.
    points = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            stamp = Time()
            stamp.sec = int(row.get("stamp_sec") or 0)
            stamp.nanosec = int(row.get("stamp_nanosec") or 0)

            points.append(
                (
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                    stamp,
                )
            )

    return points
