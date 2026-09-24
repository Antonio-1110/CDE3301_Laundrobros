#!/usr/bin/env python3

"""
Build, publish, save and load ToF scan point clouds.

Shared by the live recorder (recorder_node.py), the offline replay
tool (replay.py) and detection, so RViz sees identical PointCloud2
data whether the scan is happening live or being replayed from a
saved CSV file, and every stage reads the same CSV schema.
"""

import csv

from builtin_interfaces.msg import Time
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

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


def build_cloud_with_intensity(frame_id, stamp, xyz_points, intensities):
    """
    Build a PointCloud2 with a per-point intensity field.

    Like build_cloud(), but with a per-point `intensity` field, so
    RViz can colour points by group (set the PointCloud2 display's
    Color Transformer to "Intensity", Channel Name to "intensity").

    Used to tell separate laundry clusters apart in one cloud -
    otherwise every cluster renders in a single flat colour and a
    multi-cluster result is unreadable.
    """
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(
            name='intensity',
            offset=12,
            datatype=PointField.FLOAT32,
            count=1,
        ),
    ]

    points = [
        (float(x), float(y), float(z), float(i))
        for (x, y, z), i in zip(xyz_points, intensities)
    ]

    return point_cloud2.create_cloud(header, fields, points)


# Columns beyond the original (x, y, z, stamp) schema.
#
# A ToF reading is a RAY, not a point: the endpoint alone throws
# away where the beam started and how far it actually travelled.
# Recording the ray lets the residual be evaluated in the sensor's
# native 1-D range space (where its noise actually lives, rather
# than as an anisotropic ellipsoid in Cartesian space), and lets
# calibrate_extrinsics.py look for the theta-dependent signature of
# a mis-measured sensor mounting - neither of which is recoverable
# from the endpoint after the fact.
#
# j7 is carried alongside because the sensor sweeps with joint 7,
# so it is the natural independent variable for that calibration:
# an extrinsic error tracks J7, while a bucket-pose error does not,
# and that is the only thing distinguishing them (see
# bucket_model.fit_report).
EXTENDED_COLUMNS = ['raw_range', 'ox', 'oy', 'oz', 'j7']

BASE_COLUMNS = ['x', 'y', 'z', 'stamp_sec', 'stamp_nanosec']


def save_xyz_csv(path, points):
    # points: iterable of (x, y, z, stamp), stamp being a
    # builtin_interfaces/Time (or anything with .sec/.nanosec) so
    # replay.py can reconstruct the real capture cadence.
    #
    # Optionally (x, y, z, stamp, raw_range, ox, oy, oz, j7) - the
    # extended schema above. The two are distinguished by tuple
    # length, taken from the first row, so existing callers need no
    # change and older CSVs stay readable.
    points = list(points)

    # A base point is a 4-tuple (x, y, z, stamp); the stamp
    # expands to two columns, so column count != tuple length.
    extended = bool(points) and len(points[0]) == 4 + len(EXTENDED_COLUMNS)

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(
            BASE_COLUMNS + (EXTENDED_COLUMNS if extended else [])
        )

        for point in points:

            x, y, z, stamp = point[:4]

            writer.writerow(
                [x, y, z, stamp.sec, stamp.nanosec] + list(point[4:])
            )


def load_xyz_csv(path):
    # Returns (x, y, z, stamp) tuples. Missing/older-format stamp
    # columns default to zero.
    #
    # Deliberately still a 4-tuple even for extended-schema files:
    # replay.py unpacks these positionally, and the extra
    # columns have no bearing on the geometric view. Use
    # load_scan_csv() to get at them.
    points = []

    with open(path, newline='') as f:
        reader = csv.DictReader(f)

        for row in reader:
            stamp = Time()
            stamp.sec = int(row.get('stamp_sec') or 0)
            stamp.nanosec = int(row.get('stamp_nanosec') or 0)

            points.append(
                (
                    float(row['x']),
                    float(row['y']),
                    float(row['z']),
                    stamp,
                )
            )

    return points


def load_scan_csv(path):
    """
    Load a scan CSV as a dict of flat columns, ray columns included.

    Load a scan CSV as a dict of flat lists, including the extended
    ray columns when the file has them.

    Always provides "x", "y", "z", "stamp_sec" and "stamp_nanosec".
    Each key in EXTENDED_COLUMNS is present only if that column
    exists in the file, so a caller can test for it directly and
    give a clear "this scan predates ray recording" message rather
    than silently analysing zeros.
    """
    with open(path, newline='') as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames or []

        available = BASE_COLUMNS + [
            name for name in EXTENDED_COLUMNS if name in fieldnames
        ]

        columns = {name: [] for name in available}

        for row in reader:

            for name in available:
                value = row.get(name)
                columns[name].append(
                    float(value) if value not in (None, '') else float('nan')
                )

    return columns
