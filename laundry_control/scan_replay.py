#!/usr/bin/env python3

"""
scan_replay.py

Republishes a previously saved ToF scan (see scan_move.py
--save-points / TofScanRecorder.save_csv) as a PointCloud2, so it
can be revisited in RViz without re-running the physical scan.

Points were recorded in the robot base frame ("link_base" by
default), which does not move - so replaying them next to a
live or recorded robot model (and whatever obstacle geometry is
loaded, e.g. the bucket/table) still lines them up correctly.

Usage:
    python3 scan_replay.py path/to/scan.csv
    python3 scan_replay.py path/to/scan.csv --topic scan_record/points --frame link_base

Note: the RViz obstacle/robot model is only an approximation of
the real setup (see README) - treat point positions as
approximate relative to real-world obstacles too.
"""

import argparse

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2

from .scan_cloud_util import build_cloud, load_xyz_csv, POINT_CLOUD_QOS


class ScanReplayNode(Node):
    def __init__(self, csv_path, topic, frame_id):
        super().__init__("scan_replay")

        self.frame_id = frame_id
        # Loaded as (x, y, z, stamp) tuples; stamp is the original
        # ToF capture time (unused for the one-shot replay below,
        # but preserved in the CSV for future timing-accurate replay).
        self.points = load_xyz_csv(csv_path)

        self.get_logger().info(
            f"Loaded {len(self.points)} points from {csv_path}"
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            topic,
            POINT_CLOUD_QOS,
        )

        self.publish_cloud()

    def publish_cloud(self):
        xyz_points = [(x, y, z) for x, y, z, _ in self.points]

        cloud = build_cloud(
            frame_id=self.frame_id,
            stamp=self.get_clock().now().to_msg(),
            xyz_points=xyz_points,
        )

        self.publisher.publish(cloud)

        self.get_logger().info(
            "Published replayed scan cloud "
            "(TRANSIENT_LOCAL - late RViz subscribers will still see it)."
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Replay a saved ToF scan as a PointCloud2 for RViz."
    )

    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to a CSV file saved by TofScanRecorder.save_csv().",
    )

    parser.add_argument(
        "--topic",
        type=str,
        default="scan_record/points",
        help=(
            "Topic to publish on (default: scan_record/points, "
            "same as the live recorder, so the same RViz display works)."
        ),
    )

    parser.add_argument(
        "--frame",
        type=str,
        default="link_base",
        help="Frame the saved points are expressed in (default: link_base).",
    )

    return parser


def main():
    args = build_parser().parse_args()

    rclpy.init()

    node = ScanReplayNode(args.csv_path, args.topic, args.frame)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
