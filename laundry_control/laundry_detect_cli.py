#!/usr/bin/env python3

"""
laundry_detect_cli.py

CLI for laundry_detect.py: diff a candidate scan against a
baseline (empty-bucket) scan, print a ranked report of detected
laundry clusters, and optionally publish those same clustered
points to RViz for visual sanity-checking.

Each candidate point is flagged by how far it sits ABOVE the
baseline at its nearest baseline neighbor, then flagged points are
clustered - see laundry_detect.compute_deviation().

Usage:
    laundry_detect path/to/scan.csv
    laundry_detect path/to/scan.csv --baseline baseline_scans/baseline.csv
    laundry_detect path/to/scan.csv --publish

The report-only path (no --publish) needs no ROS graph at all -
rclpy is only imported once --publish is requested, so this also
works as a plain offline check against downloaded CSVs.
"""

import argparse

from .laundry_detect import (
    DEFAULT_CLUSTER_RADIUS_M,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_THRESHOLD_M,
    detect_laundry,
    load_points_xyz,
)

DEFAULT_BASELINE_PATH = "baseline_scans/baseline.csv"
DEFAULT_TOPIC = "scan_record/deviations"
DEFAULT_FRAME = "link_base"


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Diff a scan against a baseline (empty-bucket) scan "
            "to locate laundry."
        )
    )

    parser.add_argument(
        "candidate_csv",
        type=str,
        help="Path to the scan CSV to search for laundry in.",
    )

    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE_PATH,
        help=(
            "Path to the empty-bucket baseline scan CSV "
            f"(default: {DEFAULT_BASELINE_PATH})."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD_M,
        help=(
            "Minimum height (metres) above the baseline for a point "
            f"to count as a deviation (default: {DEFAULT_THRESHOLD_M})."
        ),
    )

    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=DEFAULT_CLUSTER_RADIUS_M,
        help=(
            "Radius (metres) within which deviating points are "
            f"linked into the same cluster (default: "
            f"{DEFAULT_CLUSTER_RADIUS_M})."
        ),
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=(
            "Minimum number of points for a cluster to be "
            f"reported (default: {DEFAULT_MIN_CLUSTER_SIZE})."
        ),
    )

    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Publish the reported clusters' points "
            "as a PointCloud2 for RViz."
        ),
    )

    parser.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help=f"Topic to publish on (default: {DEFAULT_TOPIC}).",
    )

    parser.add_argument(
        "--frame",
        type=str,
        default=DEFAULT_FRAME,
        help=(
            "Frame the saved points are expressed in "
            f"(default: {DEFAULT_FRAME})."
        ),
    )

    return parser


def print_report(baseline_csv, candidate_csv, clusters, args):

    baseline_count = load_points_xyz(baseline_csv).shape[0]
    candidate_count = load_points_xyz(candidate_csv).shape[0]

    print(f"Loaded baseline: {baseline_csv} ({baseline_count} pts)")
    print(f"Loaded candidate: {candidate_csv} ({candidate_count} pts)")

    total_flagged = sum(cluster.size for cluster in clusters)

    print(
        f"Found {len(clusters)} cluster(s) covering "
        f"{total_flagged} deviating point(s) "
        f"(threshold={args.threshold:.3f}m, "
        f"radius={args.cluster_radius:.3f}m, "
        f"min_size={args.min_cluster_size})"
    )

    if not clusters:
        print(
            "No laundry detected. Nothing in the candidate scan "
            "deviated from the baseline by more than --threshold."
        )
        return

    print()

    for i, cluster in enumerate(clusters, start=1):

        cx, cy, cz = cluster.centroid
        ex, ey, ez = cluster.extent
        hx, hy, hz = cluster.highest_point

        print(
            f"  #{i}  size={cluster.size}  "
            f"centroid=({cx:.3f}, {cy:.3f}, {cz:.3f})  "
            f"mean_dev={cluster.mean_deviation_m:.3f}m"
        )

        print(
            f"      bbox=({ex:.3f}, {ey:.3f}, {ez:.3f})  "
            f"highest=({hx:.3f}, {hy:.3f}, {hz:.3f})"
        )


def publish_deviations(args, clusters):
    """
    Publish the points belonging to the reported clusters (i.e.
    exactly what print_report() counted), so what you see in RViz
    matches the report one-to-one.
    """

    # Imported here, not at module scope, so the report-only path
    # above never requires a ROS graph/environment.
    import numpy as np
    import rclpy
    from rclpy.node import Node

    from sensor_msgs.msg import PointCloud2

    from .scan_cloud_util import build_cloud_with_intensity, POINT_CLOUD_QOS

    if clusters:
        cluster_xyz = np.concatenate(
            [cluster.points for cluster in clusters],
            axis=0,
        )

        # 1-based so the first cluster is still distinguishable from
        # the background when RViz maps intensity onto a colour ramp
        # starting at zero.
        cluster_ids = np.concatenate(
            [
                np.full(cluster.size, i, dtype=np.float64)
                for i, cluster in enumerate(clusters, start=1)
            ]
        )
    else:
        cluster_xyz = np.empty((0, 3))
        cluster_ids = np.empty((0,))

    class DeviationPublisherNode(Node):

        def __init__(self):
            super().__init__("laundry_detect_cli")

            self.publisher = self.create_publisher(
                PointCloud2,
                args.topic,
                POINT_CLOUD_QOS,
            )

            self.publish_cloud()

        def publish_cloud(self):

            cloud = build_cloud_with_intensity(
                frame_id=args.frame,
                stamp=self.get_clock().now().to_msg(),
                xyz_points=[
                    (float(x), float(y), float(z))
                    for x, y, z in cluster_xyz
                ],
                intensities=cluster_ids,
            )

            self.publisher.publish(cloud)

            self.get_logger().info(
                f"Published {cluster_xyz.shape[0]} clustered "
                f"point(s) across {len(clusters)} cluster(s) on "
                f"'{args.topic}' (TRANSIENT_LOCAL - late RViz "
                "subscribers will still see it). Each cluster "
                "carries its 1-based index as 'intensity' - set the "
                "display's Color Transformer to Intensity to tell "
                "them apart."
            )

    rclpy.init()

    node = DeviationPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():

    args = build_parser().parse_args()

    clusters = detect_laundry(
        baseline_csv=args.baseline,
        candidate_csv=args.candidate_csv,
        threshold_m=args.threshold,
        cluster_radius_m=args.cluster_radius,
        min_cluster_size=args.min_cluster_size,
    )

    print_report(args.baseline, args.candidate_csv, clusters, args)

    if args.publish:
        publish_deviations(args, clusters)


if __name__ == "__main__":
    main()
