#!/usr/bin/env python3

"""
laundry_detect_cli.py

CLI for laundry_detect.py: build a model of the empty bucket from
one or more baseline scans, locate laundry in a candidate scan
against it, print a ranked report, and optionally publish the
detected points to RViz for visual sanity-checking.

Each candidate point is judged by how far it INTRUDES past the
modelled empty-bucket wall, in the bucket's own cylindrical
coordinates, against a locally measured noise sigma - see
laundry_detect.compute_intrusion(). Flagged points are then
clustered on the unwrapped bucket surface.

This also prints the two sanity gates that the model rests on, on
every run, because they are cheap and because a bad cone fit or a
starved occupancy grid silently poisons every number below them:

    - how far the fitted cone had to move from the URDF/mesh seed
    - how well the (s, theta) grid is actually covered

Look at both before believing a result, especially the first few
times after the bucket or the sensor has been touched.

Usage:
    laundry_detect path/to/scan.csv
    laundry_detect path/to/scan.csv --baseline baseline_scans
    laundry_detect path/to/scan.csv --publish

The report-only path (no --publish) needs no ROS graph at all -
rclpy is only imported once --publish is requested, so this also
works as a plain offline check against downloaded CSVs.
"""

import argparse

from .bucket_model import (
    build_baseline_surface,
    fit_report,
    occupancy_summary,
)
from .laundry_detect import (
    DEFAULT_ABS_FLOOR_M,
    DEFAULT_CLUSTER_RADIUS_M,
    DEFAULT_K_SIGMA,
    DEFAULT_MIN_CLUSTER_SIZE,
    DEFAULT_MIN_EXTENT_M,
    DEFAULT_MIN_VOLUME_M3,
    detect_laundry,
    load_baseline_scans,
    load_points_xyz,
)

# A DIRECTORY by default, not a single file: the model wants 8-10
# empty scans so that the per-cell noise map - and therefore the
# calibrated threshold - is worth anything. A single CSV still
# works and still detects, but its sigma map is almost entirely the
# pooled fallback.
DEFAULT_BASELINE_PATH = "baseline_scans"
DEFAULT_TOPIC = "scan_record/deviations"
DEFAULT_FRAME = "link_base"


def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Locate laundry in a scan by measuring how far it "
            "intrudes past a fitted model of the empty bucket."
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
            "Empty-bucket baseline scan CSV, or a directory of them "
            f"(default: {DEFAULT_BASELINE_PATH})."
        ),
    )

    parser.add_argument(
        "--k-sigma",
        type=float,
        default=DEFAULT_K_SIGMA,
        help=(
            "How many local noise sigmas a point must intrude past "
            "the modelled wall to be flagged "
            f"(default: {DEFAULT_K_SIGMA})."
        ),
    )

    parser.add_argument(
        "--abs-floor",
        type=float,
        default=DEFAULT_ABS_FLOOR_M,
        help=(
            "Absolute minimum intrusion (metres) regardless of "
            "sigma, so a cell with a luckily-small sigma cannot "
            f"fire on nothing (default: {DEFAULT_ABS_FLOOR_M})."
        ),
    )

    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=DEFAULT_CLUSTER_RADIUS_M,
        help=(
            "Radius (metres, along the bucket wall) within which "
            "intruding points are linked into the same cluster "
            f"(default: {DEFAULT_CLUSTER_RADIUS_M})."
        ),
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help=(
            "Pre-filter on raw point count; the real gates are "
            "--min-extent and --min-volume "
            f"(default: {DEFAULT_MIN_CLUSTER_SIZE})."
        ),
    )

    parser.add_argument(
        "--min-extent",
        type=float,
        default=DEFAULT_MIN_EXTENT_M,
        help=(
            "Minimum cluster footprint (metres) across the bucket "
            f"wall (default: {DEFAULT_MIN_EXTENT_M})."
        ),
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=DEFAULT_MIN_VOLUME_M3,
        help=(
            "Minimum integrated intrusion volume (cubic metres) "
            f"for a cluster to be reported "
            f"(default: {DEFAULT_MIN_VOLUME_M3})."
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


def print_model_report(surface, baseline_scans, baseline_path):
    """
    Print the cone fit and grid occupancy.

    Both are sanity gates, not decoration. A fit that moved
    centimetres from the URDF/mesh seed means the bucket pose or
    the sensor extrinsics are wrong; a grid whose median cell count
    is far below the confidence threshold means the per-cell sigma
    is mostly the pooled fallback in disguise, and the bins want
    widening.
    """

    total = sum(scan.shape[0] for scan in baseline_scans)

    print(
        f"Baseline model: {len(baseline_scans)} scan(s) from "
        f"{baseline_path} ({total} pts)"
    )

    if len(baseline_scans) < 5:
        print(
            f"  NOTE: only {len(baseline_scans)} baseline scan(s). "
            "Per-cell sigma needs ~8-10 empty scans to mean much; "
            "below that most cells fall back to the pooled sigma "
            "and the threshold is not really calibrated."
        )

    print()
    print(fit_report(surface.cone))
    print()
    print(occupancy_summary(surface))
    print()


def print_report(candidate_csv, clusters, args):

    candidate_count = load_points_xyz(candidate_csv).shape[0]

    print(f"Loaded candidate: {candidate_csv} ({candidate_count} pts)")

    total_flagged = sum(cluster.size for cluster in clusters)

    print(
        f"Found {len(clusters)} cluster(s) covering "
        f"{total_flagged} intruding point(s) "
        f"(k_sigma={args.k_sigma:.1f}, "
        f"abs_floor={args.abs_floor:.3f}m, "
        f"radius={args.cluster_radius:.3f}m, "
        f"min_extent={args.min_extent:.3f}m, "
        f"min_volume={args.min_volume:.2e}m3)"
    )

    if not clusters:
        print(
            "No laundry detected. Nothing in the candidate scan "
            "intruded past the modelled bucket wall by enough to "
            "clear the local noise."
        )
        return

    print()

    for i, cluster in enumerate(clusters, start=1):

        cx, cy, cz = cluster.centroid
        ex, ey, ez = cluster.extent
        hx, hy, hz = cluster.highest_point

        confidence = "" if cluster.confident else "  [LOW CONFIDENCE]"

        print(
            f"  #{i}  size={cluster.size}  "
            f"vol={cluster.volume_m3 * 1e6:.1f}cm3  "
            f"centroid=({cx:.3f}, {cy:.3f}, {cz:.3f}){confidence}"
        )

        print(
            f"      mean_intrusion={cluster.mean_deviation_m:.3f}m  "
            f"max={cluster.max_intrusion_m:.3f}m  "
            f"wall_extent={cluster.surface_extent_m:.3f}m"
        )

        print(
            f"      bbox=({ex:.3f}, {ey:.3f}, {ez:.3f})  "
            f"highest=({hx:.3f}, {hy:.3f}, {hz:.3f})"
        )

    if any(not cluster.confident for cluster in clusters):
        print()
        print(
            "Low-confidence clusters sit mostly in thinly-sampled "
            "parts of the baseline grid. They are reported rather "
            "than suppressed on purpose: a missed item costs more "
            "than a wasted look, and sparse coverage tends to "
            "coincide with the awkward spots laundry actually ends "
            "up in. Take more baseline scans to firm them up."
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

    # The surface is built here rather than inside detect_laundry()
    # so the fit and occupancy reports can be printed before any
    # detection result is shown.
    baseline_scans = load_baseline_scans(args.baseline)
    surface = build_baseline_surface(baseline_scans)

    print_model_report(surface, baseline_scans, args.baseline)

    clusters = detect_laundry(
        baseline_csv=args.baseline,
        candidate_csv=args.candidate_csv,
        k_sigma=args.k_sigma,
        abs_floor_m=args.abs_floor,
        cluster_radius_m=args.cluster_radius,
        min_cluster_size=args.min_cluster_size,
        min_extent_m=args.min_extent,
        min_volume_m3=args.min_volume,
        surface=surface,
    )

    print_report(args.candidate_csv, clusters, args)

    if args.publish:
        publish_deviations(args, clusters)


if __name__ == "__main__":
    main()
