#!/usr/bin/env python3

"""
Human-readable detection reports, and publishing detections to RViz.

Used by `laundry detect`, `laundry run` and `laundry grasp`.

Every report starts with the two sanity gates the model rests on,
because they are cheap and because a bad cone fit or a starved
occupancy grid silently poisons every number below them:

    - how far the fitted cone had to move from the URDF/mesh seed
    - how well the (u, theta) grid is actually covered

Look at both before believing a result, especially the first few
times after the bucket or the sensor has been touched.

The report functions need no ROS graph at all; rclpy is only
imported by publish_clusters(), so offline detection works on a
machine with no ROS graph running.
"""

from .bucket_model import fit_report, occupancy_summary
from .. import config

DEFAULT_TOPIC = 'scan_record/deviations'


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
        f'Baseline model: {len(baseline_scans)} scan(s) from '
        f'{baseline_path} ({total} pts)'
    )

    if len(baseline_scans) < 5:
        print(
            f'  NOTE: only {len(baseline_scans)} baseline scan(s). '
            'Per-cell sigma needs ~8-10 empty scans to mean much; '
            'below that most cells fall back to the pooled sigma '
            'and the threshold is not really calibrated.'
        )

    print()
    print(fit_report(surface.cone))
    print()
    print(occupancy_summary(surface))
    print()


def print_report(candidate_csv, candidate_count, clusters, params):
    """
    Print the ranked cluster report for one candidate scan.

    params is the dict of detector keyword arguments the clusters
    came from (see perception.detect.detect_on_points), echoed so
    every report records the operating point that produced it.
    """
    print(f'Loaded candidate: {candidate_csv} ({candidate_count} pts)')

    total_flagged = sum(cluster.size for cluster in clusters)

    print(
        f'Found {len(clusters)} cluster(s) covering '
        f'{total_flagged} intruding point(s) '
        f"(k_sigma={params['k_sigma']:.1f}, "
        f"abs_floor={params['abs_floor_m']:.3f}m, "
        f"radius={params['cluster_radius_m']:.3f}m, "
        f"min_extent={params['min_extent_m']:.3f}m, "
        f"min_volume={params['min_volume_m3']:.2e}m3)"
    )

    if not clusters:
        print(
            'No laundry detected. Nothing in the candidate scan '
            'intruded past the modelled bucket wall by enough to '
            'clear the local noise.'
        )
        return

    print()

    for i, cluster in enumerate(clusters, start=1):

        cx, cy, cz = cluster.centroid
        ex, ey, ez = cluster.extent
        hx, hy, hz = cluster.highest_point

        confidence = '' if cluster.confident else '  [LOW CONFIDENCE]'

        gx, gy, gz = cluster.target_point

        print(
            f'  #{i}  size={cluster.size}  '
            f'vol={cluster.volume_m3 * 1e6:.1f}cm3  '
            f'peak={cluster.peak_sigma:.1f}sigma  '
            f'grasp=({gx:.3f}, {gy:.3f}, {gz:.3f}){confidence}'
        )

        print(
            f'      mean_intrusion={cluster.mean_deviation_m:.3f}m  '
            f'max={cluster.max_intrusion_m:.3f}m  '
            f'wall_extent={cluster.surface_extent_m:.3f}m'
        )

        print(f'      centroid=({cx:.3f}, {cy:.3f}, {cz:.3f})')

        print(
            f'      bbox=({ex:.3f}, {ey:.3f}, {ez:.3f})  '
            f'highest=({hx:.3f}, {hy:.3f}, {hz:.3f})'
        )

    if any(not cluster.confident for cluster in clusters):
        print()
        print(
            'Low-confidence clusters sit mostly in thinly-sampled '
            'parts of the baseline grid. They are reported rather '
            'than suppressed on purpose: a missed item costs more '
            'than a wasted look, and sparse coverage tends to '
            'coincide with the awkward spots laundry actually ends '
            'up in. Take more baseline scans to firm them up.'
        )


def publish_clusters(clusters, topic=DEFAULT_TOPIC, frame=config.BASE_FRAME):
    """
    Publish the reported clusters' points as a PointCloud2 for RViz.

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

    from ..scan.cloud_io import build_cloud_with_intensity, POINT_CLOUD_QOS

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
            super().__init__('laundry_detect')

            self.publisher = self.create_publisher(
                PointCloud2,
                topic,
                POINT_CLOUD_QOS,
            )

            self.publish_cloud()

        def publish_cloud(self):

            cloud = build_cloud_with_intensity(
                frame_id=frame,
                stamp=self.get_clock().now().to_msg(),
                xyz_points=[
                    (float(x), float(y), float(z))
                    for x, y, z in cluster_xyz
                ],
                intensities=cluster_ids,
            )

            self.publisher.publish(cloud)

            self.get_logger().info(
                f'Published {cluster_xyz.shape[0]} clustered '
                f'point(s) across {len(clusters)} cluster(s) on '
                f"'{topic}' (TRANSIENT_LOCAL - late RViz "
                'subscribers will still see it). Each cluster '
                "carries its 1-based index as 'intensity' - set the "
                "display's Color Transformer to Intensity to tell "
                'them apart.'
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
