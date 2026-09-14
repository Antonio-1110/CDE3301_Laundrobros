#!/usr/bin/env python3

"""
laundry_detect.py

Locate laundry inside the bucket by diffing a scan against a
baseline (empty-bucket) scan.

The wrist-mounted ToF sensor only ever reports "distance to
whatever the beam hit first" - there is no way to tell a bucket
wall/floor return apart from a laundry-item return using a single
reading. Comparing against a known-empty baseline scan is the only
practical way to do so with this sensor: any point in a later scan
that lands significantly closer to the sensor than what the empty
bucket produced at roughly the same spot indicates something is now
in the beam's path.

This module is intentionally free of any live ROS node/graph
dependency (no rclpy.Node, no topics/services) - it only reads
already-saved CSVs (via scan_cloud_util.load_xyz_csv) and does
numpy/scipy geometry, so it can be unit-tested and reused from a
plain offline script without a running ROS system.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .scan_cloud_util import load_xyz_csv

# Defaults tied to the physical scan geometry (see scan_move.py):
#
#   - scan_move.py's default insertion step is 3 cm, so consecutive
#     sweep passes over the SAME empty-bucket surface are ~3 cm
#     apart. DEFAULT_THRESHOLD_M must stay below that so the normal
#     inter-pass gap in the baseline cloud is never mistaken for a
#     deviation, while staying above the VL53L0X's noise and the
#     xArm7/TF repeatability (both sub-cm).
#
#   - DEFAULT_CLUSTER_RADIUS_M must exceed the 3 cm step so that
#     points hit on the same laundry item across adjacent sweep
#     passes still link into one cluster, while staying much
#     smaller than the bucket/scan scale so distinct items don't
#     get merged together.
DEFAULT_THRESHOLD_M = 0.02
DEFAULT_CLUSTER_RADIUS_M = 0.04
DEFAULT_MIN_CLUSTER_SIZE = 4


def load_points_xyz(csv_path: str) -> np.ndarray:
    """
    Load a scan CSV (see scan_cloud_util.save_xyz_csv) and return
    an (N, 3) float64 array of just the x, y, z columns.

    Capture timestamps are dropped: points carry no per-point
    sweep-angle/joint-state metadata to begin with, so timing isn't
    needed for a purely spatial diff.
    """

    points = load_xyz_csv(csv_path)

    if not points:
        raise ValueError(f"No points found in {csv_path!r}.")

    return np.array(
        [(x, y, z) for x, y, z, _stamp in points],
        dtype=np.float64,
    )


@dataclass
class DeviationResult:
    """
    Per-candidate-point nearest-neighbor distance to the baseline
    cloud, and the resulting deviation mask.
    """

    distances: np.ndarray
    mask: np.ndarray
    threshold_m: float


def compute_deviation(
    baseline_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    threshold_m: float = DEFAULT_THRESHOLD_M,
) -> DeviationResult:
    """
    For every point in candidate_xyz, find its Euclidean distance
    to the nearest point in baseline_xyz.

    Vectorized: a single cKDTree is built on baseline_xyz and
    queried for all candidate points at once (no per-point Python
    loop), which is trivial at the ~1000-1200 points per scan this
    package produces.
    """

    if baseline_xyz.shape[0] == 0:

        raise ValueError(
            "baseline_xyz has no points; cannot compute deviation."
        )

    tree = cKDTree(baseline_xyz)

    distances, _indices = tree.query(
        candidate_xyz,
        k=1,
        workers=-1,
    )

    distances = np.atleast_1d(distances)

    mask = distances >= threshold_m

    return DeviationResult(
        distances=distances,
        mask=mask,
        threshold_m=threshold_m,
    )


def cluster_points(
    points_xyz: np.ndarray,
    radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> List[np.ndarray]:
    """
    Group points_xyz into spatial clusters via a radius graph +
    connected components (points within radius_m of one another are
    linked, transitively, into the same cluster).

    Returns a list of INDEX arrays into points_xyz (not coordinate
    arrays), one per surviving cluster, sorted by descending size.
    Clusters smaller than min_cluster_size are dropped - this also
    naturally discards isolated noise points (which form their own
    singleton components), with no separate outlier-handling path
    needed.
    """

    n = points_xyz.shape[0]

    if n < min_cluster_size:
        return []

    tree = cKDTree(points_xyz)

    pairs = tree.query_pairs(
        r=radius_m,
        output_type="ndarray",
    )

    if pairs.shape[0] == 0:
        # No point is within radius_m of any other: every point is
        # its own singleton component, so nothing survives
        # min_cluster_size (unless min_cluster_size is 1).
        adjacency = coo_matrix((n, n))

    else:

        rows = pairs[:, 0]
        cols = pairs[:, 1]

        adjacency = coo_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n, n),
        )

    _n_components, labels = connected_components(
        adjacency,
        directed=False,
    )

    clusters = []

    for label in np.unique(labels):

        member_indices = np.flatnonzero(labels == label)

        if member_indices.shape[0] >= min_cluster_size:
            clusters.append(member_indices)

    clusters.sort(
        key=lambda indices: indices.shape[0],
        reverse=True,
    )

    return clusters


@dataclass
class ClusterSummary:
    """
    Summary of one detected laundry cluster, in base_frame metres.
    """

    points: np.ndarray
    centroid: np.ndarray
    size: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    extent: np.ndarray
    highest_point: np.ndarray
    mean_deviation_m: float


def summarize_clusters(
    cluster_indices: List[np.ndarray],
    points_xyz: np.ndarray,
    deviation_distances: Optional[np.ndarray] = None,
) -> List[ClusterSummary]:
    """
    Build a ClusterSummary per cluster.

    cluster_indices:
        As returned by cluster_points() - index arrays into
        points_xyz.

    deviation_distances:
        Nearest-neighbor distances aligned index-for-index with
        points_xyz (e.g. DeviationResult.distances restricted to
        the same deviating-point subset that was clustered), used
        to compute mean_deviation_m per cluster. If omitted,
        mean_deviation_m is NaN.

    Order of the input cluster_indices is preserved (cluster_points
    already sorts largest-first).
    """

    summaries = []

    for indices in cluster_indices:

        member_points = points_xyz[indices]

        bbox_min = member_points.min(axis=0)
        bbox_max = member_points.max(axis=0)

        highest_point = member_points[
            np.argmax(member_points[:, 2])
        ]

        if deviation_distances is not None:
            mean_deviation_m = float(
                deviation_distances[indices].mean()
            )
        else:
            mean_deviation_m = float("nan")

        summaries.append(
            ClusterSummary(
                points=member_points,
                centroid=member_points.mean(axis=0),
                size=member_points.shape[0],
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                extent=bbox_max - bbox_min,
                highest_point=highest_point,
                mean_deviation_m=mean_deviation_m,
            )
        )

    return summaries


def detect_laundry(
    baseline_csv: str,
    candidate_csv: str,
    threshold_m: float = DEFAULT_THRESHOLD_M,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> List[ClusterSummary]:
    """
    End-to-end: load both CSVs, compute per-point deviation from
    the baseline, cluster the deviating candidate points, and
    return per-cluster summaries sorted largest-first.

    This is the single function both the CLI and any future
    test/analysis script should call - the building blocks above
    are exposed individually mainly so they can be unit-tested in
    isolation (e.g. cluster_points() against synthetic points with
    no CSV/disk involved at all).
    """

    baseline_xyz = load_points_xyz(baseline_csv)
    candidate_xyz = load_points_xyz(candidate_csv)

    deviation = compute_deviation(
        baseline_xyz,
        candidate_xyz,
        threshold_m=threshold_m,
    )

    deviating_xyz = candidate_xyz[deviation.mask]
    deviating_distances = deviation.distances[deviation.mask]

    cluster_indices = cluster_points(
        deviating_xyz,
        radius_m=cluster_radius_m,
        min_cluster_size=min_cluster_size,
    )

    return summarize_clusters(
        cluster_indices,
        deviating_xyz,
        deviating_distances,
    )
