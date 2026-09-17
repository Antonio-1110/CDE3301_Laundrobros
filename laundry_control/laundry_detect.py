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

# Defaults tied to the physical scan geometry (see scan_move.py) and
# to a real calibration run (17 sept 2026, calibrate_threshold.py,
# empty-vs-empty for noise and empty-vs-known-laundry for signal),
# measured with the CURRENT one-sided signed-z metric:
#
#   - Noise (two empty-bucket scans over the same path) topped out
#     at 1.86cm, with p95 ~0.9cm and p99 ~1.4cm - far tighter than
#     the 2.7cm p90 measured before the metric went one-sided,
#     because the old symmetric/unsigned distance was charging both
#     tails of registration error against the threshold.
#
#   - Signal (known laundry present) only separates from that noise
#     in its upper tail: p99 ~5.3cm, max ~5.7cm. Most points in a
#     laundry scan still hit bare bucket, so the bulk of the
#     distribution legitimately looks like noise.
#
#   - DEFAULT_THRESHOLD_M is set from the END-TO-END behaviour
#     (deviation + clustering), not from the per-point noise
#     ceiling, because cluster_points()'s min_cluster_size filter
#     turns out to reject false positives far more effectively than
#     the threshold does: scattered noise points rarely land within
#     DEFAULT_CLUSTER_RADIUS_M of three other noise points, while a
#     real item's flagged points are contiguous by construction.
#     Sweeping the threshold on the 17 sept scans, the empty-vs-
#     empty pair produced its first false cluster at 0.010 and was
#     clean from 0.012 up, while the laundry scan's cluster grew
#     from 14 points at 0.020 to 18 at 0.012. 0.015 sits between
#     the two: ~50% more points on a real item than 0.020 gave,
#     with meaningful margin above the 0.010 breakdown - margin
#     worth keeping, since "clean" is so far based on a single
#     empty-vs-empty pair.
#
#     Sparse coverage of an item is acceptable here - downstream can
#     interpolate/hull a partial cluster, but it can't recover an
#     item that was never flagged.
#
#   - DEFAULT_CLUSTER_RADIUS_M must exceed the 3 cm step so that
#     points hit on the same laundry item across adjacent sweep
#     passes still link into one cluster, while staying much
#     smaller than the bucket/scan scale so distinct items don't
#     get merged together.
DEFAULT_THRESHOLD_M = 0.015
DEFAULT_CLUSTER_RADIUS_M = 0.04
DEFAULT_MIN_CLUSTER_SIZE = 4

# XY radius of the neighbourhood compute_deviation() takes the local
# baseline surface height from. Matches scan_move.py's 3cm insertion
# step, so it spans roughly one sweep pass either side: wide enough
# to always contain baseline samples wherever the scan had coverage,
# narrow enough not to drag in a wall standing well away laterally.
DEFAULT_SURFACE_RADIUS_M = 0.03

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
    Per-candidate-point height above the local baseline surface,
    and the resulting deviation mask.

    heights is that height in metres: positive means the candidate
    point stands proud of the empty bucket's surface at the same
    lateral position, which is what a laundry item does. Points
    with no baseline coverage nearby carry -inf (see
    compute_deviation).
    """

    heights: np.ndarray
    mask: np.ndarray
    threshold_m: float
    radius_m: float


def compute_deviation(
    baseline_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    threshold_m: float = DEFAULT_THRESHOLD_M,
    radius_m: float = DEFAULT_SURFACE_RADIUS_M,
) -> DeviationResult:
    """
    For every candidate point, measure how far it stands ABOVE the
    empty bucket's surface at the same lateral (x, y) position, and
    flag it when that height reaches threshold_m.

    The local surface is taken as the HIGHEST baseline point within
    radius_m in XY. Not the nearest baseline point's z, and not a
    mean over neighbours - both of those are wrong on this bucket,
    for the same underlying reason:

        z is not single-valued over (x, y) here. Measured on a real
        baseline, 65% of points sit in neighbourhoods spanning more
        in z than the detector's whole noise ceiling, up to 13cm
        inside a single 3cm cell, because the bucket walls are
        steep in base_frame. Comparing a point against the single
        nearest neighbour therefore compares it against whichever
        part of the wall happens to be closest in 3D - typically a
        point LOWER down the same wall. The candidate then reads as
        "above the baseline" while actually sitting below the
        wall's local top, and a patch of bare bucket wall gets
        reported as laundry. That is exactly what produced four
        spurious clusters (all measuring 2.5-8.7cm BELOW the local
        surface) on the 17 sept run.

    Taking the local maximum instead asks the question that
    actually matters - "is this point above everything the empty
    bucket ever presented here?" - which only a real object resting
    in the bucket can be true of. It is also one-sided by
    construction: an item can only intercept the beam early and
    stand proud of the surface, never sink below it.

    Points with no baseline point within radius_m get -inf and are
    never flagged: with no local reference there is nothing to
    judge them against, and guessing from a far-away baseline point
    is what the old nearest-neighbour metric did wrong.
    """

    if baseline_xyz.shape[0] == 0:

        raise ValueError(
            "baseline_xyz has no points; cannot compute deviation."
        )

    n = candidate_xyz.shape[0]

    heights = np.full(n, -np.inf, dtype=np.float64)

    if n == 0:
        return DeviationResult(
            heights=heights,
            mask=np.zeros(0, dtype=bool),
            threshold_m=threshold_m,
            radius_m=radius_m,
        )

    tree = cKDTree(baseline_xyz[:, :2])

    neighbourhoods = tree.query_ball_point(
        candidate_xyz[:, :2],
        r=radius_m,
        workers=-1,
    )

    for i, indices in enumerate(neighbourhoods):

        if not indices:
            continue

        heights[i] = candidate_xyz[i, 2] - baseline_xyz[indices, 2].max()

    mask = heights >= threshold_m

    return DeviationResult(
        heights=heights,
        mask=mask,
        threshold_m=threshold_m,
        radius_m=radius_m,
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
    deviation_heights: Optional[np.ndarray] = None,
) -> List[ClusterSummary]:
    """
    Build a ClusterSummary per cluster.

    cluster_indices:
        As returned by cluster_points() - index arrays into
        points_xyz.

    deviation_heights:
        Heights above the local baseline surface, aligned
        index-for-index with points_xyz (e.g.
        DeviationResult.heights restricted to the same
        deviating-point subset that was clustered), used to compute
        mean_deviation_m per cluster. If omitted, mean_deviation_m
        is NaN.

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

        if deviation_heights is not None:
            mean_deviation_m = float(
                deviation_heights[indices].mean()
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
    surface_radius_m: float = DEFAULT_SURFACE_RADIUS_M,
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
        radius_m=surface_radius_m,
    )

    deviating_xyz = candidate_xyz[deviation.mask]
    deviating_heights = deviation.heights[deviation.mask]

    cluster_indices = cluster_points(
        deviating_xyz,
        radius_m=cluster_radius_m,
        min_cluster_size=min_cluster_size,
    )

    return summarize_clusters(
        cluster_indices,
        deviating_xyz,
        deviating_heights,
    )
