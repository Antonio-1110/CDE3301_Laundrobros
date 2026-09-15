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
# to a real baseline-vs-laundry comparison (14 sept 2026):
#
#   - Measured nearest-neighbor deviation distances on a real scan
#     showed baseline noise/repeatability sitting at p90 ~2.7cm,
#     jumping to p95 ~6.1cm once real laundry points are reached -
#     i.e. a clear gap between ~3cm and ~6cm. DEFAULT_THRESHOLD_M
#     sits in that gap, well above sensor noise/xArm7/TF
#     repeatability and above the normal inter-pass gap from
#     scan_move.py's 3cm insertion step, so it doesn't flag the
#     empty-bucket baseline's own sampling gaps as deviations.
#
#   - DEFAULT_CLUSTER_RADIUS_M must exceed the 3 cm step so that
#     points hit on the same laundry item across adjacent sweep
#     passes still link into one cluster, while staying much
#     smaller than the bucket/scan scale so distinct items don't
#     get merged together.
DEFAULT_THRESHOLD_M = 0.04
DEFAULT_CLUSTER_RADIUS_M = 0.04
DEFAULT_MIN_CLUSTER_SIZE = 4

# compute_cell_deviation()'s defaults. Averaging several points'
# z within a cell suppresses random per-point sensor noise (it's
# zero-mean, so it washes out), while a real item - even a subtle
# one - shifts every point in its footprint the same direction, so
# the cell mean still moves even when individual points wouldn't
# have cleared DEFAULT_THRESHOLD_M on their own. That's what lets
# this mode catch smaller/subtler items the point-wise mode misses.
#
#   - DEFAULT_CELL_SIZE_M matches scan_move.py's 3cm insertion step:
#     fine enough that a genuinely small item still gets its own
#     cell(s) rather than being averaged together with surrounding
#     untouched-baseline points (which would wash the signal back
#     toward zero), coarse enough to actually gather multiple points
#     per cell given this package's sparse (~1000-1200 point) scans.
#   - DEFAULT_CELL_THRESHOLD_M is set below DEFAULT_THRESHOLD_M
#     specifically because cell averaging is expected to suppress
#     noise variance - start here and tune against real data the
#     same way DEFAULT_THRESHOLD_M itself was calibrated.
#   - DEFAULT_CELL_MIN_POINTS requires at least 2 points before
#     trusting a cell's mean at all (a lone point gets no averaging
#     benefit - it behaves exactly like the point-wise mode).
#   - DEFAULT_CELL_BASELINE_K matches grasp_plan.py's
#     DEFAULT_BASELINE_K for the same reason: averages over roughly
#     one step-radius patch of the baseline around each cell.
DEFAULT_CELL_SIZE_M = 0.03
DEFAULT_CELL_THRESHOLD_M = 0.02
DEFAULT_CELL_MIN_POINTS = 2
DEFAULT_CELL_BASELINE_K = 5


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


@dataclass
class CellDeviationResult:
    """
    Per-candidate-point cell-mean deviation and the resulting mask.

    Unlike DeviationResult (one distance per point, computed
    independently), every point sharing a cell gets the SAME
    cell_deviations value and the SAME mask outcome - the decision
    is made once per cell, not once per point.
    """

    cell_deviations: np.ndarray
    mask: np.ndarray
    threshold_m: float
    cell_size_m: float


def compute_cell_deviation(
    baseline_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    threshold_m: float = DEFAULT_CELL_THRESHOLD_M,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    min_points_per_cell: int = DEFAULT_CELL_MIN_POINTS,
    baseline_k: int = DEFAULT_CELL_BASELINE_K,
) -> CellDeviationResult:
    """
    Grid candidate_xyz into cell_size_m x cell_size_m cells over
    (x, y) (an NDT-style "box each cell" pass, simplified to a mean
    rather than a full per-cell Gaussian/covariance, since this
    package's scans are too sparse - often 1-3 points per cell - to
    estimate a stable variance): for each cell with at least
    min_points_per_cell candidate points, compare that cell's mean z
    against a local baseline estimate (mean z of the baseline_k
    nearest baseline points to the cell's own mean (x, y) - the same
    local-lookup technique grasp_plan.estimate_baseline_depth()
    already uses). Every point in a cell whose mean deviates by at
    least threshold_m is flagged, all with that cell's single shared
    deviation value.

    Cells with fewer than min_points_per_cell candidate points are
    left unflagged (not enough of them to trust a cell mean - see
    the module-level defaults comment for why a lone point is best
    left to the point-wise compute_deviation() instead).
    """

    n = candidate_xyz.shape[0]

    mask = np.zeros(n, dtype=bool)
    cell_deviations = np.zeros(n, dtype=np.float64)

    if n == 0:
        return CellDeviationResult(
            cell_deviations=cell_deviations,
            mask=mask,
            threshold_m=threshold_m,
            cell_size_m=cell_size_m,
        )

    if baseline_xyz.shape[0] == 0:

        raise ValueError(
            "baseline_xyz has no points; cannot compute deviation."
        )

    cell_indices = np.floor(
        candidate_xyz[:, :2] / cell_size_m
    ).astype(np.int64)

    _unique_cells, inverse, counts = np.unique(
        cell_indices,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    baseline_tree = cKDTree(baseline_xyz[:, :2])
    baseline_k_eff = min(baseline_k, baseline_xyz.shape[0])

    # inverse.reshape(-1) guards against a numpy version quirk where
    # return_inverse can come back with an extra trailing dimension.
    inverse = inverse.reshape(-1)

    for cell_id in range(counts.shape[0]):

        if counts[cell_id] < min_points_per_cell:
            continue

        point_mask = inverse == cell_id

        cell_points = candidate_xyz[point_mask]
        mean_xy = cell_points[:, :2].mean(axis=0)
        mean_z = cell_points[:, 2].mean()

        _distances, baseline_indices = baseline_tree.query(
            mean_xy,
            k=baseline_k_eff,
        )

        baseline_indices = np.atleast_1d(baseline_indices)
        baseline_mean_z = baseline_xyz[baseline_indices, 2].mean()

        deviation = abs(mean_z - baseline_mean_z)

        cell_deviations[point_mask] = deviation

        if deviation >= threshold_m:
            mask[point_mask] = True

    return CellDeviationResult(
        cell_deviations=cell_deviations,
        mask=mask,
        threshold_m=threshold_m,
        cell_size_m=cell_size_m,
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


def detect_laundry_by_cell(
    baseline_csv: str,
    candidate_csv: str,
    threshold_m: float = DEFAULT_CELL_THRESHOLD_M,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    min_points_per_cell: int = DEFAULT_CELL_MIN_POINTS,
    baseline_k: int = DEFAULT_CELL_BASELINE_K,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> List[ClusterSummary]:
    """
    Same end-to-end shape as detect_laundry(), but flags points via
    compute_cell_deviation() (cell-mean comparison) instead of
    compute_deviation() (raw point-to-nearest-point comparison) -
    see compute_cell_deviation()'s docstring for why this can catch
    smaller/subtler items the point-wise mode misses as noise.

    Clustering and summarization are identical to detect_laundry()
    (same cluster_points()/summarize_clusters() calls) - only the
    flagging step differs, so results from the two modes are
    directly comparable on the same data.
    """

    baseline_xyz = load_points_xyz(baseline_csv)
    candidate_xyz = load_points_xyz(candidate_csv)

    deviation = compute_cell_deviation(
        baseline_xyz,
        candidate_xyz,
        threshold_m=threshold_m,
        cell_size_m=cell_size_m,
        min_points_per_cell=min_points_per_cell,
        baseline_k=baseline_k,
    )

    deviating_xyz = candidate_xyz[deviation.mask]
    deviating_distances = deviation.cell_deviations[deviation.mask]

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
