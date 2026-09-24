#!/usr/bin/env python3

"""
Locate laundry inside the bucket by comparing a scan against the empty-bucket model.

The model is bucket_model.BaselineSurface; run from the terminal as
`laundry detect` (cli.py).

HOW THIS WORKS, AND WHY IT CHANGED
----------------------------------
The wrist-mounted ToF sensor only ever reports "distance to
whatever the beam hit first" - there is no way to tell a bucket
wall return apart from a laundry return using a single reading. So
detection is still fundamentally "compare against the empty
bucket". What changed is the coordinate system that comparison
happens in.

This module used to ask, for each candidate point, "is it higher in
z than the tallest baseline point within 3cm in (x, y)?". That
question is ill-posed on a bucket lying on its side: z is not
single-valued over (x, y), so a steep wall packs 13cm of z into one
3cm cell (see bucket_model.py for the full argument). Worse, the
test was one-sided in z, which is only correct on the bucket FLOOR
- laundry against a side wall displaces x rather than z, and
laundry on the ceiling lowers z, so both were invisible to it.

Detection now happens in the bucket's own cylindrical coordinates:

    intrusion = r_expected(s, theta) - r_measured

r_expected comes from bucket_model.BaselineSurface (a robustly
fitted cone plus a residual field learned from several empty
scans). r over (s, theta) is single-valued everywhere, so this test
is well-posed on the floor, the walls AND the ceiling, and it is
one-sided by construction: an object can only ever intercept the
beam EARLY, never late.

Two further consequences worth knowing:

  - There is no "no baseline coverage here" blind spot any more.
    The old code gave such points -inf and could never flag them,
    which turned every gap in the scan path into a region laundry
    could hide in. The fitted model is dense everywhere; thinly
    covered cells are marked low-confidence and reported, not
    silently dropped.

  - The threshold is a calibrated statistic (k * sigma from the
    per-cell noise map) rather than one global constant tuned on a
    single pair of scans.

This module is intentionally free of any live ROS node/graph
dependency (no rclpy.Node, no topics/services) - it only reads
already-saved CSVs (via scan.cloud_io.load_xyz_csv) and does
numpy/scipy geometry, so it can be unit-tested and reused from a
plain offline script without a running ROS system.
"""

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .bucket_model import (
    BaselineSurface,
    build_baseline_surface,
    to_cylindrical,
)
from ..scan.cloud_io import load_xyz_csv

# ---------------------------------------------------------------
# Detection thresholds
#
# These are STARTING POINTS, not calibrated values. Unlike the old
# global 1.5cm constant, the primary threshold is now a multiple of
# the locally measured noise sigma, so it adapts to the fact that
# residual noise varies strongly with incidence angle and coverage
# across the bucket.
#
# Run `laundry evaluate` (perception/evaluate.py) to set them properly: it does
# leave-one-out over the empty baselines (every cluster it reports
# is a false positive) and known-object runs (every miss is a false
# negative), and sweeps these two numbers over both.
#
# Bias toward RECALL when choosing an operating point. The costs
# are asymmetric: a false positive costs one wasted look, a false
# negative leaves laundry in the bucket, which is the failure the
# whole machine exists to prevent.
# ---------------------------------------------------------------

# Intrusion must exceed this many local sigmas to be flagged.
DEFAULT_K_SIGMA = 4.0

# ...and also this absolute floor, whichever is larger. Without it,
# a cell that happens to have an artificially small sigma (a handful
# of samples that agreed by luck) would fire on nothing at all.
DEFAULT_ABS_FLOOR_M = 0.008

# A point reading FURTHER out than the model wall by more than this
# is discarded rather than judged: it is behind the bucket surface,
# which is physically impossible, so it is a stray/specular return
# or a bad TF lookup. Generous, because the wall itself has real
# outward scatter.
DEFAULT_BEHIND_WALL_MARGIN_M = 0.03

# Clustering happens in plain 3D. An earlier version clustered on
# the bucket surface unwrapped into 2D, which was necessary when
# the model was a bare cone; once the model became a full profile
# (flat closed end + corner + wall) that chart stopped working -
# unwrapping with the local radius distorts axial distances by up
# to 17% at large theta as the cone tapers, and the cap's centre is
# a coordinate singularity where every theta collapses to a point.
#
# 3D has neither problem, plus no seam and no special case at the
# wall/cap corner. It was checked against the unwrapped version on
# items planted all the way round the bucket and gave identical
# clusters every time: at a 4cm radius the chord-versus-geodesic
# difference on a 0.2m-radius bucket is under 0.1%, far too small
# to change any linkage.
#
# The radius must exceed scan.pattern's 3cm insertion step so that
# points hit on the same item across adjacent sweep passes still
# link into one cluster, while staying well under the bucket scale
# so that distinct items don't get merged.
DEFAULT_CLUSTER_RADIUS_M = 0.04

# Cheap pre-filter only. The real gates are physical (extent and
# volume, below) - an absolute point count is a poor discriminator
# here because the scan's point density is strongly non-uniform, so
# the same physical item yields ten points in one region and three
# in another.
DEFAULT_MIN_CLUSTER_SIZE = 3

# Physical gates, applied to the cluster's footprint on the bucket
# wall and to its integrated intrusion volume. Volume is the
# quantity that actually matters: it is what distinguishes a sock
# from a noise speckle, and it is what grasp planning cares about.
DEFAULT_MIN_EXTENT_M = 0.025
DEFAULT_MIN_VOLUME_M3 = 1.0e-5


def load_points_xyz(csv_path: str) -> np.ndarray:
    """
    Load a scan CSV (see scan.cloud_io.save_xyz_csv) and return
    an (N, 3) float64 array of just the x, y, z columns.

    Capture timestamps and (in the extended schema) the ray/joint
    columns are dropped here: this is the plain geometric view,
    which is all detection needs. scan.cloud_io.load_scan_csv()
    exposes the extra columns.
    """

    points = load_xyz_csv(csv_path)

    if not points:
        raise ValueError(f"No points found in {csv_path!r}.")

    return np.array(
        [(p[0], p[1], p[2]) for p in points],
        dtype=np.float64,
    )


def load_baseline_scans(
    baseline: Union[str, Sequence[str]],
) -> List[np.ndarray]:
    """
    Load one or more empty-bucket baseline scans.

    `baseline` may be a single CSV path, a DIRECTORY of CSVs, or an
    explicit sequence of paths. The directory form is the intended
    one - the whole point of the new model is that it is built from
    8-10 empty scans, because that is what makes the per-cell sigma
    map (and therefore a calibrated threshold) possible at all.

    A single CSV still works so that existing call sites and older
    captures keep running, but it yields a sigma map that is almost
    entirely the pooled fallback. Treat its thresholds with
    suspicion.
    """

    if isinstance(baseline, str):

        if os.path.isdir(baseline):
            paths = sorted(glob.glob(os.path.join(baseline, "*.csv")))

            if not paths:
                raise ValueError(
                    f"No .csv files found in baseline directory "
                    f"{baseline!r}."
                )

        else:
            paths = [baseline]

    else:
        paths = list(baseline)

        if not paths:
            raise ValueError("Empty baseline path sequence.")

    return [load_points_xyz(path) for path in paths]


def load_baseline_xyz(
    baseline: Union[str, Sequence[str]],
) -> np.ndarray:
    """
    All baseline points pooled into one (N, 3) array.

    Anything that wants a single cloud of
    empty-bucket points rather than the per-scan split the model
    builder uses, and callers should not have to care whether
    `baseline` names one CSV or a directory of them.
    """

    return np.concatenate(load_baseline_scans(baseline), axis=0)


@dataclass
class IntrusionResult:
    """
    Per-candidate-point intrusion into the empty-bucket surface,
    and the resulting detection mask.

    intrusion_m:
        r_expected - r_measured, in metres. POSITIVE means the beam
        was intercepted inside the bucket wall, which only a real
        object can cause. Replaces the old `heights` field, which
        measured height above a local z maximum and was only
        meaningful on the bucket floor.

    mask:
        Points that cleared both the k-sigma and absolute-floor
        tests AND passed the geometric gate.

    in_bounds:
        Points that passed the geometric gate (inside the fitted
        axial extent, not behind the wall). Points outside it are
        never flagged, and are broken out separately so a scan that
        is mostly out of bounds - a sign the cone fit or the bucket
        pose is wrong - is visible rather than looking like a clean
        empty result.

    confident:
        False where the point landed in a thinly-sampled cell. Such
        points are still eligible for flagging; this only marks how
        much to trust the verdict.
    """

    intrusion_m: np.ndarray
    mask: np.ndarray
    in_bounds: np.ndarray
    confident: np.ndarray
    sigma_m: np.ndarray
    u: np.ndarray
    s: np.ndarray
    theta: np.ndarray
    r: np.ndarray
    k_sigma: float
    abs_floor_m: float


def compute_intrusion(
    candidate_xyz: np.ndarray,
    surface: BaselineSurface,
    k_sigma: float = DEFAULT_K_SIGMA,
    abs_floor_m: float = DEFAULT_ABS_FLOOR_M,
    behind_wall_margin_m: float = DEFAULT_BEHIND_WALL_MARGIN_M,
) -> IntrusionResult:
    """
    Measure how far each candidate point intrudes past the modelled
    empty-bucket surface, and flag the ones that clear the noise.

    The surface is the fitted MERIDIAN PROFILE, so this one test
    covers the flat closed end, the corner and the lateral wall
    alike - intrusion is perpendicular distance to that profile,
    signed by whether the point is inside the bucket volume.

    The geometric gate is deliberately limited to physically
    impossible places - beyond the mouth, or behind the wall. It
    does NOT suppress regions that are known to be noisy. Silencing
    awkward regions would hand back exactly the failure this
    rewrite set out to remove: the bucket mouth is both where the
    fit is weakest AND where a sock is most likely to be caught, so
    a position-based veto would blind the detector precisely where
    it matters. Thinly-sampled cells get a larger sigma (so they
    are harder to trip) and are reported as low-confidence, which
    is the honest version of the same caution.
    """

    candidate_xyz = np.asarray(candidate_xyz, dtype=np.float64)

    if candidate_xyz.ndim != 2 or candidate_xyz.shape[1] != 3:
        raise ValueError(
            f"candidate_xyz must be (N, 3); got {candidate_xyz.shape}."
        )

    n = candidate_xyz.shape[0]

    profile = surface.profile
    cone = surface.cone

    s, theta, r = to_cylindrical(candidate_xyz, cone)

    if n == 0:
        empty_f = np.zeros(0, dtype=np.float64)
        empty_b = np.zeros(0, dtype=bool)

        return IntrusionResult(
            intrusion_m=empty_f,
            mask=empty_b,
            in_bounds=empty_b.copy(),
            confident=empty_b.copy(),
            sigma_m=empty_f.copy(),
            u=empty_f.copy(),
            s=s,
            theta=theta,
            r=r,
            k_sigma=k_sigma,
            abs_floor_m=abs_floor_m,
        )

    u, raw_intrusion = profile.project(s, r)

    offset_mean, sigma, confident = surface.expected_offset(u, theta)

    intrusion = raw_intrusion - offset_mean

    if profile.has_cap:
        # The cap plane already bounds the far end: anything past it
        # reads as outside the volume and is caught below.
        within_extent = s <= cone.s_max
    else:
        # Wall-only profile, so nothing stops a point far beyond the
        # unscanned closed end from looking plausibly "inside" the
        # cone. Bound it explicitly.
        within_extent = (s >= cone.s_min) & (s <= cone.s_max)

    not_behind_wall = intrusion >= -behind_wall_margin_m

    in_bounds = within_extent & not_behind_wall

    threshold = np.maximum(k_sigma * sigma, abs_floor_m)

    mask = in_bounds & (intrusion >= threshold)

    return IntrusionResult(
        intrusion_m=intrusion,
        mask=mask,
        in_bounds=in_bounds,
        confident=confident,
        sigma_m=sigma,
        u=u,
        s=s,
        theta=theta,
        r=r,
        k_sigma=k_sigma,
        abs_floor_m=abs_floor_m,
    )


def cluster_points(
    points: np.ndarray,
    radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> List[np.ndarray]:
    """
    Group points into spatial clusters via a radius graph +
    connected components (points within radius_m of one another are
    linked, transitively, into the same cluster).

    Dimension-agnostic, but detection feeds it plain 3D base-frame
    points (see DEFAULT_CLUSTER_RADIUS_M for why not the unwrapped
    surface any more).

    Returns a list of INDEX arrays into points (not coordinate
    arrays), one per surviving cluster, sorted by descending size.
    Clusters smaller than min_cluster_size are dropped - this also
    naturally discards isolated noise points (which form their own
    singleton components), with no separate outlier-handling path
    needed.
    """

    n = points.shape[0]

    if n < min_cluster_size:
        return []

    tree = cKDTree(points)

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


def cluster_volume_m3(
    u: np.ndarray,
    theta: np.ndarray,
    intrusion_m: np.ndarray,
    surface: BaselineSurface,
) -> float:
    """
    Integrate a cluster's intrusion over the bucket surface to get
    the volume of material standing proud of the empty bucket.

    Deliberately computed per GRID CELL, not per point: each
    occupied cell contributes (its peak intrusion) x (its area). A
    naive sum over points would scale with how densely the scan
    happened to sample that patch, which varies several-fold across
    the helical path - so the same sock would score very differently
    depending on where it sat.

    HOW DENSITY-INDEPENDENT THIS ACTUALLY IS, measured: quadrupling
    the point count (3000 -> 12000 over the whole bucket) moves a
    fixed planted item's volume by about 25%, against the ~300% a
    raw point count would move. Above roughly 6000 points it is
    stable to a few percent. So this is much better than a count
    but not exact - it under-reads a thinly sampled item, because
    cells its footprint covers but no point landed in contribute
    nothing.

    The interior-hole pass below recovers part of that: a cell with
    no points of its own but well surrounded by cells that have
    some is inside the item, and is filled from its neighbours.
    Only WELL-surrounded cells (4 of the 8 neighbours occupied)
    qualify, so this fills holes without inflating the item's
    boundary outward.
    """

    if u.size == 0:
        return 0.0

    i, j = surface.cell_indices(u, theta)

    n_arc = surface.n_arc
    n_angular = surface.n_angular

    peak = np.zeros((n_arc, n_angular), dtype=np.float64)
    np.maximum.at(peak, (i, j), intrusion_m)

    occupied = np.zeros((n_arc, n_angular), dtype=bool)
    occupied[i, j] = True

    neighbour_sum = np.zeros_like(peak)
    neighbour_count = np.zeros_like(peak)

    values = np.where(occupied, peak, 0.0)

    for d_theta in (-1, 0, 1):

        # theta wraps around the bucket; u does not.
        rolled_values = np.roll(values, d_theta, axis=1)
        rolled_occupied = np.roll(occupied, d_theta, axis=1)

        for d_u in (-1, 0, 1):

            if d_u == 0:
                shifted_values = rolled_values
                shifted_occupied = rolled_occupied

            else:
                shifted_values = np.zeros_like(rolled_values)
                shifted_occupied = np.zeros_like(rolled_occupied)

                if d_u == -1:
                    shifted_values[:-1] = rolled_values[1:]
                    shifted_occupied[:-1] = rolled_occupied[1:]
                else:
                    shifted_values[1:] = rolled_values[:-1]
                    shifted_occupied[1:] = rolled_occupied[:-1]

            neighbour_sum += shifted_values
            neighbour_count += shifted_occupied

    interior = (~occupied) & (neighbour_count >= 4)

    peak[interior] = neighbour_sum[interior] / neighbour_count[interior]
    occupied = occupied | interior

    cell_area = surface.cell_area()

    return float(
        (peak * np.broadcast_to(cell_area, peak.shape))[occupied].sum()
    )


@dataclass
class ClusterSummary:
    """
    Summary of one detected laundry cluster, in base_frame metres.

    The first eight fields are the original contract and are
    unchanged, because grasp.plan.compute_grasp_target() reads
    centroid and the grasp/pipeline stages print the rest.
    mean_deviation_m is retained as the mean INTRUSION, which is the
    direct analogue of what it used to mean.
    """

    points: np.ndarray
    centroid: np.ndarray
    size: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    extent: np.ndarray
    highest_point: np.ndarray
    mean_deviation_m: float

    # Added by the model-based detector.
    volume_m3: float = float("nan")
    max_intrusion_m: float = float("nan")
    surface_extent_m: float = float("nan")
    confident: bool = True


def summarize_clusters(
    cluster_indices: List[np.ndarray],
    points_xyz: np.ndarray,
    intrusion_m: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
    surface: Optional[BaselineSurface] = None,
    confident: Optional[np.ndarray] = None,
) -> List[ClusterSummary]:
    """
    Build a ClusterSummary per cluster.

    cluster_indices:
        As returned by cluster_points() - index arrays into
        points_xyz.

    intrusion_m:
        Per-point intrusion past the modelled surface, aligned
        index-for-index with points_xyz, used for
        mean_deviation_m/max_intrusion_m. If omitted, both are NaN.

    u, theta, surface:
        Supply all three to also get volume_m3, the physical gate
        detect_laundry() filters on. Without them it stays NaN and
        only the extent and point-count filters apply.

    confident:
        Per-point confidence from IntrusionResult. A cluster is
        reported confident only if a majority of its points are -
        one stray low-confidence point should not discredit an
        otherwise solid detection, but a cluster living mostly in
        thinly-sampled cells genuinely is a weaker claim.

    Order of the input cluster_indices is preserved (cluster_points
    already sorts largest-first).
    """

    summaries = []

    have_surface = u is not None and theta is not None and surface is not None

    for indices in cluster_indices:

        member_points = points_xyz[indices]

        bbox_min = member_points.min(axis=0)
        bbox_max = member_points.max(axis=0)

        highest_point = member_points[
            np.argmax(member_points[:, 2])
        ]

        if intrusion_m is not None:
            member_intrusion = intrusion_m[indices]
            mean_deviation_m = float(member_intrusion.mean())
            max_intrusion_m = float(member_intrusion.max())
        else:
            member_intrusion = None
            mean_deviation_m = float("nan")
            max_intrusion_m = float("nan")

        # Straight 3D bounding-box diagonal. An earlier version
        # measured this on the unwrapped surface, which reported the
        # entire circumference for anything straddling the theta
        # seam - and theta = 0 is the top of the bucket, so that was
        # every ceiling item. In 3D there is no seam to straddle.
        surface_extent_m = float(np.linalg.norm(bbox_max - bbox_min))

        volume_m3 = float("nan")

        if have_surface and member_intrusion is not None:
            volume_m3 = cluster_volume_m3(
                u[indices],
                theta[indices],
                member_intrusion,
                surface,
            )

        if confident is not None:
            is_confident = bool(confident[indices].mean() >= 0.5)
        else:
            is_confident = True

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
                volume_m3=volume_m3,
                max_intrusion_m=max_intrusion_m,
                surface_extent_m=surface_extent_m,
                confident=is_confident,
            )
        )

    return summaries


def detect_on_points(
    candidate_xyz: np.ndarray,
    surface: BaselineSurface,
    k_sigma: float = DEFAULT_K_SIGMA,
    abs_floor_m: float = DEFAULT_ABS_FLOOR_M,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_extent_m: float = DEFAULT_MIN_EXTENT_M,
    min_volume_m3: float = DEFAULT_MIN_VOLUME_M3,
):
    """
    Run the detector over an already-loaded point array.

    Returns (clusters, intrusion_result): clusters sorted by
    descending volume, and the per-point IntrusionResult they came
    from (evaluation needs both).

    This is the one implementation of the detector; detect_laundry()
    wraps it for CSV input. Evaluation calls it directly because it
    re-runs the same points against many thresholds, and re-reading
    the file each time would dominate the runtime.
    """
    result = compute_intrusion(
        candidate_xyz,
        surface,
        k_sigma=k_sigma,
        abs_floor_m=abs_floor_m,
    )

    if not result.mask.any():
        return [], result

    flagged_xyz = candidate_xyz[result.mask]

    cluster_indices = cluster_points(
        flagged_xyz,
        radius_m=cluster_radius_m,
        min_cluster_size=min_cluster_size,
    )

    summaries = summarize_clusters(
        cluster_indices,
        flagged_xyz,
        result.intrusion_m[result.mask],
        u=result.u[result.mask],
        theta=result.theta[result.mask],
        surface=surface,
        confident=result.confident[result.mask],
    )

    kept = [
        summary
        for summary in summaries
        if summary.surface_extent_m >= min_extent_m
        and summary.volume_m3 >= min_volume_m3
    ]

    kept.sort(key=lambda summary: summary.volume_m3, reverse=True)

    return kept, result


def detect_laundry(
    baseline_csv: Union[str, Sequence[str]],
    candidate_csv: str,
    k_sigma: float = DEFAULT_K_SIGMA,
    abs_floor_m: float = DEFAULT_ABS_FLOOR_M,
    cluster_radius_m: float = DEFAULT_CLUSTER_RADIUS_M,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    min_extent_m: float = DEFAULT_MIN_EXTENT_M,
    min_volume_m3: float = DEFAULT_MIN_VOLUME_M3,
    surface: Optional[BaselineSurface] = None,
) -> List[ClusterSummary]:
    """
    End-to-end: build (or accept) the empty-bucket model, measure
    how far the candidate scan intrudes past it, cluster the
    intruding points, and return per-cluster summaries.

    baseline_csv:
        A CSV path, a DIRECTORY of baseline CSVs, or a sequence of
        paths. See load_baseline_scans - a directory of 8-10 empty
        scans is the intended input.

    surface:
        A pre-built BaselineSurface, which skips loading and fitting
        entirely. Pass this when detecting repeatedly against one
        model (the fit is the expensive part), and in leave-one-out
        validation, where the held-out scan must not have
        contributed to the model judging it.

    Results are sorted by descending VOLUME, not point count, so the
    first entry is the physically largest find - which is the one
    the grasp stage should try first.

    This is the single function the CLI and any test/analysis script
    should call; the building blocks above are exposed individually
    mainly so they can be unit-tested in isolation.
    """

    if surface is None:
        surface = build_baseline_surface(
            load_baseline_scans(baseline_csv)
        )

    candidate_xyz = load_points_xyz(candidate_csv)

    clusters, _result = detect_on_points(
        candidate_xyz,
        surface,
        k_sigma=k_sigma,
        abs_floor_m=abs_floor_m,
        cluster_radius_m=cluster_radius_m,
        min_cluster_size=min_cluster_size,
        min_extent_m=min_extent_m,
        min_volume_m3=min_volume_m3,
    )

    return clusters
