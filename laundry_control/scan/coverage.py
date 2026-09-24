#!/usr/bin/env python3

"""
How much of the bucket a scan path actually sees, measured or simulated.

The scan's coverage is the hard ceiling on detection: an item no beam
reaches cannot be found by any threshold (see perception/
synthetic_eval.py, which reports those misses separately). This
module puts numbers on it in two ways:

  measured_coverage()  - from real scans: which parts of the bucket
                         surface fall inside at least one beam's cone
                         footprint.

  simulate_path()      - from a scan PATH: the beams a given
                         depth/step/sweep/phase would produce, cast
                         against the fitted bucket. Validated against
                         the measured coverage of the current path
                         (`laundry evaluate --coverage` prints both),
                         so alternative paths can be compared before
                         anyone spends arm time on them.

Coverage is area-based, not point-based: a surface point counts as
seen if it lies within some beam's footprint on the wall (radius
range * tan(half FOV), ~2.2cm at 10cm range), because that is what the
sensor integrates over. The detector's per-cell residual grid is a
stricter, point-based view of the same thing (see
bucket_model.occupancy_summary).
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..config import (
    TOF_FIELD_OF_VIEW_RAD,
    TOF_MAX_RANGE_M,
    TOF_MIN_RANGE_M,
    TOF_SENSOR_OFFSET_X,
    TOF_SENSOR_OFFSET_Z,
)
from ..perception.bucket_model import _axis_basis, to_cylindrical
from ..perception.synthetic import (
    BOTTOM_FLANGE_POSITION,
    BOTTOM_TOOL_Z,
    INTER_FLANGE_POSITION,
    INTER_TOOL_Z,
    Rays,
)

# Boresight (link7 +X) at the recorded poses' own J7 angle, from the
# same forward kinematics as the flange positions.
INTER_BORESIGHT = np.array([-0.011, 0.009, -1.000])
BOTTOM_BORESIGHT = np.array([-0.019, -0.673, -0.739])

# Measured on the real baselines: 14 inward strokes in ~15.5s at the
# default velocity scaling, sampled at 20Hz.
DEFAULT_SAMPLES_PER_STROKE = 22
# Per move of the detour (entry tilt, stationary sweep, exit tilt):
# roughly 300 of a real scan's ~1000 readings between the three.
DEFAULT_SAMPLES_PER_BOTTOM_SWEEP = 100

# Regions, by theta from the top of the bucket (0) to the floor (180),
# symmetric about the vertical - the same bands synthetic_eval uses.
REGION_BANDS_DEG = {
    'floor': (150.0, 180.0),
    'lower_wall': (90.0, 150.0),
    'upper_wall': (30.0, 90.0),
    'ceiling': (0.0, 30.0),
}

SURFACE_PITCH_M = 0.01


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _rotate(vector, axis, angle):
    """Rodrigues rotation of vector(s) about a unit axis by angle(s)."""
    axis = _unit(axis)
    vector = np.atleast_2d(vector)
    angle = np.atleast_1d(angle)[:, None]

    return (
        vector * np.cos(angle)
        + np.cross(axis, vector) * np.sin(angle)
        + axis * (vector @ axis)[:, None] * (1.0 - np.cos(angle))
    )


@dataclass
class ScanPath:
    """A scan.pattern-style path, as geometry (no timing beyond sampling)."""

    depth_m: float = 0.42
    step_m: float = 0.03
    sweep_deg: float = 150.0
    # Offset of the outward strokes' axial positions, as a fraction of
    # step: 0.5 interleaves them halfway between the inward ones.
    outward_phase: float = 0.0
    # Centre of the sweep, relative to INTER's J7 (0 = floor-facing).
    sweep_centre_deg: float = 0.0
    bottom_detour: bool = True
    samples_per_stroke: int = DEFAULT_SAMPLES_PER_STROKE


def _stroke_beams(path):
    """Return (origins, directions) for every sample of every stroke."""
    tool_z = _unit(INTER_TOOL_Z)
    half = np.deg2rad(path.sweep_deg) / 2.0
    centre = np.deg2rad(path.sweep_centre_deg)

    strokes = int(round(path.depth_m / path.step_m))
    progress = (np.arange(path.samples_per_stroke) + 0.5) / path.samples_per_stroke

    origins = []
    directions = []

    for leg in ('in', 'out'):
        for index in range(strokes):
            if leg == 'in':
                start = index * path.step_m
                end = start + path.step_m
            else:
                start = path.depth_m - index * path.step_m
                end = start - path.step_m
                start -= path.outward_phase * path.step_m
                end -= path.outward_phase * path.step_m

            sign = 1.0 if index % 2 == 0 else -1.0
            if leg == 'out':
                sign = -sign

            depth = np.clip(start + (end - start) * progress, 0.0, path.depth_m)
            angle = centre + sign * (-half + 2.0 * half * progress)

            direction = _rotate(INTER_BORESIGHT, tool_z, angle)

            flange = INTER_FLANGE_POSITION + np.outer(depth, tool_z)
            origin = (
                flange
                + TOF_SENSOR_OFFSET_Z * tool_z
                + TOF_SENSOR_OFFSET_X * direction
            )

            origins.append(origin)
            directions.append(direction)

    if path.bottom_detour:
        samples = max(
            DEFAULT_SAMPLES_PER_BOTTOM_SWEEP,
            int(round(DEFAULT_SAMPLES_PER_BOTTOM_SWEEP * path.sweep_deg / 150.0)),
        )
        origins_d, directions_d = _detour_beams(path, half, centre, samples)
        origins.append(origins_d)
        directions.append(directions_d)

    return np.concatenate(origins), np.concatenate(directions)


def _detour_beams(path, half, centre, samples=DEFAULT_SAMPLES_PER_BOTTOM_SWEEP):
    """
    Beams of the BOTTOM detour: tilt in while sweeping, sweep, tilt out.

    The entry and exit moves are joint-space, so the true wrist path
    between the deepest stroke pose and BOTTOM is not a straight
    line; it is approximated by interpolating the flange position,
    tool axis and J7-reference boresight linearly between the two.
    That is good enough for area coverage (which only needs roughly
    where the beams land); `laundry evaluate --coverage` prints the
    simulated current path next to the measured one to show how good.
    """
    deep_z = _unit(INTER_TOOL_Z)
    deep_flange = INTER_FLANGE_POSITION + path.depth_m * deep_z

    bottom_z = _unit(BOTTOM_TOOL_Z)

    def beams(blend, angle):
        blend = np.atleast_1d(blend)[:, None]
        tool_z = (1.0 - blend) * deep_z + blend * bottom_z
        tool_z /= np.linalg.norm(tool_z, axis=1)[:, None]

        reference = (1.0 - blend) * INTER_BORESIGHT + blend * BOTTOM_BORESIGHT
        reference -= np.einsum('ij,ij->i', reference, tool_z)[:, None] * tool_z
        reference /= np.linalg.norm(reference, axis=1)[:, None]

        direction = np.stack([
            _rotate(ref, axis, a)[0]
            for ref, axis, a in zip(reference, tool_z, angle)
        ])

        flange = (1.0 - blend) * deep_flange + blend * BOTTOM_FLANGE_POSITION

        origin = (
            flange
            + TOF_SENSOR_OFFSET_Z * tool_z
            + TOF_SENSOR_OFFSET_X * direction
        )

        return origin, direction

    ramp = np.linspace(0.0, 1.0, samples)
    sweep = np.linspace(-half, half, samples)

    entry = beams(ramp, centre - sweep)
    at_bottom = beams(np.ones(samples), centre + sweep)
    exit_ = beams(ramp[::-1], centre - sweep)

    return (
        np.concatenate([entry[0], at_bottom[0], exit_[0]]),
        np.concatenate([entry[1], at_bottom[1], exit_[1]]),
    )


# The J7 rate the current path runs at (150 deg per ~1.1s stroke).
# Wider sweeps are assumed to keep it - the strokes slow down rather
# than spin J7 faster - so samples and duration scale with the sweep.
CURRENT_DEG_PER_SAMPLE = 150.0 / DEFAULT_SAMPLES_PER_STROKE


def path_for_sweep(sweep_deg, **kwargs):
    """Return a ScanPath with stroke sampling scaled to keep J7's rate."""
    samples = max(
        DEFAULT_SAMPLES_PER_STROKE,
        int(round(sweep_deg / CURRENT_DEG_PER_SAMPLE)),
    )
    return ScanPath(sweep_deg=sweep_deg, samples_per_stroke=samples, **kwargs)


def estimated_duration_s(path, rate_hz=20.0):
    """Return a rough scan duration: 20Hz samples plus the detour."""
    strokes = 2 * int(round(path.depth_m / path.step_m))
    detour = 3 * DEFAULT_SAMPLES_PER_BOTTOM_SWEEP * path.sweep_deg / 150.0
    return (strokes * path.samples_per_stroke + detour) / rate_hz


CANDIDATE_PATHS = {
    'current (150deg, 3cm step)': ScanPath(),
    'outward strokes interleaved': ScanPath(outward_phase=0.5),
    'step 2cm': ScanPath(step_m=0.02),
    'sweep 210deg': path_for_sweep(210.0),
    'sweep 270deg': path_for_sweep(270.0),
    'sweep 360deg': path_for_sweep(360.0),
}


def cast_rays(origins, directions, profile, step_m=0.001):
    """
    March each beam until it leaves the bucket; return its range.

    NaN where the beam never meets the wall within the sensor's
    usable range - the sensor node drops those readings too.
    """
    distances = np.arange(TOF_MIN_RANGE_M, TOF_MAX_RANGE_M + step_m, step_m)

    ranges = np.full(origins.shape[0], np.nan)

    for start in range(0, origins.shape[0], 256):
        o = origins[start:start + 256]
        d = directions[start:start + 256]

        samples = o[:, None, :] + distances[None, :, None] * d[:, None, :]
        flat = samples.reshape(-1, 3)

        s, _theta, r = to_cylindrical(flat, profile.cone)
        _u, intrusion = profile.project(s, r)

        inside = (intrusion > 0.0) & (s <= profile.cone.s_max)
        inside = inside.reshape(samples.shape[:2])

        exited = ~inside
        first = np.argmax(exited, axis=1)
        hit = exited[np.arange(exited.shape[0]), first] & (first > 0)

        ranges[start:start + 256][hit] = distances[first[hit]]

    return ranges


def simulate_path(path, profile):
    """Return the Rays a ScanPath would record against this bucket."""
    origins, directions = _stroke_beams(path)

    ranges = cast_rays(origins, directions, profile)

    keep = np.isfinite(ranges)

    return Rays(
        origin=origins[keep],
        direction=directions[keep],
        range_m=ranges[keep],
        from_bottom=np.zeros(int(keep.sum()), dtype=bool),
    )


def surface_samples(profile, pitch_m=SURFACE_PITCH_M):
    """
    Return (points, theta_deg_from_top, on_cap) sampling the bucket surface.

    The lateral wall is sampled on an (s, theta) grid at roughly equal
    area per sample; the closed end (if modelled) on a square grid.
    """
    cone = profile.cone
    e1, e2 = _axis_basis(cone.axis_dir)

    s_values = np.arange(cone.s_min, cone.s_max, pitch_m)

    points = []
    thetas = []

    for s in s_values:
        radius = float(cone.radius_at(s))
        count = max(8, int(2.0 * np.pi * radius / pitch_m))
        theta = (np.arange(count) + 0.5) * 2.0 * np.pi / count

        radial = np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2
        points.append(cone.axis_point + s * cone.axis_dir + radius * radial)
        thetas.append(theta)

    wall = np.concatenate(points)
    theta = np.concatenate(thetas)

    # Fold theta to "degrees from the top", 0..180, both sides alike.
    from_top = np.degrees(np.minimum(theta, 2.0 * np.pi - theta))

    on_cap = np.zeros(wall.shape[0], dtype=bool)

    if profile.has_cap:
        s_cap = float(profile.vertices[0][0])
        r_cap = float(profile.vertices[1][1])
        grid = np.arange(-r_cap, r_cap + 1e-9, pitch_m)
        gu, gv = np.meshgrid(grid, grid, indexing='ij')
        disc = gu ** 2 + gv ** 2 <= r_cap ** 2
        cap = (
            cone.axis_point
            + s_cap * cone.axis_dir
            + np.outer(gu[disc], e1)
            + np.outer(gv[disc], e2)
        )

        wall = np.concatenate([wall, cap])
        from_top = np.concatenate([from_top, np.full(cap.shape[0], np.nan)])
        on_cap = np.concatenate([on_cap, np.ones(cap.shape[0], dtype=bool)])

    return wall, from_top, on_cap


def footprint_seen(surface_points, rays, half_angle_rad=TOF_FIELD_OF_VIEW_RAD / 2.0):
    """Return a mask of surface points inside at least one beam's footprint."""
    endpoints = rays.endpoint
    footprint = rays.range_m * np.tan(half_angle_rad)

    tree = cKDTree(endpoints)

    seen = np.zeros(surface_points.shape[0], dtype=bool)

    candidates = tree.query_ball_point(surface_points, float(footprint.max()))

    for index, beams in enumerate(candidates):
        if not beams:
            continue

        beams = np.asarray(beams)
        gap = np.linalg.norm(endpoints[beams] - surface_points[index], axis=1)

        seen[index] = bool(np.any(gap <= footprint[beams]))

    return seen


def coverage_by_region(profile, rays):
    """Return {region: fraction of that region's surface seen}."""
    points, from_top, on_cap = surface_samples(profile)

    seen = footprint_seen(points, rays)

    result = {}

    for region, (low, high) in REGION_BANDS_DEG.items():
        band = (~on_cap) & (from_top >= low) & (from_top <= high)
        result[region] = float(seen[band].mean()) if band.any() else float('nan')

    if on_cap.any():
        result['closed_end'] = float(seen[on_cap].mean())

    result['whole_bucket'] = float(seen.mean())

    return result


def measured_coverage(profile, scans):
    """Return coverage_by_region() for the pooled beams of real scans."""
    from ..perception.synthetic import reconstruct_rays

    rays = [reconstruct_rays(scan) for scan in scans]

    pooled = Rays(
        origin=np.concatenate([r.origin for r in rays]),
        direction=np.concatenate([r.direction for r in rays]),
        range_m=np.concatenate([r.range_m for r in rays]),
        from_bottom=np.concatenate([r.from_bottom for r in rays]),
    )

    return coverage_by_region(profile, pooled)
