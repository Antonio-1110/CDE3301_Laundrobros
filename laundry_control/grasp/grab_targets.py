#!/usr/bin/env python3

"""
The sweep's grabs as placed by hand: scan_plans/grab_targets.yaml.

Each grab is given relative to the bucket, so moving the bucket
(config.OBSTACLES) moves the grabs with it:

    depth_m          from the CLOSED end along the bucket axis
    floor_angle_deg  around the axis from the floor's lowest line,
                     positive toward link_base +x
    height_m         contact point in from the wall (above the floor)
    tilt_deg         tool axis from straight-at-the-wall, leaning
                     toward the closed end
    approach_m       how far the gripper backs off along its own axis
                     above the grab, before the straight descent

in visiting order. `laundry plan edit-grabs` (grasp/grab_editor.py)
edits them by dragging in RViz; `laundry plan bake retrieve` solves
each one exactly as placed (grasp/retrieve_grid.py) - no search over
heights or tilts - and reports any it cannot reach.

Everything here is plain geometry and file I/O, no ROS.
"""

import os

import numpy as np
import yaml

from .. import config
from ..perception.bucket_model import _axis_basis, seed_cone

TARGETS_VERSION = 1

FIELDS = ('depth_m', 'floor_angle_deg', 'height_m', 'tilt_deg', 'approach_m')

# What a dragged grab is clamped to. Height runs to most of the
# drum's ~36-45 cm diameter; negative tilt leans toward the mouth.
LIMITS = {
    'depth_m': (0.02, 0.50),
    'floor_angle_deg': (-90.0, 90.0),
    'height_m': (0.01, 0.30),
    'tilt_deg': (-30.0, 75.0),
    'approach_m': (0.0, 0.15),
}

# Stored to the millimetre and the half degree.
_ROUNDING = {
    'depth_m': 3, 'height_m': 3, 'approach_m': 3,
    'floor_angle_deg': 1, 'tilt_deg': 1,
}

# The starting layout for a ~2/3-full drum (pile top ~25 cm above the
# floor): the old grid's 12 spots, 8 cm up rather than 1.5-4 cm, each
# at the least tilt that reached it with a 5 cm approach (fake
# controller, 2026-10-06). Higher costs tilt - the 15 cm gripper and
# its approach meet the drum's ceiling: vertical grabs reach ~5-8 cm,
# 11 cm needs 30-45 deg, 14 cm 45-60 deg. A first layout to drag
# from.
_DEFAULT_TILT_BY_DEPTH = {0.40: 15.0, 0.30: 30.0, 0.20: 30.0, 0.10: 45.0}


def path():
    """Return <repo>/scan_plans/grab_targets.yaml."""
    return config.grab_targets_path()


def clean(target):
    """Return target with every field clamped to LIMITS and rounded."""
    result = {}

    for field in FIELDS:
        low, high = LIMITS[field]
        value = min(max(float(target[field]), low), high)

        if field.endswith('_deg'):
            value = round(value * 2.0) / 2.0
        else:
            value = round(value, _ROUNDING[field])

        result[field] = value

    return result


def default_targets():
    """Return the starting layout (see _DEFAULT_TILT_BY_DEPTH)."""
    return [
        clean({
            'depth_m': depth,
            'floor_angle_deg': angle,
            'height_m': 0.08,
            'tilt_deg': tilt,
            'approach_m': 0.05,
        })
        for depth, tilt in _DEFAULT_TILT_BY_DEPTH.items()
        for angle in (0.0, -20.0, 20.0)
    ]


def load(file_path=None):
    """Return the targets in visiting order; [] if the file is absent."""
    file_path = file_path or path()

    if not os.path.isfile(file_path):
        return []

    with open(file_path) as handle:
        document = yaml.safe_load(handle) or {}

    if document.get('version') != TARGETS_VERSION:
        raise ValueError(f'{file_path!r} is an unknown grab-targets version.')

    return [clean(target) for target in document.get('grabs', [])]


def save(targets, file_path=None):
    """Write the targets (visiting order) as YAML."""
    file_path = file_path or path()

    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    document = {
        'version': TARGETS_VERSION,
        'grabs': [clean(target) for target in targets],
    }

    with open(file_path, 'w') as handle:
        handle.write(
            "# The sweep's grabs, in visiting order "
            '(grasp/grab_targets.py).\n'
            '# Edit by dragging: `laundry plan edit-grabs`; then '
            '`./rebake.sh retrieve`.\n'
        )
        yaml.safe_dump(document, handle, sort_keys=False, width=100)


def _frame(floor_angle_deg, cone):
    """Return (axis, outward) at a floor angle, as retrieve_grid uses them."""
    axis = cone.axis_dir
    up, side = _axis_basis(axis)

    if side[0] < 0.0:
        side = -side

    phi = np.radians(floor_angle_deg)

    return axis, -np.cos(phi) * up + np.sin(phi) * side


def geometry(target, cone=None):
    """Return (contact point, tool +Z, claw +X) in link_base."""
    from .retrieve_grid import grab_geometry

    return grab_geometry(
        target['depth_m'], target['floor_angle_deg'],
        target['height_m'], target['tilt_deg'], cone,
    )


def stack(target, cone=None):
    """
    Return the points along the gripper, in link_base.

    contact:  where the claw closes
    flange:   the wrist at the grab, GRIPPER_OFFSET_Z back along the tool
    approach: the wrist at the approach pose, approach_m further back
    """
    contact, tool_z, _ = geometry(target, cone)

    flange = contact - config.GRIPPER_OFFSET_Z * tool_z

    return {
        'contact': contact,
        'flange': flange,
        'approach': flange - target['approach_m'] * tool_z,
        'tool_z': tool_z,
    }


def wall_margin(point, cone=None):
    """Return how far a point is inside the wall (m; < 0 = outside)."""
    from ..perception.bucket_model import to_cylindrical

    cone = cone or seed_cone()
    s, _, r = to_cylindrical(np.asarray(point, dtype=float)[None], cone)

    return float(cone.radius_at(s[0]) - r[0])


def from_pose(contact, tool_z, approach_m, cone=None):
    """
    Return the target whose contact point and tool axis these are.

    The inverse of geometry(); a tool axis with a sideways component
    (which geometry() never produces) only counts by its part in the
    plane of the bucket axis and the radius.
    """
    from .retrieve_grid import floor_position

    cone = cone or seed_cone()
    depth, angle, height = floor_position(contact, cone)
    axis, outward = _frame(angle, cone)

    tool_z = np.asarray(tool_z, dtype=float)
    tilt = np.degrees(np.arctan2(-tool_z @ axis, tool_z @ outward))

    return clean({
        'depth_m': depth,
        'floor_angle_deg': angle,
        'height_m': height,
        'tilt_deg': tilt,
        'approach_m': approach_m,
    })


def approach_from_handle(target, handle_point, cone=None):
    """Return approach_m for an approach handle dragged to handle_point."""
    contact, tool_z, _ = geometry(target, cone)
    back = float((contact - np.asarray(handle_point, dtype=float)) @ tool_z)

    return back - config.GRIPPER_OFFSET_Z


def _fill_rows(fraction, cone, depth_step_m):
    """
    Return, per depth, where the laundry's level top meets the walls.

    A list of (left, right) points, None where the level misses that
    cross-section. The level fills fraction of the drum's height,
    measured at mid-depth from the floor's lowest line.
    """
    axis = cone.axis_dir
    up, side = _axis_basis(axis)

    mid = 0.5 * cone.s_max
    mid_centre = cone.axis_point + mid * axis
    level_z = (
        mid_centre[2] + (2.0 * fraction - 1.0) * cone.radius_at(mid) * up[2]
    )

    rows = []

    for depth in np.arange(0.0, cone.s_max + 1e-9, depth_step_m):
        centre = cone.axis_point + depth * axis
        radius = cone.radius_at(depth)

        # The level plane cuts this cross-section's circle at height
        # h along `up`; up is within a few degrees of world z here.
        h = (level_z - centre[2]) / up[2]

        if abs(h) >= radius:
            rows.append(None)
            continue

        half = np.sqrt(radius**2 - h**2)
        middle = centre + h * up
        rows.append((middle - half * side, middle + half * side))

    return rows


def fill_surface(fraction, cone=None, depth_step_m=0.02):
    """
    Return the laundry's top surface as triangles ((N, 3), 3 rows each).

    What the drum looks like when the sweep starts, for display only;
    see _fill_rows.
    """
    rows = _fill_rows(fraction, cone or seed_cone(), depth_step_m)
    triangles = []

    for a, b in zip(rows, rows[1:]):
        if a is None or b is None:
            continue

        triangles += [a[0], a[1], b[0], a[1], b[1], b[0]]

    return np.array(triangles).reshape(-1, 3)


def fill_outline(fraction, cone=None, depth_step_m=0.02, cross_every=5):
    """
    Return the laundry's top as line segments ((N, 3), 2 rows each).

    Its edges along both walls and a line across every cross_every
    depth steps: the fill_surface() outline, which, unlike a solid
    surface, does not catch the clicks meant for the grabs below it.
    """
    rows = _fill_rows(fraction, cone or seed_cone(), depth_step_m)
    segments = []

    for index, (a, b) in enumerate(zip(rows, rows[1:])):
        if a is None or b is None:
            continue

        segments += [a[0], b[0], a[1], b[1]]

        if index % cross_every == 0:
            segments += [a[0], a[1]]

    return np.array(segments).reshape(-1, 3)


def bucket_wireframe(cone=None, ring_every_m=0.1, line_every_deg=30.0):
    """
    Return the bucket as line segments ((N, 3), 2 rows each).

    Rings round the axis every ring_every_m from the closed end to the
    mouth, lines along the wall every line_every_deg, and a cross on
    the closed end: a see-through bucket that does not catch clicks
    the way the solid planning-scene mesh does.
    """
    cone = cone or seed_cone()
    axis = cone.axis_dir
    up, side = _axis_basis(axis)

    def at(depth, angle):
        return (
            cone.axis_point + depth * axis
            + cone.radius_at(depth) * (np.cos(angle) * up + np.sin(angle) * side)
        )

    segments = []
    ring = np.radians(np.arange(0.0, 360.0 + 1e-9, 5.0))
    depths = sorted(set(
        list(np.arange(0.0, cone.s_max, ring_every_m)) + [cone.s_max]
    ))

    for depth in depths:
        for a, b in zip(ring, ring[1:]):
            segments += [at(depth, a), at(depth, b)]

    for angle in np.radians(np.arange(0.0, 360.0, line_every_deg)):
        segments += [at(0.0, angle), at(cone.s_max, angle)]

    for angle in (0.0, np.pi / 2):
        segments += [at(0.0, angle), at(0.0, angle + np.pi)]

    return np.array(segments).reshape(-1, 3)


def describe(target):
    """One line for logs and the RViz label."""
    return (
        f'{target["depth_m"] * 100:.0f} cm deep, '
        f'{target["floor_angle_deg"]:+.0f} deg, '
        f'{target["height_m"] * 100:.1f} cm up, '
        f'tilt {target["tilt_deg"]:.0f} deg, '
        f'approach {target["approach_m"] * 100:.0f} cm'
    )
