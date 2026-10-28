#!/usr/bin/env python3

"""
The sensorless sweep's grabs, in layers from the floor up (config.PILE).

WHY
---
`laundry preplanned` (and the start of `laundry clear`) runs on a drum
with laundry in it, so there is a pile: grabbing at the drum floor
would drive the claw through all of it. The grabs go into the pile from
its top down, and where its top is depends on how full the drum is.

WHAT
----
generate() turns config.PILE into the grab list of scan_plans/
grab_targets.yaml by solving on the MoveIt scene, once for every
layer:

  - Layer N's grabs aim at layer_heights_m[N - 1] above the floor, one
    per spot (depth x sideways offset, the claw down). A spot takes the
    HIGHEST height down to layer_window_m under that (never under
    floor_clearance_m) that some tilt reaches, at the widest sideways
    offset that fits (no nearer the middle than MIN_SIDE_M); else it
    is left out of that layer.
  - The list is in visiting order: the top layer first, mouth first
    within a layer. Each grab carries its layer number.

At run time `--fill N` picks the layers: select_layers() keeps layer N
and every layer under it, so the sweep starts in the pile, not above
it, and follows it down as it empties - one bake for any fill.

A grab's height_m (grab_targets.py) is its distance in from the wall
along the radius at its floor angle; the pile's level is vertical, so
vertical_height() and radial_height() convert. Every grab is checked
as grab_targets.clean() rounds it, with retrieve_grid's checks, so
the bake of the same list does not fail on rounding.

The geometry is plain numpy; the solving needs an arm with
compute_ik and first_invalid_state (arm/controller).
"""

import numpy as np

from . import grab_targets, retrieve_grid
from .. import config
from ..bucket import seed_cone

# Heights are tried this far apart.
HEIGHT_STEP_M = 0.01

# A grab whose wrist sits within this of the wall's radius, or beyond
# it, cannot fit: skipped without asking IK (the arm links are padded
# by 3 cm besides).
MIN_FLANGE_MARGIN_M = 0.0

# A side spot that does not fit this far out tries this much nearer the
# middle, and so on - but no nearer than MIN_SIDE_M: about half the
# claw's open width (~13 cm), so it never repeats the middle's grab.
LATERAL_STEP_M = 0.01
MIN_SIDE_M = 0.06


def vertical_height(depth_m, floor_angle_deg, height_m, cone=None):
    """
    Return a grab's height above the floor's lowest line (m).

    height_m is the distance in from the wall along the radius at the
    floor angle; at 0 degrees the two are the same.
    """
    cone = cone or seed_cone()
    radius = cone.radius_at(depth_m)

    return float(
        radius - (radius - height_m) * np.cos(np.radians(floor_angle_deg))
    )


def radial_height(depth_m, floor_angle_deg, vertical_m, cone=None):
    """Return the height_m whose vertical_height() is vertical_m."""
    cone = cone or seed_cone()
    radius = cone.radius_at(depth_m)

    return float(
        radius - (radius - vertical_m) / np.cos(np.radians(floor_angle_deg))
    )


def fill_fraction(top_height_m=None, cone=None):
    """
    Return the pile's top as a fraction of the drum's height.

    As the RViz editor draws it (grab_targets.fill_surface), measured
    at mid-depth.
    """
    cone = cone or seed_cone()
    # Default: as full as the top layer reaches into, ~5 cm over it.
    top = (
        max(config.PILE['layer_heights_m']) + 0.05
        if top_height_m is None else top_height_m
    )

    return float(top / (2.0 * cone.radius_at(0.5 * cone.s_max)))


def pile_signature(pile=None):
    """Return config.PILE as plain data: what a generated list is stamped with."""
    pile = pile or config.PILE

    return {
        key: (
            [float(v) for v in value]
            if isinstance(value, (list, tuple)) else float(value)
        )
        for key, value in pile.items()
    }


def pile_changed(generated, pile=None):
    """Return True if a list generated from `generated` is not for PILE now."""
    if generated is None:
        return False

    return generated.get('pile') != pile_signature(pile)


def _spots(pile):
    return [
        (depth, lateral)
        for depth in pile['depths_m']
        for lateral in pile['lateral_m']
    ]


def _inward(lateral_m):
    """Yield lateral_m, then each step closer to the middle, to MIN_SIDE_M."""
    if abs(lateral_m) < 1e-9:
        yield 0.0
        return

    steps = int(round(abs(lateral_m) / LATERAL_STEP_M))
    lowest = int(round(MIN_SIDE_M / LATERAL_STEP_M))

    for k in range(steps, min(lowest, steps) - 1, -1):
        yield float(np.copysign(k * LATERAL_STEP_M, lateral_m))


def _candidates(depth_m, lateral_m, aim_m, lowest_m, pile, cone):
    """Yield (vertical height, target) from the aim down, tilts smallest first."""
    low = grab_targets.LIMITS['height_m']

    vertical = aim_m

    while vertical >= lowest_m - 1e-9:
        if low[0] <= vertical <= low[1]:
            for tilt in pile['tilts_deg']:
                target = grab_targets.clean({
                    'depth_m': depth_m,
                    'floor_angle_deg': retrieve_grid.angle_about_axis(
                        depth_m, lateral_m, vertical, cone
                    ),
                    'lateral_m': lateral_m,
                    'height_m': vertical,
                    'tilt_deg': tilt,
                    'approach_m': pile['approach_m'],
                })

                yield vertical, target

        vertical -= HEIGHT_STEP_M


def target_vertical(target, cone=None):
    """Return a target's height above the floor's lowest line (m)."""
    if 'lateral_m' in target:
        return float(target['height_m'])

    return vertical_height(
        target['depth_m'], target['floor_angle_deg'], target['height_m'], cone
    )


def find_grab(
    arm, depth_m, lateral_m, aim_m, lowest_m, pile=None, seeds=None,
    cone=None,
):
    """
    Return the widest, highest-reachable grab at one spot as (target, reason).

    A down grab (grab_targets) lateral_m sideways of the middle: the
    first sideways offset, from lateral_m in LATERAL_STEP_M steps
    toward the middle (no nearer than MIN_SIDE_M), at which any height
    from aim_m down to lowest_m passes retrieve_grid's checks, at its
    highest such height and least tilt: (target, None); else (None,
    why the last try failed). The drum's ceiling comes down and in toward the closed
    end, so the deeper spots fit less far out (fake controller,
    2026-10-27: +/-12 cm at 32 cm deep, +/-9 at 17). Needs the gripper
    padded (retrieve_grid.gripper_padding).
    """
    pile = pile or config.PILE
    cone = cone or seed_cone()
    seeds = seeds or retrieve_grid._seeds()
    inter = np.array(config.get_named_pose('inter'))

    reason = "nothing in the layer's heights"

    for lateral in _inward(lateral_m):
        for _vertical, target in _candidates(
            depth_m, lateral, aim_m, lowest_m, pile, cone
        ):
            contact, tool_z, claw_x = grab_targets.geometry(target, cone)
            flange = contact - config.GRIPPER_OFFSET_Z * tool_z

            if grab_targets.wall_margin(flange, cone) < MIN_FLANGE_MARGIN_M:
                reason = 'the wrist does not fit under the drum wall'
                continue

            found, why = retrieve_grid._try_pose(
                arm, contact, tool_z, claw_x, target['approach_m'], seeds,
                inter,
            )

            if found is not None:
                return target, None

            reason = why

    return None, reason


def generate(arm, pile=None, log=print):
    """
    Return (targets, misses, layers) for every layer, in visiting order.

    targets: the grab_targets.yaml list, each with its 'layer' (1 =
    lowest), top layer first; misses: (layer, depth_m, lateral_m,
    reason) for spots a layer found nothing for; layers: per layer,
    top first, the heights its grabs reached (m).
    """
    pile = pile or config.PILE
    cone = seed_cone()
    seeds = retrieve_grid._seeds()

    heights = pile['layer_heights_m']
    targets, misses, layers = [], [], []

    with retrieve_grid.gripper_padding(arm):
        for number in range(len(heights), 0, -1):
            aim = heights[number - 1]
            lowest = max(
                aim - pile['layer_window_m'], pile['floor_clearance_m']
            )
            found = []

            for depth, lateral in _spots(pile):
                target, reason = find_grab(
                    arm, depth, lateral, aim, lowest, pile, seeds, cone
                )

                if target is None:
                    misses.append((number, depth, lateral, reason))
                    log(
                        f'  layer {number}: {depth * 100:.0f} cm deep, '
                        f'{lateral * 100:+.0f} cm sideways: NOT REACHABLE - '
                        f'{reason}'
                    )
                    continue

                target = dict(target, layer=number)
                found.append(target)

                log(f'  layer {number}: {grab_targets.describe(target)}')

            targets += found
            layers.append([target_vertical(t, cone) for t in found])

    return targets, misses, layers


def select_layers(grabs, fill):
    """
    Return the grabs `--fill fill` sweeps: layer fill and every one under.

    In the list's own (visiting) order. A list without layers (placed
    by hand, `laundry plan edit-grabs`) is returned whole; fill None
    keeps everything.
    """
    if fill is None or not any('layer' in grab for grab in grabs):
        return list(grabs)

    return [grab for grab in grabs if grab.get('layer', 0) <= fill]
