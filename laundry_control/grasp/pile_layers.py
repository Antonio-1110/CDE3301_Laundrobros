#!/usr/bin/env python3

"""
The sensorless sweep's grabs, from how full the drum is (config.PILE).

WHY
---
`laundry preplanned` runs when the drum is already full and nothing has
been taken out, so there is a pile: grabbing at the drum floor would
drive the claw through all of it. The grabs belong in the pile's upper
part, and where that is depends on how full the drum is.

WHAT
----
generate() turns config.PILE into the grab list of scan_plans/
grab_targets.yaml, in visiting order, by solving on the MoveIt scene:

  - Layer 1 aims at the pile's top, top_height_m, sink_m below its
    surface. The 15 cm gripper and its approach meet the drum's
    ceiling, so a spot usually cannot reach that high: each spot takes
    the HIGHEST height some tilt reaches (heights 1 cm apart from
    the aim down to min_height_m, tilts smallest first).
  - Layer 2 aims layer_step_m below what the spot's layer 1 actually
    reached, and so on, until a layer would go below min_height_m.
    Laundry lower than that is for the scan-then-grasp stage; a pile
    that low gets no sweep at all.
  - The sweep visits layer 1 over all spots, mouth first, then layer
    2, so each pass takes from a pile that the one before has
    lowered.

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

# Layers are capped, so a bad PILE cannot loop.
MAX_LAYERS = 8

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
    top = config.PILE['top_height_m'] if top_height_m is None else top_height_m

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


def _candidates(depth_m, lateral_m, aim_m, pile, cone):
    """Yield (vertical height, target) from the aim down, tilts smallest first."""
    low = grab_targets.LIMITS['height_m']

    vertical = aim_m

    while vertical >= pile['min_height_m'] - 1e-9:
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


def find_grab(arm, depth_m, lateral_m, aim_m, pile=None, seeds=None, cone=None):
    """
    Return the widest, highest-reachable grab at one spot as (target, reason).

    A down grab (grab_targets) lateral_m sideways of the middle: the
    first sideways offset, from lateral_m in LATERAL_STEP_M steps toward
    the middle (no nearer than MIN_SIDE_M), at which any height from aim_m down to PILE's
    min_height_m passes retrieve_grid's checks, at its highest such
    height and least tilt: (target, None); else (None, why the last try
    failed). The drum's ceiling comes down and in toward the closed
    end, so the deeper spots fit less far out (fake controller,
    2026-10-27: +/-12 cm at 32 cm deep, +/-9 at 17). Needs the gripper
    padded (retrieve_grid.gripper_padding).
    """
    pile = pile or config.PILE
    cone = cone or seed_cone()
    seeds = seeds or retrieve_grid._seeds()
    inter = np.array(config.get_named_pose('inter'))

    reason = 'the aim is below the lowest sweep height'

    for lateral in _inward(lateral_m):
        for _vertical, target in _candidates(
            depth_m, lateral, aim_m, pile, cone
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
    Return (targets, misses, layers) for the pile, in visiting order.

    targets: the grab_targets.yaml list; misses: (layer, depth_m,
    lateral_m, reason) for spots a layer found nothing for; layers:
    per layer, the vertical heights its grabs reached (m).
    """
    pile = pile or config.PILE
    cone = seed_cone()
    seeds = retrieve_grid._seeds()

    top = pile['top_height_m']
    aim = top - pile['sink_m']

    # Where each spot's last layer reached; a spot with none left drops out.
    reached = {}
    targets, misses, layers = [], [], []

    with retrieve_grid.gripper_padding(arm):
        for layer in range(1, MAX_LAYERS + 1):
            found = []

            for spot in _spots(pile):
                if layer == 1:
                    spot_aim = aim
                elif spot in reached:
                    spot_aim = reached[spot] - pile['layer_step_m']
                else:
                    continue

                if spot_aim < pile['min_height_m'] - 1e-9:
                    reached.pop(spot, None)
                    continue

                target, reason = find_grab(
                    arm, spot[0], spot[1], spot_aim, pile, seeds, cone
                )

                if target is None:
                    misses.append((layer, spot[0], spot[1], reason))
                    reached.pop(spot, None)
                    log(
                        f'  layer {layer}: {spot[0] * 100:.0f} cm deep, '
                        f'{spot[1] * 100:+.0f} cm sideways: NOT REACHABLE - '
                        f'{reason}'
                    )
                    continue

                vertical = target_vertical(target, cone)
                reached[spot] = vertical
                found.append((vertical, target))

                log(
                    f'  layer {layer}: {grab_targets.describe(target)} '
                    f'({vertical * 100:.0f} cm above the floor)'
                )

            if not found:
                break

            targets += [target for _vertical, target in found]
            layers.append([vertical for vertical, _target in found])

    return targets, misses, layers
