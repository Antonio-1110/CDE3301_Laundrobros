#!/usr/bin/env python3

"""
The grasp stage: pick a reachable target and go and get it.

Given detected clusters and the bucket model they were detected
against:

    move to INTER (the reference orientation all the offset math in
    config/grasp.plan assumes, and where reachability is probed from)
    -> walk clusters best-first until one has a reachable grasp
       target (grasp.plan.compute_grasp_target)
    -> open gripper -> move_to_pose() so the gripper's contact point
       lands on the target -> close gripper -> retract to INTER
    -> (drop=True) DROP -> open gripper (release) -> INTER

Run from the terminal as `laundry grasp targets.json`, or as part of
`laundry run`.

The gripper is a GripperClient (gripper_node must be running
separately - it owns the GPIO) or, with --fake-hardware, a
hardware.fake.FakeGripper.
"""

from dataclasses import dataclass
from typing import Any

from . import retrieve_grid
from .plan import compute_grasp_target
from ..arm.transfers import go_to


def rank_clusters(clusters):
    """
    Order detected clusters best-target-first.

    Largest by integrated intrusion VOLUME, ties broken by point
    count.

    Volume rather than point count, because the scan's point
    density is strongly non-uniform - the helical path samples some
    parts of the bucket several times more densely than others - so
    cluster.size partly measures where an item happened to sit
    rather than how much fabric is there. volume_m3 is integrated
    per grid cell and is density-independent, so it compares two
    items fairly wherever they landed. It is also the quantity a
    gripper actually cares about.

    Callers should walk this list rather than committing to its
    first entry: being the biggest cluster does not make a target
    reachable, and an unreachable one is no reason to abandon a
    scan that found other candidates.
    """
    return sorted(
        clusters,
        key=lambda cluster: (cluster.volume_m3, cluster.size),
        reverse=True,
    )


def select_target_cluster(clusters):
    """Return the single best-ranked cluster, or None if there are none."""
    ranked = rank_clusters(clusters)

    return ranked[0] if ranked else None


def describe_grasp(grasp):
    """One-line summary of a GraspTarget, for the log."""
    return (
        f'Grasp target: TCP={grasp.tcp_position}, '
        f'orientation={grasp.orientation}, '
        f'contact point={grasp.grasp_point}, '
        f'sink={grasp.sink_amount_m * 100.0:.1f}cm '
        f'(fraction={grasp.sink_fraction_used}, '
        f'gap={grasp.gap_m * 100.0:.1f}cm, '
        f'reachability={grasp.reachability_fraction:.3f})'
    )


def execute_grasp(arm, gripper, grasp, drop=False):
    """
    Carry out a planned grasp; True if every step succeeded.

    Assumes the arm is at INTER (where grasp was planned from).

    With drop=True the item is carried to DROP and released. The
    retract to INTER before DROP is deliberate: the grasp pose is
    deep inside the bucket at an arbitrary computed position, so
    going straight to DROP would sweep the arm (and whatever it is
    now holding) sideways through the bucket wall. INTER is the
    withdrawn pose the scan itself starts and ends at, so it is a
    known-clear waypoint out.
    """
    print('Opening gripper...')

    if not gripper.open_blocking():
        print('Gripper did not confirm it opened (is gripper_node running?); '
              'aborting before the approach.')

        return False

    x, y, z = grasp.tcp_position

    if not arm.move_to_pose(x, y, z, orientation=grasp.orientation):
        print('Failed to reach grasp target; aborting.')

        return False

    print('Closing gripper...')

    closed = gripper.close_blocking()

    if not closed:
        # Still retract: the arm must not be left down in the bucket.
        print('Gripper did not confirm it closed; retracting without the '
              'item.')

    print('Retracting to INTER...')

    if not go_to(arm, 'inter'):
        print('Failed to retract to INTER; aborting.')

        return False

    if not closed:
        return False

    if not drop:
        return True

    print('Moving to DROP...')

    if not go_to(arm, 'drop'):
        print('Failed to reach DROP; aborting.')

        return False

    print('Opening gripper to release the item...')

    released = gripper.open_blocking()

    if not released:
        print('Gripper did not confirm it opened at DROP; the item may '
              'still be held.')

    print('Returning to INTER...')

    return go_to(arm, 'inter') and released


@dataclass
class GraspPlan:
    """
    How the next item will be taken.

    kind 'floor': a grid-style grab (retrieve_grid.plan_floor_grab),
    plan is its dict. kind 'cartesian': the tilted reach from INTER
    (plan.compute_grasp_target), plan is a GraspTarget.
    """

    kind: str
    cluster: Any
    plan: Any


def plan_grasp(arm, clusters, surface, grabs=None):
    """
    Return a GraspPlan for the best-ranked reachable cluster, or None.

    Assumes the arm is at INTER. Per cluster, best first:

      1. a grab like the grab grid's - IK over the item, a straight
         descent, reached from the nearest baked grab_NN - when the
         grid is baked (grabs defaults to retrieve.yaml's) and the
         item is within config.DETECTED_GRAB_MAX_ANGLE_DEG of the
         floor's lowest line (default: the whole lower half). Measured on the fake
         controller, the Cartesian reach below could not get to a
         towel in the middle of the floor at any sink depth: its one
         straight tool line from INTER runs into the wall.
      2. else the Cartesian reach (grasp/plan.py), for items away from
         the floor.
    """
    if grabs is None:
        grabs, _stamp = retrieve_grid.load()

    for cluster in rank_clusters(clusters):
        point = getattr(cluster, 'target_point', cluster.centroid)

        print(
            f'Trying cluster: size={cluster.size} target=('
            + ', '.join(f'{v:.3f}' for v in point)
            + f') mean_dev={cluster.mean_deviation_m:.3f}m'
        )

        if grabs:
            floor = retrieve_grid.plan_floor_grab(arm, point, grabs)

            if floor is not None:
                return GraspPlan('floor', cluster, floor)

        grasp = compute_grasp_target(cluster, surface, arm)

        if grasp is not None:
            return GraspPlan('cartesian', cluster, grasp)

        print('  unreachable; trying the next cluster.')

    return None


def execute_plan(arm, gripper, plan, drop=False):
    """Carry out a GraspPlan; True if every step succeeded."""
    if plan.kind == 'floor':
        return retrieve_grid.run_floor_grab(
            arm, gripper, plan.plan, go_to, drop=drop
        )

    return execute_grasp(arm, gripper, plan.plan, drop=drop)


def describe_plan(plan):
    """One-line summary of a GraspPlan, for the log."""
    if plan.kind == 'cartesian':
        return describe_grasp(plan.plan)

    grab = plan.plan

    return (
        f'Floor grab via {grab["via"]}: contact={grab["contact"]}, '
        f'{grab["floor_angle_deg"]:+.0f} deg, '
        f'{grab["height_m"] * 100:.1f} cm from the wall, '
        f'tilt {grab["tilt_deg"]:.0f} deg'
    )


def grasp_best(arm, gripper, clusters, surface, drop=False, dry_run=False):
    """
    Move to INTER, plan against the clusters best-first, then execute.

    dry_run stops after printing the grasp plan - the arm still moves
    to INTER (the reachability probe must start from the real
    starting state), but never approaches, grips or drops.

    Returns True on success (or a successful dry run).
    """
    print('Moving to INTER (reference orientation for grasp math)...')

    if not go_to(arm, 'inter'):
        print('Failed to reach INTER; aborting.')

        return False

    plan = plan_grasp(arm, clusters, surface)

    if plan is None:
        print(
            f'None of the {len(clusters)} detected cluster(s) '
            'yielded a reachable grasp; aborting.'
        )

        return False

    print(describe_plan(plan))

    if dry_run:
        print('--dry-run: stopping before the approach.')

        return True

    return execute_plan(arm, gripper, plan, drop=drop)
