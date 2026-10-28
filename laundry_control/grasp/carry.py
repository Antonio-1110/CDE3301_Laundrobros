#!/usr/bin/env python3

"""
Carry the laundry with the claw turned, and move the claw on arrival.

TURNED: the jaws close along link7 Y, and every grab sets link7 X
along the bucket axis - at INTER, X points down - so a claw carried as
it grabbed closes sideways, and the laundry hangs in a pinch all the
way out and to DROP. Turned config.CARRY_J7_TURN_DEG on J7 the jaws
close up and down instead: the lower one is under the laundry. Every
grasp (retrieve_grid.run, the sweep; retrieve_grid.run_floor_grab,
detected items; grasp.execute.execute_grasp, `laundry grasp`) carries
its laundry out and to DROP as ONE baked move, and the turn rides on
it (transfers.prepare_go_to(turn_rad), arm/joint_path.add_j7_turn):
J7 turns as the lift starts and is back by DROP, so the arm never
stops to turn, and the claw opens unturned. A carry too short to turn
within J7's limits, or that would collide turned, is made unturned.

ON ARRIVAL: the claw is told to close or open just before the move
into the grab or into DROP is planned to end (config.
CLAW_COMMAND_LEAD_SEC), not once the controller reports the arm
there, ~0.45 s later on the rig. The arm still waits for the claw's
ack before it moves on.
"""

import math

from .. import config


def turn_rad():
    """Return the carry turn on J7 (rad); 0 carries unturned."""
    return math.radians(config.CARRY_J7_TURN_DEG)


def on_arrival(start):
    """
    Return (before_end, started) for a prepared move(before_end=...).

    before_end starts the claw move (start: e.g. gripper.start_close)
    as the arm arrives; started then holds its Pending. wait() gets
    the result.
    """
    started = []

    return (
        (config.CLAW_COMMAND_LEAD_SEC, lambda: started.append(start())),
        started,
    )


def wait(started, start):
    """Wait for the claw move started on arrival; start it now if it was not."""
    return (started[0] if started else start()).wait()
