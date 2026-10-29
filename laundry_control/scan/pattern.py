#!/usr/bin/env python3

"""
The bucket scan: helical tool-Z strokes with a J7 twist, plus a BOTTOM detour.

scan() only MOVES the arm. The ToF readings are captured by the
separate scan_recorder_node (scan/recorder_node.py) through TF while
this runs; `recorder` is only used to checkpoint and finally save
that node's CSV.

Run from the terminal as `laundry scan` (cli/stages.py).

scan() reads as the sequence of steps; each step is a method of
_ScanRun, which carries the options and the arm's progress between
them.
"""

import argparse
import contextlib
import math
import os
import time

import numpy as np
import rclpy
from scipy.spatial.transform import Rotation

from . import segments
from .strokes import MAX_STEP_M, stroke_plan
from .. import config
from ..arm.joint_path import MIN_JERK_PEAK
from ..arm.transfers import entry_joints, go_to

# A min-jerk move's peak acceleration, as a multiple of distance /
# duration^2 (10 / sqrt(3)).
MIN_JERK_PEAK_ACCELERATION = 10.0 / math.sqrt(3.0)

# Defaults for every scan parameter, shared by scan() and the CLI so
# the two can never drift apart. Baselines and detection scans must
# see the wall the same way: same depth range, sweep and J7 speed.
# The step may be coarser than the baselines' (the model is per wall
# cell, and every stroke views the wall square-on), as long as the
# stroke speed scales with it - see scaled_stroke_speed().
DEFAULT_DEPTH_M = 0.42
# How far the arm moves straight in from INTER, without recording or
# turning J7, before the strokes start: at INTER the sensor is at the
# bucket mouth, and readings taken there (and on the way out) caught
# the lip and things past it - every out-of-bucket point in the
# 2026-09-28 scans came from the first or last 5% of the scan. At 6 cm
# the sensor starts ~4-5 cm inside. Changing it changes the scan path
# (re-collect baselines), but needs no re-bake: the strokes are
# planned live and the end scan pivots at `depth` regardless.
DEFAULT_ENTRY_DEPTH_M = 0.06
# Speed scaling for the straight moves into and out of the bucket that
# nothing is recorded on: a scan's entry and exit (entry_depth) and the
# end-scan-only pass. Faster than the strokes (DEFAULT_VELOCITY), which
# are slow only so each J7 sweep gets enough readings.
STRAIGHT_IN_OUT_VELOCITY = 0.1
# The LARGEST spacing between strokes (scan/strokes.py spreads them
# evenly over depth - entry depth); at most strokes.MAX_STEP_M.
DEFAULT_STEP_M = 0.03
DEFAULT_SWEEP_DEG = 150.0
#
# Stroke velocity/acceleration scaling: 0.03, down from 0.1. Measured
# on the fake controller, a 3cm stroke takes 0.95s at 0.1 and 2.68s
# at 0.03, so at the ToF's fixed 20Hz a scan gets ~2.1x the readings
# (~94s instead of ~46s). It also brings the J7 twist - which MoveIt
# does not rate-limit (config.JOINT7_MAX_VELOCITY_RAD_S) - from ~157
# deg/s, above J7's 123 deg/s limit, down to ~56 deg/s. In simulation
# the denser scan cut the grasp point's median error ~1.0 -> 0.65cm;
# 0.02 gained nothing more. The committed baselines were recorded at
# 0.1: re-collect them at this speed (HARDWARE_TESTS.md D).
#
# These are the speeds for the DEFAULT step. A stroke's duration is set
# by its tool-Z distance (the J7 twist rides on MoveIt's timing,
# arm/trajectory.add_joint7_twist), so a coarser step at the same
# scaling would just take longer per stroke and save no time. The CLI
# therefore scales both with the step (scaled_stroke_speed): 12 cm at
# 0.12 sweeps J7 as fast as 3 cm at 0.03, in a quarter of the strokes.
#
# 2026-10-28: 0.045 (was 0.03), 1.5x faster, matched to the ESP32's
# ToF rate: a reading every ~33 ms (TOF_TIMING_BUDGET_US 33000, ~30 Hz)
# against the 20 Hz the 0.03 was chosen for, so each J7 sweep still
# gets as many readings. Each reading now smears over a 1.5x wider arc
# of its window, so the baselines must be RE-COLLECTED at this speed.
# With the Pi's own I2C sensor (20 Hz, tof_source:=i2c) a sweep gets a
# third fewer readings: pass --velocity 0.03 --acceleration 0.03.
# 0.045 put the J7 twist just over J7's limits (127 deg/s against 123,
# 10.4 rad/s^2 against 10), so every pass was slowed 1.04x anyway, with
# a warning: 0.043 is that speed without it.
DEFAULT_VELOCITY = 0.043
DEFAULT_ACCELERATION = 0.043


def scaled_stroke_speed(step, base=DEFAULT_VELOCITY):
    """
    Return the stroke velocity/acceleration scaling for this step.

    base * step / DEFAULT_STEP_M, capped at 1: every stroke then takes
    about as long as a default one, so J7 sweeps at the baselines' speed
    and each sweep gets as many readings.
    """
    return min(1.0, base * step / DEFAULT_STEP_M)


DEFAULT_ROTATION_VELOCITY = 0.5
DEFAULT_ROTATION_ACCELERATION = 0.5

# The scan's own joint moves around the strokes - in with the J7
# offset, the turnaround, back out to INTER - are mostly J7 turning,
# so they run at this fraction of J7's limits (_ScanRun.
# _quick_joint_move) instead of the 45 deg/s of a general straight
# joint move (config.LINEAR_JOINT_MOVE_MAX_VELOCITY_RAD_S): the
# 150 deg turnaround took 7.3 s on the rig (2026-10-28), ~2.5 s now.
QUICK_MOVE_J7_FRACTION = 0.9
DEFAULT_CARTESIAN_STEP_M = 0.005
DEFAULT_PAUSE_SEC = 0.0
DEFAULT_SAVE_INTERVAL_SEC = 5.0
# How often a scan logs its progress (_ScanRun.reporting), seconds.
PROGRESS_PERIOD_SEC = 5.0

# How the closed end is covered: 'precession' (the FULL scan, baked -
# see scan/endcap.py), 'none' (the QUICK scan, the CLI default: no end
# scan, only a J7 turnaround at the deepest stroke - the tilting end
# scan swings the wrist and elbow low, rubbing laundry on the lower
# walls, and the quick scan never does; it sees ~86% of the floor and
# lower walls and none of the closed end, and is judged against the
# baselines' strokes, scan/segments.py) or 'bottom' (the old
# recorded-pose detour, only when asked for). Baselines are always
# collected full (scan/baselines.run_one_scan asks for it).
END_SCANS = ('precession', 'none', 'bottom')
DEFAULT_END_SCAN = 'none'

# Coarse first (`laundry run` / `clear`, quick scans - pipeline.
# scan_and_detect): a pass at each of these steps, coarsest first,
# before the scan's own --step, which alone (with the end scan) may
# say the bucket is empty. Each pass hands over inside the bucket
# (stay_inside/start_inside), and laundry found stops the search.
#
# Measured offline 2026-10-26 (issue #10; leave-one-out on the
# 2026-10-22 baselines, synthetic items, each simulated 12 cm beam
# replaced by the held-out scan's nearest real reading - pessimistic
# for flat cloth, which needs many readings: the same method gives
# 75% at 3 cm against the real scan's 95%):
#
#   pass          false alarms  bundle towel shirt  sock  flat  grasp err
#   3 cm (real)      0/8         100   100   100    99    95   0.5-0.9 cm
#   12 cm            0/40        100   100   100    81    19   1.2-1.6 cm
#   12 cm, inward    0/40        100    98    85    48    13   2.6-2.8 cm
#
# On real laundry (the 10 rig quick scans of 2026-10-22, resampled
# the same way but matched by ray, not by wall point): the 12 cm pass
# found 6 of the 7 items the 3 cm scan found (not the smallest, 353
# cm^3, which the 3 cm inward strokes alone miss too), and nothing in
# the 3 empty ones. But it splits a big pile into several clusters:
# the biggest lay 2-7 cm from the 3 cm scan's (1-9 cm on the inward
# strokes alone) - on the pile for the 3000-7000 cm^3 heaps, maybe
# not for a ~900 cm^3 item. So coarse passes always run whole: no check
# at the bottom, whose early exit would save only ~8 s at 12 cm (fake
# controller: leaving from the bottom ~16 s, against ~24 s to finish
# the pass and leave from the entry pose) and lose the outward
# strokes' aim. On the 3 cm pass it saves ~33 s for little aim.
#
# A 12 cm pass takes ~38 s on the rig against ~97 s at 3 cm, so the
# bigger items are found in about a third of the time; what it misses
# the 3 cm pass still finds, for ~30 s more than scanning dense only.
DEFAULT_COARSE_STEPS_M = (0.12,)

# How `run` / `clear` search with quick scans (pipeline.scan_and_detect):
#
#   'in-out' (default since 2026-10-28): ONE trip into the bucket. The
#       coarsest step's strokes on the way in, a detection at the
#       bottom, then - nothing found - the end scan right there (the
#       arm is already at its pivot) and the scan's own step on the
#       way out; a CONFIDENT find (CONFIDENT_MIN_*) - out on the
#       coarse strokes too, no end scan; a weaker find - out at the
#       scan's own step to confirm and aim, no end scan. Empty bucket: ~105 s against
#       ~175 s for the ladder (estimates from simulated stroke and
#       end-scan times); the bucket is called empty after the dense
#       outward strokes, the coarse inward ones and the end scan.
#       Unlike the ladder's dense pass, the dense strokes run one way
#       only: they cross nothing on the way in, so the wall between
#       their helix turns is seen by the coarse strokes alone - check
#       small-item recall with `laundry evaluate` before trusting an
#       "empty" from it.
#   'ladder' (2026-10-26): a whole coarse pass, then a whole pass at
#       the scan's step, then the end scan alone (pipeline.scan_and_detect).
SEARCHES = ('in-out', 'ladder')
DEFAULT_SEARCH = 'in-out'

# In-out search, at the bottom: a find this big AND this strong (and
# in well-sampled cells, ClusterSummary.confident) is CONFIDENT, and
# the arm comes out on the coarse strokes too - a coarse pass in and
# out aims to ~1.2-1.6 cm (issue #10, offline), inside the claw's
# ~3 cm. A weaker find comes out dense, to confirm and aim. Chosen
# from: the rig's laundry (2026-10-21) 2400 cm^3 at 45 sigma; synthetic
# towels a median ~1600 cm^3; socks a median ~120 cm^3; false clusters
# at most 61 cm^3 and 5 sigma (issue #22). Tune them on real in-out
# scans.
CONFIDENT_MIN_VOLUME_M3 = 500.0e-6
CONFIDENT_MIN_PEAK_SIGMA = 10.0


def scan(
    arm,
    depth=DEFAULT_DEPTH_M,
    entry_depth=DEFAULT_ENTRY_DEPTH_M,
    step=DEFAULT_STEP_M,
    sweep_deg=DEFAULT_SWEEP_DEG,
    velocity=DEFAULT_VELOCITY,
    acceleration=DEFAULT_ACCELERATION,
    rotation_velocity=DEFAULT_ROTATION_VELOCITY,
    rotation_acceleration=DEFAULT_ROTATION_ACCELERATION,
    cartesian_step=DEFAULT_CARTESIAN_STEP_M,
    pause=DEFAULT_PAUSE_SEC,
    recorder=None,
    save_interval=DEFAULT_SAVE_INTERVAL_SEC,
    end_scan='precession',
    end_plan=None,
    end_scan_time_scale=1.0,
    start_inside=False,
    stay_inside=False,
    check=None,
    outward_step=None,
    outward_velocity=None,
    outward_acceleration=None,
    outward_when_found=False,
):
    """
    Perform the complete xArm7 scanning sequence.

    Parameters
    ----------
    arm:
        Existing XArm7Controller.

    depth:
        Maximum insertion depth in metres.

        Default:
            0.42 m

    entry_depth:
        Straight insertion from INTER before the strokes start (and
        the retraction after they end), neither recorded nor
        swinging J7, so every reading is taken inside the bucket.

        Default:
            0.06 m

    step:
        The LARGEST linear distance per scan stroke, at most
        strokes.MAX_STEP_M (0.12 m). The strokes are spread evenly
        from entry_depth to depth (scan/strokes.stroke_plan), so each
        may be a little shorter. Over the default, pass velocity and
        acceleration from scaled_stroke_speed(step), or the scan only
        gets slower, not faster.

        Default:
            0.03 m

    sweep_deg:
        J7 rotation during each scan stroke.

        Default:
            150 degrees

        The initial offset is automatically sweep_deg / 2.

    velocity:
        MoveIt velocity scaling.

    acceleration:
        MoveIt acceleration scaling.

    rotation_velocity:
        MoveIt velocity scaling used for the non-insertion
        motions: the initial J7 offset and the BOTTOM detour
        (entry, stationary sweep, and exit/turnaround). Since
        none of these involve simultaneous insertion/retraction
        they can run faster than the interleaved scan strokes.

        Default:
            0.5

    rotation_acceleration:
        MoveIt acceleration scaling for the same non-insertion
        motions.

        Default:
            0.5

    cartesian_step:
        Cartesian interpolation resolution passed to the controller.

    pause:
        Optional pause between motions in seconds.

    recorder:
        ScanRecorderClient (or hardware.fake.FakeRecorder) used to
        checkpoint and finally save scan_recorder_node's CSV, or
        None to only move the arm.

    save_interval:
        Seconds between fire-and-forget checkpoint saves during the
        scan; <= 0 disables them.

    end_scan:
        How the closed end is covered at maximum depth: 'precession'
        (default) replays the baked coning sweep in end_plan (see
        scan/endcap.py); 'none' (the quick scan) only turns J7 for
        the way out; 'bottom' runs the old recorded-pose BOTTOM
        detour and is only used when asked for.

    end_plan:
        The endcap.EndcapPlan to replay. Required for 'precession'.

    end_scan_time_scale:
        Replay the precession end scan (and its entry/exit moves)
        slower than baked, in (0, 1] - for cautious first runs on
        the real arm. Does not change the path.

    start_inside, stay_inside:
        For back-to-back scans (baseline collect). stay_inside ends
        the scan with J7 turned back but still entry_depth in, instead
        of going out to INTER; start_inside then starts the next scan
        right there, skipping INTER and the straight entry. Neither is
        recorded, so the recorded path is unchanged. start_inside
        refuses (moves nothing) unless the flange is at that pose.

    check:
        Called with no arguments at the deepest point, after the
        inward strokes, once the readings so far are saved to the
        recorder's CSV - for the pipeline to detect on the inward
        pass while the arm stands there. If it returns True the scan
        stops: no end scan, no outward strokes, straight out to INTER
        (even with stay_inside). None never stops.

    outward_step:
        The outward strokes' largest spacing (default: step), so a
        scan can go in coarse and come out dense (the pipeline's
        in-out search). Spread evenly like the inward ones.

    outward_velocity, outward_acceleration:
        The outward strokes' speed scaling (default: velocity and
        acceleration); scale it with outward_step as
        scaled_stroke_speed does.

    outward_when_found:
        With check: when it finds laundry, skip the end scan but still
        come out on the outward strokes (they sharpen the aim)
        instead of straight out. check() may return a step (metres)
        instead of True: come out at that step (with the speed
        scaled to it) instead of outward_step.


    Notes
    -----
    The scan pattern:

    Assuming:

        sweep_deg = 150

    Start (not recorded):

        INTER, J7 = 0 deg; straight in by entry_depth.

    Initial positioning (recording from here):

        0 -> -75 deg

        No linear movement.

    Inward:

        -75 -> +75     while inserting one step
        +75 -> -75     while inserting one step
        -75 -> +75     while inserting one step
        ...

    End scan (also performs the turnaround phase shift):

        At maximum depth the gripper stops the sensor reaching the
        closed end with radial beams, so the end is covered
        separately.

        'precession' (default): a baked, collision-checked joint
            trajectory tilts the tool axis in a cone about the
            deepest flange position so the beam sweeps concentric
            arcs over the lower closed end (scan/endcap.py). The
            arm then returns to the stroke configuration with J7
            turned to the turnaround target -- the side OPPOSITE
            where the inward scan ended (relative to INTER's
            reference).

        'bottom' (only when requested): tilt up to the recorded
            BOTTOM pose while sweeping J7, sweep again there, and
            tilt back landing J7 on the same turnaround target.

        Outward:

        Alternate +/-150 degree strokes while retracting
        one step at a time.

    No further stationary rotations occur on the way out.

    Leaving (not recorded): back at entry_depth, J7 back to INTER's,
    then straight out to INTER.

    """
    error = _invalid_option(
        depth, end_scan, end_plan, end_scan_time_scale, sweep_deg,
        velocity, acceleration, rotation_velocity, rotation_acceleration,
    )

    if error:
        arm.get_logger().error(error)
        return False

    try:
        number_of_steps, step_used = stroke_plan(depth, entry_depth, step)
        out_steps, out_step_used = stroke_plan(
            depth, entry_depth, outward_step or step
        )
    except ValueError as exc:
        arm.get_logger().error(str(exc))
        return False

    run = _ScanRun(
        arm,
        recorder,
        number_of_steps=number_of_steps,
        step=step_used,
        sweep_deg=sweep_deg,
        entry_depth=entry_depth,
        velocity=velocity,
        acceleration=acceleration,
        rotation_velocity=rotation_velocity,
        rotation_acceleration=rotation_acceleration,
        cartesian_step=cartesian_step,
        pause=pause,
        save_interval=save_interval,
        start_inside=start_inside,
        stay_inside=stay_inside,
        outward=(
            out_steps, out_step_used,
            velocity if outward_velocity is None else outward_velocity,
            acceleration if outward_acceleration is None
            else outward_acceleration,
        ),
    )

    run.log_configuration(depth, max_step=step, end_scan=end_scan)

    def sequence(end_scan):
        # The sequence (see Notes above); each step logs its own failure.
        if not (
            run.enter()                                     # 1, 1b
            and run.initial_offset()                        # 2
            and run.strokes(inward=True)                    # 3
        ):
            return False

        if check is not None:
            stop = run.check_at_bottom(check)               # 3b

            if stop is None:
                return False

            if stop and not outward_when_found:
                return (
                    run.stop_and_save()
                    and run.leave(stay_inside=False)
                    and run.report()
                )

            if stop:
                # Found: no end scan (it only certifies an empty bucket);
                # turn round and come out on the outward strokes, which
                # sharpen the inward strokes' aim - at the step check()
                # returned, if it returned one (a confident find needs no
                # dense strokes to aim).
                end_scan = 'none'

                if not isinstance(stop, bool):
                    if not run.set_outward_step(float(stop), depth):
                        return False

        phase_twist = run.end_scan(
            end_scan, end_plan, depth, end_scan_time_scale
        )                                                   # 4

        return (
            phase_twist is not None
            and run.strokes(inward=False, after_phase_twist=phase_twist)  # 5
            and run.stop_and_save()                         # 6
            and run.leave()                                 # 7
            and run.report()                                # 8
        )

    with run.reporting():
        return sequence(end_scan)


def at_entry_pose(arm, entry_depth):
    """
    Return True if the flange is entry_depth straight in from INTER.

    Where a scan's entry would have put it (and its leave(stay_inside)
    left it): INTER's orientation, J7 included, within 2 deg, and
    within 5 mm of INTER moved entry_depth along its tool Z. Logs why
    not.
    """
    log = arm.get_logger()
    current = arm.get_current_joints()
    here = None if current is None else arm.compute_fk(current)
    inter = arm.compute_fk(config.get_named_pose('inter'))

    if here is None or inter is None:
        log.error('Could not compute FK to check the start pose.')
        return False

    rotation = Rotation.from_quat(inter[1])
    expected = inter[0] + entry_depth * rotation.as_matrix()[:, 2]
    off_mm = float(np.linalg.norm(here[0] - expected)) * 1000.0
    off_deg = math.degrees(
        (Rotation.from_quat(here[1]) * rotation.inv()).magnitude()
    )

    if off_mm > 5.0 or off_deg > 2.0:
        log.error(
            f'Start inside: the flange is {off_mm:.1f} mm / '
            f'{off_deg:.1f} deg from the entry pose (INTER, '
            f'{entry_depth * 100:.0f} cm in); not scanning from here.'
        )
        return False

    return True


def _invalid_option(
    depth, end_scan, end_plan, end_scan_time_scale, sweep_deg,
    velocity, acceleration, rotation_velocity, rotation_acceleration,
):
    """Return why scan()'s options are unusable, or None if they are fine."""
    if depth <= 0.0:
        return 'depth must be greater than zero.'

    if end_scan not in END_SCANS:
        return f'end_scan must be one of {END_SCANS}; got {end_scan!r}.'

    if end_scan == 'precession' and end_plan is None:
        return "end_scan='precession' needs a baked plan (laundry plan bake)."

    if not 0.0 < end_scan_time_scale <= 1.0:
        return 'end_scan_time_scale must be in (0, 1].'

    if sweep_deg <= 0.0:
        return 'sweep_deg must be greater than zero.'

    for name, value in (
        ('velocity', velocity),
        ('acceleration', acceleration),
        ('rotation_velocity', rotation_velocity),
        ('rotation_acceleration', rotation_acceleration),
    ):
        if not 0.0 < value <= 1.0:
            return f'{name} must be in the range (0, 1].'

    return None


class _ScanRun:
    """
    One scan() in progress: its options, and where the arm has got to.

    Each step method moves the arm through one part of the sequence
    and returns True, or logs why it failed and returns False.
    """

    def __init__(
        self, arm, recorder, number_of_steps, step, sweep_deg, entry_depth,
        velocity, acceleration, rotation_velocity, rotation_acceleration,
        cartesian_step, pause, save_interval, start_inside=False,
        stay_inside=False, outward=None,
    ):
        self.arm = arm
        self.log = arm.get_logger()
        self.recorder = recorder

        self.number_of_steps = number_of_steps
        self.step = step
        self.sweep_deg = sweep_deg
        self.half_sweep = sweep_deg / 2.0
        self.entry_depth = entry_depth

        self.velocity = velocity
        self.acceleration = acceleration
        self.rotation_velocity = rotation_velocity
        self.rotation_acceleration = rotation_acceleration
        self.cartesian_step = cartesian_step
        self.pause = pause
        self.save_interval = save_interval
        self.start_inside = start_inside
        self.stay_inside = stay_inside
        # The outward strokes: (count, step, velocity, acceleration).
        self.outward = outward or (
            number_of_steps, step, velocity, acceleration
        )

        # Nominal insertion depth, for the log only.
        self.current_depth = 0.0
        self.last_save_time = arm.get_clock().now()

        # Progress (reporting()): the phase now, and when it started.
        self.phase = 'starting'
        self.started = self.phase_started = time.monotonic()
        self.pass_info = None

        # enter() turned J7 to the initial offset on its way in.
        self.offset_done = False

    def set_phase(self, phase):
        """Log a new phase of the scan, and time it for the progress line."""
        now = time.monotonic()
        self.phase, self.phase_started = phase, now
        self.pass_info = None
        self.log.info(f'[scan {now - self.started:5.1f} s] {phase}')

    def _progress(self):
        now = time.monotonic()
        line = (
            f'[scan {now - self.started:5.1f} s] ... {self.phase} '
            f'({now - self.phase_started:.0f} s)'
        )

        # During a pass of strokes (one trajectory): how far along it
        # is, from the motion's planned duration.
        started = getattr(self.arm, 'motion_started', 0.0)
        duration = getattr(self.arm, 'motion_duration', 0.0)

        if (
            self.pass_info is not None and duration > 0.0
            and started >= self.phase_started
        ):
            start, count, step, inward = self.pass_info
            done = min(max((now - started) / duration, 0.0), 1.0)
            depth = start + (1.0 if inward else -1.0) * done * count * step
            line += (
                f': stroke {min(int(done * count) + 1, count)}/{count}, '
                f'~{depth:.2f} m, ~{max(duration - (now - started), 0.0):.0f} '
                's left in the pass'
            )

        self.log.info(line)

    @contextlib.contextmanager
    def reporting(self, period_sec=PROGRESS_PERIOD_SEC):
        """
        Quiet the per-motion detail and log progress every period_sec.

        The arm's per-stroke lines (joint targets, Cartesian path
        fractions, the J7 twist) go to DEBUG for the scan's duration
        (XArm7Controller.quiet_motion); a progress line replaces them.
        The timer fires whenever the node spins - while it waits on
        each motion.
        """
        quiet_before = getattr(self.arm, 'quiet_motion', False)
        self.arm.quiet_motion = True
        timer = (
            self.arm.create_timer(period_sec, self._progress)
            if period_sec > 0 else None
        )

        try:
            yield
        finally:
            if timer is not None:
                timer.cancel()
                self.arm.destroy_timer(timer)
            self.arm.quiet_motion = quiet_before

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def log_configuration(self, depth, max_step, end_scan='?'):
        """Log the scan's parameters before anything moves."""
        for line in (
            '========================================',
            'XARM7 SCAN SEQUENCE',
            f'Depth            : {depth:.3f} m',
            f'Entry depth      : {self.entry_depth:.3f} m',
            f'Linear step      : {self.step:.4f} m (at most {max_step:.3f})',
            f'Number of strokes: {self.number_of_steps}',
            f'Outward          : {self.outward[0]} strokes of '
            f'{self.outward[1]:.4f} m',
            f'Wrist sweep      : {self.sweep_deg:.1f} deg',
            f'Stroke speed     : {self.velocity:.3f} in, '
            f'{self.outward[2]:.3f} out (MoveIt scaling)',
            f'End scan         : {end_scan}',
            f'Initial offset   : {-self.half_sweep:+.1f} deg',
            '========================================',
        ):
            self.log.info(line)

    def mark_segment(self, segment):
        """Label the readings from now on as this scan segment."""
        if self.recorder is not None and not self.recorder.set_segment(segment):
            self.log.error(
                'Could not label the end-scan readings in the recorder; '
                'this scan cannot be split into strokes and end scan.'
            )

    def set_recording(self, on):
        """Start or pause scan_recorder_node's recording; False if it would not."""
        if self.recorder is None or self.recorder.set_recording(on):
            return True

        self.log.error(
            f"Could not {'start' if on else 'stop'} scan_recorder_node's "
            'recording (a bring-up started before the recorder had '
            "a 'recording' parameter? Restart it)."
        )
        return False

    def wait(self):
        """
        Pause between motions.

        Spins the node (rather than a plain time.sleep) so pending
        service futures (e.g. recorder.save_async()) get processed
        even if this stroke happens to be the last one before a
        long pause. ToF capture itself runs in scan_recorder_node,
        a separate process, so it needs no help from this spin.
        """
        if self.pause <= 0.0:
            return

        deadline = self.arm.get_clock().now() + rclpy.duration.Duration(
            seconds=self.pause
        )

        while self.arm.get_clock().now() < deadline:
            rclpy.spin_once(self.arm, timeout_sec=0.05)

    def checkpoint(self):
        """
        Checkpoint the recorder's CSV every save_interval seconds.

        Periodically ask scan_recorder_node to checkpoint its CSV,
        at a much slower cadence than point capture/publishing, via
        a fire-and-forget service call (see ScanRecorderClient).
        """
        if self.recorder is None or self.save_interval <= 0.0:
            return

        now = self.arm.get_clock().now()

        if (now - self.last_save_time).nanoseconds * 1e-9 >= self.save_interval:
            self.recorder.save_async()
            self.last_save_time = now

    def _quick_joint_move(self, target):
        """
        Move on a straight joint line to target, as fast as J7 allows.

        For the moves around the strokes, where J7 does most of the
        turning: timed so the fastest joint peaks at QUICK_MOVE_J7_
        FRACTION of J7's speed AND acceleration limits (min-jerk peaks
        at 1.875 x the mean speed, 5.77 x distance / duration^2).
        Collision-checked like any move_joints_linear; False if refused.
        """
        current = self.arm.get_current_joints()

        if current is None:
            return False

        distance = float(np.abs(np.subtract(target, current)).max())

        if distance < 1e-4:
            return True

        velocity = QUICK_MOVE_J7_FRACTION * config.JOINT7_MAX_VELOCITY_RAD_S
        acceleration = (
            QUICK_MOVE_J7_FRACTION * config.JOINT7_MAX_ACCELERATION_RAD_S2
        )
        duration = max(
            MIN_JERK_PEAK * distance / velocity,
            math.sqrt(MIN_JERK_PEAK_ACCELERATION * distance / acceleration),
        )

        return self.arm.move_joints_linear(
            list(target), max_velocity_rad_s=MIN_JERK_PEAK * distance / duration,
        )

    def _tool_z(self, distance):
        """Move straight along tool Z, unrecorded and without twist."""
        return self.arm.move_tool_z(
            distance,
            max_step=self.cartesian_step,
            velocity=STRAIGHT_IN_OUT_VELOCITY,
            acceleration=STRAIGHT_IN_OUT_VELOCITY,
        )

    # ---------------------------------------------------------
    # 1. INTER, then straight in; recording starts inside
    #
    # INTER establishes the known starting pose (and its nominal J7
    # orientation of 0 degrees) that the rest of the sequence,
    # starting with the initial offset, assumes. Nothing is
    # recorded until the sensor is inside the bucket.
    # ---------------------------------------------------------

    def at_entry_pose(self):
        """Return True if the flange is entry_depth straight in from INTER."""
        return at_entry_pose(self.arm, self.entry_depth)

    def enter(self):
        """Go to INTER, move straight in by entry_depth, start recording."""
        self.set_phase(
            'entering' if not self.start_inside
            else 'starting at the entry pose'
        )

        if not self.set_recording(False):
            return False

        if self.start_inside:
            # The previous scan stayed in (leave(), stay_inside).
            self.log.info('========== START INSIDE (entry pose) ==========')

            if not self.at_entry_pose():
                return False

            self.current_depth = self.entry_depth

        else:
            self.log.info('========== MOVE TO INTER ==========')

            # Baked transfer from HOME/DROP, else a straight checked
            # joint move, else the planner (arm/transfers.go_to).
            if not go_to(self.arm, 'inter'):
                self.log.error('Move to INTER failed.')
                return False

            self.wait()

        # 1b. In by entry_depth. Where it is clear, one move turning J7
        # to the initial offset on the way (2. is then done): ~1.3 s
        # against ~4.2 s straight in and then turning (rig,
        # 2026-10-28). Else straight in along tool Z, J7 unchanged.
        if self.entry_depth > 0.0 and not self.start_inside:
            self.log.info(
                f'========== ENTRY ({self.entry_depth:.3f} m) =========='
            )
            target = entry_joints(self.arm, self.entry_depth)

            if target is not None:
                target = list(target)
                target[6] -= math.radians(self.half_sweep)

            if target is not None and self._quick_joint_move(target):
                self.log.info(
                    f'In with J7 turned to {-self.half_sweep:+.0f} deg.'
                )
                self.offset_done = True
            elif not self._tool_z(self.entry_depth):
                self.log.error('Straight entry into the bucket failed.')
                return False

            self.current_depth = self.entry_depth
            self.wait()

        # Recording starts here, with the sensor inside: drop whatever
        # the recorder picked up before it was paused.
        if self.recorder is not None and not self.recorder.clear_blocking():
            self.log.error('Could not clear the recorder.')
            return False

        if not self.set_recording(True):
            return False

        self.last_save_time = self.arm.get_clock().now()

        return True

    # ---------------------------------------------------------
    # 2. INITIAL J7 OFFSET
    #
    # INTER is assumed to have the nominal wrist orientation of
    # 0 degrees. Move 0 -> -75 for the default 150-degree sweep.
    # rotate_joint7() keeps J1-J6 where they are.
    # ---------------------------------------------------------

    def initial_offset(self):
        """Turn J7 to -half the sweep, on the spot (unless enter() did)."""
        if self.offset_done:
            return True

        self.set_phase(f'turning J7 to {-self.half_sweep:+.0f} deg')

        self.log.info('========== INITIAL OFFSET ==========')
        self.log.info(f'Stationary J7 rotation: {-self.half_sweep:+.1f} deg')

        current = self.arm.get_current_joints()

        if current is not None:
            turned = list(current)
            turned[6] -= math.radians(self.half_sweep)

            if self._quick_joint_move(turned):
                self.wait()
                return True

        if not self.arm.rotate_joint7(
            delta_deg=-self.half_sweep,
            velocity=self.rotation_velocity,
            acceleration=self.rotation_acceleration,
        ):
            self.log.error('Initial J7 positioning failed.')
            return False

        self.wait()

        return True

    # ---------------------------------------------------------
    # 3 and 5. THE STROKES
    #
    # Inward, from -75 deg: -75 -> +75 (+150) while inserting one
    # step, then +75 -> -75 (-150), alternating.
    #
    # Outward, after the end scan's phase shift (9 strokes: inward
    # ends at +75, the phase shift turns +75 -> -75), start the
    # other way: -75 -> +75 (+150), then +75 -> -75, ... There are
    # NO stationary resets between strokes, and (without --pause) a
    # pass is one trajectory, so the arm never stands at a turnaround.
    # ---------------------------------------------------------

    def strokes(self, inward, after_phase_twist=0.0):
        """Run the inward strokes, or the outward ones after the end scan."""
        count, step = (
            (self.number_of_steps, self.step) if inward else self.outward[:2]
        )
        end = (
            self.current_depth + count * step if inward
            else self.current_depth - count * step
        )
        self.set_phase(
            f"{'inward' if inward else 'outward'} strokes: {count} x "
            f'{step * 100:.1f} cm, {self.current_depth:.2f} -> '
            f'{max(end, 0.0):.2f} m'
        )
        # For the progress line: where this pass starts, and its strokes.
        self.pass_info = (self.current_depth, count, step, inward)

        if inward:
            label, name, sign = 'IN', 'Inward', 1.0
            self.log.info('========== INWARD SCAN ==========')
            direction = 1.0
            count, step = self.number_of_steps, self.step
            velocity, acceleration = self.velocity, self.acceleration
        else:
            label, name, sign = 'OUT', 'Outward', -1.0
            self.log.info('========== OUTWARD SCAN ==========')
            # Toward the other side from where J7 stands now.
            direction = (
                1.0 if self._inward_end_angle() + after_phase_twist < 0.0
                else -1.0
            )
            count, step, velocity, acceleration = self.outward

        if self.pause <= 0.0:
            return self._strokes_in_one(
                label, name, sign, direction, inward,
                count, step, velocity, acceleration,
            )

        for i in range(count):
            twist = direction * self.sweep_deg
            next_depth = self.current_depth + sign * step

            # Avoid ugly floating-point logging near zero.
            if abs(next_depth) < 1e-9:
                next_depth = 0.0

            self.log.info(
                f'[{label} {i + 1}/{count}] '
                f'{self.current_depth:.3f} -> {next_depth:.3f} m | '
                f'J7 {twist:+.1f} deg'
            )

            if not self.arm.move_tool_z_with_twist(
                distance=sign * step,
                twist_deg=twist,
                max_step=self.cartesian_step,
                velocity=velocity,
                acceleration=acceleration,
            ):
                self.log.error(f'{name} stroke {i + 1} failed.')
                return False

            self.current_depth = next_depth
            direction *= -1.0

            self.wait()
            self.checkpoint()

        if inward:
            self.log.info(
                f'Maximum scan depth reached: {self.current_depth:.3f} m'
            )

        return True

    def _strokes_in_one(
        self, label, name, sign, direction, inward, count, step, velocity,
        acceleration,
    ):
        """
        Run the strokes() pass as one trajectory (no --pause).

        Same strokes, but planned up front and executed together
        (arm.move_tool_z_strokes), so the arm does not stand at each
        J7 turnaround between them (~0.5 s each on the rig). The
        recorder checkpoints only after the pass.
        """
        strokes = []
        depth = self.current_depth

        for i in range(count):
            twist = direction * self.sweep_deg
            next_depth = depth + sign * step

            # Avoid ugly floating-point logging near zero.
            if abs(next_depth) < 1e-9:
                next_depth = 0.0

            # All planned at once: the progress line follows them as
            # they run (reporting()).
            self.log.debug(
                f'[{label} {i + 1}/{count}] '
                f'{depth:.3f} -> {next_depth:.3f} m | J7 {twist:+.1f} deg'
            )

            strokes.append((sign * step, twist))
            depth = next_depth
            direction *= -1.0

        if not self.arm.move_tool_z_strokes(
            strokes,
            max_step=self.cartesian_step,
            velocity=velocity,
            acceleration=acceleration,
        ):
            self.log.error(f'{name} strokes failed.')
            return False

        self.current_depth = depth
        self.checkpoint()

        if inward:
            self.log.info(
                f'Maximum scan depth reached: {self.current_depth:.3f} m'
            )

        return True

    # ---------------------------------------------------------
    # 3b. CHECK AT THE BOTTOM (only with scan(check=...))
    #
    # The inward pass alone has seen the whole depth range once, so
    # the pipeline can detect on it before the turnaround and the
    # outward pass spend their half of the scan. The arm stands still
    # while the readings are saved, and detection takes milliseconds.
    # ---------------------------------------------------------

    def check_at_bottom(self, check):
        """Save the readings so far and ask check(); None on failure."""
        self.set_phase('checking for laundry at the bottom')

        self.log.info('========== CHECK AT THE BOTTOM ==========')

        if self.recorder is not None and not self.recorder.save_blocking():
            self.log.error('Could not save the inward pass to check it.')
            return None

        answer = check()

        # A step (a number, not a bool) means found, come out at it.
        return answer if not isinstance(answer, bool) and answer else bool(answer)

    def set_outward_step(self, step, depth):
        """
        Come out at `step` instead; True, or False if it is invalid.

        The speed scales from the outward strokes' own, as
        scaled_stroke_speed does, so J7 sweeps as fast as on them.
        """
        count, old_step, velocity, acceleration = self.outward

        try:
            new_count, new_step = stroke_plan(depth, self.entry_depth, step)
        except ValueError as exc:
            self.log.error(str(exc))
            return False

        ratio = new_step / old_step
        self.outward = (
            new_count, new_step,
            min(1.0, velocity * ratio), min(1.0, acceleration * ratio),
        )
        self.log.info(
            f'Coming out at {new_step * 100:.1f} cm: {new_count} strokes.'
        )

        return True

    # ---------------------------------------------------------
    # 4. END SCAN (also performs the turnaround phase shift)
    #
    # Default: the baked precession sweep (scan/endcap.py,
    # _precession_end_scan). The quick scan ('none') only turns J7
    # around on the spot. With end_scan='bottom', the old detour
    # (_bottom_detour):
    #
    # The sensor is mounted at the wrist, and the end effector
    # keeps the arm from inserting far enough for the sensor to
    # see the very last part of the bucket. To cover that area,
    # raise the TCP angle by moving J1-J6 to the recorded BOTTOM
    # configuration (at BOTTOM's own J7 the boresight points down
    # and toward the closed end, ~48 deg below horizontal - see
    # config.BOTTOM_RECORDED), and sweep J7 throughout:
    #
    #   Entry  : tilt up to BOTTOM while sweeping J7 to the side
    #            OPPOSITE where the inward scan ended (relative
    #            to BOTTOM's own reference angle). This alone
    #            covers most of the sweep width.
    #
    #   At BOTTOM: one stationary full sweep_deg-wide J7 sweep,
    #            back to the side matching where the inward scan
    #            ended (relative to BOTTOM's own reference).
    #
    #   Exit   : tilt back down to the pre-detour J1-J6
    #            configuration while landing J7 directly on the
    #            turnaround target -- the side OPPOSITE where the
    #            inward scan ended (relative to INTER's reference).
    #
    # Landing the exit move on the turnaround target performs the
    # phase shift for free as part of the tilt-down motion, so no
    # separate stationary turnaround rotation is needed afterward:
    # the outward scan can begin immediately.
    # ---------------------------------------------------------

    def _inward_end_angle(self):
        """Return J7's nominal angle (deg) after the inward strokes."""
        # Started at -half_sweep: an odd number of strokes ends at
        # +half_sweep, an even number at -half_sweep.
        return self.half_sweep * (1.0 if self.number_of_steps % 2 else -1.0)

    def end_scan(self, end_scan, end_plan, depth, time_scale):
        """Cover the closed end and turn around; the phase twist, or None."""
        self.set_phase(
            'turning round at the bottom' if end_scan == 'none'
            else f'end scan ({end_scan})'
        )

        nominal_end_angle = self._inward_end_angle()

        self.log.info(f'Nominal wrist orientation: {nominal_end_angle:+.1f} deg')

        self.log.info(f'========== END SCAN ({end_scan}) ==========')

        pre_end_joints = self.arm.get_current_joints()

        if pre_end_joints is None:
            self.log.error('Could not read joint state before the end scan.')
            return None

        phase_sign = 1.0 if nominal_end_angle > 0.0 else -1.0

        # Turnaround target: opposite side from where the inward scan
        # ended, relative to INTER's reference. Computed here (rather
        # than executed as its own stationary move) because the end
        # scan's exit move performs it directly.
        phase_twist = -self.sweep_deg if nominal_end_angle > 0.0 else +self.sweep_deg

        if end_scan == 'none' and abs(self.outward[1] - self.step) > 1e-6:
            # The turnaround only staggers the outward strokes between
            # the inward ones; at a different step they do not line up
            # anyway, so come straight out from where J7 stands (7.3 s
            # saved on the rig, 2026-10-28).
            self.log.info(
                'No turnaround: the outward strokes are '
                f'{self.outward[1] * 100:.1f} cm, the inward '
                f'{self.step * 100:.1f} cm.'
            )
            return 0.0

        self.log.info(
            f'Turnaround phase shift (folded into end-scan exit): '
            f'{phase_twist:+.1f} deg'
        )

        if end_scan == 'bottom':
            success = _bottom_detour(
                self.arm,
                pre_end_joints,
                phase_sign,
                phase_twist,
                self.half_sweep,
                self.sweep_deg,
                self.rotation_velocity,
                self.rotation_acceleration,
                self.wait,
            )
        elif end_scan == 'none':
            # The quick scan: no end scan, only the turnaround, as a J7
            # turn on the spot (straight, collision-checked joint move),
            # so the outward strokes still interleave with the inward.
            turned = list(pre_end_joints)
            turned[6] += math.radians(phase_twist)
            success = self._quick_joint_move(turned)
        else:
            success = _precession_end_scan(
                self.arm,
                end_plan,
                depth,
                pre_end_joints,
                phase_twist,
                time_scale,
                on_replay=lambda active: self.mark_segment(
                    segments.END if active else segments.STROKES
                ),
            )

        if not success:
            return None

        self.wait()

        return phase_twist

    # ---------------------------------------------------------
    # 6. STOP RECORDING, SAVE
    #
    # Blocking is fine (and necessary) here: the arm is stationary,
    # and this is the one save that must land.
    # ---------------------------------------------------------

    def stop_and_save(self):
        """Stop recording and save the recorder's CSV."""
        if not self.set_recording(False):
            return False

        if self.recorder is not None:
            self.recorder.save_blocking()
            self.log.info('Final scan checkpoint saved.')

        return True

    # ---------------------------------------------------------
    # 7. LEAVE: J7 back to INTER's (inside), straight out, INTER
    # ---------------------------------------------------------

    def leave(self, stay_inside=None):
        """
        Turn J7 back, move straight out of the bucket, go to INTER.

        From wherever the strokes got to: entry_depth after the
        outward pass, the full depth after a check at the bottom.
        stay_inside: override the scan's own (None keeps it).
        """
        self.set_phase('leaving the bucket')

        if stay_inside is None:
            stay_inside = self.stay_inside

        if stay_inside and self.entry_depth > 0.0:
            # Back to the entry pose, where the next scan starts
            # (start_inside): J7 turned back, still entry_depth in.
            # Solved from INTER, not just J7 unwound: the strokes'
            # Cartesian moves drift a few mm / deg (6 mm, 2 deg on
            # the fake controller), which leaving via INTER hides.
            self.log.info('========== STAY INSIDE (entry pose) ==========')

            target = entry_joints(self.arm, self.entry_depth)

            if target is None or not self._quick_joint_move(list(target)):
                self.log.error('Could not move back to the entry pose.')
                return False

            return True

        self.log.info('========== RETURN TO INTER ==========')

        # From the entry depth: one move back to INTER, J7 unwinding
        # on the way, where it is clear (~1.3 s against 6.6 s for
        # unwinding, straight out, then INTER; rig, 2026-10-28).
        if (
            0.0 < self.current_depth <= self.entry_depth + 1e-3
            and self._quick_joint_move(config.get_named_pose('inter'))
        ):
            self.current_depth = 0.0
            self.wait()
            return True

        # (Not > 0: the strokes' sums leave ~1e-17 with no entry depth.)
        if self.current_depth > 1e-6:
            current = self.arm.get_current_joints()

            if current is None:
                return False

            unwound = list(current)
            unwound[6] = config.get_named_pose('inter')[6]

            if not self.arm.move_joints_linear(unwound):
                self.log.error(
                    'Could not turn J7 back before leaving the bucket.'
                )
                return False

            if not self._tool_z(-self.current_depth):
                self.log.error('Straight exit from the bucket failed.')
                return False

            self.current_depth = 0.0

        if not go_to(self.arm, 'inter'):
            self.log.error('Return to INTER failed.')
            return False

        self.wait()

        return True

    # ---------------------------------------------------------
    # 8. COMPLETE
    # ---------------------------------------------------------

    def report(self):
        """Log the end of the scan."""
        if abs(self.current_depth) < 1e-9:
            self.current_depth = 0.0

        for line in (
            '========================================',
            'SCAN COMPLETE',
            f'Took {time.monotonic() - self.started:.1f} s',
            f'Final nominal depth: {self.current_depth:.4f} m',
            '========================================',
        ):
            self.log.info(line)

        return True


def end_scan_only(
    arm,
    recorder=None,
    depth=DEFAULT_DEPTH_M,
    end_plan=None,
    end_scan_time_scale=1.0,
    cartesian_step=DEFAULT_CARTESIAN_STEP_M,
    entry_depth=DEFAULT_ENTRY_DEPTH_M,
    start_inside=False,
    **_scan_options,
):
    """
    Run the end scan on its own: straight in, the end scan, straight out.

    For after a quick scan that found nothing: that scan's strokes plus
    these end-scan readings are a full scan's readings (scan/
    segments.py), without swinging through the bucket twice. The
    recorder is NOT cleared - its quick-scan points stay - and only the
    end scan's replay is recorded, labelled END; recording is off
    afterwards. The caller pins the CSV and saves (pipeline).

    start_inside: the quick scan stayed in (scan(stay_inside=True)), so
    start from its entry pose, entry_depth in, instead of INTER;
    refuses, moving nothing, unless the flange is there.

    MOVES THE ARM. Returns True on success.
    """
    from .endcap import run_plan

    log = arm.get_logger()

    if end_plan is None:
        log.error('The end-scan pass needs a baked end scan (laundry plan bake).')
        return False

    def set_recording(on):
        return recorder is None or recorder.set_recording(on)

    def replay(active):
        if recorder is None:
            return

        recorder.set_segment(segments.END if active else segments.STROKES)
        recorder.set_recording(active)

    if not set_recording(False):
        log.error("Could not stop scan_recorder_node's recording.")
        return False

    log.info('========== END SCAN ONLY: STRAIGHT IN ==========')

    if start_inside:
        if not at_entry_pose(arm, entry_depth):
            return False

        inserted = entry_depth

    else:
        if not go_to(arm, 'inter'):
            log.error('Move to INTER failed.')
            return False

        inserted = 0.0

    if not arm.move_tool_z(
        depth - inserted,
        max_step=cartesian_step,
        velocity=STRAIGHT_IN_OUT_VELOCITY,
        acceleration=STRAIGHT_IN_OUT_VELOCITY,
    ):
        log.error(f'Straight insertion to {depth:.3f} m failed.')
        return False

    log.info('========== END SCAN ONLY: END SCAN ==========')

    if not run_plan(
        arm, end_plan, depth, time_scale=end_scan_time_scale,
        on_replay=replay,
    ):
        log.error('End scan failed.')
        return False

    log.info('========== END SCAN ONLY: STRAIGHT OUT ==========')

    if not arm.move_tool_z(
        -depth,
        max_step=cartesian_step,
        velocity=STRAIGHT_IN_OUT_VELOCITY,
        acceleration=STRAIGHT_IN_OUT_VELOCITY,
    ):
        log.error('Straight retraction from the end scan failed.')
        return False

    if not go_to(arm, 'inter'):
        log.error('Return to INTER failed.')
        return False

    return True


def _precession_end_scan(
    arm, plan, depth, pre_end_joints, phase_twist, time_scale,
    on_replay=None,
):
    """
    Replay the baked precession end scan, then turn J7 for the way out.

    See scan/endcap.py. Every motion here is planner-free and
    collision-checked: a straight joint move onto the plan, the plan
    itself, and a straight joint move to the stroke configuration the
    inward pass ended at with J7 turned by phase_twist - the same
    turnaround target the BOTTOM detour's exit landed on.
    """
    from .endcap import run_plan

    arm.get_logger().info(
        f'Precession end scan: {plan.duration_s:.1f} s baked trajectory '
        f'(rings {", ".join(f"{a:g}deg" for a, _lo, _hi in plan.rings)})'
        + (f', replayed at {time_scale:g}x speed' if time_scale < 1.0 else '')
    )

    if not run_plan(
        arm, plan, depth, time_scale=time_scale, on_replay=on_replay
    ):
        arm.get_logger().error('Precession end scan failed.')
        return False

    exit_joints = list(pre_end_joints)
    exit_joints[6] = pre_end_joints[6] + math.radians(phase_twist)

    arm.get_logger().info(
        'Returning to the stroke configuration, turned around for the '
        'outward scan.'
    )

    if not arm.move_joints_linear(exit_joints, time_scale=time_scale):
        arm.get_logger().error('Return from the end scan failed.')
        return False

    return True


def _bottom_detour(
    arm,
    pre_bottom_joints,
    phase_sign,
    phase_twist,
    half_sweep,
    sweep_deg,
    rotation_velocity,
    rotation_acceleration,
    wait_between_movements,
):
    """
    Cover the closed end the old way: tilt to BOTTOM, sweep J7, tilt back.

    Only used with end_scan='bottom'. Kept for comparison and as a
    fallback; the precession end scan sees much more of the closed
    end (scan/endcap.py), and its motion is baked rather than
    planned, so it is identical every run. These moves go through
    move_joints(), i.e. Pilz PTP if loaded, else OMPL - which takes a
    different route every time.
    """
    # Entry: sweep to the side OPPOSITE nominal_end_angle,
    # relative to BOTTOM's own reference angle.
    bottom = config.get_named_pose('bottom')
    entry_joints = list(bottom)

    entry_joints[6] = (
        bottom[6]
        + math.radians(-phase_sign * half_sweep)
    )

    arm.get_logger().info(
        'Entering BOTTOM configuration (sweeping while tilting).'
    )

    success = arm.move_joints(
        entry_joints,
        velocity=rotation_velocity,
        acceleration=rotation_acceleration,
    )

    if not success:

        arm.get_logger().error(
            'Move to BOTTOM configuration failed.'
        )

        return False

    wait_between_movements()

    # Stationary full-width sweep at BOTTOM, back to the side
    # matching nominal_end_angle, relative to BOTTOM's reference.
    bottom_sweep_twist = phase_sign * sweep_deg

    arm.get_logger().info(
        f'Stationary BOTTOM sweep: '
        f'{bottom_sweep_twist:+.1f} deg'
    )

    success = arm.rotate_joint7(
        delta_deg=bottom_sweep_twist,
        velocity=rotation_velocity,
        acceleration=rotation_acceleration,
    )

    if not success:

        arm.get_logger().error(
            'Stationary BOTTOM sweep failed.'
        )

        return False

    wait_between_movements()

    # Exit: revert J1-J6 to the pre-detour configuration while
    # landing J7 on the turnaround target, so the outward scan
    # can start immediately with no further stationary rotation.
    exit_joints = list(pre_bottom_joints)

    exit_joints[6] = (
        pre_bottom_joints[6]
        + math.radians(phase_twist)
    )

    arm.get_logger().info(
        'Reverting to pre-BOTTOM tilt '
        '(already turned around for the outward scan).'
    )

    success = arm.move_joints(
        exit_joints,
        velocity=rotation_velocity,
        acceleration=rotation_acceleration,
    )

    if not success:

        arm.get_logger().error(
            'Revert from BOTTOM configuration failed.'
        )

        return False

    return True


# =============================================================
# TERMINAL INTERFACE
#
# The argument definitions live here, next to the defaults, so
# `laundry scan`, `laundry run` and `laundry baseline collect` all
# accept (and forward) exactly the same scan options.
# =============================================================

def _stroke_spacing(text):
    """Parse --step, rejecting anything over strokes.MAX_STEP_M."""
    value = float(text)

    if not 0.0 < value <= MAX_STEP_M + 1e-9:
        raise argparse.ArgumentTypeError(
            f'must be more than 0 and at most {MAX_STEP_M} m '
            f'({MAX_STEP_M * 100:g} cm); got {text}'
        )

    return value


def add_quick_full_flags(parser):
    """Add --quick / --full: shorthands for --end-scan none / precession."""
    parser.add_argument(
        '--quick', dest='end_scan', action='store_const', const='none',
        help='The quick scan: same as --end-scan none.',
    )
    parser.add_argument(
        '--full', dest='end_scan', action='store_const', const='precession',
        help='The full scan: same as --end-scan precession.',
    )


def _coarse_steps(text):
    """Parse --coarse: comma-separated stroke spacings, or 'none'."""
    if text.strip().lower() == 'none':
        return ()

    return tuple(_stroke_spacing(part) for part in text.split(','))


def add_search_arguments(parser):
    """Add --search, --coarse and --no-bottom-check (`laundry run` / `clear`)."""
    default = ','.join(f'{step:g}' for step in DEFAULT_COARSE_STEPS_M)

    parser.add_argument(
        '--search',
        choices=SEARCHES,
        default=DEFAULT_SEARCH,
        help=(
            "Quick scans: 'in-out' goes in once - the coarsest --coarse "
            'step on the way in, a check at the bottom, the end scan there '
            'if nothing was found, and --step on the way out; '
            "'ladder' runs a whole coarse pass, then a whole --step pass, "
            f'then the end scan alone (default: {DEFAULT_SEARCH}).'
        ),
    )

    parser.add_argument(
        '--coarse',
        type=_coarse_steps,
        default=DEFAULT_COARSE_STEPS_M,
        help=(
            'Quick scans: first scan at these stroke spacings in metres, '
            'coarsest first and each over --step, and only scan at --step '
            "if they find nothing; 'none' scans at --step only "
            f'(default: {default}).'
        ),
    )

    parser.add_argument(
        '--no-bottom-check',
        dest='bottom_check',
        action='store_false',
        help=(
            "Quick scans: do not detect on the --step pass's inward "
            'strokes at the deepest point (laundry found there skips the '
            'turnaround and outward strokes, and the arm goes straight '
            'out to grab it; coarse passes always run whole).'
        ),
    )


def coarse_steps_from_args(args):
    """Return --coarse checked against --step; ValueError if not coarser."""
    for step in args.coarse:
        if not step > args.step:
            raise ValueError(
                f'--coarse {step:g} is not coarser than --step {args.step:g}.'
            )

    return tuple(sorted(args.coarse, reverse=True))


def add_scan_arguments(parser, end_scan_default=DEFAULT_END_SCAN):
    """
    Add every scan() tuning option to an argparse parser.

    end_scan_default: the command's own --end-scan default (`laundry
    run` defaults to the quick scan, 'none').
    """
    parser.add_argument(
        '--end-scan',
        choices=END_SCANS,
        default=end_scan_default,
        help=(
            "How to cover the closed end: 'precession' (the full scan: "
            "the baked coning sweep), 'none' (the quick scan: no end "
            "scan, no tilting; judged against the baselines' strokes) "
            "or 'bottom' (the old BOTTOM-pose detour). Default: "
            f"'{end_scan_default}'."
        ),
    )

    add_quick_full_flags(parser)

    parser.add_argument(
        '--end-plan',
        type=str,
        default=None,
        help='Baked end-scan plan (default: <repo>/scan_plans/endcap.yaml).',
    )

    parser.add_argument(
        '--end-scan-speed',
        type=float,
        default=1.0,
        help=(
            'Replay the precession end scan at this fraction of its baked '
            'speed, (0, 1] (default: 1). Use e.g. 0.3 for first runs on '
            'the rig.'
        ),
    )

    parser.add_argument(
        '--depth',
        type=float,
        default=DEFAULT_DEPTH_M,
        help=(
            'Maximum insertion depth in metres '
            f'(default: {DEFAULT_DEPTH_M}).'
        ),
    )

    parser.add_argument(
        '--entry-depth',
        type=float,
        default=DEFAULT_ENTRY_DEPTH_M,
        help=(
            'Straight insertion from INTER before the strokes start, '
            f'not recorded, in metres (default: {DEFAULT_ENTRY_DEPTH_M}).'
        ),
    )

    parser.add_argument(
        '--step',
        type=_stroke_spacing,
        default=DEFAULT_STEP_M,
        help=(
            'Largest distance between scan strokes in metres, at most '
            f'{MAX_STEP_M} (default: {DEFAULT_STEP_M}); the strokes are '
            'spread evenly from --entry-depth to --depth. Coarser is '
            'faster but misses small items (12 cm: about half the '
            'socks); the stroke speed scales with it unless --velocity '
            'is given.'
        ),
    )

    parser.add_argument(
        '--sweep',
        type=float,
        default=DEFAULT_SWEEP_DEG,
        help=(
            'J7 rotation per scan stroke in degrees '
            f'(default: {DEFAULT_SWEEP_DEG:g}).'
        ),
    )

    parser.add_argument(
        '--velocity',
        dest='scan_velocity',
        type=float,
        default=None,
        help=(
            'MoveIt velocity scaling for the scan strokes (default: '
            f'{DEFAULT_VELOCITY} at the default --step, scaled with it: '
            f'{scaled_stroke_speed(MAX_STEP_M):g} at {MAX_STEP_M:g}).'
        ),
    )

    parser.add_argument(
        '--acceleration',
        dest='scan_acceleration',
        type=float,
        default=None,
        help=(
            'MoveIt acceleration scaling for the scan strokes (default: '
            f'{DEFAULT_ACCELERATION} at the default --step, scaled with '
            'it like --velocity).'
        ),
    )

    parser.add_argument(
        '--rotation-velocity',
        type=float,
        default=DEFAULT_ROTATION_VELOCITY,
        help=(
            'MoveIt velocity scaling for the stationary '
            f'(non-linear) J7 rotations (default: {DEFAULT_ROTATION_VELOCITY}).'
        ),
    )

    parser.add_argument(
        '--rotation-acceleration',
        type=float,
        default=DEFAULT_ROTATION_ACCELERATION,
        help=(
            'MoveIt acceleration scaling for the stationary '
            '(non-linear) J7 rotations '
            f'(default: {DEFAULT_ROTATION_ACCELERATION}).'
        ),
    )

    parser.add_argument(
        '--cartesian-step',
        type=float,
        default=DEFAULT_CARTESIAN_STEP_M,
        help=(
            'MoveIt Cartesian interpolation step '
            f'in metres (default: {DEFAULT_CARTESIAN_STEP_M}).'
        ),
    )

    parser.add_argument(
        '--pause',
        type=float,
        default=DEFAULT_PAUSE_SEC,
        help=(
            'Optional pause between motions in seconds '
            f'(default: {DEFAULT_PAUSE_SEC:g}).'
        ),
    )

    parser.add_argument(
        '--save-interval',
        type=float,
        default=DEFAULT_SAVE_INTERVAL_SEC,
        help=(
            'Seconds between CSV checkpoint saves requested from '
            'scan_recorder_node during the scan '
            f'(default: {DEFAULT_SAVE_INTERVAL_SEC:g}).'
        ),
    )


def scan_kwargs_from_args(args):
    """
    Turn parsed add_scan_arguments() options into scan() keywords.

    Loads the end-scan plan when the precession end scan is selected,
    so a missing or mismatched plan is reported before the arm moves.
    """
    end_plan = None

    if args.end_scan == 'none':
        # The quick scan itself needs no plan; `run` and `clear` use it
        # for their end-scan-only pass when a quick scan finds nothing.
        from .endcap import default_plan_path, EndcapPlan

        path = args.end_plan or default_plan_path()

        if os.path.isfile(path):
            end_plan = EndcapPlan.load(path)

    if args.end_scan == 'precession':
        from .endcap import default_plan_path, EndcapPlan

        path = args.end_plan or default_plan_path()

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'No baked end-scan plan at {path}. Bake one '
                '(`laundry plan bake`, with MoveIt running) or pass '
                '--end-scan bottom.'
            )

        end_plan = EndcapPlan.load(path)

        if abs(end_plan.depth_m - args.depth) > 1e-6:
            raise ValueError(
                f'{path} was baked for --depth {end_plan.depth_m:.3f}; '
                f'this scan uses {args.depth:.3f}. Re-bake with '
                f'`laundry plan bake --depth {args.depth:.3f}`.'
            )

    return {
        'end_scan': args.end_scan,
        'end_plan': end_plan,
        'end_scan_time_scale': args.end_scan_speed,
        'depth': args.depth,
        'entry_depth': args.entry_depth,
        'step': args.step,
        'sweep_deg': args.sweep,
        # Unless given, both scale with the step (scaled_stroke_speed).
        'velocity': (
            args.scan_velocity if args.scan_velocity is not None
            else scaled_stroke_speed(args.step, DEFAULT_VELOCITY)
        ),
        'acceleration': (
            args.scan_acceleration if args.scan_acceleration is not None
            else scaled_stroke_speed(args.step, DEFAULT_ACCELERATION)
        ),
        'rotation_velocity': args.rotation_velocity,
        'rotation_acceleration': args.rotation_acceleration,
        'cartesian_step': args.cartesian_step,
        'pause': args.pause,
        'save_interval': args.save_interval,
    }
