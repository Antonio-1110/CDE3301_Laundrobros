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
import math
import os

import rclpy

from . import segments
from .strokes import MAX_STEP_M, stroke_plan
from .. import config
from ..arm.transfers import go_to

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
DEFAULT_VELOCITY = 0.03
DEFAULT_ACCELERATION = 0.03


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
DEFAULT_CARTESIAN_STEP_M = 0.005
DEFAULT_PAUSE_SEC = 0.0
DEFAULT_SAVE_INTERVAL_SEC = 5.0

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
    )

    run.log_configuration(depth, max_step=step)

    # The sequence (see Notes above); each step logs its own failure.
    if not (
        run.enter()                                     # 1, 1b
        and run.initial_offset()                        # 2
        and run.strokes(inward=True)                    # 3
    ):
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
        cartesian_step, pause, save_interval,
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

        # Nominal insertion depth, for the log only.
        self.current_depth = 0.0
        self.last_save_time = arm.get_clock().now()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def log_configuration(self, depth, max_step):
        """Log the scan's parameters before anything moves."""
        for line in (
            '========================================',
            'XARM7 SCAN SEQUENCE',
            f'Depth            : {depth:.3f} m',
            f'Entry depth      : {self.entry_depth:.3f} m',
            f'Linear step      : {self.step:.4f} m (at most {max_step:.3f})',
            f'Number of strokes: {self.number_of_steps}',
            f'Wrist sweep      : {self.sweep_deg:.1f} deg',
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

    def _tool_z(self, distance):
        """Move straight along tool Z at the stroke speed, without twist."""
        return self.arm.move_tool_z(
            distance,
            max_step=self.cartesian_step,
            velocity=self.velocity,
            acceleration=self.acceleration,
        )

    # ---------------------------------------------------------
    # 1. INTER, then straight in; recording starts inside
    #
    # INTER establishes the known starting pose (and its nominal J7
    # orientation of 0 degrees) that the rest of the sequence,
    # starting with the initial offset, assumes. Nothing is
    # recorded until the sensor is inside the bucket.
    # ---------------------------------------------------------

    def enter(self):
        """Go to INTER, move straight in by entry_depth, start recording."""
        self.log.info('========== MOVE TO INTER ==========')

        if not self.set_recording(False):
            return False

        # Baked transfer from HOME/DROP, else a straight checked joint
        # move, else the planner (arm/transfers.go_to).
        if not go_to(self.arm, 'inter'):
            self.log.error('Move to INTER failed.')
            return False

        self.wait()

        # 1b. Straight in along tool Z, J7 unchanged.
        if self.entry_depth > 0.0:
            self.log.info(
                f'========== ENTRY ({self.entry_depth:.3f} m) =========='
            )

            if not self._tool_z(self.entry_depth):
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
        """Turn J7 to -half the sweep, on the spot."""
        self.log.info('========== INITIAL OFFSET ==========')
        self.log.info(f'Stationary J7 rotation: {-self.half_sweep:+.1f} deg')

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
    # NO stationary resets between strokes.
    # ---------------------------------------------------------

    def strokes(self, inward, after_phase_twist=0.0):
        """Run the inward strokes, or the outward ones after the end scan."""
        if inward:
            label, name, sign = 'IN', 'Inward', 1.0
            self.log.info('========== INWARD SCAN ==========')
            direction = 1.0
        else:
            label, name, sign = 'OUT', 'Outward', -1.0
            self.log.info('========== OUTWARD SCAN ==========')
            direction = 1.0 if after_phase_twist < 0.0 else -1.0

        for i in range(self.number_of_steps):
            twist = direction * self.sweep_deg
            next_depth = self.current_depth + sign * self.step

            # Avoid ugly floating-point logging near zero.
            if abs(next_depth) < 1e-9:
                next_depth = 0.0

            self.log.info(
                f'[{label} {i + 1}/{self.number_of_steps}] '
                f'{self.current_depth:.3f} -> {next_depth:.3f} m | '
                f'J7 {twist:+.1f} deg'
            )

            if not self.arm.move_tool_z_with_twist(
                distance=sign * self.step,
                twist_deg=twist,
                max_step=self.cartesian_step,
                velocity=self.velocity,
                acceleration=self.acceleration,
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

    def end_scan(self, end_scan, end_plan, depth, time_scale):
        """Cover the closed end and turn around; the phase twist, or None."""
        # Started at -half_sweep: an odd number of strokes ends at
        # +half_sweep, an even number at -half_sweep.
        if self.number_of_steps % 2 == 1:
            nominal_end_angle = +self.half_sweep
        else:
            nominal_end_angle = -self.half_sweep

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
            success = self.arm.move_joints_linear(turned)
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

    def leave(self):
        """Turn J7 back, move straight out of the bucket, go to INTER."""
        self.log.info('========== RETURN TO INTER ==========')

        if self.entry_depth > 0.0:
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

            if not self._tool_z(-self.entry_depth):
                self.log.error('Straight exit from the bucket failed.')
                return False

            self.current_depth -= self.entry_depth

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
            f'Final nominal depth: {self.current_depth:.4f} m',
            '========================================',
        ):
            self.log.info(line)

        return True


# Speed scaling for the end-scan-only pass's straight moves in and
# out: nothing is recorded or twisted on them.
END_ONLY_VELOCITY = 0.1


def end_scan_only(
    arm,
    recorder=None,
    depth=DEFAULT_DEPTH_M,
    end_plan=None,
    end_scan_time_scale=1.0,
    cartesian_step=DEFAULT_CARTESIAN_STEP_M,
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

    if not go_to(arm, 'inter'):
        log.error('Move to INTER failed.')
        return False

    if not arm.move_tool_z(
        depth,
        max_step=cartesian_step,
        velocity=END_ONLY_VELOCITY,
        acceleration=END_ONLY_VELOCITY,
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
        velocity=END_ONLY_VELOCITY,
        acceleration=END_ONLY_VELOCITY,
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
