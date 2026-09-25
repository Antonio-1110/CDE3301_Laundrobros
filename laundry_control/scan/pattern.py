#!/usr/bin/env python3

"""
The bucket scan: helical tool-Z strokes with a J7 twist, plus a BOTTOM detour.

scan() only MOVES the arm. The ToF readings are captured by the
separate scan_recorder_node (scan/recorder_node.py) through TF while
this runs; `recorder` is only used to checkpoint and finally save
that node's CSV.

Run from the terminal as `laundry scan` (cli.py).
"""

import math
import os

import rclpy

from ..arm.transfers import go_to
from ..config import BOTTOM

# Defaults for every scan parameter, shared by scan() and the CLI so
# the two can never drift apart. Baselines and detection scans MUST
# use the same values: the detector's learned residual field is only
# valid for the trajectory it was learned on.
DEFAULT_DEPTH_M = 0.42
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
DEFAULT_VELOCITY = 0.03
DEFAULT_ACCELERATION = 0.03
DEFAULT_ROTATION_VELOCITY = 0.5
DEFAULT_ROTATION_ACCELERATION = 0.5
DEFAULT_CARTESIAN_STEP_M = 0.005
DEFAULT_PAUSE_SEC = 0.0
DEFAULT_SAVE_INTERVAL_SEC = 5.0

# How the closed end is covered: 'precession' (default, baked - see
# scan/endcap.py) or 'bottom' (the old recorded-pose detour, only
# when asked for).
END_SCANS = ('precession', 'bottom')
DEFAULT_END_SCAN = 'precession'


def scan(
    arm,
    depth=DEFAULT_DEPTH_M,
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

    step:
        Linear distance travelled during each scan stroke.

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
        scan/endcap.py); 'bottom' runs the old recorded-pose BOTTOM
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

    Start:

        J7 = 0 deg

    Initial positioning:

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

    """
    # =========================================================
    # Validate input
    # =========================================================

    if depth <= 0.0:
        arm.get_logger().error(
            'depth must be greater than zero.'
        )
        return False

    if end_scan not in END_SCANS:
        arm.get_logger().error(
            f'end_scan must be one of {END_SCANS}; got {end_scan!r}.'
        )
        return False

    if end_scan == 'precession' and end_plan is None:
        arm.get_logger().error(
            "end_scan='precession' needs a baked plan (laundry plan bake)."
        )
        return False

    if not 0.0 < end_scan_time_scale <= 1.0:
        arm.get_logger().error('end_scan_time_scale must be in (0, 1].')
        return False

    if step <= 0.0:
        arm.get_logger().error(
            'step must be greater than zero.'
        )
        return False

    if step > depth:
        arm.get_logger().error(
            'step cannot be greater than depth.'
        )
        return False

    if sweep_deg <= 0.0:
        arm.get_logger().error(
            'sweep_deg must be greater than zero.'
        )
        return False

    if not 0.0 < velocity <= 1.0:
        arm.get_logger().error(
            'velocity must be in the range (0, 1].'
        )
        return False

    if not 0.0 < acceleration <= 1.0:
        arm.get_logger().error(
            'acceleration must be in the range (0, 1].'
        )
        return False

    if not 0.0 < rotation_velocity <= 1.0:
        arm.get_logger().error(
            'rotation_velocity must be in the range (0, 1].'
        )
        return False

    if not 0.0 < rotation_acceleration <= 1.0:
        arm.get_logger().error(
            'rotation_acceleration must be in the range (0, 1].'
        )
        return False

    # =========================================================
    # Determine number of strokes
    # =========================================================

    step_count_float = depth / step

    number_of_steps = round(
        step_count_float
    )

    if not math.isclose(
        step_count_float,
        number_of_steps,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        arm.get_logger().error(
            'depth must currently be an exact multiple '
            'of step.'
        )

        arm.get_logger().error(
            f'depth = {depth:.4f} m'
        )

        arm.get_logger().error(
            f'step  = {step:.4f} m'
        )

        return False

    # =========================================================
    # Derived values
    # =========================================================

    half_sweep = sweep_deg / 2.0

    # Used only for logging our nominal insertion depth.
    current_depth = 0.0

    # =========================================================
    # Scan configuration
    # =========================================================

    arm.get_logger().info(
        '========================================'
    )

    arm.get_logger().info(
        'XARM7 SCAN SEQUENCE'
    )

    arm.get_logger().info(
        f'Depth            : {depth:.3f} m'
    )

    arm.get_logger().info(
        f'Linear step      : {step:.3f} m'
    )

    arm.get_logger().info(
        f'Number of strokes: {number_of_steps}'
    )

    arm.get_logger().info(
        f'Wrist sweep      : {sweep_deg:.1f} deg'
    )

    arm.get_logger().info(
        f'Initial offset   : {-half_sweep:+.1f} deg'
    )

    arm.get_logger().info(
        '========================================'
    )

    # =========================================================
    # Helper
    # =========================================================

    def wait_between_movements():
        """
        Pause between motions.

        Spins the node (rather than a plain time.sleep) so pending
        service futures (e.g. recorder.save_async()) get processed
        even if this stroke happens to be the last one before a
        long pause. ToF capture itself runs in scan_recorder_node,
        a separate process, so it needs no help from this spin.
        """
        if pause <= 0.0:
            return

        deadline = arm.get_clock().now() + rclpy.duration.Duration(
            seconds=pause
        )

        while arm.get_clock().now() < deadline:
            rclpy.spin_once(arm, timeout_sec=0.05)

    last_save_time = arm.get_clock().now()

    def maybe_checkpoint_save():
        """
        Checkpoint the recorder's CSV every save_interval seconds.

        Periodically ask scan_recorder_node to checkpoint its CSV,
        at a much slower cadence than point capture/publishing, via
        a fire-and-forget service call (see ScanRecorderClient).
        """
        nonlocal last_save_time

        if recorder is None or save_interval <= 0.0:
            return

        now = arm.get_clock().now()

        if (now - last_save_time).nanoseconds * 1e-9 >= save_interval:
            recorder.save_async()
            last_save_time = now

    # =========================================================
    # 1. MOVE TO INTER
    #
    # Establishes the known starting pose (and its nominal J7
    # orientation of 0 degrees) that the rest of the sequence,
    # starting with the initial offset below, assumes.
    # =========================================================

    arm.get_logger().info(
        '========== MOVE TO INTER =========='
    )

    # Baked transfer from HOME/DROP, else a straight checked joint
    # move, else the planner (arm/transfers.go_to).
    success = go_to(arm, 'inter')

    if not success:

        arm.get_logger().error(
            'Move to INTER failed.'
        )

        return False

    wait_between_movements()

    # =========================================================
    # 2. INITIAL J7 OFFSET
    #
    # INTER position is assumed to have the desired nominal
    # wrist orientation of 0 degrees.
    #
    # Move:
    #
    #        0 -> -75
    #
    # for the default 150-degree sweep.
    #
    # rotate_joint7() gets the CURRENT J1-J7 state and keeps
    # J1-J6 at their current positions.
    # =========================================================

    arm.get_logger().info(
        '========== INITIAL OFFSET =========='
    )

    arm.get_logger().info(
        f'Stationary J7 rotation: '
        f'{-half_sweep:+.1f} deg'
    )

    success = arm.rotate_joint7(
        delta_deg=-half_sweep,
        velocity=rotation_velocity,
        acceleration=rotation_acceleration,
    )

    if not success:

        arm.get_logger().error(
            'Initial J7 positioning failed.'
        )

        return False

    wait_between_movements()

    # =========================================================
    # 3. INWARD SCAN
    #
    # Starting orientation:
    #
    #        -75 deg
    #
    # Therefore first movement is:
    #
    #        -75 -> +75
    #
    #        +150 deg
    #
    # Then alternate:
    #
    #        +150
    #        -150
    #        +150
    #        -150
    #        ...
    # =========================================================

    arm.get_logger().info(
        '========== INWARD SCAN =========='
    )

    inward_direction = +1.0

    for i in range(number_of_steps):

        twist = (
            inward_direction
            * sweep_deg
        )

        next_depth = (
            current_depth
            + step
        )

        arm.get_logger().info(
            f'[IN {i + 1}/{number_of_steps}] '
            f'{current_depth:.3f} -> '
            f'{next_depth:.3f} m | '
            f'J7 {twist:+.1f} deg'
        )

        success = arm.move_tool_z_with_twist(
            distance=step,
            twist_deg=twist,
            max_step=cartesian_step,
            velocity=velocity,
            acceleration=acceleration,
        )

        if not success:

            arm.get_logger().error(
                f'Inward stroke {i + 1} failed.'
            )

            return False

        current_depth = next_depth

        # Alternate:
        #
        # +150 -> -150 -> +150 ...
        inward_direction *= -1.0

        wait_between_movements()
        maybe_checkpoint_save()

    # =========================================================
    # Inward endpoint
    # =========================================================

    arm.get_logger().info(
        f'Maximum scan depth reached: '
        f'{current_depth:.3f} m'
    )

    # Because we started at -half_sweep:
    #
    # Odd number of strokes:
    #
    #     -75 -> +75
    #     ...
    #     final = +75
    #
    # Even number:
    #
    #     final = -75

    if number_of_steps % 2 == 1:

        nominal_end_angle = +half_sweep

    else:

        nominal_end_angle = -half_sweep

    arm.get_logger().info(
        f'Nominal wrist orientation: '
        f'{nominal_end_angle:+.1f} deg'
    )

    # =========================================================
    # 4. END SCAN (also performs the turnaround phase shift)
    #
    # Default: the baked precession sweep (scan/endcap.py,
    # _precession_end_scan). With end_scan='bottom', the old detour
    # below (_bottom_detour):
    #
    # The sensor is mounted at the wrist, and the end effector
    # keeps the arm from inserting far enough for the sensor to
    # see the very last part of the bucket. To cover that area,
    # raise the TCP angle by moving J1-J6 to the recorded BOTTOM
    # configuration (at BOTTOM's own J7 the boresight points down
    # and toward the closed end, ~48 deg below horizontal - see
    # config.BOTTOM), and sweep J7 throughout:
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
    # =========================================================

    arm.get_logger().info(
        f'========== END SCAN ({end_scan}) =========='
    )

    pre_bottom_joints = arm.get_current_joints()

    if pre_bottom_joints is None:

        arm.get_logger().error(
            'Could not read joint state before the end scan.'
        )

        return False

    phase_sign = 1.0 if nominal_end_angle > 0.0 else -1.0

    # Turnaround target: opposite side from where the inward scan
    # ended, relative to INTER's reference. Computed here (rather
    # than executed as its own stationary move) because the
    # detour's exit move performs it directly.
    if nominal_end_angle > 0.0:

        phase_twist = -sweep_deg

    else:

        phase_twist = +sweep_deg

    arm.get_logger().info(
        f'Turnaround phase shift (folded into end-scan exit): '
        f'{phase_twist:+.1f} deg'
    )

    if end_scan == 'bottom':
        success = _bottom_detour(
            arm,
            pre_bottom_joints,
            phase_sign,
            phase_twist,
            half_sweep,
            sweep_deg,
            rotation_velocity,
            rotation_acceleration,
            wait_between_movements,
        )
    else:
        success = _precession_end_scan(
            arm,
            end_plan,
            depth,
            pre_bottom_joints,
            phase_twist,
            end_scan_time_scale,
        )

    if not success:
        return False

    wait_between_movements()

    # =========================================================
    # 5. OUTWARD SCAN
    #
    # After the phase shift, start by rotating in the
    # OPPOSITE direction.
    #
    # Example for 9 strokes:
    #
    # inward ends:
    #
    #       +75
    #
    # phase:
    #
    #       +75 -> -75      (-150)
    #
    # first outward:
    #
    #       -75 -> +75      (+150)
    #
    # second outward:
    #
    #       +75 -> -75      (-150)
    #
    # etc.
    #
    # There are NO additional stationary resets.
    # =========================================================

    arm.get_logger().info(
        '========== OUTWARD SCAN =========='
    )

    if phase_twist < 0.0:

        outward_direction = +1.0

    else:

        outward_direction = -1.0

    for i in range(number_of_steps):

        twist = (
            outward_direction
            * sweep_deg
        )

        next_depth = (
            current_depth
            - step
        )

        # Avoid ugly floating-point logging near zero.
        if abs(next_depth) < 1e-9:
            next_depth = 0.0

        arm.get_logger().info(
            f'[OUT {i + 1}/{number_of_steps}] '
            f'{current_depth:.3f} -> '
            f'{next_depth:.3f} m | '
            f'J7 {twist:+.1f} deg'
        )

        success = arm.move_tool_z_with_twist(
            distance=-step,
            twist_deg=twist,
            max_step=cartesian_step,
            velocity=velocity,
            acceleration=acceleration,
        )

        if not success:

            arm.get_logger().error(
                f'Outward stroke {i + 1} failed.'
            )

            return False

        current_depth = next_depth

        # Alternate:
        #
        # +150 -> -150 -> +150 ...
        outward_direction *= -1.0

        wait_between_movements()
        maybe_checkpoint_save()

    # =========================================================
    # 6. RETURN TO INTER
    # =========================================================

    arm.get_logger().info(
        '========== RETURN TO INTER =========='
    )

    success = go_to(arm, 'inter')

    if not success:

        arm.get_logger().error(
            'Return to INTER failed.'
        )

        return False

    wait_between_movements()

    # =========================================================
    # 7. COMPLETE
    # =========================================================

    if abs(current_depth) < 1e-9:
        current_depth = 0.0

    arm.get_logger().info(
        '========================================'
    )

    arm.get_logger().info(
        'SCAN COMPLETE'
    )

    arm.get_logger().info(
        f'Final nominal depth: '
        f'{current_depth:.4f} m'
    )

    if recorder is not None:

        # Blocking is fine (and necessary) here: the arm is already
        # stationary, and this is the last chance to persist this
        # scan's points before the caller shuts the node down.
        recorder.save_blocking()

        arm.get_logger().info(
            'Final scan checkpoint saved.'
        )

    arm.get_logger().info(
        '========================================'
    )

    return True


def _precession_end_scan(
    arm, plan, depth, pre_end_joints, phase_twist, time_scale
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

    if not run_plan(arm, plan, depth, time_scale=time_scale):
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
    entry_joints = list(BOTTOM)

    entry_joints[6] = (
        BOTTOM[6]
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

def add_scan_arguments(parser):
    """Add every scan() tuning option to an argparse parser."""
    parser.add_argument(
        '--end-scan',
        choices=END_SCANS,
        default=DEFAULT_END_SCAN,
        help=(
            "How to cover the closed end: 'precession' (default, the "
            "baked coning sweep) or 'bottom' (the old BOTTOM-pose "
            'detour).'
        ),
    )

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
        '--step',
        type=float,
        default=DEFAULT_STEP_M,
        help=(
            'Linear distance per scan stroke in metres '
            f'(default: {DEFAULT_STEP_M}).'
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
        default=DEFAULT_VELOCITY,
        help=(
            'MoveIt velocity scaling for the scan strokes '
            f'(default: {DEFAULT_VELOCITY}).'
        ),
    )

    parser.add_argument(
        '--acceleration',
        dest='scan_acceleration',
        type=float,
        default=DEFAULT_ACCELERATION,
        help=(
            'MoveIt acceleration scaling for the scan strokes '
            f'(default: {DEFAULT_ACCELERATION}).'
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
        'step': args.step,
        'sweep_deg': args.sweep,
        'velocity': args.scan_velocity,
        'acceleration': args.scan_acceleration,
        'rotation_velocity': args.rotation_velocity,
        'rotation_acceleration': args.rotation_acceleration,
        'cartesian_step': args.cartesian_step,
        'pause': args.pause,
        'save_interval': args.save_interval,
    }
