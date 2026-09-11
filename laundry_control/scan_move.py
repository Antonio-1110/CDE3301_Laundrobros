#!/usr/bin/env python3

import argparse
import math
import time

import rclpy

from arm_position import INTER
from move import XArm7Controller
from scan_record import TofScanRecorder


def scan(
    arm,
    depth=0.45,
    step=0.05,
    sweep_deg=150.0,
    velocity=0.1,
    acceleration=0.1,
    rotation_velocity=0.5,
    rotation_acceleration=0.5,
    cartesian_step=0.005,
    pause=0.0,
    recorder=None,
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
            0.45 m

    step:
        Linear distance travelled during each scan stroke.

        Default:
            0.05 m

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
        MoveIt velocity scaling used for the two stationary
        (non-linear) J7 rotations: the initial offset and the
        turnaround phase shift. Since these involve no
        simultaneous linear motion they can run faster than
        the interleaved scan strokes.

        Default:
            0.5

    rotation_acceleration:
        MoveIt acceleration scaling for the stationary J7
        rotations.

        Default:
            0.5

    cartesian_step:
        Cartesian interpolation resolution passed to move.py.

    pause:
        Optional pause between motions in seconds.


    Scan pattern
    ------------

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

    Turnaround:

        Perform ONE stationary 150-degree rotation to move
        J7 to the opposite side of the sweep.

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
            "depth must be greater than zero."
        )
        return False

    if step <= 0.0:
        arm.get_logger().error(
            "step must be greater than zero."
        )
        return False

    if step > depth:
        arm.get_logger().error(
            "step cannot be greater than depth."
        )
        return False

    if sweep_deg <= 0.0:
        arm.get_logger().error(
            "sweep_deg must be greater than zero."
        )
        return False

    if not 0.0 < velocity <= 1.0:
        arm.get_logger().error(
            "velocity must be in the range (0, 1]."
        )
        return False

    if not 0.0 < acceleration <= 1.0:
        arm.get_logger().error(
            "acceleration must be in the range (0, 1]."
        )
        return False

    if not 0.0 < rotation_velocity <= 1.0:
        arm.get_logger().error(
            "rotation_velocity must be in the range (0, 1]."
        )
        return False

    if not 0.0 < rotation_acceleration <= 1.0:
        arm.get_logger().error(
            "rotation_acceleration must be in the range (0, 1]."
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
            "depth must currently be an exact multiple "
            "of step."
        )

        arm.get_logger().error(
            f"depth = {depth:.4f} m"
        )

        arm.get_logger().error(
            f"step  = {step:.4f} m"
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
        "========================================"
    )

    arm.get_logger().info(
        "XARM7 SCAN SEQUENCE"
    )

    arm.get_logger().info(
        f"Depth            : {depth:.3f} m"
    )

    arm.get_logger().info(
        f"Linear step      : {step:.3f} m"
    )

    arm.get_logger().info(
        f"Number of strokes: {number_of_steps}"
    )

    arm.get_logger().info(
        f"Wrist sweep      : {sweep_deg:.1f} deg"
    )

    arm.get_logger().info(
        f"Initial offset   : {-half_sweep:+.1f} deg"
    )

    arm.get_logger().info(
        "========================================"
    )

    # =========================================================
    # Helper
    # =========================================================

    def wait_between_movements():
        """
        Pause between motions.

        Spins the node (rather than a plain time.sleep) so that,
        if a recorder is attached, ToF readings keep being
        captured while the arm is stationary too.
        """

        if pause <= 0.0:
            return

        deadline = arm.get_clock().now() + rclpy.duration.Duration(
            seconds=pause
        )

        while arm.get_clock().now() < deadline:
            rclpy.spin_once(arm, timeout_sec=0.05)

    # =========================================================
    # 1. INITIAL J7 OFFSET
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
        "========== INITIAL OFFSET =========="
    )

    arm.get_logger().info(
        f"Stationary J7 rotation: "
        f"{-half_sweep:+.1f} deg"
    )

    success = arm.rotate_joint7(
        delta_deg=-half_sweep,
        velocity=rotation_velocity,
        acceleration=rotation_acceleration,
    )

    if not success:

        arm.get_logger().error(
            "Initial J7 positioning failed."
        )

        return False

    wait_between_movements()

    # =========================================================
    # 2. INWARD SCAN
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
        "========== INWARD SCAN =========="
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
            f"[IN {i + 1}/{number_of_steps}] "
            f"{current_depth:.3f} -> "
            f"{next_depth:.3f} m | "
            f"J7 {twist:+.1f} deg"
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
                f"Inward stroke {i + 1} failed."
            )

            return False

        current_depth = next_depth

        # Alternate:
        #
        # +150 -> -150 -> +150 ...
        inward_direction *= -1.0

        wait_between_movements()

    # =========================================================
    # Inward endpoint
    # =========================================================

    arm.get_logger().info(
        f"Maximum scan depth reached: "
        f"{current_depth:.3f} m"
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
        f"Nominal wrist orientation: "
        f"{nominal_end_angle:+.1f} deg"
    )

    # =========================================================
    # 3. TURNAROUND PHASE SHIFT
    #
    # ONE stationary rotation.
    #
    # If inward ended at:
    #
    #       +75
    #
    # move:
    #
    #       +75 -> -75
    #
    #       -150 deg
    #
    #
    # If inward ended at:
    #
    #       -75
    #
    # move:
    #
    #       -75 -> +75
    #
    #       +150 deg
    #
    #
    # This puts the outward scan on the opposite angular
    # phase from the inward scan.
    # =========================================================

    arm.get_logger().info(
        "========== TURNAROUND =========="
    )

    if nominal_end_angle > 0.0:

        phase_twist = -sweep_deg

    else:

        phase_twist = +sweep_deg

    arm.get_logger().info(
        f"Stationary phase shift: "
        f"{phase_twist:+.1f} deg"
    )

    success = arm.rotate_joint7(
        delta_deg=phase_twist,
        velocity=rotation_velocity,
        acceleration=rotation_acceleration,
    )

    if not success:

        arm.get_logger().error(
            "Turnaround phase shift failed."
        )

        return False

    wait_between_movements()

    # =========================================================
    # 4. OUTWARD SCAN
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
        "========== OUTWARD SCAN =========="
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
            f"[OUT {i + 1}/{number_of_steps}] "
            f"{current_depth:.3f} -> "
            f"{next_depth:.3f} m | "
            f"J7 {twist:+.1f} deg"
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
                f"Outward stroke {i + 1} failed."
            )

            return False

        current_depth = next_depth

        # Alternate:
        #
        # +150 -> -150 -> +150 ...
        outward_direction *= -1.0

        wait_between_movements()

    # =========================================================
    # 5. RETURN TO INTER
    # =========================================================

    arm.get_logger().info(
        "========== RETURN TO INTER =========="
    )

    success = arm.move_joints(INTER)

    if not success:

        arm.get_logger().error(
            "Return to INTER failed."
        )

        return False

    wait_between_movements()

    # =========================================================
    # 6. COMPLETE
    # =========================================================

    if abs(current_depth) < 1e-9:
        current_depth = 0.0

    arm.get_logger().info(
        "========================================"
    )

    arm.get_logger().info(
        "SCAN COMPLETE"
    )

    arm.get_logger().info(
        f"Final nominal depth: "
        f"{current_depth:.4f} m"
    )

    if recorder is not None:

        arm.get_logger().info(
            f"ToF points recorded: "
            f"{len(recorder.points_base_frame)}"
        )

    arm.get_logger().info(
        "========================================"
    )

    return True


# =============================================================
# TERMINAL INTERFACE
# =============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "Perform an xArm7 inward/outward scanning "
            "sequence using XArm7Controller."
        )
    )

    parser.add_argument(
        "--depth",
        type=float,
        default=0.45,
        help=(
            "Maximum insertion depth in metres "
            "(default: 0.45)."
        ),
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help=(
            "Linear distance per scan stroke in metres "
            "(default: 0.05)."
        ),
    )

    parser.add_argument(
        "--sweep",
        type=float,
        default=150.0,
        help=(
            "J7 rotation per scan stroke in degrees "
            "(default: 150)."
        ),
    )

    parser.add_argument(
        "--velocity",
        type=float,
        default=0.1,
        help=(
            "MoveIt velocity scaling "
            "(default: 0.1)."
        ),
    )

    parser.add_argument(
        "--acceleration",
        type=float,
        default=0.1,
        help=(
            "MoveIt acceleration scaling "
            "(default: 0.1)."
        ),
    )

    parser.add_argument(
        "--rotation-velocity",
        type=float,
        default=0.5,
        help=(
            "MoveIt velocity scaling for the stationary "
            "(non-linear) J7 rotations (default: 0.5)."
        ),
    )

    parser.add_argument(
        "--rotation-acceleration",
        type=float,
        default=0.5,
        help=(
            "MoveIt acceleration scaling for the stationary "
            "(non-linear) J7 rotations (default: 0.5)."
        ),
    )

    parser.add_argument(
        "--cartesian-step",
        type=float,
        default=0.005,
        help=(
            "MoveIt Cartesian interpolation step "
            "in metres (default: 0.005)."
        ),
    )

    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help=(
            "Optional pause between motions in seconds "
            "(default: 0)."
        ),
    )

    parser.add_argument(
        "--save-points",
        type=str,
        default=None,
        help=(
            "Optional CSV path to save recorded ToF scan "
            "points (base frame) to after the scan."
        ),
    )

    return parser


def main():

    args = build_parser().parse_args()

    rclpy.init()

    arm = XArm7Controller()
    recorder = TofScanRecorder(arm)

    success = False

    try:

        success = scan(
            arm=arm,
            depth=args.depth,
            step=args.step,
            sweep_deg=args.sweep,
            velocity=args.velocity,
            acceleration=args.acceleration,
            rotation_velocity=args.rotation_velocity,
            rotation_acceleration=args.rotation_acceleration,
            cartesian_step=args.cartesian_step,
            pause=args.pause,
            recorder=recorder,
        )

        if args.save_points:
            recorder.save_csv(args.save_points)

            arm.get_logger().info(
                f"Saved scan points to {args.save_points}"
            )

    except KeyboardInterrupt:

        arm.get_logger().warning(
            "Scan interrupted by user."
        )

        success = False

    finally:

        arm.destroy_node()
        rclpy.shutdown()

    raise SystemExit(
        0 if success else 1
    )


if __name__ == "__main__":
    main()