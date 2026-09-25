#!/usr/bin/env python3

"""
Report where the flange's insertion axis (link7 local +Z) points right now.

Run it at INTER (`laundry move inter`, then `laundry check-flange`).

At INTER, local +Z is the scan's insertion axis: every tool-Z stroke
travels along it, and the ToF sensor sweeps around it. It should
point INTO the bucket along the bucket's own axis. The old version
of this check expected +Z to point straight down, which stopped
being true once the bucket was laid on its side - at the recorded
INTER, +Z is horizontal along -Y, so it would always have reported
"NOT VERTICAL".

What matters now is the angle between the insertion axis and the
bucket axis, and how far the sensor therefore drifts relative to the
bucket over a full-depth scan. The bucket axis is the one fitted
from baseline_scans/ (perception.bucket_model.fit_cone), falling back
to the configured bucket (config.OBSTACLES) when no baselines load.

Measured with the recorded INTER (fake-controller FK): 4.6 deg to the
old bucket-model seed, ~6 deg to the fitted axis (5.8-6.5 deg across runs, within
MoveIt's joint tolerance; the real bucket's axis rises ~7 deg toward
the mouth while INTER inserts horizontally), i.e. ~4.5cm of drift over
the 0.42m scan. That is not an error in itself - the baselines were
recorded along this same path, so the detector's model already
accounts for it - but re-recording INTER changes the path and
invalidates every baseline.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros

from .geometry import angle_between_deg, tool_z_from_quaternion
from .. import config
from ..perception.bucket_model import fit_cone, seed_axis_direction
from ..perception.detect import load_baseline_scans
from ..scan.pattern import DEFAULT_DEPTH_M

# Verdict bands on the sensor's drift relative to the bucket axis
# over a full-depth scan (scan.pattern.DEFAULT_DEPTH_M).
ALIGNED_DRIFT_M = 0.01
CLOSE_DRIFT_M = 0.03


def describe_alignment(tool_z, bucket_axis, depth_m=DEFAULT_DEPTH_M):
    """
    Return (misalignment_deg, elevation_deg, drift_m, verdict) for tool +Z.

    misalignment_deg is measured against the direction INTO the
    bucket, i.e. -bucket_axis (the axis points from the closed end
    toward the mouth). elevation_deg is +Z's angle above the
    horizontal plane. drift_m is how far the insertion line moves
    sideways relative to the bucket axis over depth_m of insertion.
    """
    into_bucket = -np.asarray(bucket_axis, dtype=np.float64)

    misalignment = angle_between_deg(tool_z, into_bucket)

    elevation = math.degrees(math.asin(max(-1.0, min(1.0, tool_z[2]))))

    drift = depth_m * math.sin(math.radians(misalignment))

    if drift < ALIGNED_DRIFT_M:
        verdict = 'ALIGNED: the scan runs along the bucket axis.'
    elif drift < CLOSE_DRIFT_M:
        verdict = 'CLOSE: the scan runs nearly along the bucket axis.'
    else:
        verdict = (
            'OFFSET: the scan drifts noticeably relative to the bucket '
            'axis. Fine as long as the baselines were recorded with this '
            'same INTER - changing INTER changes the path and invalidates '
            'them.'
        )

    return misalignment, elevation, drift, verdict


def bucket_axis_for_check():
    """Return (axis, source): the fitted bucket axis, else the configured one."""
    try:
        points = np.concatenate(load_baseline_scans(config.baseline_dir()))
        return fit_cone(points).axis_dir, f'fitted from {config.baseline_dir()}'
    except (OSError, ValueError):
        return seed_axis_direction(), 'configured bucket (no baselines loaded)'


class FlangeChecker(Node):

    def __init__(self):
        super().__init__('flange_checker')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def check(self):
        """Wait for TF, then print the insertion-axis report."""
        while not self.tf_buffer.can_transform(
            config.BASE_FRAME,
            config.FLANGE_LINK,
            rclpy.time.Time(),
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        tf = self.tf_buffer.lookup_transform(
            config.BASE_FRAME,
            config.FLANGE_LINK,
            rclpy.time.Time(),
        )

        zx, zy, zz = tool_z_from_quaternion(tf.transform.rotation)

        axis, source = bucket_axis_for_check()

        misalignment, elevation, drift, verdict = describe_alignment(
            (zx, zy, zz), axis
        )

        t = tf.transform.translation

        print()
        print(f'Flange position in {config.BASE_FRAME}:')
        print(f'  ({t.x:+.4f}, {t.y:+.4f}, {t.z:+.4f}) m')
        print()
        print(f'Flange/tool +Z (insertion axis) in {config.BASE_FRAME}:')
        print(f'  X = {zx:+.6f}')
        print(f'  Y = {zy:+.6f}')
        print(f'  Z = {zz:+.6f}')
        print()
        print(f'Elevation above horizontal = {elevation:+.3f} degrees')
        print(f'Angle to bucket axis ({source}) = {misalignment:.3f} degrees')
        print(
            f'Drift over a {DEFAULT_DEPTH_M:.2f} m scan = '
            f'{drift * 100.0:.1f} cm'
        )
        print()
        print(verdict)


def main():
    rclpy.init()

    node = FlangeChecker()

    try:
        node.check()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
