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
bucket axis. The bucket axis used here is the URDF seed
(perception.bucket_model.seed_axis_direction), not a fitted one; the
cone fit printed by `laundry detect` says how far the real bucket
sits from that seed.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
import tf2_ros

from .. import config
from ..perception.bucket_model import seed_axis_direction
from .geometry import angle_between_deg, tool_z_from_quaternion

# Verdict bands for the insertion-vs-bucket-axis misalignment. At
# the full 0.42m insertion depth, 1 deg of misalignment moves the
# sensor ~7mm off the intended line; 5 deg moves it ~37mm, which is
# enough to change which part of the wall each beam lands on.
VERY_GOOD_DEG = 1.0
GOOD_DEG = 3.0
WARNING_DEG = 5.0


def describe_alignment(tool_z, bucket_axis):
    """
    Return (misalignment_deg, elevation_deg, verdict) for a tool +Z vector.

    misalignment_deg is measured against the direction INTO the
    bucket, i.e. -bucket_axis (the axis points from the closed end
    toward the mouth). elevation_deg is +Z's angle above the
    horizontal plane.
    """
    into_bucket = -np.asarray(bucket_axis, dtype=np.float64)

    misalignment = angle_between_deg(tool_z, into_bucket)

    elevation = math.degrees(math.asin(max(-1.0, min(1.0, tool_z[2]))))

    if misalignment < VERY_GOOD_DEG:
        verdict = 'VERY GOOD: insertion axis is along the bucket axis.'
    elif misalignment < GOOD_DEG:
        verdict = 'GOOD: close to the bucket axis.'
    elif misalignment < WARNING_DEG:
        verdict = 'WARNING: noticeable misalignment with the bucket axis.'
    else:
        verdict = (
            'MISALIGNED: INTER (or the bucket pose) should probably be '
            'adjusted.'
        )

    return misalignment, elevation, verdict


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

        misalignment, elevation, verdict = describe_alignment(
            (zx, zy, zz), seed_axis_direction()
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
        print(
            f'Angle to bucket axis (URDF seed) = {misalignment:.3f} degrees'
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
