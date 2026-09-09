#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import tf2_ros


class FlangeChecker(Node):

    def __init__(self):
        super().__init__("flange_checker")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

    def check(self):

        # Wait until TF becomes available
        while not self.tf_buffer.can_transform(
            "link_base",
            "link7",
            rclpy.time.Time(),
        ):
            rclpy.spin_once(self, timeout_sec=0.1)

        tf = self.tf_buffer.lookup_transform(
            "link_base",
            "link7",
            rclpy.time.Time(),
        )

        q = tf.transform.rotation

        # Tool local +Z axis expressed in link_base coordinates.
        zx = 2.0 * (q.x * q.z + q.w * q.y)
        zy = 2.0 * (q.y * q.z - q.w * q.x)
        zz = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)

        # Normalize
        length = math.sqrt(
            zx * zx +
            zy * zy +
            zz * zz
        )

        zx /= length
        zy /= length
        zz /= length

        # Angle from vertical.
        #
        # abs() means both +Z upward and +Z downward
        # count as vertical.
        vertical = max(-1.0, min(1.0, abs(zz)))

        tilt = math.degrees(
            math.acos(vertical)
        )

        print()
        print("Flange/tool +Z vector in link_base:")
        print(
            f"  X = {zx:+.6f}"
        )
        print(
            f"  Y = {zy:+.6f}"
        )
        print(
            f"  Z = {zz:+.6f}"
        )

        print()
        print(
            f"Tilt from vertical = {tilt:.3f} degrees"
        )

        if zz < 0:
            print("Tool +Z points mostly DOWN.")
        else:
            print("Tool +Z points mostly UP.")

        print()

        if tilt < 1.0:
            print("VERY GOOD: essentially vertical.")

        elif tilt < 3.0:
            print("GOOD: close to vertical.")

        elif tilt < 5.0:
            print("WARNING: noticeable tilt.")

        else:
            print("NOT VERTICAL: INTER should probably be adjusted.")


def main():

    rclpy.init()

    node = FlangeChecker()

    try:
        node.check()

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()