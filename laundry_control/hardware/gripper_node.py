#!/usr/bin/env python3

"""
Standalone node that owns the gripper's servo hardware (see servo.py).

Callers like the grasp stage never need direct GPIO access - the
same split already used for the ToF sensor (tof_sensor.py) and the
scan recorder (scan/recorder_node.py).

Services (std_srvs/Trigger):

    open_gripper  - move the servo to the `open_angle_deg` param.
    close_gripper - move the servo to the `close_angle_deg` param.

Which raw servo angle actually opens/closes the physical claw
depends on how it's linked to the servo horn - the defaults
(config.GRIPPER_OPEN_ANGLE_DEG / GRIPPER_CLOSE_ANGLE_DEG) are
calibrated for the current hardware; override via --ros-args if
that ever changes, rather than editing code:

    ros2 run laundry_control gripper_node --ros-args \\
        -p open_angle_deg:=20.0 -p close_angle_deg:=160.0

HARDWARE ONLY. Usage:
    ros2 run laundry_control gripper_node
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from ..config import GRIPPER_CLOSE_ANGLE_DEG, GRIPPER_OPEN_ANGLE_DEG
from .gripper_client import CLOSE_SERVICE, OPEN_SERVICE


class GripperNode(Node):

    def __init__(self):
        super().__init__('gripper_node')

        self.declare_parameter('open_angle_deg', GRIPPER_OPEN_ANGLE_DEG)
        self.declare_parameter('close_angle_deg', GRIPPER_CLOSE_ANGLE_DEG)

        self._open_srv = self.create_service(
            Trigger, OPEN_SERVICE, self._open_callback
        )

        self._close_srv = self.create_service(
            Trigger, CLOSE_SERVICE, self._close_callback
        )

        self.get_logger().info('gripper_node ready.')

    def _open_callback(self, request, response):
        return self._move_to(
            self.get_parameter('open_angle_deg').value, response
        )

    def _close_callback(self, request, response):
        return self._move_to(
            self.get_parameter('close_angle_deg').value, response
        )

    def _move_to(self, angle, response):
        try:
            # Inside the try: the first servo call is what actually
            # initialises the GPIO pin factory (see servo.py), so a
            # hardware/permission problem there must fail this one
            # service call, not crash the whole node.
            from .servo import set_servo_angle

            set_servo_angle(angle)

        except Exception as exc:
            response.success = False
            response.message = f'Failed to set servo to {angle} deg: {exc}'
            self.get_logger().error(response.message)
            return response

        response.success = True
        response.message = f'Servo set to {angle} deg.'
        return response


def main(args=None):
    rclpy.init(args=args)

    node = GripperNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
