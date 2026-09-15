#!/usr/bin/env python3

"""
gripper_node.py

Standalone node that owns the gripper's servo hardware (see
servo_motor.py), so callers like retrieve.py never need direct
GPIO access - the same split already used for the ToF
sensor (tof_sensor.py's ToFSensorNode) and the scan recorder
(scan_recorder_node.py).

Services (std_srvs/Trigger):

    open_gripper  - move the servo to the `open_angle_deg` param.
    close_gripper - move the servo to the `close_angle_deg` param.

Which raw servo angle actually opens/closes the physical claw
depends on how it's linked to the servo horn - defaults below
(45/100 deg) are calibrated for the current hardware; override via
--ros-args if that ever changes, rather than editing code:

    ros2 run laundry_control gripper_node --ros-args \\
        -p open_angle_deg:=20.0 -p close_angle_deg:=160.0

Usage:
    ros2 run laundry_control gripper_node
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class GripperNode(Node):

    def __init__(self):
        super().__init__("gripper_node")

        self.declare_parameter("open_angle_deg", 55.0)
        self.declare_parameter("close_angle_deg", 100.0)

        self._open_srv = self.create_service(
            Trigger, "open_gripper", self._open_callback
        )

        self._close_srv = self.create_service(
            Trigger, "close_gripper", self._close_callback
        )

        self.get_logger().info("gripper_node ready.")

    def _open_callback(self, request, response):
        return self._move_to(
            self.get_parameter("open_angle_deg").value, response
        )

    def _close_callback(self, request, response):
        return self._move_to(
            self.get_parameter("close_angle_deg").value, response
        )

    def _move_to(self, angle, response):

        try:
            # Imported here, not at module scope, so this node's
            # own class can still be imported/inspected without
            # requiring gpiozero/lgpio to be installed - same
            # reasoning tof_sensor.py documents for deferring its
            # own hardware imports into ToFSensor.__init__. Also
            # inside the try: importing servo_motor is what
            # actually initializes the GPIO pin factory
            # (Device.pin_factory = LGPIOFactory() runs at import
            # time), so a hardware/permission problem there must
            # fail this one service call, not crash the whole node.
            from .servo_motor import set_servo_angle

            set_servo_angle(angle)

        except Exception as exc:
            response.success = False
            response.message = f"Failed to set servo to {angle} deg: {exc}"
            self.get_logger().error(response.message)
            return response

        response.success = True
        response.message = f"Servo set to {angle} deg."
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


if __name__ == "__main__":
    main()
