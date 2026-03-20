import rospy
from xarm_msgs.srv import Move, SetInt16, SetAxis

class LaundryArm:
    def __init__(self):
        rospy.init_node('laundry_commander')
        rospy.loginfo("Initializing Laundry Robot Control...")

        # 1. Setup Service Proxies
        self.move_joint = rospy.ServiceProxy('/xarm/move_joint', Move)
        self.set_state = rospy.ServiceProxy('/xarm/set_state', SetInt16)
        self.set_mode = rospy.ServiceProxy('/xarm/set_mode', SetInt16)
        self.clear_err = rospy.ServiceProxy('/xarm/clear_err', SetInt16)

    def ready_robot(self):
        """Clears errors and sets the robot to Position Mode (0)"""
        rospy.loginfo("Clearing errors and setting state...")
        self.clear_err(0)
        self.set_mode(0)
        self.set_state(0)
        rospy.sleep(1) # Give the hardware a second to breathe

    def move_to_laundry_bin(self):
        """Moves to a specific joint configuration"""
        # [J1, J2, J3, J4, J5, J6, J7] in Radians
        # Note: Be careful with J2 (the second 0). Don't let it hit 100% load!
        target_joints = [0.0, -0.2, 0.0, 0.5, 0.0, 0.4, 0.0]
        
        rospy.loginfo("Moving to Laundry Bin...")
        try:
            # Arguments: pose, vel, acc, time, radii
            response = self.move_joint(target_joints, 0.3, 5.0, 0, 0)
            if response.ret == 0:
                rospy.loginfo("Move Successful!")
            else:
                rospy.logerr("Move Failed with code: %s", response.ret)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

if __name__ == '__main__':
    try:
        robot = LaundryArm()
        robot.ready_robot()
        
        # Start with a safe nudge
        robot.move_to_laundry_bin()
        
    except rospy.ROSInterruptException:
        pass