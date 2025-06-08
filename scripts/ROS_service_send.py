#!/usr/bin/env python3
from Arm_Lib import Arm_Device
import rospy
from dofbot_msgs.srv import GrabStatus, GrabStatusResponse
import time
import sys
import signal
import atexit

class BlockPicker:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('block_picker', anonymous=True)
        
        # Initialize the arm
        self.arm = Arm_Device()
        
        # Create service server
        self.grab_service = rospy.Service('grab_status', GrabStatus, self.handle_grab_status)
        
        # Initialize grab status
        self.grab_success = False
        
        # Setup cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, lambda signum, frame: self.cleanup(True))
        signal.signal(signal.SIGTERM, lambda signum, frame: self.cleanup(True))
        
        # Enable torque
        self.arm.Arm_serial_set_torque(1)
        time.sleep(0.1)

    def handle_grab_status(self, req):
        """Service handler for grab status requests"""
        return GrabStatusResponse(self.grab_success)

    def check_grip_resistance(self):
        """Check if the gripper is experiencing resistance"""
        # Read current gripper position
        current_pos = self.arm.Arm_serial_servo_read(5)
        # If position is significantly different from target, there's resistance
        return current_pos < 170  # Adjust threshold as needed

    def pick_block(self):
        """Attempt to pick up the block"""
        try:
            # Reach down to block
            print("Reaching for block...")
            self.arm.Arm_serial_servo_write6(90, 90, 45, 45, 90, 90, 600)
            time.sleep(0.6)
            
            # Close gripper
            print("Closing gripper...")
            self.arm.Arm_serial_servo_write6(90, 90, 45, 45, 180, 90, 600)
            time.sleep(0.6)
            
            # Check for resistance
            self.grab_success = self.check_grip_resistance()
            
            if self.grab_success:
                print("Block successfully grabbed!")
                # Lift block
                self.arm.Arm_serial_servo_write6(90, 90, 90, 0, 180, 90, 600)
                time.sleep(0.6)
            else:
                print("Failed to grab block")
                # Return to home position
                self.home_position()
            
            return self.grab_success
            
        except Exception as e:
            print(f"Error during pick operation: {e}")
            return False

    def place_block(self):
        """Place the block down"""
        try:
            # Move to placement position
            print("Placing block...")
            self.arm.Arm_serial_servo_write6(90, 90, 45, 45, 180, 90, 600)
            time.sleep(0.6)
            
            # Open gripper
            self.arm.Arm_serial_servo_write6(90, 90, 45, 45, 90, 90, 600)
            time.sleep(0.6)
            
            # Return to home position
            self.home_position()
            return True
            
        except Exception as e:
            print(f"Error during place operation: {e}")
            return False

    def home_position(self):
        """Move to home position"""
        self.arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
        time.sleep(2)

    def cleanup(self, signal_received=False):
        """Cleanup function"""
        try:
            print("Starting cleanup...")
            self.home_position()
            self.arm.Arm_serial_set_torque(0)
            print("Cleanup completed")
            
            if signal_received:
                sys.exit(0)
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
            if signal_received:
                sys.exit(1)

def main():
    picker = BlockPicker()
    
    try:
        # Attempt to pick up the block
        if picker.pick_block():
            print("Waiting for second robot to acknowledge...")
            rospy.spin()  # Keep service running until shutdown
        else:
            print("Failed to pick up block")
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        picker.cleanup()

if __name__ == '__main__':
    main()

