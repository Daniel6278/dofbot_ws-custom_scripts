#Test grab and lift block from central position

#!/usr/bin/env python3
from Arm_Lib import Arm_Device
import time
import sys

# Initialize the arm
arm = Arm_Device()
HOME_POSITION = [90, 90, 90, 90, 90, 90]

# MODIFIED CLEANUP FUNCTION:
def cleanup():
    try:
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)  # Increase movement time
        time.sleep(5)  # Extra buffer for completion
        print("End of cleanup movement")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        arm.Arm_serial_set_torque(0)  # Single torque disable point
        print("Arm safely deactivated")

def main():
    try:
        # Move individual servos
        # Parameters: (servo_id, angle, time_in_ms)
        # servo_id: 1-6 (each joint)
        # angle: 0-180 degrees (0-270 for servo 5)
        # time: how long to take for the movement
        
        # Reach down
        print("Reach Down")
        arm.Arm_serial_servo_write6(90, 45, 45, 45, 90, 90, 1500)
        time.sleep(3)
        
        # grab item
        print("Grab")
        arm.Arm_serial_servo_write6(90, 45, 45, 45, 90, 180, 1000)
        time.sleep(3)
        
        # Return to home position
        print("Lift")
        arm.Arm_serial_servo_write6(90, 90, 90, 0, 90, 90, 1000)
        time.sleep(3)
        print("End of main movement")
        # Move all servos at once
        # Parameters: (angle1, angle2, angle3, angle4, angle5, angle6, time_in_ms)
        #arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        #time.sleep(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup()  # Make this the ONLY cleanup handler
    sys.exit(0)
