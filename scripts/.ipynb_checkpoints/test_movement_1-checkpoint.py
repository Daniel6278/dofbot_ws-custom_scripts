#!/usr/bin/env python3
from Arm_Lib import Arm_Device
import time

# Initialize the arm
arm = Arm_Device()

def main():
    try:
        # Move individual servos
        # Parameters: (servo_id, angle, time_in_ms)
        # servo_id: 1-6 (each joint)
        # angle: 0-180 degrees (0-270 for servo 5)
        # time: how long to take for the movement
        
        # Move joints
        print("Moving Servo 1 (ROTATE BASE)")
        arm.Arm_serial_servo_write(1, 90, 1000)
        time.sleep(2)

        print("Moving Servo 2 (LOWEST SERVO)")
        arm.Arm_serial_servo_write(2, 45, 1000)
        time.sleep(2)

        print("Moving Servo 3 (MIDDLE SERVO)")
        arm.Arm_serial_servo_write(3, 45, 1000)
        time.sleep(2)

        print("Moving Servo 4 (TOP SERVO)")
        arm.Arm_serial_servo_write(4, 45, 1000)
        time.sleep(2)

        print("Moving Servo 5 (WRIST ROTATION)")
        arm.Arm_serial_servo_write(5, 45, 1000)
        time.sleep(2)

        print("Moving Servo 6 (CLAMP)")
        arm.Arm_serial_servo_write(6, 45, 1000)
        time.sleep(2)
        
        # Move all servos at once
        # Parameters: (angle1, angle2, angle3, angle4, angle5, angle6, time_in_ms)
        #arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        #time.sleep(1)
        
        # Return to home position
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1000)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # You might want to relax the servos when done
        arm.Arm_serial_set_torque(0)

if __name__ == "__main__":
    main()