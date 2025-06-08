#!/usr/bin/env python3
from Arm_Lib import Arm_Device
import time
import sys
import signal
import atexit

# Initialize the arm
arm = Arm_Device()

def cleanup():
    """Cleanup function to ensure safe shutdown"""
    try:
        # Disable torque before exiting
        arm.Arm_serial_set_torque(0)
        print("Torque disabled")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def calibrate_servo(servo_id):
    """
    Calibrate a single servo
    Args:
        servo_id: The ID of the servo to calibrate (1-6)
    """
    print(f"\nCalibrating servo {servo_id}")
    print("1. First, disable torque to allow manual movement")
    arm.Arm_serial_set_torque(0)
    time.sleep(0.1)
    
    print(f"2. Manually move servo {servo_id} to its center position")
    print("   The arm should be in a natural, upright position")
    input("Press Enter when ready to proceed...")
    
    print("3. Enabling torque and saving calibration")
    arm.Arm_serial_set_torque(1)
    time.sleep(0.1)
    
    # Save the calibration for this servo
    arm.Arm_serial_servo_write_offset_switch(servo_id)
    time.sleep(0.1)
    
    # Check calibration status
    status = arm.Arm_serial_servo_write_offset_state()
    if status == 1:
        print(f"Servo {servo_id} calibration successful!")
    elif status == 2:
        print(f"Warning: Servo {servo_id} calibration value out of range")
    else:
        print(f"Error: Servo {servo_id} not detected")

def main():
    try:
        print("=== DOFBOT Robot Arm Calibration ===")
        print("This script will help you calibrate each servo of the robot arm.")
        print("IMPORTANT: Follow these steps carefully!")
        print("1. The robot should be powered on and connected")
        print("2. Each servo will be calibrated one at a time")
        print("3. You will need to manually position each servo")
        print("4. The arm should be in a natural, upright position when calibrating")
        print("\nWARNING: This process will clear all previous calibration values!")
        
        input("\nPress Enter to begin calibration (or Ctrl+C to cancel)...")
        
        # First, clear all previous calibrations
        print("\nClearing all previous calibrations...")
        arm.Arm_serial_servo_write_offset_switch(0)
        time.sleep(0.1)
        
        # Calibrate each servo one by one
        for servo_id in range(1, 7):
            calibrate_servo(servo_id)
        
        print("\nCalibration complete!")
        print("Testing final positions...")
        
        # Test the calibration by moving to home position
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
        time.sleep(2)
        
        print("\nIf the arm is not in the correct position, you may need to:")
        print("1. Run this calibration script again")
        print("2. Check for mechanical issues")
        print("3. Ensure all servos are properly connected")
        
    except KeyboardInterrupt:
        print("\nCalibration cancelled by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cleanup()

if __name__ == "__main__":
    # Register cleanup handlers
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda signum, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    
    main() 