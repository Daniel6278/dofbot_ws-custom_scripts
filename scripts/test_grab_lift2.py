#!/usr/bin/env python3
from Arm_Lib import Arm_Device
import time
import sys
import signal
import atexit

# Initialize the arm
arm = Arm_Device()
HOME_POSITION = [90, 90, 90, 90, 90, 90]

# Flag to prevent multiple cleanups
cleanup_done = False

def release_bus():
    """Explicitly release the I2C bus"""
    try:
        if hasattr(arm, 'bus'):
            arm.bus.close()
        print("I2C bus released")
    except Exception as e:
        print(f"Error releasing I2C bus: {e}")

def cleanup(signal_received=False):
    """Enhanced cleanup function with guard against multiple calls"""
    global cleanup_done
    if cleanup_done:
        return
    
    try:
        print("Starting cleanup...")
        # First move to home position
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
        time.sleep(2)  # Wait for movement
        
        # Then disable torque
        arm.Arm_serial_set_torque(0)
        print("Torque disabled")
        
        # Finally release the I2C bus
        release_bus()
        
        cleanup_done = True  # Mark cleanup as done
        print("Cleanup completed")
        
        if signal_received:
            sys.exit(0)
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
        cleanup_done = True  # Mark as done even if there was an error
        if signal_received:
            sys.exit(1)

def grab_set_position():
    arm.Arm_serial_servo_write6(90, 180, 0, 0, 90, 180, 2000)
    time.sleep(3)

def close_grip():
    #TODO
def open_grip():
    #TODO
def down_position():
    #TODO
    
def grab_middle():
    print("Grabbing middle, white square")
    grab_set_position()
    arm.Arm_serial_servo_write6(90, 90, 0, 0, 0, 90, 2000)
    time.sleep(3)
    arm.Arm_serial_servo_write6(90, 70, 0, 70, 90, 90, 2000)
    time.sleep(5)
    print("grab_middle finished")
    

def home_position():
    arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
    time.sleep(3)
    

# Register the cleanup function for both normal exit and signals
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda signum, frame: cleanup(True))
signal.signal(signal.SIGTERM, lambda signum, frame: cleanup(True))


###################################################################################
###################################################################################

def main():
    try:
        # Ensure torque is enabled at start
        # arm.Arm_serial_set_torque(1)
        # time.sleep(0.1)

        # # # Return to home position
        # arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 2000)
        time.sleep(3)
        
        # Reach down
        # print("Reach Down")
        # arm.Arm_serial_servo_write6(90, 90, 45, 45, 90, 90, 2000)
        # time.sleep(3)
        
        # # grab item
        # print("Grab")
        # arm.Arm_serial_servo_write6(90, 90, 20, 0, 90, 90, 2000)
        # time.sleep(3)
        # arm.Arm_serial_servo_write6(90, 60, 20, 0, 90, 90, 2000)
        # time.sleep(3)
        # arm.Arm_serial_servo_write6(90, 60, 20, 0, 90, 120, 2000)
        # time.sleep(3)

        # release_bus()
        # print("\n1. Resetting torque state...")
        # arm.Arm_serial_set_torque(0)
        # time.sleep(1)

        print("\nEnabling torque...")
        arm.Arm_serial_set_torque(1)
        time.sleep(1)
        
        home_position()
        grab_middle()
        home_position()
        
        
    except Exception as e:
        print(f"An error occurred in main: {e}")
        raise  # Re-raise the exception to trigger cleanup
#####################################################################################
#####################################################################################

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if not cleanup_done:  # Only call cleanup if it hasn't been done yet
            cleanup()