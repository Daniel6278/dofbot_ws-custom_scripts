#!/usr/bin/env python3
from Arm_Lib import Arm_Device
import time
import os
import subprocess

def kill_existing_processes():
    # Kill any existing arm processes
    try:
        subprocess.run(['pkill', '-f', 'YahboomArm.py'], check=False)
        time.sleep(1)  # Wait for processes to be killed
    except Exception as e:
        print(f"Error killing processes: {e}")

def reset_i2c():
    try:
        # Reset I2C bus (you might need sudo privileges for this)
        os.system('sudo i2cdetect -y 7')  # This can help reset the I2C bus
        time.sleep(0.5)
    except Exception as e:
        print(f"Error resetting I2C: {e}")

def reset_arm():
    kill_existing_processes()
    reset_i2c()
    
    try:
        arm = Arm_Device()
        # Move to home position
        arm.Arm_serial_servo_write6(90, 90, 90, 90, 90, 90, 1500)
        time.sleep(1.5)
        # Disable torque
        arm.Arm_serial_set_torque(0)
        # Explicitly close the I2C bus
        if hasattr(arm, 'bus'):
            arm.bus.close()
        print("Arm reset completed successfully")
    except Exception as e:
        print(f"Error during arm reset: {e}")
    finally:
        # Force close any remaining connections
        try:
            if 'arm' in locals() and hasattr(arm, 'bus'):
                arm.bus.close()
        except:
            pass

if __name__ == "__main__":
    reset_arm()